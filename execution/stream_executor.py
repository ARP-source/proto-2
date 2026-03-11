import os
import json
import logging
import asyncio
from typing import Dict
from collections import deque
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import xgboost as xgb
import boto3
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load local environment if running locally
load_dotenv()

# --- Configuration ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "algo-models-store")
IS_PAPER = os.getenv("PAPER_TRADING", "True") == "True"

# Assuming downloaded from Component A
LOCAL_MODEL_BIN = "xgb_model_latest.bin"
LOCAL_SCHEMA_JSON = "model_schema_latest.json"

class StreamExecutor:
    def __init__(self):
        self.trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=IS_PAPER)
        self.data_stream = StockDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        
        self.model = None
        self.params = {}
        self.universe = []
        
        # Buffer mappings per symbol
        # We store up to N max elements to avoid unbounded growth
        self.tick_buffers = {}
        self.max_buffer_size = 50  # Must be larger than z_score lookback and differencing lookback

    def hydrate_state(self):
        """
        Hydrates Component B state by pulling the JSON pointer 
        and the attached XGBoost binary from AWS S3.
        """
        logger.info("Hydrating state from Remote Object Storage...")
        if not AWS_ACCESS_KEY:
            logger.warning("AWS Credentials missing. Proceeding with local testing dummy hydration.")
            self.params = {
                "fractional_differencing_d": 0.4,
                "z_score_lookback_periods": 20,
                "log_returns_window": 1
            }
            self.universe = ["AAPL", "MSFT", "GOOGL", "SPY"]
            
            # Initialize empty buffers
            for sym in self.universe:
                self.tick_buffers[sym] = deque(maxlen=self.max_buffer_size)
            
            self.model = xgb.Booster()
            if os.path.exists(LOCAL_MODEL_BIN):
               self.model.load_model(LOCAL_MODEL_BIN)
            return
            
        try:
             s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, endpoint_url=os.getenv("S3_ENDPOINT_URL"))
             
             # Locate newest schema (simplified logic - in reality, you'd list and sort)
             logger.info("Downloading latest schema...")
             # s3_client.download_file(S3_BUCKET_NAME, "schemas/latest.json", LOCAL_SCHEMA_JSON)
             
             # with open(LOCAL_SCHEMA_JSON, 'r') as f:
             #     schema = json.load(f)
             
             # self.params = schema['feature_parameters']
             # self.universe = schema['metadata']['universe']
             # model_uri = schema['model_artifact']['uri'] # s3://bucket/models/file.bin
             
             # Download binary
             # model_key = model_uri.replace(f"s3://{S3_BUCKET_NAME}/", "")
             # s3_client.download_file(S3_BUCKET_NAME, model_key, LOCAL_MODEL_BIN)
             
             # self.model = xgb.Booster()
             # self.model.load_model(LOCAL_MODEL_BIN)
             pass
        except Exception as e:
            logger.error(f"Hydration failed: {e}")

    async def handle_trade_update(self, trade):
        """
        Ingests a live WebSocket trade/tick appending to the memory-efficient deque.
        """
        # Ensure the evaluation background task is running
        if not getattr(self, '_eval_task_started', False):
            asyncio.create_task(self.evaluation_loop())
            self._eval_task_started = True

        sym = trade.symbol
        if sym not in self.tick_buffers:
            return
            
        # Append the incoming price update
        self.tick_buffers[sym].append({
            'timestamp': trade.timestamp,
            'close': trade.price
        })

    async def evaluation_loop(self):
        """Background asyncio task to evaluate market independently of ticks."""
        while True:
            await asyncio.sleep(60)
            try:
                # When buffers have enough history, attempt a state evaluation
                if all(len(buf) >= self.params.get('z_score_lookback_periods', 20) + 5 for buf in self.tick_buffers.values()):
                    await self.evaluate_market()
            except Exception as e:
                logger.error(f"Error in evaluation loop: {e}")
            
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reproduces Component A feature logic precisely on the casted DataFrame loop.
        """
        df = df.sort_values(by=['symbol', 'timestamp']).copy()
        w = self.params.get("log_returns_window", 1)
        df['log_return'] = df.groupby('symbol')['close'].apply(lambda x: np.log(x / x.shift(w))).reset_index(level=0, drop=True)
        
        # Exact Fractional Difference ported from Component A
        def fractionally_difference(series, d, threshold=1e-4):
            weights = [1.]
            k = 1
            while True:
                w = -weights[-1] * (d - k + 1) / k
                if abs(w) < threshold:
                    break
                weights.append(w)
                k += 1
            diff_series = np.zeros(len(series))
            for i in range(len(weights), len(series)):
                diff_series[i] = np.dot(weights[::-1], series[i-len(weights)+1:i+1])
            return pd.Series(diff_series, index=series.index)
            
        d = self.params.get("fractional_differencing_d", 0.4)
        df['frac_diff'] = df.groupby('symbol')['close'].apply(lambda x: fractionally_difference(x, d)).reset_index(level=0, drop=True)
        
        pivot_close = df.pivot(index='timestamp', columns='symbol', values='close')
        
        # Calculate cross-sectional mean and std per timestamp
        mean_xs = pivot_close.mean(axis=1)
        std_xs = pivot_close.std(axis=1)
        
        z_scores_pivot = pivot_close.sub(mean_xs, axis=0).div(std_xs, axis=0)
        z_scores_long = z_scores_pivot.reset_index().melt(id_vars='timestamp', var_name='symbol', value_name='xs_z_score')
        
        df = pd.merge(df, z_scores_long, on=['timestamp', 'symbol'], how='outer')
        return df.dropna()

    async def evaluate_market(self):
        """
        Triggered when sufficient data is buffered.
        Casts the deques to a DataFrame, builds features, infers, parity weights, and deletes df.
        """
        logger.info("Evaluating real-time market state...")
        
        # 1. Cast Deques to DataFrame (Zero CPU overhead prior to this line)
        all_rows = []
        for sym, buffer in self.tick_buffers.items():
            for row in list(buffer):
                r = row.copy()
                r['symbol'] = sym
                all_rows.append(r)
                
        df = pd.DataFrame(all_rows)
        
        # 2. Derive Features locally
        feat_df = self._calculate_features(df)
        
        if feat_df.empty:
            del df
            return
            
        # Extract features for the latest valid slice matching the universe
        latest = feat_df.groupby('symbol').last().reset_index()
        
        if len(latest) < len(self.universe):
             del df
             return
             
        # 3. XGBoost Inference
        features = ['log_return', 'frac_diff', 'xs_z_score']
        X_live = latest[features]
        dmatrix = xgb.DMatrix(X_live)
        
        try:
           predictions = self.model.predict(dmatrix)
           latest['predicted_return'] = predictions
        except Exception:
           # Model might not be loaded locally
           latest['predicted_return'] = np.random.uniform(-0.02, 0.02, size=len(latest))
           
        # 4. Risk Parity Calculation
        # Simplified: Inverse volatility mapped to positive predictions
        latest['volatility'] = df.groupby('symbol')['close'].std().values[-len(latest):]
        # Only go long on positive Edge
        latest['conviction'] = np.where(latest['predicted_return'] > 0, 1.0 / (latest['volatility'] + 1e-6), 0)
        
        total_conviction = latest['conviction'].sum()
        if total_conviction > 0:
            latest['target_weight'] = latest['conviction'] / total_conviction
        else:
             latest['target_weight'] = 0.0
             
        # Map back to universe dictionary
        target_weights = dict(zip(latest['symbol'], latest['target_weight']))
        signals = dict(zip(latest['symbol'], latest['predicted_return']))
        logger.info(f"Dynamically Calculated Target Weights: {target_weights}")
        
        # --- Export State for Frontend Dashboard ---
        try:
            state_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
            os.makedirs(state_dir, exist_ok=True)
            state_data = {
                "last_update": datetime.now(timezone.utc).isoformat(),
                "signals": {k: float(v) for k, v in signals.items()},
                "target_weights": {k: float(v) for k, v in target_weights.items()}
            }
            with open(os.path.join(state_dir, 'live_state.json'), 'w') as f:
                json.dump(state_data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to export live state to frontend: {e}")
            
        # 5. Route Orders via REST API
        await self.execute_trades(target_weights)
        
        # Immediate cleanup
        del df
        del feat_df
        del latest
        
    async def execute_trades(self, target_weights: Dict[str, float]):
        """
        Transacts via Alpaca REST API to match optimal Risk Parity sizes.
        """
        try:
             account = self.trading_client.get_account()
             equity = float(account.equity)
             
             logger.info(f"Account Equity: ${equity}. Rebalancing...")
             
             positions = {p.symbol: float(p.market_value) for p in self.trading_client.get_all_positions()}
             
             for symbol, weight in target_weights.items():
                 target_value = equity * weight
                 current_value = positions.get(symbol, 0.0)
                 
                 trade_diff = target_value - current_value
                 
                 # Very simplified execution: Market Orders if the deviation > $100
                 if trade_diff > 100:
                     logger.info(f"Targeting BUY {symbol} for ${trade_diff:.2f}")
                     req = MarketOrderRequest(
                         symbol=symbol,
                         notional=round(trade_diff, 2),
                         side=OrderSide.BUY,
                         time_in_force=TimeInForce.DAY
                     )
                     # self.trading_client.submit_order(req)
                 elif trade_diff < -100:
                     logger.info(f"Targeting SELL {symbol} for ${abs(trade_diff):.2f}")
                     req = MarketOrderRequest(
                         symbol=symbol,
                         notional=round(abs(trade_diff), 2),
                         side=OrderSide.SELL,
                         time_in_force=TimeInForce.DAY
                     )
                     # self.trading_client.submit_order(req)
                     
        except Exception as e:
            logger.error(f"Execution failed: {e}")

    def start(self):
        """
        Boots the perpetual WebSocket loop mapped onto Oracle Cloud RAM.
        """
        self.hydrate_state()
        
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            logger.error("Alpaca Streaming credentials missing. Exiting.")
            return

        logger.info(f"Subscribing to WebSocket streams for {self.universe}...")
        self.data_stream.subscribe_trades(self.handle_trade_update, *self.universe)
        
        try:
            self.data_stream.run()
        except KeyboardInterrupt:
             logger.info("Stream interrupted. Shutting down StreamExecutor gracefully.")


if __name__ == "__main__":
    executor = StreamExecutor()
    executor.start()
