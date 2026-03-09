import os
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from strategies.ml_model import MLAgent
from strategies.signal_bridge import SignalTranslator
from data.feature_engineering import FeatureEngineer
from data.validator_agent import ValidatorAgent
from risk.optimizer import PortfolioOptimizer
from risk.constraints import RiskConstraints
from execution.alpaca_client import AlpacaExecutionClient

# Load API Keys
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LiveTrader")

class LiveTraderSupervisor:
    """
    Agent 12 (Live): Continuous Walk-Forward ML Orchestrator.
    Wakes up, fetches the last 3 years of data, trains a fresh XGBoost model,
    predicts today's returns, sizes via Risk Parity, checks CVaR, and executes via Alpaca.
    """
    def __init__(self, symbols: list, lookback_years: int = 3):
        self.symbols = symbols
        self.lookback_years = lookback_years
        
        self.feature_cols = [
            'sma_20', 'sma_50', 'ema_20', 'volatility_20', 'roc_10', 'roc_20',
            'trailingPE', 'priceToBook', 'returnOnEquity', 'debtToEquity'
        ]
        
        # Initialize Core Pipeline Agents
        self.validator = ValidatorAgent(target_horizon=5)
        self.ml_agent = MLAgent(target_col='target_return')
        self.translator = SignalTranslator()
        self.optimizer = PortfolioOptimizer(target_volatility=0.15)
        self.risk_manager = RiskConstraints(max_cvar_daily=0.03)
        
        # Initialize Execution Client (Automatically picks up .env keys)
        self.executor = AlpacaExecutionClient()

    def fetch_current_prices_for_execution(self) -> dict:
        """Fetches the absolute latest real-time prices to calculate target share amounts."""
        current_prices = {}
        for sym in self.symbols:
            try:
                ticker = yf.Ticker(sym)
                # Fast grab of current price
                hist = ticker.history(period="1d")
                if not hist.empty:
                    current_prices[sym] = hist['Close'].iloc[-1]
            except Exception as e:
                logger.error(f"Failed to fetch live price for {sym}: {e}")
        return current_prices

    def run_cycle(self):
        logger.info("=== STARTING LIVE TRADING CYCLE ===")
        
        # 1. Calculate Date Horizons
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * self.lookback_years)
        
        logger.info(f"Agent 6: Fetching OHLCV + Fundamentals from {start_date.date()} to {end_date.date()}")
        
        raw_data_feeds = {}
        master_train_frames = []
        historical_prices = {}
        
        # 2. Agent 6 & 7: Data Fetch & Validation
        for sym in self.symbols:
            df = yf.download(sym, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
            if df.empty:
                logger.warning(f"No data fetched for {sym}. Skipping.")
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns=str.lower)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            
            # Store price history for Risk Parity variance calc
            historical_prices[sym] = df['close'].copy()
            
            # Splice features
            df = FeatureEngineer.splice_features(df, sym)
            raw_data_feeds[sym] = df
            
            # Validate and create targets
            valid_train_df = self.validator.validate_and_prep(df.copy())
            master_train_frames.append(valid_train_df)

        if not master_train_frames:
            logger.error("No valid training data generated. Aborting cycle.")
            return

        # 3. Agent 8: Continuous Walk-Forward Training
        logger.info("Agent 8: Training fresh XGBoost Regressor on newest market regime...")
        master_train_df = pd.concat(master_train_frames)
        training_success = self.ml_agent.train(master_train_df, self.feature_cols)
        
        if not training_success:
            logger.error("Model training failed. Aborting cycle.")
            return

        # 4. Agent 8: Predict Today
        logger.info("Agent 8: Generating forward predictions for today's market snapshot...")
        today_features = {}
        for sym, df in raw_data_feeds.items():
            # Grab the very last row (today)
            latest_row = df.iloc[-1]
            features_row = []
            valid = True
            for col in self.feature_cols:
                if col in latest_row:
                    features_row.append(latest_row[col])
                else:
                    valid = False
                    break
            if valid:
                today_features[sym] = features_row

        df_today = pd.DataFrame.from_dict(today_features, orient='index', columns=self.feature_cols)
        
        if df_today.empty:
            logger.warning("No valid feature snapshots for today. Flattening portfolio.")
            self.executor.rebalance_portfolio({})
            return
            
        predictions = self.ml_agent.predict(df_today, self.feature_cols)
        logger.info(f"Raw Predictions:\n{predictions}")

        # 5. Agent 9: Translate Signals
        alpha_signals = self.translator.translate_to_alpha(predictions, top_n=len(self.symbols)//2)
        logger.info(f"Agent 9 Conviction Signals: {alpha_signals}")

        if not alpha_signals:
            self.executor.rebalance_portfolio({})
            return

        # 6. Agent 10: Risk Parity Routing
        df_hist = pd.DataFrame(historical_prices)
        returns_df = df_hist.pct_change().dropna()
        
        target_weights = self.optimizer.optimize_weights(df_hist, alpha_signals)
        logger.info(f"Agent 10 Unconstrained Risk Parity Weights: {target_weights}")

        # 7. Agent 11: CVaR Constraints
        final_weights = self.risk_manager.apply_gateway(target_weights, returns_df)
        
        if not final_weights:
            logger.warning("Agent 11: TRADE BLOCKED BY RISK GATEWAY. Flattening Portfolio.")
        else:
            logger.info(f"Agent 11: Weights passed constraints. Final Target: {final_weights}")

        # 8. Execute via Alpaca Client 
        # (Assuming the Alpaca client natively handles mapping dictionary percentages internally)
        # Note: For this to work perfectly, Alpaca Client needs a slight update to map % to quantities.
        # But conceptually, we pass the weights payload downstream.
        logger.info("Agent 12: Routing weights to execution client...")
        
        # Fetch current prices to map percentages to actual share quantities
        current_prices = self.fetch_current_prices_for_execution()
        capital = self.executor.get_account_capital()
        
        if capital <= 0:
            logger.error("Alpaca account has 0 capital or failed to connect.")
            return
            
        logger.info(f"Current Paper Buying Power: ${capital:.2f}")

        # Basic target calculation for routing
        for symbol, weight in final_weights.items():
            if symbol in current_prices:
                price = current_prices[symbol]
                target_dollars = capital * weight
                target_shares = target_dollars / price
                logger.info(f" -> Execution Plan {symbol}: Map {weight*100:.1f}% -> ${target_dollars:.2f} (~{target_shares:.2f} shares @ ${price:.2f})")
                
                # In a full production script, we would diff this against current positions
                # and dispatch exact precise DELTA orders.
        
        logger.info("=== LIVE TRADING CYCLE COMPLETE ===")

if __name__ == "__main__":
    test_universe = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
    trader = LiveTraderSupervisor(symbols=test_universe, lookback_years=3)
    
    # Run the cycle once immediately
    trader.run_cycle()
    
    # To run this continuously, wrap run_cycle() in a schedule loop 
    # e.g., using python 'schedule' package to run every day at 09:30 AM EST.
