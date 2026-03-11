import os
import time
import logging
from typing import Dict
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlpacaExecutionClient:
    """
    Handles live/paper order execution via Alpaca REST API.
    Implements a basic TWAP slice execution logic for routing large trades.
    """
    def __init__(self, api_key: str = None, api_secret: str = None, paper: bool = True):
        self.api_key = api_key or os.getenv("APCA_API_KEY_ID", "")
        self.api_secret = api_secret or os.getenv("APCA_API_SECRET_KEY", "")
        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=paper)

    def get_account_capital(self) -> float:
        """Returns the current buying power / equity."""
        try:
            account = self.trading_client.get_account()
            return float(account.equity)
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return 0.0

    def execute_twap(self, symbol: str, total_qty: float, side: OrderSide, slices: int = 5, interval_seconds: int = 10):
        """
        Executes a basic Time-Weighted Average Price (TWAP) algorithm to slice
        a large order into smaller child orders.
        """
        if total_qty <= 0 or slices <= 0:
            return

        slice_qty = total_qty / slices
        
        logger.info(f"Starting TWAP for {symbol}: {total_qty} units over {slices} slices every {interval_seconds}s")
        
        for i in range(slices):
            # For the last slice, execute whatever remains to avoid rounding errors
            if i == slices - 1:
                qty_to_execute = total_qty - (slice_qty * i)
            else:
                qty_to_execute = slice_qty
                
            qty_to_execute = max(0.0001, round(qty_to_execute, 4)) # Handle fractional minimums
                
            try:
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty_to_execute,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
                self.trading_client.submit_order(order_data=order_data)
                logger.info(f"[{i+1}/{slices}] Executed TWAP slice: {qty_to_execute} {symbol}")
            except Exception as e:
                logger.error(f"TWAP slice failed for {symbol}: {e}")
                
            if i < slices - 1:
                time.sleep(interval_seconds)

    def rebalance_portfolio(self, target_weights: Dict[str, float], current_prices: Dict[str, float] = None):
        """
        Takes target constrained weights from the Risk Manager and executes them.
        Diffs against current positions and dispatches delta orders.
        """
        if not target_weights:
            logger.info("Empty target weights provided. Flattening portfolio.")
            self.trading_client.close_all_positions(cancel_orders=True)
            return
            
        capital = self.get_account_capital()
        if capital <= 0:
            logger.error("No capital available for rebalance.")
            return
            
        # Get current positions
        try:
            current_positions = {p.symbol: float(p.qty) for p in self.trading_client.get_all_positions()}
        except Exception as e:
            logger.error(f"Failed to fetch open positions: {e}")
            current_positions = {}
            
        logger.info(f"Target rebalance weights: {target_weights}")
        
        if not current_prices:
            logger.error("Current prices must be provided to map weights to quantities.")
            return

        LARGE_ORDER_THRESHOLD = 500  # Example threshold for TWAP (shares)

        for symbol, weight in target_weights.items():
            if symbol not in current_prices:
                logger.warning(f"No current price for {symbol}, safely skipping.")
                continue

            current_price = current_prices[symbol]
            target_notional = capital * weight
            target_qty = int(target_notional / current_price) # Alpaca prefers whole shares or standard fractions
            
            current_qty = current_positions.get(symbol, 0.0)
            delta_qty = target_qty - current_qty
            
            if abs(delta_qty) < 1.0: # Ignore fractional share rounding noise for now
                continue
                
            side = OrderSide.BUY if delta_qty > 0 else OrderSide.SELL
            abs_delta = abs(delta_qty)
            
            logger.info(f"Dispatching Order for {symbol}: Delta={delta_qty} (Target: {target_qty}, Current: {current_qty})")

            if abs_delta > LARGE_ORDER_THRESHOLD:
                # Slices logic for large blocks
                self.execute_twap(symbol, abs_delta, side, slices=5, interval_seconds=10)
            else:
                # Standard dispatch
                try:
                    order_data = MarketOrderRequest(
                        symbol=symbol,
                        qty=abs_delta,
                        side=side,
                        time_in_force=TimeInForce.DAY
                    )
                    self.trading_client.submit_order(order_data=order_data)
                except Exception as e:
                    logger.error(f"Failed to submit order for {symbol}: {e}")

        # Handle sells for targets that dropped to 0 weight
        for symbol, current_qty in current_positions.items():
            if symbol not in target_weights or target_weights[symbol] == 0:
                logger.info(f"Liquidating position {symbol} since target weight is 0.")
                try:
                    self.trading_client.close_position(symbol)
                except Exception as e:
                    logger.error(f"Failed to close position {symbol}: {e}")
