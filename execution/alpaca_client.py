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
        self.api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self.api_secret = api_secret or os.getenv("ALPACA_SECRET_KEY", "")
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

    def rebalance_portfolio(self, target_weights: Dict[str, float]):
        """
        Takes target constrained weights from the Risk Manager and executes them.
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
        current_positions = {p.symbol: float(p.qty) for p in self.trading_client.get_all_positions()}
        
        # We need current prices to convert weights to quantities (or use notional value API if available)
        # For simplicity in this shell, assuming we have a mechanism to fetch current price:
        # Pseucode: prices = self.get_current_prices(target_weights.keys())
        
        logger.info(f"Target rebalance weights: {target_weights}")
        # Implementation of diffing current_positions vs target quantities goes here
        # For large diffs, route to execute_twap() instead of submit_order() directly
        
        # Example dummy dispatch:
        # for symbol, weight in target_weights.items():
        #     target_notional = capital * weight
        #     qty = target_notional / current_price
        #     delta = qty - current_positions.get(symbol, 0)
        #     if abs(delta) > LARGE_ORDER_THRESHOLD:
        #         self.execute_twap(symbol, delta, side, slices=10)
        #     else:
        #         self.trading_client.submit_order(...)
        pass
