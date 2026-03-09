import os
import json
import logging
import asyncio
from typing import List, Optional
import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockTradesRequest, StockQuotesRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.live import StockDataStream
from execution.state_manager import StateManager
from data.storage import DataStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    """
    Handles fetching historical data and streaming live quotes/trades
    from Alpaca. Routes live data through Redis and stores in Clickhouse.
    """
    def __init__(self, api_key: str = None, api_secret: str = None, paper: bool = True):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self.api_secret = api_secret or os.getenv("ALPACA_SECRET_KEY", "")
        
        # REST client
        self.hist_client = StockHistoricalDataClient(self.api_key, self.api_secret)
        
        # WebSocket Streaming client
        self.stream = StockDataStream(self.api_key, self.api_secret)
        
        # Storage and messaging
        self.state_manager = StateManager()
        self.storage = DataStorage()
        
    def fetch_historical_bars(self, symbols: List[str], start_time: datetime, end_time: datetime, timeframe: TimeFrame = TimeFrame.Day) -> Optional[pd.DataFrame]:
        """Fetches historical OHLCV data."""
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start_time,
            end=end_time
        )
        try:
            bars = self.hist_client.get_stock_bars(request_params)
            return bars.df
        except Exception as e:
            logger.error(f"Error fetching historical bars: {e}")
            return None

    def _quote_handler(self, q):
        """Callback for incoming quotes."""
        quote_data = {
            "type": "quote",
            "symbol": q.symbol,
            "timestamp": q.timestamp.isoformat(),
            "bid": q.bid_price,
            "ask": q.ask_price,
            "bid_size": q.bid_size,
            "ask_size": q.ask_size
        }
        # Publish to StateManager (Redis) for low latency signal routing
        self.state_manager.publish(f"market_data_{q.symbol}", quote_data)
        
        # Optionally batch into Clickhouse
        # In a real async loop we would queue these before batch inserting
        df = pd.DataFrame([quote_data])
        # Need to strip 'type' and handle timestamp parsing internally
        # self.storage.insert_ticks(clean_df)
        
    def _trade_handler(self, t):
        """Callback for incoming trades."""
        trade_data = {
            "type": "trade",
            "symbol": t.symbol,
            "timestamp": t.timestamp.isoformat(),
            "price": t.price,
            "size": t.size,
            "conditions": t.conditions
        }
        self.state_manager.publish(f"market_data_{t.symbol}", trade_data)

    async def start_streaming(self, symbols: List[str]):
        """Starts WebSocket subscriptions to stock quotes and trades."""
        logger.info(f"Subscribing to {symbols} streams...")
        self.stream.subscribe_quotes(self._quote_handler, *symbols)
        self.stream.subscribe_trades(self._trade_handler, *symbols)
        
        await self.stream._run_forever()

if __name__ == "__main__":
    # Example usage (will block forever if streaming)
    # ingestion = DataIngestion()
    # asyncio.run(ingestion.start_streaming(["AAPL", "TSLA"]))
    pass
