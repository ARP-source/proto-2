import os
import pandas as pd
from datetime import datetime
import clickhouse_connect
from arcticdb import Arctic, QueryBuilder

class DataStorage:
    """
    Handles historical tick data storage in ClickHouse and 
    Pandas DataFrame versioning via ArcticDB.
    """
    def __init__(self, ch_host='localhost', ch_port=8123, 
                 arctic_uri='lmdb://./arctic_data'):
        # Clickhouse init
        try:
            self.ch_client = clickhouse_connect.get_client(
                host=ch_host, 
                port=ch_port,
                username=os.getenv('CLICKHOUSE_USER', 'default'),
                password=os.getenv('CLICKHOUSE_PASSWORD', '')
            )
            self._init_clickhouse_schema()
        except Exception as e:
            print(f"Warning: ClickHouse connection failed. {e}")
            self.ch_client = None
            
        # ArcticDB init
        try:
            self.arctic = Arctic(arctic_uri)
            if 'features' not in self.arctic.list_libraries():
                self.arctic.create_library('features')
            self.feature_lib = self.arctic['features']
        except Exception as e:
            print(f"Warning: ArcticDB initialization failed. {e}")
            self.arctic = None

    def _init_clickhouse_schema(self):
        """Initializes the required tables in ClickHouse for tick data."""
        if not self.ch_client:
            return
            
        create_query = """
        CREATE TABLE IF NOT EXISTS ticks (
            symbol String,
            timestamp DateTime64(3),
            bid Float64,
            ask Float64,
            bid_size Float64,
            ask_size Float64
        ) ENGINE = MergeTree()
        ORDER BY (symbol, timestamp)
        """
        self.ch_client.command(create_query)

    def insert_ticks(self, df: pd.DataFrame):
        """Inserts tick data into ClickHouse.
        Expects df with columns: symbol, timestamp, bid, ask, bid_size, ask_size"""
        if self.ch_client and not df.empty:
            self.ch_client.insert_df('ticks', df)
            
    def get_ticks(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Retrieves tick data from ClickHouse."""
        if not self.ch_client:
            return pd.DataFrame()
            
        query = f"""
        SELECT * FROM ticks 
        WHERE symbol = '{symbol}' 
        AND timestamp >= '{start.strftime("%Y-%m-%d %H:%M:%S")}'
        AND timestamp <= '{end.strftime("%Y-%m-%d %H:%M:%S")}'
        ORDER BY timestamp
        """
        return self.ch_client.query_df(query)

    def save_features(self, symbol: str, df: pd.DataFrame, metadata: dict = None):
        """Version-controls a feature DataFrame using ArcticDB."""
        if self.feature_lib is not None:
            self.feature_lib.write(symbol, df, metadata=metadata)
            
    def load_features(self, symbol: str, version: int = None) -> pd.DataFrame:
        """Retrieves a specific version of a feature DataFrame."""
        if self.feature_lib is not None:
            if version:
                return self.feature_lib.read(symbol, version=version).data
            return self.feature_lib.read(symbol).data
        return pd.DataFrame()
