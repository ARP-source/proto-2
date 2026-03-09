import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Calculates rolling SMA, EMA, and rolling standard deviation (volatility)
    for trading signals.
    """
    
    @staticmethod
    def add_features(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """
        Adds basic technical features to the DataFrame in-place.
        """
        if df.empty or price_col not in df.columns:
            return df
            
        # Moving Averages
        df['sma_20'] = df[price_col].rolling(window=20).mean()
        df['sma_50'] = df[price_col].rolling(window=50).mean()
        df['ema_20'] = df[price_col].ewm(span=20, adjust=False).mean()
        
        # Volatility
        df['volatility_20'] = df[price_col].rolling(window=20).std()
        
        # Momentum (Rate of Change)
        df['roc_10'] = df[price_col].pct_change(periods=10)
        
        return df

    @staticmethod
    def calculate_cointegration_features(df_x: pd.DataFrame, df_y: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """
        Calculates spread and z-score for two potentially cointegrated series.
        Used for Statistical Arbitrage (Pairs Trading).
        """
        # Ensure indices match
        common_idx = df_x.index.intersection(df_y.index)
        x = df_x.loc[common_idx, price_col]
        y = df_y.loc[common_idx, price_col]
        
        # Basic hedge ratio (could be calculated dynamically via OLS)
        hedge_ratio = 1.0 
        
        spread = y - (hedge_ratio * x)
        
        df_spread = pd.DataFrame(index=common_idx)
        df_spread['spread'] = spread
        df_spread['spread_mean'] = spread.rolling(window=20).mean()
        df_spread['spread_std'] = spread.rolling(window=20).std()
        df_spread['z_score'] = (spread - df_spread['spread_mean']) / df_spread['spread_std']
        
        return df_spread
