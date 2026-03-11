import os
import finnhub
import pandas as pd
import numpy as np
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Calculates rolling SMA, EMA, rolling standard deviation (volatility),
    and splices in fundamental data (Value/Quality factors) for ML pipelines.
    """
    
    @staticmethod
    def add_technical_features(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
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
        df['roc_20'] = df[price_col].pct_change(periods=20)
        
        return df

    @staticmethod
    def fetch_fundamentals(symbol: str) -> dict:
        """
        Agent 6: Fetch static fundamental data from Finnhub.
        Using trailing PE, Price-to-Book, and ROE as Value/Quality factors.
        """
        try:
            api_key = os.getenv("FINNHUB_API_KEY", "")
            if not api_key:
                logger.warning("FINNHUB_API_KEY not found in environment.")
                return {'trailingPE': np.nan, 'priceToBook': np.nan, 'returnOnEquity': np.nan, 'debtToEquity': np.nan}
                
            finnhub_client = finnhub.Client(api_key=api_key)
            
            # Use 'basic-financials' endpoint (metric section)
            metrics = finnhub_client.company_basic_financials(symbol, 'all')
            if metrics and 'metric' in metrics:
                m = metrics['metric']
                fundamentals = {
                    'trailingPE': m.get('peInclExtraTTM', np.nan),
                    'priceToBook': m.get('pbAnnual', np.nan),
                    'returnOnEquity': m.get('roeTTM', np.nan),
                    'debtToEquity': m.get('totalDebt/totalEquityAnnual', np.nan)
                }
                return fundamentals
            else:
                return {'trailingPE': np.nan, 'priceToBook': np.nan, 'returnOnEquity': np.nan, 'debtToEquity': np.nan}
        except Exception as e:
            logger.warning(f"Failed to fetch fundamentals for {symbol}: {e}")
            return {'trailingPE': np.nan, 'priceToBook': np.nan, 'returnOnEquity': np.nan, 'debtToEquity': np.nan}

    @staticmethod
    def splice_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Combines technicals and static fundamentals into a single feature set.
        """
        df = FeatureEngineer.add_technical_features(df)
        fundamentals = FeatureEngineer.fetch_fundamentals(symbol)
        
        for key, val in fundamentals.items():
            df[key] = val
            
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
