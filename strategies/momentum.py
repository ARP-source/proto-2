import pandas as pd
import numpy as np
from typing import List, Dict

class CrossSectionalMomentum:
    """
    Ranks a universe of equities based on momentum (Rate of Change)
    and goes long the top N and short the bottom N.
    """
    def __init__(self, top_n: int = 10, bottom_n: int = 10, momentum_period: int = 20):
        self.top_n = top_n
        self.bottom_n = bottom_n
        self.momentum_period = momentum_period

    def calculate_momentum(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """
        Calculates the momentum metric (e.g., n-period return).
        Expects a MultiIndex DataFrame (symbol, date) or similar panel data.
        """
        # Calculate percentage change over the momentum period
        momentum = df.groupby(level='symbol')[price_col].pct_change(self.momentum_period)
        return momentum

    def generate_signals(self, prices_snapshot: Dict[str, float], historical_prices: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        Generates target weights for the current snapshot.
        prices_snapshot: {symbol: current_price}
        historical_prices: {symbol: Series of historical prices}
        """
        momentum_scores = {}
        
        for symbol, prices in historical_prices.items():
            if len(prices) >= self.momentum_period:
                # Simple n-period return
                current = prices_snapshot.get(symbol, prices.iloc[-1])
                past = prices.iloc[-self.momentum_period]
                if past > 0:
                    momentum_scores[symbol] = (current - past) / past
                    
        if not momentum_scores:
            return {}
            
        # Rank the scores
        sorted_scores = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        
        target_weights = {}
        total_longs = min(self.top_n, len(sorted_scores))
        total_shorts = min(self.bottom_n, len(sorted_scores))
        
        if total_longs == 0 or total_shorts == 0:
            return {}

        long_weight_per_asset = 1.0 / total_longs
        short_weight_per_asset = -1.0 / total_shorts
        
        # Assign Longs (Top N)
        for i in range(total_longs):
            target_weights[sorted_scores[i][0]] = long_weight_per_asset
            
        # Assign Shorts (Bottom N)
        for i in range(len(sorted_scores) - total_shorts, len(sorted_scores)):
            target_weights[sorted_scores[i][0]] = short_weight_per_asset
            
        return target_weights
