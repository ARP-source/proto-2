import pandas as pd
import numpy as np
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from typing import Dict, List

class PortfolioOptimizer:
    """
    Uses PyPortfolioOpt for Mean-Variance Optimization to size positions.
    """
    def __init__(self, target_volatility: float = 0.15):
        self.target_volatility = target_volatility

    def optimize_weights(self, price_data: pd.DataFrame, current_signals: Dict[str, float]) -> Dict[str, float]:
        """
        Takes historical price data and target signals (direction/conviction) from a strategy.
        Outputs optimal constrained weights.
        price_data: DataFrame with dates as index and symbols as columns.
        current_signals: Dict mapping symbol to a directional score (e.g. momentum score) 
                         which we can use as expected returns if we assume scores = returns.
        """
        # If we have less than 2 valid assets, just return equal weight locally
        valid_assets = [s for s in current_signals.keys() if s in price_data.columns]
        if len(valid_assets) < 2:
            return {s: 1.0/len(valid_assets) for s in valid_assets} if valid_assets else {}
            
        subset_prices = price_data[valid_assets]
        
        # Calculate expected returns from historical prices
        # Alternatively, we could map current_signals directly as expected returns
        mu = expected_returns.mean_historical_return(subset_prices)
        
        # Calculate the covariance matrix
        S = risk_models.sample_cov(subset_prices)
        
        try:
            # We allow shorting (weight bounds -1 to 1) 
            ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
            
            # Maximize Sharpe ratio
            raw_weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()
            return dict(cleaned_weights)
            
        except Exception as e:
            # Fallback to naive sizing based on signals if optimization fails 
            print(f"Optimization failed: {e}. Falling back to signal-based sizing.")
            total_abs_signal = sum(abs(v) for v in current_signals.values())
            if total_abs_signal == 0:
                return {}
            return {k: v/total_abs_signal for k, v in current_signals.items()}
