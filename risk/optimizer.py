import pandas as pd
import numpy as np
from typing import Dict

class PortfolioOptimizer:
    """
    Agent 10: Executor - Risk & Sizing (Risk Parity)
    Replaces Mean-Variance Optimization with an Inverse Volatility (Risk Parity) approach.
    Ensures that lower volatility assets receive higher capital allocation 
    to equalize risk contribution across the portfolio.
    """
    def __init__(self, target_volatility: float = 0.15):
        self.target_volatility = target_volatility

    def optimize_weights(self, price_data: pd.DataFrame, current_signals: Dict[str, float]) -> Dict[str, float]:
        """
        Takes historical price data and target signals (direction/conviction) from Agent 9.
        Outputs optimal Risk Parity weights multiplied by the Alpha signal direction.
        price_data: DataFrame with dates as index and symbols as columns.
        current_signals: Dict mapping symbol to a directional conviction score (-1.0 to 1.0).
        """
        valid_assets = [s for s in current_signals.keys() if s in price_data.columns]
        if not valid_assets:
            return {}
            
        subset_prices = price_data[valid_assets]
        
        # Calculate daily returns
        returns = subset_prices.pct_change().dropna()
        if returns.empty:
            return {}
            
        # Calculate annualized volatility for each asset (assuming ~252 trading days)
        volatilities = returns.std() * np.sqrt(252)
        
        # Compute Inverse Volatility (Risk Parity Base Weight)
        inv_vol = 1.0 / volatilities
        
        # In case of 0 volatility, replace inf with 0
        inv_vol.replace([np.inf, -np.inf], 0.0, inplace=True)
        
        sum_inv_vol = inv_vol.sum()
        if sum_inv_vol == 0:
            return {}
            
        base_risk_parity_weights = inv_vol / sum_inv_vol
        
        # Apply Alpha Signals (Direction & Magnitude) to the Risk Parity base
        final_weights = {}
        for symbol in valid_assets:
            # Alpha signal (-1.0 to 1.0)
            alpha_signal = current_signals.get(symbol, 0.0)
            
            # Final allocation = Risk Parity Weight * Alpha Conviction
            allocated_weight = base_risk_parity_weights[symbol] * alpha_signal
            final_weights[symbol] = allocated_weight
            
        return final_weights
