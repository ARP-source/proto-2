import pandas as pd
import numpy as np
from typing import Dict

class RiskConstraints:
    """
    Implements hard risk constraints, specifically Conditional Value at Risk (CVaR).
    """
    def __init__(self, max_cvar_daily: float = 0.03, confidence_level: float = 0.95):
        """
        max_cvar_daily: Maximum allowed expected shortfall (e.g., 0.03 means 3% daily drop max)
        confidence_level: the alpha for VaR/CVaR (e.g., 0.95)
        """
        self.max_cvar_daily = max_cvar_daily
        self.confidence_level = confidence_level

    def check_cvar_constraint(self, target_weights: Dict[str, float], historical_returns: pd.DataFrame) -> bool:
        """
        Calculates the projected portfolio CVaR based on historical returns and target weights.
        Returns True if the portfolio is SAFE (CVaR <= max_cvar_daily), False if it is BLOCKED.
        """
        if not target_weights or historical_returns.empty:
            return True # Trivial pass

        # Align weights with available assets in returns
        valid_assets = [s for s in target_weights.keys() if s in historical_returns.columns]
        if not valid_assets:
            return True
            
        weights_array = np.array([target_weights[s] for s in valid_assets])
        subset_returns = historical_returns[valid_assets]
        
        # Calculate historical daily portfolio returns
        portfolio_returns = subset_returns.dot(weights_array)
        
        # Calculate Value at Risk (VaR)
        var_threshold = np.percentile(portfolio_returns.dropna(), (1 - self.confidence_level) * 100)
        
        # Calculate Conditional Value at Risk (CVaR / Expected Shortfall)
        # CVaR is the expected return given that the return is less than VaR
        tail_returns = portfolio_returns[portfolio_returns <= var_threshold]
        
        if len(tail_returns) == 0:
            cvar = var_threshold
        else:
            cvar = tail_returns.mean()
            
        # cvar is usually negative (a loss).
        # We check if the absolute loss exceeds our maximum allowed drawdown.
        projected_drawdown = abs(cvar)
        
        is_safe = projected_drawdown <= self.max_cvar_daily
        
        if not is_safe:
            print(f"TRADE BLOCKED: Projected CVaR ({projected_drawdown:.2%}) exceeds limit ({self.max_cvar_daily:.2%})")
            
        return is_safe

    def apply_gateway(self, target_weights: Dict[str, float], historical_returns: pd.DataFrame) -> Dict[str, float]:
        """
        Risk Gateway: If weights pass constraints, return them. If blocked, return empty dict (flatten).
        """
        if self.check_cvar_constraint(target_weights, historical_returns):
            return target_weights
        else:
            return {} # Block the trade / close positions
