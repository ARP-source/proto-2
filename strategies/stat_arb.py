import pandas as pd
import numpy as np
from typing import Dict, Tuple

class StatisticalArbitrageStrategy:
    """
    Statistical Arbitrage using Co-integration (Pairs Trading).
    Generates target allocations based on Z-Score of the spread.
    """
    def __init__(self, entry_z_score: float = 2.0, exit_z_score: float = 0.0):
        self.entry_z_score = entry_z_score
        self.exit_z_score = exit_z_score
        
    def generate_signals(self, z_score: float) -> Tuple[float, float]:
        """
        Takes a current z_score for a pair (X, Y) and outputs target weights.
        Returns (weight_x, weight_y).
        
        Logic: 
        If Z-Score > entry (Spread is too wide, Y is overvalued relative to X):
            Short Y, Long X
        If Z-Score < -entry (Spread is too narrow, Y is undervalued):
            Long Y, Short X
        If |Z-Score| drops below exit_z_score:
            Flatten positions
        """
        weight_x, weight_y = 0.0, 0.0
        
        if z_score > self.entry_z_score:
            weight_y = -0.5
            weight_x = 0.5
        elif z_score < -self.entry_z_score:
            weight_y = 0.5
            weight_x = -0.5
            
        return weight_x, weight_y

    def run(self, df_spread: pd.DataFrame) -> pd.DataFrame:
        """
        Runs logic sequentially over a dataframe (mostly for backtester injection).
        Expects a z_score column.
        """
        signals = df_spread.copy()
        signals['weight_x'] = 0.0
        signals['weight_y'] = 0.0
        signals['position'] = 0 # 1 means Long Y/Short X, -1 means Short Y/Long X
        
        current_pos = 0
        
        for i, row in signals.iterrows():
            z = row['z_score']
            if pd.isna(z):
                continue
                
            # Entry logic
            if z > self.entry_z_score:
                current_pos = -1
            elif z < -self.entry_z_score:
                current_pos = 1
                
            # Exit logic
            if current_pos == -1 and z < self.exit_z_score:
                current_pos = 0
            elif current_pos == 1 and z > -self.exit_z_score:
                current_pos = 0
                
            if current_pos == -1:
                signals.at[i, 'weight_y'] = -0.5
                signals.at[i, 'weight_x'] = 0.5
            elif current_pos == 1:
                signals.at[i, 'weight_y'] = 0.5
                signals.at[i, 'weight_x'] = -0.5
                
            signals.at[i, 'position'] = current_pos
            
        return signals
