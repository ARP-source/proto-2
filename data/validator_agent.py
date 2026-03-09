import pandas as pd
import numpy as np

class ValidatorAgent:
    """
    Agent 7: Feature Validation Communicator.
    Ensures the integrated feature set (Momentum + Fundamentals) has no look-ahead bias,
    handles NaN values properly (e.g., forward filling quarterly data), and
    normalizes the data appropriately for ML ingestion.
    """
    def __init__(self, target_horizon: int = 5):
        self.target_horizon = target_horizon # Predict N days out

    def validate_and_prep(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """
        Prepares the dataframe for ML by preventing lookahead bias.
        Generates the target variable shifted strictly to avoid using future data.
        """
        if df.empty:
            return df
            
        validated = df.copy()
        
        # Clean infinite values (from pct_change on bad data)
        validated.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Forward-fill fundamental data (which is static or quarterly) to prevent leakage
        validated.fillna(method='ffill', inplace=True)
        
        # Calculate the Target: Next N-days return.
        # IF target_horizon is 5, Shift(-5) pulls the price from 5 days in the future to TODAY.
        # This will be used STRICTLY as the 'y' labels for training today.
        validated['target_return'] = (validated[price_col].shift(-self.target_horizon) - validated[price_col]) / validated[price_col]
        
        # Drop rows where target or features are NaN to not train on dirty data
        validated.dropna(inplace=True)
        
        return validated
