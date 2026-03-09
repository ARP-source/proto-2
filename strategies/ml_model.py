import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger(__name__)

class MLAgent:
    """
    Agent 8: Executor - Machine Learning Layer
    Trains an XGBoost model on the engineered rolling history 
    to predict the next N-period forward return.
    """
    def __init__(self, target_col: str = 'target_return'):
        self.target_col = target_col
        # We use a regressor to predict continuous forward returns
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror'
        )

    def train(self, df: pd.DataFrame, feature_cols: list):
        """
        Trains the XGBoost model on validated historical data.
        """
        if df.empty or self.target_col not in df.columns:
            logger.error(f"Cannot train ML Model: Missing target column '{self.target_col}' or empty dataset.")
            return False

        # Drop any remaining NaNs to ensure XGBoost receives clean data
        clean_df = df.dropna(subset=feature_cols + [self.target_col])
        if clean_df.empty:
            logger.warning("After dropping NaNs on training data, DataFrame is empty.")
            return False

        X = clean_df[feature_cols]
        y = clean_df[self.target_col]

        logger.info(f"Training XGBoost Regressor on {len(X)} samples with {len(feature_cols)} features...")
        self.model.fit(X, y)
        return True

    def predict(self, df_current: pd.DataFrame, feature_cols: list) -> pd.Series:
        """
        Generates forward-looking predictions for the current point-in-time snapshot.
        """
        if df_current.empty:
            return pd.Series(dtype=float)
            
        # For prediction, we might have NaNs in fundamentals if they dropped out, 
        # XGBoost can handle NaNs inherently but we should ideally ensure the columns exist
        X_pred = df_current[feature_cols]
        predictions = self.model.predict(X_pred)
        
        return pd.Series(predictions, index=df_current.index)
