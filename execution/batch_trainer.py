import os
import json
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import xgboost as xgb
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import boto3
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load local environment if running locally
load_dotenv()

# --- Configuration ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "algo-models-store")

# Feature Parameters Contract
FEATURE_PARAMS = {
    "fractional_differencing_d": 0.4,
    "z_score_lookback_periods": 20,
    "log_returns_window": 1
}

# The universe of assets
UNIVERSE = ["AAPL", "MSFT", "GOOGL", "SPY"]
TIMEZONE = "America/New_York"

def get_t_minus_1_data(client, symbols, days_back=30):
    """
    Fetches daily closed market data for T-1 up to `days_back` days ago.
    Uses strict America/New_York alignment.
    """
    ny_tz = ZoneInfo(TIMEZONE)
    now = datetime.now(ny_tz)
    
    # Target T-1 (yesterday)
    end_date = (now - timedelta(days=1)).replace(hour=16, minute=0, second=0, microsecond=0)
    start_date = (end_date - timedelta(days=days_back)).replace(hour=9, minute=30, second=0, microsecond=0)
    
    logger.info(f"Fetching data from {start_date} to {end_date}")
    
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date
    )
    
    bars = client.get_stock_bars(req).df
    if bars.empty:
         raise ValueError("No data returned from Alpaca.")
    
    # Simplify multi-index
    bars = bars.reset_index()
    # Ensure correct tz on timestamp
    if bars['timestamp'].dt.tz is None:
        bars['timestamp'] = bars['timestamp'].dt.tz_localize('UTC').dt.tz_convert(TIMEZONE)
    else:
        bars['timestamp'] = bars['timestamp'].dt.tz_convert(TIMEZONE)
        
    return bars

def fractionally_difference(series, d, threshold=1e-4):
    """
    Applies fractional differencing to preserve memory while achieving stationarity.
    """
    weights = [1.]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1

    diff_series = np.zeros(len(series))
    for i in range(len(weights), len(series)):
        # Calculate trailing sum applied with the weight expansion
        diff_series[i] = np.dot(weights[::-1], series[i-len(weights)+1:i+1])
        
    return pd.Series(diff_series, index=series.index)

def engineer_features(df):
    """
    Engineers stationary features based on the predefined parameters.
    df must be a flattened dataframe with ['symbol', 'timestamp', 'close', ...] columns.
    """
    logger.info("Engineering stationary features...")
    df = df.sort_values(by=['symbol', 'timestamp'])
    
    # Log Returns
    w = FEATURE_PARAMS["log_returns_window"]
    df['log_return'] = df.groupby('symbol')['close'].apply(lambda x: np.log(x / x.shift(w))).reset_index(level=0, drop=True)
    
    # Fractional Differencing on Close prices
    d = FEATURE_PARAMS["fractional_differencing_d"]
    df['frac_diff'] = df.groupby('symbol')['close'].apply(lambda x: fractionally_difference(x, d)).reset_index(level=0, drop=True)
    
    # Cross-sectional Z-Scores (requires pivoting on timestamp)
    pivot_close = df.pivot(index='timestamp', columns='symbol', values='close')
    
    # Calculate cross-sectional mean and std per timestamp
    mean_xs = pivot_close.mean(axis=1)
    std_xs = pivot_close.std(axis=1)
    
    z_scores_pivot = pivot_close.sub(mean_xs, axis=0).div(std_xs, axis=0)
    # Melt back to long format mapped by timestamp and symbol
    z_scores_long = z_scores_pivot.reset_index().melt(id_vars='timestamp', var_name='symbol', value_name='xs_z_score')
    
    # Merge back into original DataFrame
    df = pd.merge(df, z_scores_long, on=['timestamp', 'symbol'], how='outer')
    
    # Define Target: Next period's log return (Forward 1 step)
    df['target_return'] = df.groupby('symbol')['log_return'].shift(-1)
    
    df.dropna(inplace=True)
    return df

def train_xgboost_model(df):
    """
    Trains a walk-forward XGBoost model on the engineered dataset.
    """
    logger.info("Training XGBoost model...")
    # Features
    # Needs temporal split to prevent data leakage
    df_sorted = df.sort_values(by='timestamp')
    X = df_sorted[features]
    y = df_sorted['target_return']
    
    # Simplistic direct train (A real implementation would use TimeSeriesSplit for walk-forward CV)
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 3,
        'learning_rate': 0.05,
        'subsample': 0.8
    }
    
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    
    model = xgb.train(
        params=params, 
        dtrain=dtrain, 
        num_boost_round=100, 
        evals=evals, 
        early_stopping_rounds=10, 
        verbose_eval=False
    )
    
    return model

def serialize_and_upload(model):
    """
    Serializes XGBoost model to a standalone binary file and uploads it with 
    the parameter schema to Object Storage.
    """
    date_str = datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y%M%d")
    model_version = f"xgb_v1_0_{date_str}"
    
    # 1. Save binary model artifact
    bin_filename = f"{model_version}.bin"
    model.save_model(bin_filename)
    
    # Create pointer URI
    uri_pointer = f"s3://{S3_BUCKET_NAME}/models/{bin_filename}"
    
    # 2. Build parameter schema JSON
    state_payload = {
        "metadata": {
            "generated_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
            "model_version": model_version,
            "universe": UNIVERSE,
            "timezone": TIMEZONE
        },
        "feature_parameters": FEATURE_PARAMS,
        "model_artifact": {
            "format": "xgboost_binary",
            "uri": uri_pointer
        }
    }
    
    schema_filename = f"model_state_{date_str}.json"
    with open(schema_filename, 'w') as f:
        json.dump(state_payload, f, indent=4)
        
    logger.info(f"Uploading artifacts to S3 bucket: {S3_BUCKET_NAME}")
    try:
        if AWS_ACCESS_KEY and AWS_SECRET_KEY:
             s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, endpoint_url=os.getenv("S3_ENDPOINT_URL"))
             
             s3_client.upload_file(bin_filename, S3_BUCKET_NAME, f"models/{bin_filename}")
             s3_client.upload_file(schema_filename, S3_BUCKET_NAME, f"schemas/{schema_filename}")
             logger.info("Upload sequence complete.")
        else:
            logger.warning("AWS Credentials not found. Skipping S3 upload for local testing.")
            logger.info(f"Local Schema saved: {schema_filename}")
            logger.info(f"Local Binary saved: {bin_filename}")
            
    except Exception as e:
         logger.error(f"S3 Upload failed: {e}")
         
    # Cleanup local binary if uploaded
    if os.path.exists(bin_filename) and AWS_ACCESS_KEY:
        os.remove(bin_filename)
    if os.path.exists(schema_filename) and AWS_ACCESS_KEY:
        os.remove(schema_filename)


def run_batch_pipeline():
    logger.info("Initializing Batch Pipeline...")
    
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        logger.error("Alpaca API credentials missing. Exiting.")
        return
        
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    
    try:
        df = get_t_minus_1_data(client, UNIVERSE, days_back=100)
        engineered_df = engineer_features(df)
        model = train_xgboost_model(engineered_df)
        serialize_and_upload(model)
        
        logger.info("Batch Pipeline executed successfully.")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")


if __name__ == "__main__":
    run_batch_pipeline()
