# Alpha Vanguard Hedge Fund Algo

Alpha Vanguard is a machine learning-driven quantitative trading system designed for automated execution. It uses fundamental data (via Finnhub) and price actions to predict returns, optimizes a portfolio using risk parity constraints, and executes dynamically via Alpaca.

## Architecture & Pipeline
1. **Data Ingestion**: Pulls OHLCV and technical indicators, supplemented with static fundamentals (P/E, P/B, ROE) from Finnhub.
2. **ML Model**: Trains a walk-forward XGBoost Regressor directly on incoming data to predict near-term horizon returns.
3. **Signal Translation**: Translates model predictions into convictions (Alpha).
4. **Risk & Optimization**: Constructs a Risk Parity portfolio. Applies Maximum CVaR constraints before generating target weights.
5. **Execution Orchestration**: Runs on a daily schedule, checking real-time positions, diffing target sizes, and dispatching precisely sliced execution orders via Alpaca REST API. Tracks trailing Sharpe and Max Drawdown.

## Prerequisites & Installation

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

### Environment Variables (.env)
You must create a `.env` file in the root directory with the following keys:
- `FINNHUB_API_KEY`: Finnhub API Key for fundamentals.
- `APCA_API_KEY_ID`: Your Alpaca Paper Trading API Key.
- `APCA_API_SECRET_KEY`: Your Alpaca Paper Trading Secret.

## Running in Production

To run the continuous daily cycle, simply run:
```bash
python live_trader.py
```
The trader will wake up, evaluate the market, dispatch orders, log metrics, and schedule itself for the next trading day.
