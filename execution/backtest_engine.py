import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
from strategies.momentum import CrossSectionalMomentum
from strategies.ml_model import MLAgent
from strategies.signal_bridge import SignalTranslator
from data.feature_engineering import FeatureEngineer
from data.validator_agent import ValidatorAgent
from risk.optimizer import PortfolioOptimizer
from risk.constraints import RiskConstraints
from execution.slippage import SlippageCommission

class MLStrategyBT(bt.Strategy):
    """
    Agent 12: Supervisor - Orchestrator
    Backtrader wrapper for the new XGBoost predictive Machine Learning pipeline.
    Routes data continuously from A6->A7->A8->A9->A10->A11.
    """
    params = (
        ('momentum_period', 20),
        ('rebalance_days', 20),
        ('target_volatility', 0.15),
        ('max_cvar', 0.03),
        ('ml_model', None), # Pre-trained A8 Executor
        ('feature_cols', None)
    )

    def __init__(self):
        self.optimizer = PortfolioOptimizer(target_volatility=self.p.target_volatility)
        self.risk_manager = RiskConstraints(max_cvar_daily=self.p.max_cvar)
        self.translator = SignalTranslator()
        
        self.days_since_rebalance = 0

    def next(self):
        self.days_since_rebalance += 1
        
        # Only rebalance every N days
        if self.days_since_rebalance < self.p.rebalance_days:
            return
            
        self.days_since_rebalance = 0
        
        # Build price history snapshot for the models
        prices_snapshot = {}
        historical_prices = {}
        
        current_features = {}
        
        for d in self.datas:
            symbol = d._name
            if len(d) >= self.p.momentum_period:
                prices_snapshot[symbol] = d.close[0]
                hist = [d.close[-i] for i in range(self.p.momentum_period-1, -1, -1)]
                historical_prices[symbol] = pd.Series(hist)
                
                # Fetch pre-computed features from the data feed lines
                try:
                    features_row = []
                    for col in self.p.feature_cols:
                        # Extract exact value from the data line by dynamically getting the attribute
                        features_row.append(getattr(d, col)[0])
                    current_features[symbol] = features_row
                except Exception as e:
                    pass
                
        if not current_features:
            return
            
        # Agent 8: Predict
        df_current = pd.DataFrame.from_dict(current_features, orient='index', columns=self.p.feature_cols)
        predictions = self.p.ml_model.predict(df_current, self.p.feature_cols)
        
        # Agent 9: Translate to Alpha Convicton
        alpha_signals = self.translator.translate_to_alpha(predictions, top_n=5)
        
        if not alpha_signals:
            return
            
        # Agent 10: Risk Parity Routing
        df_hist = pd.DataFrame(historical_prices)
        returns_df = df_hist.pct_change().dropna()
        
        # Optimize weights using Inverse Volatility logic
        target_weights = self.optimizer.optimize_weights(df_hist, alpha_signals)
        
        # Agent 11: Constrain
        final_weights = self.risk_manager.apply_gateway(target_weights, returns_df)
        
        # Execute routing
        if not final_weights:
            for d in self.datas:
                self.order_target_percent(d, target=0.0)
            return
            
        for d in self.datas:
            symbol = d._name
            target = final_weights.get(symbol, 0.0)
            self.order_target_percent(d, target=target)

class MLDataFeed(bt.feeds.PandasData):
    """
    Custom data feed to include additional feature columns for the ML pipeline.
    """
    lines = ('sma_20', 'sma_50', 'ema_20', 'volatility_20', 'roc_10', 'roc_20',
             'trailingPE', 'priceToBook', 'returnOnEquity', 'debtToEquity')
    
    params = (
        ('sma_20', -1), ('sma_50', -1), ('ema_20', -1), ('volatility_20', -1),
        ('roc_10', -1), ('roc_20', -1),
        ('trailingPE', -1), ('priceToBook', -1), ('returnOnEquity', -1), ('debtToEquity', -1),
    )

def run_backtest(data_feeds: dict, ml_model: MLAgent, feature_cols: list, initial_cash: float = 100000.0):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    
    # Apply Square Root Slippage Model
    slippage_comm = SlippageCommission(adv=1000000.0, volatility=0.02, constant=0.1)
    cerebro.broker.addcommissioninfo(slippage_comm)

    # Add data feeds
    for symbol, df in data_feeds.items():
        data = MLDataFeed(dataname=df)
        cerebro.adddata(data, name=symbol)

    cerebro.addstrategy(MLStrategyBT, ml_model=ml_model, feature_cols=feature_cols)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')

    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    print(f'Final Portfolio Value: {final_value:.2f}')
    
    print('Sharpe Ratio:', strat.analyzers.sharpe.get_analysis().get('sharperatio'))
    print('Max Drawdown:', strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown'))
    print('Total Return:', strat.analyzers.returns.get_analysis().get('rtot'))

    # Extract daily portfolio values
    timereturns = strat.analyzers.timereturn.get_analysis()
    
    # Convert daily returns to portfolio value curve
    dates = []
    values = []
    current_val = initial_cash
    for dt, ret in sorted(timereturns.items()):
        current_val *= (1 + ret)
        dates.append(dt)
        values.append(current_val)
        
    df_values = pd.DataFrame({'date': dates, 'portfolio_value': values})
    df_values.set_index('date', inplace=True)
    df_values.to_csv('strategy_results.csv')
    print("Strategy results saved to 'strategy_results.csv'")

    return cerebro

if __name__ == '__main__':
    import yfinance as yf
    
    symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    train_end = '2022-12-31' # Pre-train on first 3 years
    
    print(f"Agent 6: Fetching historical data from yfinance {start_date} to {end_date}...")
    
    raw_data_feeds = {}
    master_train_frames = []
    feature_cols = ['sma_20', 'sma_50', 'ema_20', 'volatility_20', 'roc_10', 'roc_20',
                   'trailingPE', 'priceToBook', 'returnOnEquity', 'debtToEquity']
                   
    validator = ValidatorAgent(target_horizon=5)
    
    for sym in symbols:
        df = yf.download(sym, start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        
        # Agent 6: Splice Technicals and Fundamentals
        df = FeatureEngineer.splice_features(df, sym)
        
        # Agent 7: Validate and Prep Training Target
        train_df = df[:train_end].copy()
        test_df = df[train_end:].copy()
        
        valid_train_df = validator.validate_and_prep(train_df)
        master_train_frames.append(valid_train_df)
        
        raw_data_feeds[sym] = test_df

    # Agent 8: Train XGBoost Model
    print("Agent 8: Training XGBoost Regressor...")
    master_train_df = pd.concat(master_train_frames)
    
    ml_agent = MLAgent(target_col='target_return')
    success = ml_agent.train(master_train_df, feature_cols)
    if not success:
        print("Model training failed. Aborting.")
        exit(1)

    print("Fetching SPY benchmark for Out-of-Sample evaluation...")
    spy = yf.download('SPY', start=train_end, end=end_date, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    spy.to_csv('spy_benchmark.csv')
    print("SPY benchmark saved to 'spy_benchmark.csv'")
        
    print("Agent 12: Orchestrating Backtest...")
    run_backtest(raw_data_feeds, ml_model=ml_agent, feature_cols=feature_cols)
