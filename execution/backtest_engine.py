import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
from strategies.momentum import CrossSectionalMomentum
from risk.optimizer import PortfolioOptimizer
from risk.constraints import RiskConstraints
from execution.slippage import SlippageCommission

class MomentumStrategyBT(bt.Strategy):
    """
    Backtrader wrapper for the Cross-Sectional Momentum Strategy.
    Integrates Agent 2 (Research) and Agent 3 (Risk).
    """
    params = (
        ('momentum_period', 20),
        ('top_n', 10),
        ('bottom_n', 10),
        ('rebalance_days', 20),
        ('target_volatility', 0.15),
        ('max_cvar', 0.03)
    )

    def __init__(self):
        self.momentum_model = CrossSectionalMomentum(
            top_n=self.p.top_n, 
            bottom_n=self.p.bottom_n, 
            momentum_period=self.p.momentum_period
        )
        self.optimizer = PortfolioOptimizer(target_volatility=self.p.target_volatility)
        self.risk_manager = RiskConstraints(max_cvar_daily=self.p.max_cvar)
        
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
        
        for d in self.datas:
            symbol = d._name
            if len(d) >= self.p.momentum_period:
                prices_snapshot[symbol] = d.close[0]
                # Extract history
                hist = [d.close[-i] for i in range(self.p.momentum_period-1, -1, -1)]
                historical_prices[symbol] = pd.Series(hist)
                
        if not historical_prices:
            return
            
        # Agent 2: Signal Generation
        raw_signals = self.momentum_model.generate_signals(prices_snapshot, historical_prices)
        
        if not raw_signals:
            return
            
        # Agent 3: Risk Management (Simulated historical returns)
        # In a real environment we would pull 1 year of daily returns from DB
        df_hist = pd.DataFrame(historical_prices)
        returns_df = df_hist.pct_change().dropna()
        
        # Optimize weights
        target_weights = self.optimizer.optimize_weights(df_hist, raw_signals)
        
        # Check constraints
        final_weights = self.risk_manager.apply_gateway(target_weights, returns_df)
        
        # Agent 4: Execution Routing
        if not final_weights:
            # Flatten if blocked by risk
            for d in self.datas:
                self.order_target_percent(d, target=0.0)
            return
            
        # Execute target rebalance
        for d in self.datas:
            symbol = d._name
            target = final_weights.get(symbol, 0.0)
            self.order_target_percent(d, target=target)

def run_backtest(data_feeds: dict, initial_cash: float = 100000.0):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    
    # Apply Square Root Slippage Model
    slippage_comm = SlippageCommission(adv=1000000.0, volatility=0.02, constant=0.1)
    cerebro.broker.addcommissioninfo(slippage_comm)

    # Add data feeds
    for symbol, df in data_feeds.items():
        data = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data, name=symbol)

    cerebro.addstrategy(MomentumStrategyBT)
    
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
    start_date = '2022-01-01'
    end_date = '2024-01-01'
    
    print(f"Fetching historical data from yfinance {start_date} to {end_date}...")
    data_feeds = {}
    
    for sym in symbols:
        df = yf.download(sym, start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        data_feeds[sym] = df

    print("Fetching SPY benchmark...")
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    spy.to_csv('spy_benchmark.csv')
    print("SPY benchmark saved to 'spy_benchmark.csv'")
        
    run_backtest(data_feeds)
