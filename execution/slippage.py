import backtrader as bt
import numpy as np

class SquareRootSlippageModel:
    """
    Implements a theoretical slippage model based on the Square Root Law of Market Impact.
    Delta Price = Volatility * Constant * sqrt(Order Size / Average Daily Volume)
    """
    def __init__(self, adv: float = 1_000_000, volatility: float = 0.02, constant: float = 0.1):
        self.adv = adv
        self.volatility = volatility
        self.constant = constant

    def calculate_impact(self, order_size: float, price: float) -> float:
        """Calculates the estimated price impact in units of the asset's price."""
        if order_size == 0 or self.adv == 0:
            return 0.0
            
        fraction_of_adv = abs(order_size) / self.adv
        impact_pct = self.volatility * self.constant * np.sqrt(fraction_of_adv)
        return price * impact_pct


class SlippageCommission(bt.CommInfoBase):
    """
    Custom Backtrader Commission scheme implementing Square Root Slippage.
    """
    params = (
        ('adv', 1000000.0),
        ('volatility', 0.02),
        ('constant', 0.1),
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_FIXED),
        ('commission', 0.0) # Base commission is 0 assuming free trading (e.g. Alpaca)
    )

    def _getcommission(self, size, price, pseudoexec):
        """
        Calculates the "commission" which we will use to represent the slippage 
        in our backtests to cleanly subtract it from capital in Backtrader.
        """
        slippage_model = SquareRootSlippageModel(
            adv=self.p.adv, 
            volatility=self.p.volatility, 
            constant=self.p.constant
        )
        impact_per_share = slippage_model.calculate_impact(size, price)
        total_slippage_cost = impact_per_share * abs(size)
        
        # We model slippage as a commission cost to explicitly track it.
        return total_slippage_cost
