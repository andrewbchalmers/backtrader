"""
Strategy Generator - Creates Dynamic Trading Strategies
"""

import backtrader as bt
from itertools import product
from src.indicator_library import IndicatorFactory, IndicatorSignal, get_indicator_description


class GenericStrategy(bt.Strategy):
    """
    Generic strategy template that can use any combination of entry/exit indicators
    """
    params = (
        ('entry_indicator', None),
        ('entry_params', None),
        ('entry_type', 'crossover'),  # crossover, threshold, breakout
        ('exit_indicator', None),
        ('exit_params', None),
        ('exit_type', 'crossover'),  # crossover, stop_loss, take_profit, trailing_stop
        ('stop_loss_pct', 0.02),  # 2% stop loss
        ('take_profit_pct', 0.06),  # 6% take profit
        ('trailing_stop_pct', 0.03),  # 3% trailing stop
        ('printlog', False),
    )
    
    def __init__(self):
        # Create entry indicator
        self.entry_ind = IndicatorFactory.create_indicator(
            self.data,
            self.params.entry_indicator,
            *self.params.entry_params
        )
        
        # Create exit indicator
        self.exit_ind = IndicatorFactory.create_indicator(
            self.data,
            self.params.exit_indicator,
            *self.params.exit_params
        )
        
        # Track order and entry price
        self.order = None
        self.entry_price = None
        self.highest_price = None
        
        # For crossover strategies, we need a reference (typically price)
        self.price = self.data.close
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
                self.highest_price = order.executed.price
                if self.params.printlog:
                    print(f'BUY EXECUTED, Price: {order.executed.price:.2f}')
            elif order.issell():
                if self.params.printlog:
                    print(f'SELL EXECUTED, Price: {order.executed.price:.2f}')
                self.entry_price = None
                self.highest_price = None
        
        self.order = None
    
    def next(self):
        # Check if we have an order pending
        if self.order:
            return
        
        # Check if we are in the market
        if not self.position:
            # Entry logic
            entry_signal = self._check_entry_signal()
            if entry_signal:
                self.order = self.buy()
        else:
            # Update highest price for trailing stop
            if self.data.close[0] > self.highest_price:
                self.highest_price = self.data.close[0]
            
            # Exit logic
            exit_signal = self._check_exit_signal()
            if exit_signal:
                self.order = self.sell()
    
    def _check_entry_signal(self):
        """Check if entry conditions are met"""
        entry_type = self.params.entry_type
        
        if entry_type == 'crossover':
            # Price crosses above indicator
            return self.price[0] > self.entry_ind[0] and self.price[-1] <= self.entry_ind[-1]
        
        elif entry_type == 'crossunder':
            # Price crosses below indicator
            return self.price[0] < self.entry_ind[0] and self.price[-1] >= self.entry_ind[-1]
        
        elif entry_type == 'threshold':
            # Indicator crosses above a threshold (for oscillators like RSI)
            # For RSI: buy when RSI crosses above 30
            threshold = 30 if self.params.entry_indicator == 'rsi' else 0
            return self.entry_ind[0] > threshold and self.entry_ind[-1] <= threshold
        
        elif entry_type == 'breakout':
            # Price breaks out of Bollinger Bands or similar
            if self.params.entry_indicator == 'bb':
                # Buy when price crosses above lower band
                return self.price[0] > self.entry_ind.bot[0] and self.price[-1] <= self.entry_ind.bot[-1]
            else:
                # Generic breakout: price above indicator
                return self.price[0] > self.entry_ind[0]
        
        return False
    
    def _check_exit_signal(self):
        """Check if exit conditions are met"""
        exit_type = self.params.exit_type
        
        if exit_type == 'crossover':
            # Price crosses below indicator
            return self.price[0] < self.exit_ind[0] and self.price[-1] >= self.exit_ind[-1]
        
        elif exit_type == 'crossunder':
            # Price crosses above indicator
            return self.price[0] > self.exit_ind[0] and self.price[-1] <= self.exit_ind[-1]
        
        elif exit_type == 'stop_loss':
            # Stop loss hit
            if self.entry_price:
                stop_price = self.entry_price * (1 - self.params.stop_loss_pct)
                return self.data.close[0] < stop_price
        
        elif exit_type == 'take_profit':
            # Take profit hit
            if self.entry_price:
                target_price = self.entry_price * (1 + self.params.take_profit_pct)
                return self.data.close[0] > target_price
        
        elif exit_type == 'trailing_stop':
            # Trailing stop hit
            if self.highest_price:
                stop_price = self.highest_price * (1 - self.params.trailing_stop_pct)
                return self.data.close[0] < stop_price
        
        # Also check indicator-based exit
        if self.params.exit_indicator:
            return self.price[0] < self.exit_ind[0] and self.price[-1] >= self.exit_ind[-1]
        
        return False


class StrategyGenerator:
    """Generate all possible strategy combinations"""
    
    def __init__(self, entry_indicators, exit_indicators, config):
        """
        Initialize strategy generator
        
        Args:
            entry_indicators: List of (indicator_name, params) tuples
            exit_indicators: List of (indicator_name, params) tuples
            config: Configuration dictionary
        """
        self.entry_indicators = entry_indicators
        self.exit_indicators = exit_indicators
        self.config = config
        self.entry_types = config.get('entry_types', ['crossover', 'threshold', 'breakout'])
        self.exit_types = config.get('exit_types', ['crossover', 'stop_loss', 'take_profit'])
    
    def generate_strategies(self):
        """
        Generate all valid strategy combinations
        
        Returns:
            List of strategy configuration dictionaries
        """
        strategies = []
        strategy_id = 1
        
        # Generate all combinations
        for entry_ind, exit_ind in product(self.entry_indicators, self.exit_indicators):
            entry_name, entry_params = entry_ind
            exit_name, exit_params = exit_ind
            
            # Determine valid entry types for this indicator
            valid_entry_types = self._get_valid_entry_types(entry_name)
            
            # Determine valid exit types for this indicator
            valid_exit_types = self._get_valid_exit_types(exit_name)
            
            # Create strategy for each valid combination
            for entry_type, exit_type in product(valid_entry_types, valid_exit_types):
                strategy_config = {
                    'id': strategy_id,
                    'entry_indicator': entry_name,
                    'entry_params': entry_params,
                    'entry_type': entry_type,
                    'exit_indicator': exit_name,
                    'exit_params': exit_params,
                    'exit_type': exit_type,
                    'description': self._generate_description(
                        entry_name, entry_params, entry_type,
                        exit_name, exit_params, exit_type
                    )
                }
                strategies.append(strategy_config)
                strategy_id += 1
        
        return strategies
    
    def _get_valid_entry_types(self, indicator_name):
        """Determine valid entry types for an indicator"""
        if indicator_name in ['rsi', 'cci', 'stoch']:
            return ['threshold']  # Oscillators use threshold
        elif indicator_name == 'bb':
            return ['breakout']  # Bollinger Bands use breakout
        else:
            return ['crossover']  # Moving averages use crossover
    
    def _get_valid_exit_types(self, indicator_name):
        """Determine valid exit types for an indicator"""
        if indicator_name == 'atr':
            return ['trailing_stop']  # ATR is good for trailing stops
        elif indicator_name in ['rsi', 'cci', 'stoch']:
            return ['threshold', 'stop_loss']
        else:
            return ['crossover', 'stop_loss', 'take_profit']
    
    def _generate_description(self, entry_ind, entry_params, entry_type,
                            exit_ind, exit_params, exit_type):
        """Generate human-readable strategy description"""
        entry_desc = get_indicator_description(entry_ind, entry_params)
        exit_desc = get_indicator_description(exit_ind, exit_params)
        
        return f"Entry: {entry_type.upper()} on {entry_desc} | Exit: {exit_type.upper()} on {exit_desc}"
    
    def get_strategy_count(self):
        """Get total number of strategies that will be generated"""
        count = 0
        for entry_ind, exit_ind in product(self.entry_indicators, self.exit_indicators):
            entry_name, _ = entry_ind
            exit_name, _ = exit_ind
            valid_entry_types = self._get_valid_entry_types(entry_name)
            valid_exit_types = self._get_valid_exit_types(exit_name)
            count += len(valid_entry_types) * len(valid_exit_types)
        return count


def create_strategy_class(strategy_config):
    """
    Create a backtrader strategy class from configuration
    
    Args:
        strategy_config: Strategy configuration dictionary
        
    Returns:
        Configured strategy class
    """
    params = {
        'entry_indicator': strategy_config['entry_indicator'],
        'entry_params': strategy_config['entry_params'],
        'entry_type': strategy_config['entry_type'],
        'exit_indicator': strategy_config['exit_indicator'],
        'exit_params': strategy_config['exit_params'],
        'exit_type': strategy_config['exit_type'],
    }
    
    # Create a new strategy class with these parameters
    class DynamicStrategy(GenericStrategy):
        params = tuple(params.items())
    
    return DynamicStrategy
