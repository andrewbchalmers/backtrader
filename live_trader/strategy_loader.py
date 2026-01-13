# strategy_loader.py
"""
Dynamically load trading strategies and extract signals
"""

import importlib.util
import pandas as pd
import numpy as np


class StrategyLoader:
    """Dynamically load any backtrader strategy"""

    def __init__(self, module_name, class_name):
        self.module_name = module_name
        self.class_name = class_name
        self.strategy_class = self._load_strategy()

    def _load_strategy(self):
        """Load strategy class from module"""
        try:
            spec = importlib.util.spec_from_file_location(
                self.module_name,
                f"{self.module_name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            strategy_class = getattr(module, self.class_name)
            print(f"✓ Loaded strategy: {self.class_name} from {self.module_name}.py")
            return strategy_class

        except Exception as e:
            print(f"❌ Failed to load strategy: {e}")
            raise

    def get_entry_signal(self, df, params):
        """
        Extract entry logic from strategy
        Returns: {'signal': bool, 'price': float, 'stop_loss': float, 'atr': float}
        """
        indicators = self._calculate_indicators(df, params)

        # Check for crossover (common pattern)
        if 'fast_sma' in indicators and 'slow_sma' in indicators:
            fast = indicators['fast_sma']
            slow = indicators['slow_sma']

            # Crossover detected
            if fast.iloc[-1] > slow.iloc[-1] and fast.iloc[-2] <= slow.iloc[-2]:
                # Check trend filter if present
                if 'trend_ma' in indicators:
                    trend = indicators['trend_ma']
                    if trend.iloc[-1] <= trend.iloc[-2]:  # Declining trend
                        return {'signal': False}

                # Calculate stop loss
                close = df['Close'].iloc[-1]
                atr = indicators.get('atr', pd.Series([0])).iloc[-1]

                atr_mult = params.get('atr_mult', 3.0)
                stop_pct = params.get('stop_loss_pct', 0.1)

                atr_stop = close - atr_mult * atr if atr > 0 else close * (1 - stop_pct)
                pct_stop = close * (1 - stop_pct)
                stop_loss = max(atr_stop, pct_stop)

                return {
                    'signal': True,
                    'price': close,
                    'stop_loss': stop_loss,
                    'atr': atr
                }

        return {'signal': False}

    def get_exit_signal(self, df, params, entry_price, current_stop):
        """
        Extract exit logic from strategy
        Checks recent bars to catch stops that may have been hit when system was offline

        Returns: {'signal': bool, 'price': float, 'stop_type': str, 'new_stop': float, 'bars_ago': int}
        """
        indicators = self._calculate_indicators(df, params)

        # Get parameters
        atr_mult = params.get('atr_mult', 3.0)
        stop_pct = params.get('stop_loss_pct', 0.1)
        lookback_bars = params.get('exit_lookback_bars', 5)  # Check last 5 bars by default

        # Check recent bars for stop hits (in reverse chronological order)
        for i in range(lookback_bars):
            idx = -(i + 1)  # -1, -2, -3, etc.

            if abs(idx) > len(df):
                break

            close = df['Close'].iloc[idx]
            atr_series = indicators.get('atr', pd.Series([0]))

            if abs(idx) > len(atr_series):
                continue

            atr = atr_series.iloc[idx]

            # Calculate what the stop would have been at that bar
            # ATR-based stop (trails below price)
            atr_stop = close - atr_mult * atr if atr > 0 else entry_price * (1 - stop_pct)

            # Percentage-based stop (from entry)
            pct_stop = entry_price * (1 - stop_pct)

            # The stop at that point in time (can't go below current_stop from before)
            stop_at_bar = max(atr_stop, pct_stop, current_stop)

            # Check if price closed below stop
            if close <= stop_at_bar:
                bars_ago = i
                bar_date = df.index[idx].strftime('%Y-%m-%d') if hasattr(df.index[idx], 'strftime') else str(df.index[idx])

                return {
                    'signal': True,
                    'price': close,
                    'stop_type': 'TRAILING_STOP',
                    'new_stop': stop_at_bar,
                    'bars_ago': bars_ago,
                    'bar_date': bar_date
                }

        # No stop hit in recent history, calculate current trailing stop
        close = df['Close'].iloc[-1]
        atr = indicators.get('atr', pd.Series([0])).iloc[-1]

        atr_stop = close - atr_mult * atr if atr > 0 else entry_price * (1 - stop_pct)
        pct_stop = entry_price * (1 - stop_pct)
        new_stop = max(atr_stop, pct_stop, current_stop)

        return {'signal': False, 'new_stop': new_stop}

    def _calculate_indicators(self, df, params):
        """Calculate common indicators"""
        indicators = {}

        # SMAs
        if 'fast_len' in params:
            indicators['fast_sma'] = df['Close'].rolling(window=params['fast_len']).mean()
        if 'slow_len' in params:
            indicators['slow_sma'] = df['Close'].rolling(window=params['slow_len']).mean()
        if 'trend_len' in params:
            indicators['trend_ma'] = df['Close'].rolling(window=params['trend_len']).mean()

        # ATR
        if 'atr_len' in params:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            indicators['atr'] = true_range.rolling(window=params['atr_len']).mean()

        return indicators


def calculate_warmup_days(strategy_params, default_days=300):
    """
    Automatically calculate required warmup days from strategy parameters
    Looks for common parameter names and adds safety buffer

    Args:
        strategy_params: Dictionary of strategy parameters
        default_days: Default minimum warmup days

    Returns:
        int: Recommended warmup days
    """
    max_period = 0

    # Common parameter names for lookback periods
    lookback_params = [
        'trend_len', 'slow_len', 'fast_len', 'atr_len',
        'ma_period', 'sma_period', 'ema_period', 'rsi_period',
        'bb_period', 'macd_slow', 'lookback', 'period', 'length'
    ]

    for param_name, param_value in strategy_params.items():
        if any(key in param_name.lower() for key in lookback_params):
            if isinstance(param_value, (int, float)):
                max_period = max(max_period, int(param_value))

    if max_period > 0:
        # Add 50% buffer for indicator stabilization
        recommended = int(max_period * 1.5)
        return max(recommended, default_days)  # Use at least the default
    else:
        return default_days
