# strategy_loader.py
"""
Dynamically load trading strategies and extract signals
"""

import importlib.util
import sys
import os
import pandas as pd
import numpy as np
import backtrader as bt


class SignalCapture(bt.Observer):
    """Observer to capture buy/sell signals from strategy"""
    lines = ('signal', 'price')

    def __init__(self):
        self.buy_signals = []
        self.sell_signals = []

    def next(self):
        pass


class StrategyLoader:
    """Dynamically load any backtrader strategy"""

    def __init__(self, module_name, class_name):
        self.module_name = module_name
        self.class_name = class_name
        self.strategy_class = self._load_strategy()
        self._is_ml_strategy = 'lorentzian' in class_name.lower() or 'classification' in class_name.lower()

    def _load_strategy(self):
        """Load strategy class from module"""
        try:
            # Get absolute path and create a proper module name
            module_path = os.path.abspath(f"{self.module_name}.py")
            module_name = os.path.basename(self.module_name).replace('-', '_')

            spec = importlib.util.spec_from_file_location(
                module_name,
                module_path
            )
            module = importlib.util.module_from_spec(spec)

            # Register in sys.modules BEFORE executing (required by backtrader)
            sys.modules[module_name] = module

            spec.loader.exec_module(module)

            strategy_class = getattr(module, self.class_name)
            print(f"✓ Loaded strategy: {self.class_name} from {module_path}")
            return strategy_class

        except Exception as e:
            print(f"❌ Failed to load strategy: {e}")
            raise

    def get_entry_signal(self, df, params):
        """
        Extract entry logic from strategy
        Returns: {'signal': bool, 'price': float, 'stop_loss': float, 'atr': float}
        """
        # Use backtrader-based signal detection for ML strategies
        if self._is_ml_strategy:
            return self._get_ml_strategy_signal(df, params)

        # Legacy: simple indicator-based strategies
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

    def _get_ml_strategy_signal(self, df, params):
        """
        Run the actual ML strategy through backtrader to detect signals.
        Returns: {'signal': bool, 'signal_type': str, 'price': float, 'stop_loss': float, 'bars_ago': int}
        """
        try:
            # Prepare data for backtrader
            bt_df = df.copy()
            bt_df.columns = [c.lower() for c in bt_df.columns]

            # Ensure required columns exist
            required = ['open', 'high', 'low', 'close', 'volume']
            for col in required:
                if col not in bt_df.columns:
                    return {'signal': False}

            # Create cerebro instance
            cerebro = bt.Cerebro(stdstats=False)

            # Add data
            data = bt.feeds.PandasData(
                dataname=bt_df,
                datetime=None,
                open='open',
                high='high',
                low='low',
                close='close',
                volume='volume'
            )
            cerebro.adddata(data)

            # Prepare params (remove non-strategy params, set verbose=False)
            strategy_params = params.copy()
            strategy_params['verbose'] = False

            # Create a signal-capturing wrapper strategy
            captured_signals = []

            class SignalCaptureStrategy(self.strategy_class):
                def __init__(self):
                    super().__init__()
                    self.captured_buys = []
                    self.captured_sells = []

                def _execute_buy(self):
                    # Capture the signal instead of executing
                    self.captured_buys.append({
                        'bar': len(self),
                        'date': self.data.datetime.date(0),
                        'price': self.data.close[0],
                        'prediction': self.prediction
                    })
                    # Still call parent to maintain state
                    super()._execute_buy()

                def _close_position(self, reason):
                    # Capture sell signal
                    self.captured_sells.append({
                        'bar': len(self),
                        'date': self.data.datetime.date(0),
                        'price': self.data.close[0],
                        'reason': reason
                    })
                    super()._close_position(reason)

            # Add strategy
            cerebro.addstrategy(SignalCaptureStrategy, **strategy_params)

            # Set broker with enough cash
            cerebro.broker.setcash(1000000)
            cerebro.broker.setcommission(commission=0.0)

            # Run
            results = cerebro.run()
            strat = results[0]

            # Check for recent buy signals (last 5 bars)
            total_bars = len(bt_df)
            recent_threshold = total_bars - 5

            for sig in reversed(strat.captured_buys):
                if sig['bar'] >= recent_threshold:
                    bars_ago = total_bars - sig['bar']
                    close = df['Close'].iloc[-1]
                    stop_pct = params.get('stop_loss_pct', 0.05)
                    if hasattr(stop_pct, '__float__'):
                        stop_pct = float(stop_pct)

                    return {
                        'signal': True,
                        'signal_type': 'BUY',
                        'price': sig['price'],
                        'current_price': close,
                        'stop_loss': sig['price'] * (1 - stop_pct),
                        'date': sig['date'],
                        'bars_ago': bars_ago,
                        'prediction': sig.get('prediction', 0)
                    }

            # Check for recent sell signals
            for sig in reversed(strat.captured_sells):
                if sig['bar'] >= recent_threshold:
                    bars_ago = total_bars - sig['bar']
                    return {
                        'signal': True,
                        'signal_type': 'SELL',
                        'price': sig['price'],
                        'date': sig['date'],
                        'bars_ago': bars_ago,
                        'reason': sig.get('reason', 'EXIT')
                    }

            return {'signal': False}

        except Exception as e:
            print(f"❌ Error running ML strategy signal detection: {e}")
            import traceback
            traceback.print_exc()
            return {'signal': False}

    def get_exit_signal(self, df, params, entry_price, current_stop):
        """
        Extract exit logic from strategy
        Checks recent bars to catch stops that may have been hit when system was offline

        Returns: {'signal': bool, 'price': float, 'stop_type': str, 'new_stop': float, 'bars_ago': int}
        """
        # For ML strategies, use simple percentage-based stop logic
        # (The actual exit signals are captured during buy signal detection)
        stop_pct = params.get('stop_loss_pct', 0.05)
        if hasattr(stop_pct, '__float__'):
            stop_pct = float(stop_pct)

        lookback_bars = params.get('exit_lookback_bars', 5)

        # Check recent bars for stop hits (in reverse chronological order)
        for i in range(lookback_bars):
            idx = -(i + 1)

            if abs(idx) > len(df):
                break

            close = df['Close'].iloc[idx]

            # Percentage-based stop (from entry)
            pct_stop = entry_price * (1 - stop_pct)

            # Trailing stop (can't go below current_stop)
            new_stop = max(pct_stop, current_stop)

            # Check if price closed below stop
            if close <= new_stop:
                bars_ago = i
                bar_date = df.index[idx].strftime('%Y-%m-%d') if hasattr(df.index[idx], 'strftime') else str(df.index[idx])

                return {
                    'signal': True,
                    'price': close,
                    'stop_type': 'STOP_LOSS',
                    'new_stop': new_stop,
                    'bars_ago': bars_ago,
                    'bar_date': bar_date
                }

        # No stop hit, calculate new trailing stop
        close = df['Close'].iloc[-1]
        pct_stop = entry_price * (1 - stop_pct)

        # Simple trailing: stop trails at stop_pct below highest close since entry
        # For simplicity, just use the higher of pct_stop or current_stop
        new_stop = max(pct_stop, current_stop)

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


def calculate_warmup_days(strategy_params, default_days=100):
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
        'bb_period', 'macd_slow', 'lookback', 'period', 'length',
        'max_bars_back'  # For ML strategies like Lorentzian Classification
    ]

    for param_name, param_value in strategy_params.items():
        if any(key in param_name.lower() for key in lookback_params):
            if isinstance(param_value, (int, float)):
                max_period = max(max_period, int(param_value))

    if max_period > 0:
        # Add small buffer for indicator stabilization
        recommended = max_period + 10
        return max(recommended, default_days)
    else:
        return default_days
