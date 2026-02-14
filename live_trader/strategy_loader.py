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

            # Add strategy directory to sys.path so auxiliary modules
            # (fundamental_data.py, cross_symbol_preloader.py) can be imported
            strategy_dir = os.path.dirname(module_path)
            if strategy_dir not in sys.path:
                sys.path.insert(0, strategy_dir)

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

    def get_entry_signal(self, df, params, symbol=None):
        """
        Extract entry logic from strategy
        Returns: {'signal': bool, 'price': float, 'stop_loss': float, 'atr': float}
        """
        # Use backtrader-based signal detection for ML strategies
        if self._is_ml_strategy:
            return self._get_ml_strategy_signal(df, params, symbol=symbol)

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

    def _get_ml_strategy_signal(self, df, params, symbol=None):
        """
        Run the actual ML strategy through backtrader to detect signals.
        Returns: {'signal': bool, 'signal_type': str, 'price': float, 'stop_loss': float, 'bars_ago': int}
        """
        import gc
        cerebro = None
        result = {'signal': False}

        try:
            # Prepare data for backtrader
            bt_df = df.copy()
            bt_df.columns = [c.lower() for c in bt_df.columns]

            # Ensure required columns exist
            required = ['open', 'high', 'low', 'close', 'volume']
            for col in required:
                if col not in bt_df.columns:
                    return result

            # Fix zero-range bars to prevent division-by-zero in ADX indicator
            zero_range = bt_df['high'] == bt_df['low']
            if zero_range.any():
                epsilon = bt_df['close'][zero_range] * 1e-6
                bt_df.loc[zero_range, 'high'] = bt_df.loc[zero_range, 'high'] + epsilon
                bt_df.loc[zero_range, 'low'] = bt_df.loc[zero_range, 'low'] - epsilon

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

            # Set per-symbol params for cross-symbol training and fundamentals
            if symbol:
                strategy_params['cross_symbol_target_symbol'] = symbol
                strategy_params['fundamental_symbol'] = symbol

            # Create a signal-capturing wrapper strategy
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

            # Get current date for earnings blackout check
            from datetime import date
            current_date = date.today()

            for sig in reversed(strat.captured_buys):
                if sig['bar'] >= recent_threshold:
                    bars_ago = total_bars - sig['bar']
                    close = df['Close'].iloc[-1]
                    stop_pct = params.get('stop_loss_pct', 0.05)
                    if hasattr(stop_pct, '__float__'):
                        stop_pct = float(stop_pct)

                    # Final earnings blackout check using TODAY's date
                    # (signal may have been valid on signal date but we're acting today)
                    if (params.get('use_fundamental_filter') and
                        hasattr(strat, 'fundamental_provider') and
                        strat.fundamental_provider is not None):
                        blackout_before = params.get('earnings_blackout_before', 5)
                        blackout_after = params.get('earnings_blackout_after', 2)
                        if strat.fundamental_provider.is_in_earnings_blackout(
                                current_date, blackout_before, blackout_after):
                            # In earnings blackout - skip this signal
                            continue

                    result = {
                        'signal': True,
                        'signal_type': 'BUY',
                        'price': sig['price'],
                        'current_price': close,
                        'stop_loss': sig['price'] * (1 - stop_pct),
                        'date': sig['date'],
                        'bars_ago': bars_ago,
                        'prediction': sig.get('prediction', 0)
                    }
                    break

            # Check for recent sell signals (only if no buy signal found)
            if not result['signal']:
                for sig in reversed(strat.captured_sells):
                    if sig['bar'] >= recent_threshold:
                        bars_ago = total_bars - sig['bar']
                        result = {
                            'signal': True,
                            'signal_type': 'SELL',
                            'price': sig['price'],
                            'date': sig['date'],
                            'bars_ago': bars_ago,
                            'reason': sig.get('reason', 'EXIT')
                        }
                        break

        except Exception as e:
            print(f"\n❌ Error running ML strategy signal detection: {e}")

        finally:
            # Clean up cerebro and force garbage collection
            if cerebro is not None:
                cerebro.runstop()
                del cerebro
            gc.collect()

        return result

    def get_exit_signal(self, df, params, entry_price, current_stop, symbol=None):
        """
        Extract exit logic from strategy
        For ML strategies, runs the actual strategy to detect exit signals.

        Returns: {'signal': bool, 'price': float, 'stop_type': str, 'new_stop': float, 'bars_ago': int}
        """
        # For ML strategies, run the actual strategy exit logic
        if self._is_ml_strategy:
            return self._get_ml_exit_signal(df, params, entry_price, current_stop, symbol=symbol)

        # Legacy: simple percentage-based stop logic
        stop_pct = params.get('stop_loss_pct', 0.05)
        if hasattr(stop_pct, '__float__'):
            stop_pct = float(stop_pct)

        close = df['Close'].iloc[-1]
        pct_stop = entry_price * (1 - stop_pct)
        new_stop = max(pct_stop, current_stop)

        if close <= new_stop:
            bar_date = df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else str(df.index[-1])
            return {
                'signal': True,
                'price': close,
                'stop_type': 'STOP_LOSS',
                'new_stop': new_stop,
                'bars_ago': 0,
                'bar_date': bar_date
            }

        return {'signal': False, 'new_stop': new_stop}

    def _get_ml_exit_signal(self, df, params, entry_price, current_stop, symbol=None):
        """
        Run the ML strategy to detect exit signals for held positions.
        Simulates holding a position and checks if strategy would exit.

        Returns: {'signal': bool, 'price': float, 'stop_type': str, 'new_stop': float, 'bars_ago': int}
        """
        import gc
        from datetime import date
        cerebro = None

        stop_pct = params.get('stop_loss_pct', 0.05)
        if hasattr(stop_pct, '__float__'):
            stop_pct = float(stop_pct)

        current_price = float(df['Close'].iloc[-1])
        pct_stop = entry_price * (1 - stop_pct)
        new_stop = max(pct_stop, current_stop)

        result = {'signal': False, 'new_stop': new_stop}

        # === STOP LOSS CHECK (simple current price vs entry price) ===
        # This is checked FIRST, before running the full strategy simulation
        if params.get('use_stop_loss', True):
            current_pnl_pct = (current_price - entry_price) / entry_price
            if current_pnl_pct <= -stop_pct:
                return {
                    'signal': True,
                    'price': current_price,
                    'stop_type': 'STOP LOSS HIT',
                    'new_stop': new_stop,
                    'bars_ago': 0,
                    'bar_date': date.today().strftime('%Y-%m-%d')
                }

        try:
            # Prepare data for backtrader
            bt_df = df.copy()
            bt_df.columns = [c.lower() for c in bt_df.columns]

            required = ['open', 'high', 'low', 'close', 'volume']
            for col in required:
                if col not in bt_df.columns:
                    return result

            # Fix zero-range bars to prevent division-by-zero in ADX indicator
            zero_range = bt_df['high'] == bt_df['low']
            if zero_range.any():
                epsilon = bt_df['close'][zero_range] * 1e-6
                bt_df.loc[zero_range, 'high'] = bt_df.loc[zero_range, 'high'] + epsilon
                bt_df.loc[zero_range, 'low'] = bt_df.loc[zero_range, 'low'] - epsilon

            cerebro = bt.Cerebro(stdstats=False)

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

            strategy_params = params.copy()
            strategy_params['verbose'] = False

            # Set per-symbol params for cross-symbol training and fundamentals
            if symbol:
                strategy_params['cross_symbol_target_symbol'] = symbol
                strategy_params['fundamental_symbol'] = symbol

            # Create a strategy wrapper that simulates holding a position
            parent_strategy_class = self.strategy_class
            captured_entry_price = entry_price
            captured_total_bars = len(bt_df)  # Capture for closure

            class ExitSignalStrategy(parent_strategy_class):
                def __init__(self):
                    super().__init__()
                    self.exit_signals = []
                    self.simulated_position = False
                    self.sim_entry_price = captured_entry_price
                    self.sim_entry_bar = None

                def _execute_buy(self):
                    # Track when strategy would have entered
                    if not self.simulated_position:
                        self.simulated_position = True
                        self.sim_entry_bar = len(self)
                        # Set entry_price BEFORE calling super (which overwrites it)
                        # We need to use sim_entry_price (actual entry), not bar's close
                        self.entry_bar = self.sim_entry_bar
                        # Don't call super()._execute_buy() - it would overwrite entry_price
                        # and try to place an actual order
                        self.entry_price = self.sim_entry_price
                        self._highest_since_entry = self.sim_entry_price
                        atr_val = self.label_atr[0] if len(self.label_atr) > 0 else 0
                        if atr_val > 0:
                            self._trailing_stop_level = self.sim_entry_price - (self.p.trailing_atr_mult * atr_val)
                        else:
                            self._trailing_stop_level = 0

                def _close_position(self, reason):
                    # Capture exit signal
                    if self.simulated_position:
                        current_price = self.data.close[0]
                        pnl = current_price - self.entry_price
                        self.exit_signals.append({
                            'bar': len(self),
                            'date': self.data.datetime.date(0),
                            'price': current_price,
                            'reason': reason
                        })
                        # Track loss penalty (LZC8 feature)
                        if pnl < 0 and hasattr(self.p, 'use_loss_penalty') and self.p.use_loss_penalty:
                            penalty = self.p.loss_penalty_amount if self.p.loss_penalty_amount > 0 else getattr(self, 'entry_prediction', 1.0) * 2.5
                            if hasattr(self, 'loss_penalty'):
                                self.loss_penalty += penalty
                        self.simulated_position = False

                def next(self):
                    # Run parent logic but with simulated position
                    if len(self) < 50:
                        return

                    if self.order:
                        return

                    # Check earnings exit before main logic (LZC8 feature)
                    if self.simulated_position:
                        if (hasattr(self, 'fundamental_provider') and
                            self.fundamental_provider is not None and
                            hasattr(self.p, 'use_fundamental_filter') and
                            self.p.use_fundamental_filter and
                            hasattr(self.p, 'close_before_earnings') and
                            self.p.close_before_earnings):
                            current_date = self.data.datetime.date(0)
                            days_until = self.fundamental_provider.days_until_earnings(current_date)
                            if days_until is not None and 0 < days_until <= self.p.earnings_blackout_before:
                                self._close_position("EARNINGS BLACKOUT APPROACHING")
                                return

                    self._store_features()

                    # Label generation - handle both forward and legacy labels
                    if hasattr(self.p, 'use_forward_labels') and self.p.use_forward_labels:
                        atr_val = self.label_atr[0] if len(self.label_atr) > 0 else 0
                        self.pending_labels.append((len(self), self.data.close[0], atr_val))
                        self._resolve_pending_labels()
                    elif hasattr(self, '_calculate_label_legacy'):
                        label = self._calculate_label_legacy()
                        self.label_array.append(label)
                    elif hasattr(self, '_calculate_label'):
                        label = self._calculate_label()
                        self.label_array.append(label)

                    if self.p.test_start_idx > 0 and len(self) < self.p.test_start_idx:
                        return

                    self.prediction = self._run_knn()

                    # Apply loss penalty (LZC8 feature)
                    if hasattr(self.p, 'use_loss_penalty') and self.p.use_loss_penalty:
                        if hasattr(self, 'loss_penalty') and self.loss_penalty > 0:
                            if self.prediction > 0:
                                self.prediction -= self.loss_penalty
                            elif self.prediction < 0:
                                self.prediction += self.loss_penalty
                            self.loss_penalty *= self.p.loss_penalty_decay
                            if self.loss_penalty < 0.1:
                                self.loss_penalty = 0.0

                    # Prediction normalization (LZC8 feature)
                    if hasattr(self, '_normalize_prediction'):
                        self.raw_prediction = self.prediction
                        self.prediction = self._normalize_prediction(self.raw_prediction)

                    signal_changed = self._update_signal()

                    # Simulate having a position only in the LAST FEW BARS
                    # We only care about current exit signals, not historical ones
                    bars_from_end = captured_total_bars - len(self)

                    if not self.simulated_position and bars_from_end <= 5:
                        # Start simulating position in the last 5 bars
                        self.simulated_position = True
                        self.entry_price = self.sim_entry_price
                        self.entry_bar = len(self)
                        # Initialize trailing stop tracking
                        self._highest_since_entry = self.sim_entry_price
                        atr_val = self.label_atr[0] if len(self.label_atr) > 0 else 0
                        if atr_val > 0:
                            self._trailing_stop_level = self.sim_entry_price - (self.p.trailing_atr_mult * atr_val)
                        else:
                            self._trailing_stop_level = 0

                    # Check exit conditions if we have a simulated position
                    if self.simulated_position:
                        # Manually check exit conditions (can't use self.position)
                        self._check_simulated_exit(signal_changed)

                def _check_simulated_exit(self, signal_changed):
                    """Check exit conditions for simulated position.
                    Mirrors LZC8's _check_dynamic_exit/_check_strict_exit logic.
                    NOTE: Stop loss is checked upfront in _get_ml_exit_signal() using
                    current price vs actual entry price, not here in simulation."""
                    current_price = self.data.close[0]

                    # ATR trailing stop (LZC8 feature)
                    if hasattr(self.p, 'use_trailing_atr_exit') and self.p.use_trailing_atr_exit:
                        atr_val = self.label_atr[0] if len(self.label_atr) > 0 else 0
                        if atr_val > 0:
                            if current_price > self._highest_since_entry:
                                self._highest_since_entry = current_price
                            new_stop = self._highest_since_entry - (self.p.trailing_atr_mult * atr_val)
                            if new_stop > self._trailing_stop_level:
                                self._trailing_stop_level = new_stop
                            bars_in_trade = len(self) - self.entry_bar
                            if bars_in_trade >= self.p.trailing_atr_warmup:
                                if current_price < self._trailing_stop_level:
                                    self._close_position("ATR TRAILING STOP")
                                    return

                    # Dynamic vs strict exit modes
                    if self.p.use_dynamic_exits:
                        # No kernel available: fall back to signal flip
                        if not self.p.use_kernel_filter and not self.p.use_kernel_exit:
                            if signal_changed and self.signal == -1:
                                self._close_position("SIGNAL FLIP TO BEARISH")
                            return

                        if hasattr(self, 'kernel_rq') and len(self.kernel_rq) >= 2:
                            # Kernel line exit
                            if self.p.use_kernel_exit:
                                kernel_val = self.kernel_rq.estimate[0]
                                if current_price < kernel_val:
                                    self._close_position("PRICE BELOW KERNEL")
                                    return

                            # Kernel direction change exit
                            if self.p.use_kernel_filter:
                                was_bullish = self.kernel_rq.estimate[-2] < self.kernel_rq.estimate[-1]
                                is_bearish = self.kernel_rq.estimate[-1] > self.kernel_rq.estimate[0]
                                if was_bullish and is_bearish:
                                    self._close_position("KERNEL BEARISH CHANGE")
                                    return
                    else:
                        # Strict exit mode
                        # Signal flip exit
                        if signal_changed and self.signal == -1:
                            self._close_position("SIGNAL FLIP TO BEARISH")
                            return

                        # RSI exit
                        if self.p.use_rsi_exit and hasattr(self, 'rsi_exit'):
                            rsi_val = self.rsi_exit[0]
                            if rsi_val >= self.p.rsi_overbought:
                                self._close_position(f"RSI OVERBOUGHT ({rsi_val:.1f})")
                                return

                        # Kernel line exit
                        if self.p.use_kernel_exit and hasattr(self, 'kernel_rq'):
                            kernel_val = self.kernel_rq.estimate[0]
                            if current_price < kernel_val:
                                self._close_position("PRICE BELOW KERNEL")
                                return

            cerebro.addstrategy(ExitSignalStrategy, **strategy_params)
            cerebro.broker.setcash(1000000)
            cerebro.broker.setcommission(commission=0.0)

            results = cerebro.run()
            strat = results[0]

            # Check for recent exit signals (last 5 bars)
            total_bars = len(bt_df)
            recent_threshold = total_bars - 5

            for sig in reversed(strat.exit_signals):
                if sig['bar'] >= recent_threshold:
                    bars_ago = total_bars - sig['bar']
                    bar_date = sig['date'].strftime('%Y-%m-%d') if hasattr(sig['date'], 'strftime') else str(sig['date'])

                    result = {
                        'signal': True,
                        'price': sig['price'],
                        'stop_type': sig['reason'],
                        'new_stop': new_stop,
                        'bars_ago': bars_ago,
                        'bar_date': bar_date
                    }
                    break

        except Exception as e:
            print(f"\n❌ Error running ML exit signal detection: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if cerebro is not None:
                cerebro.runstop()
                del cerebro
            gc.collect()

        return result

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


def calculate_lookback(strategy_class, strategy_params=None):
    """
    Calculate required lookback period from strategy class defaults + overrides.
    Matches the calculate_lookback logic in backtest.py.

    Args:
        strategy_class: The backtrader Strategy class (to read default params)
        strategy_params: Optional dict of parameter overrides

    Returns:
        int: Required lookback bars
    """
    from decimal import Decimal

    # Start with strategy class defaults
    params_dict = {}
    for param_name in dir(strategy_class.params):
        if not param_name.startswith('_'):
            param_value = getattr(strategy_class.params, param_name)
            params_dict[param_name] = param_value

    # Override with user-provided params
    if strategy_params:
        params_dict.update(strategy_params)

    lookback_candidates = []
    for param_name, param_value in params_dict.items():
        if isinstance(param_value, Decimal):
            try:
                param_value = int(param_value)
            except (ValueError, TypeError):
                continue
        if isinstance(param_value, int) and param_value > 0:
            exclude_patterns = ['verbose', 'plot', 'print', 'count', 'feature']
            if not any(pattern in param_name.lower() for pattern in exclude_patterns):
                lookback_candidates.append(param_value)

    # Need extra warmup for ML model training data
    max_lookback = (max(lookback_candidates) if lookback_candidates else 50) + 100

    print(f"\n📊 Calculated Lookback: {max_lookback} bars")
    print(f"   Found period parameters: {sorted(lookback_candidates, reverse=True)[:5]}")

    return max_lookback
