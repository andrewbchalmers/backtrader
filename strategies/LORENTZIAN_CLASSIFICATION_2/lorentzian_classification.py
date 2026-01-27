"""
Machine Learning: Lorentzian Classification Strategy

A backtrader implementation of the TradingView indicator by @jdehorty.

This strategy uses a K-Nearest Neighbors (KNN) machine learning algorithm with
Lorentzian distance instead of Euclidean distance for classification.

Key Features:
- Lorentzian distance metric (reduces influence of outliers)
- Multiple configurable features (RSI, Wave Trend, CCI, ADX)
- Approximate Nearest Neighbors (ANN) search algorithm
- Multiple filters (Volatility, Regime, ADX, EMA, SMA)
- Nadaraya-Watson Kernel Regression for trend confirmation
- Configurable entry/exit logic

Note on Label Alignment:
The original TradingView implementation uses backward-looking labels which
effectively creates a mean-reversion bias. This implementation preserves
that behavior for accuracy to the original, but can be configured for
forward-looking labels if desired.

Author: Backtrader implementation based on TradingView indicator by @jdehorty
"""

import math
from decimal import Decimal
from collections import deque
import backtrader as bt
import numpy as np


# =============================================================================
# Custom Indicators
# =============================================================================

class NormalizedRSI(bt.Indicator):
    """
    Normalized RSI indicator.
    Returns RSI rescaled to approximately -1 to 1 range.
    """
    lines = ('nrsi',)
    params = (
        ('period', 14),
        ('smoothing', 1),
    )

    def __init__(self):
        rsi = bt.indicators.RSI(self.data, period=self.p.period)
        if self.p.smoothing > 1:
            rsi = bt.indicators.EMA(rsi, period=self.p.smoothing)
        # Normalize: (RSI - 50) / 50 gives range -1 to 1
        self.lines.nrsi = (rsi - 50) / 50


class WaveTrend(bt.Indicator):
    """
    Wave Trend Oscillator (LazyBear style).
    Used as a momentum indicator.
    """
    lines = ('wt1', 'wt2', 'nwt')
    params = (
        ('channel_len', 10),
        ('avg_len', 21),
    )

    def __init__(self):
        hlc3 = (self.data.high + self.data.low + self.data.close) / 3
        esa = bt.indicators.EMA(hlc3, period=self.p.channel_len)
        d = bt.indicators.EMA(abs(hlc3 - esa), period=self.p.channel_len)
        # Avoid division by zero
        ci = (hlc3 - esa) / (0.015 * d + 0.0001)
        self.lines.wt1 = bt.indicators.EMA(ci, period=self.p.avg_len)
        self.lines.wt2 = bt.indicators.SMA(self.lines.wt1, period=4)
        # Normalized version (typical WT ranges from -100 to 100)
        self.lines.nwt = self.lines.wt1 / 100


class NormalizedCCI(bt.Indicator):
    """
    Normalized CCI indicator.
    Returns CCI rescaled to approximately -1 to 1 range.
    """
    lines = ('ncci',)
    params = (
        ('period', 20),
        ('smoothing', 1),
    )

    def __init__(self):
        cci = bt.indicators.CCI(self.data, period=self.p.period)
        if self.p.smoothing > 1:
            cci = bt.indicators.EMA(cci, period=self.p.smoothing)
        # Normalize: CCI typically ranges -200 to 200, divide by 200
        self.lines.ncci = cci / 200


class NormalizedADX(bt.Indicator):
    """
    Normalized ADX indicator.
    Returns ADX rescaled to 0 to 1 range.
    """
    lines = ('nadx',)
    params = (
        ('period', 14),
    )

    def __init__(self):
        adx = bt.indicators.ADX(self.data, period=self.p.period)
        # Normalize: ADX ranges 0-100, divide by 100
        self.lines.nadx = adx / 50 - 1  # Scale to -1 to 1


class RationalQuadraticKernel(bt.Indicator):
    """
    Nadaraya-Watson Kernel Regression using Rational Quadratic Kernel.

    The Rational Quadratic kernel is a mixture of Gaussian kernels
    with different length scales.
    """
    lines = ('estimate',)
    params = (
        ('lookback', 8),      # h - lookback window
        ('rel_weight', 8.0),  # r - relative weighting
        ('start_bar', 25),    # x - regression start bar
    )

    def __init__(self):
        self.addminperiod(self.p.lookback + self.p.start_bar)

    def next(self):
        lookback = min(self.p.lookback, len(self) - 1)
        if lookback < 1:
            self.lines.estimate[0] = self.data[0]
            return

        weights_sum = 0.0
        weighted_sum = 0.0

        for i in range(lookback):
            # Rational Quadratic Kernel
            # K(x) = (1 + x^2 / (2 * r * h^2))^(-r)
            w = math.pow(
                1 + (i * i) / (2 * self.p.rel_weight * self.p.lookback * self.p.lookback),
                -self.p.rel_weight
            )
            weighted_sum += self.data[-i] * w
            weights_sum += w

        if weights_sum > 0:
            self.lines.estimate[0] = weighted_sum / weights_sum
        else:
            self.lines.estimate[0] = self.data[0]


class GaussianKernel(bt.Indicator):
    """
    Nadaraya-Watson Kernel Regression using Gaussian Kernel.
    """
    lines = ('estimate',)
    params = (
        ('lookback', 8),
        ('start_bar', 25),
    )

    def __init__(self):
        self.addminperiod(self.p.lookback + self.p.start_bar)

    def next(self):
        lookback = min(self.p.lookback, len(self) - 1)
        if lookback < 1:
            self.lines.estimate[0] = self.data[0]
            return

        weights_sum = 0.0
        weighted_sum = 0.0

        for i in range(lookback):
            # Gaussian Kernel: K(x) = exp(-x^2 / (2 * h^2))
            w = math.exp(-(i * i) / (2 * self.p.lookback * self.p.lookback))
            weighted_sum += self.data[-i] * w
            weights_sum += w

        if weights_sum > 0:
            self.lines.estimate[0] = weighted_sum / weights_sum
        else:
            self.lines.estimate[0] = self.data[0]


class VolatilityFilter(bt.Indicator):
    """
    Volatility filter based on ATR comparison.
    Returns True when volatility is within acceptable range.
    """
    lines = ('filter',)
    params = (
        ('min_length', 1),
        ('max_length', 10),
    )

    def __init__(self):
        self.atr_min = bt.indicators.ATR(self.data, period=self.p.min_length)
        self.atr_max = bt.indicators.ATR(self.data, period=self.p.max_length)

    def next(self):
        # Filter passes when short-term ATR < long-term ATR (calm conditions)
        self.lines.filter[0] = 1.0 if self.atr_min[0] <= self.atr_max[0] else 0.0


class RegimeFilter(bt.Indicator):
    """
    Regime filter using Ehlers Super Smoother and highpass filter.
    Detects trending vs ranging market conditions.
    """
    lines = ('filter', 'klmf')
    params = (
        ('threshold', -0.1),
    )

    def __init__(self):
        self.addminperiod(50)  # Need warmup for the filter

    def next(self):
        # Simplified regime detection using price momentum
        if len(self) < 50:
            self.lines.filter[0] = 1.0
            self.lines.klmf[0] = 0.0
            return

        # Calculate simple momentum-based regime
        # Using rate of change as proxy for regime
        prices = [self.data.close[-i] for i in range(min(20, len(self)))]
        if len(prices) >= 20:
            momentum = (prices[0] - prices[-1]) / prices[-1] if prices[-1] != 0 else 0
            self.lines.klmf[0] = momentum
            self.lines.filter[0] = 1.0 if momentum > self.p.threshold else 0.0
        else:
            self.lines.filter[0] = 1.0
            self.lines.klmf[0] = 0.0


# =============================================================================
# Main Strategy
# =============================================================================

class LorentzianClassificationStrategy(bt.Strategy):
    """
    Machine Learning Lorentzian Classification Strategy.

    Uses K-Nearest Neighbors with Lorentzian distance metric for
    price direction classification.
    """

    params = (
        # === General Settings ===
        ('neighbors_count', 8),          # Number of neighbors for KNN
        ('max_bars_back', 2000),         # Maximum lookback for training data
        ('feature_count', 3),            # Number of features (2-5)

        # === Feature 1 (RSI) ===
        ('f1_type', 'RSI'),
        ('f1_param_a', 14),
        ('f1_param_b', 1),

        # === Feature 2 (Wave Trend) ===
        ('f2_type', 'WT'),
        ('f2_param_a', 10),
        ('f2_param_b', 11),

        # === Feature 3 (CCI) ===
        ('f3_type', 'CCI'),
        ('f3_param_a', 20),
        ('f3_param_b', 1),

        # === Feature 4 (ADX) ===
        ('f4_type', 'ADX'),
        ('f4_param_a', 20),
        ('f4_param_b', 2),

        # === Feature 5 (RSI) ===
        ('f5_type', 'RSI'),
        ('f5_param_a', 9),
        ('f5_param_b', 1),

        # === Filters ===
        ('use_volatility_filter', False),
        ('use_regime_filter', True),
        ('regime_threshold', -0.2),
        ('use_adx_filter', False),
        ('adx_threshold', 20),
        ('use_ema_filter', False),
        ('ema_period', 200),
        ('use_sma_filter', False),
        ('sma_period', 200),

        # === Kernel Settings ===
        ('use_kernel_filter', True),
        ('use_kernel_smoothing', False),
        ('kernel_lookback', 8),
        ('kernel_rel_weight', 8.0),
        ('kernel_start_bar', 25),
        ('kernel_lag', 2),

        # === Exit Settings ===
        ('use_dynamic_exits', False),
        ('bars_to_hold', 1000),  # Default holding period

        # === RSI Exit Settings ===
        ('use_rsi_exit', True),          # Enable RSI threshold exits
        ('rsi_exit_period', 14),         # RSI period for exit signals
        ('rsi_overbought', 70),          # Exit longs when RSI crosses above this
        ('rsi_oversold', 30),            # Exit shorts when RSI crosses below this

        # === Risk Management ===
        ('position_size_pct', Decimal('0.95')),
        ('stop_loss_pct', Decimal('0.05')),
        ('use_stop_loss', False),

        # === Trade Direction ===
        ('long_only', True),  # Set to False to enable short selling

        # === Display ===
        ('verbose', True),

        # === Backtest Control ===
        ('test_start_idx', 0),  # Bar index to start trading (0 = trade from start)
    )

    def __init__(self):
        """Initialize indicators and state."""
        self._init_features()
        self._init_filters()
        self._init_kernels()
        self._init_state()

    def _init_features(self):
        """Initialize feature indicators."""
        self.features = []

        feature_configs = [
            (self.p.f1_type, self.p.f1_param_a, self.p.f1_param_b),
            (self.p.f2_type, self.p.f2_param_a, self.p.f2_param_b),
            (self.p.f3_type, self.p.f3_param_a, self.p.f3_param_b),
            (self.p.f4_type, self.p.f4_param_a, self.p.f4_param_b),
            (self.p.f5_type, self.p.f5_param_a, self.p.f5_param_b),
        ]

        for i, (ftype, param_a, param_b) in enumerate(feature_configs[:self.p.feature_count]):
            feature = self._create_feature(ftype, param_a, param_b)
            self.features.append(feature)

    def _create_feature(self, ftype, param_a, param_b):
        """Create a feature indicator based on type."""
        if ftype == 'RSI':
            return NormalizedRSI(self.data, period=param_a, smoothing=param_b)
        elif ftype == 'WT':
            return WaveTrend(self.data, channel_len=param_a, avg_len=param_b)
        elif ftype == 'CCI':
            return NormalizedCCI(self.data, period=param_a, smoothing=param_b)
        elif ftype == 'ADX':
            return NormalizedADX(self.data, period=param_a)
        else:
            raise ValueError(f"Unknown feature type: {ftype}")

    def _init_filters(self):
        """Initialize filter indicators."""
        # Volatility filter
        if self.p.use_volatility_filter:
            self.volatility_filter = VolatilityFilter(self.data)

        # Regime filter
        if self.p.use_regime_filter:
            self.regime_filter = RegimeFilter(self.data, threshold=self.p.regime_threshold)

        # ADX filter
        if self.p.use_adx_filter:
            self.adx = bt.indicators.ADX(self.data, period=14)

        # EMA filter
        if self.p.use_ema_filter:
            self.ema = bt.indicators.EMA(self.data.close, period=self.p.ema_period)

        # SMA filter
        if self.p.use_sma_filter:
            self.sma = bt.indicators.SMA(self.data.close, period=self.p.sma_period)

        # RSI for exit signals
        if self.p.use_rsi_exit:
            self.rsi_exit = bt.indicators.RSI(self.data.close, period=self.p.rsi_exit_period)

    def _init_kernels(self):
        """Initialize kernel regression indicators."""
        if self.p.use_kernel_filter:
            self.kernel_rq = RationalQuadraticKernel(
                self.data.close,
                lookback=self.p.kernel_lookback,
                rel_weight=self.p.kernel_rel_weight,
                start_bar=self.p.kernel_start_bar
            )
            self.kernel_gaussian = GaussianKernel(
                self.data.close,
                lookback=self.p.kernel_lookback - self.p.kernel_lag,
                start_bar=self.p.kernel_start_bar
            )

    def _init_state(self):
        """Initialize strategy state variables."""
        # ML state
        self.feature_arrays = [deque(maxlen=self.p.max_bars_back) for _ in range(self.p.feature_count)]
        self.label_array = deque(maxlen=self.p.max_bars_back)

        # Trading state
        self.order = None
        self.signal = 0  # 1 = long, -1 = short, 0 = neutral
        self.bars_held = 0
        self.entry_bar = 0
        self.entry_price = 0
        self.prediction = 0

        # ML Prediction Accuracy Tracking
        # Store predictions as: (bar_idx, prediction, price_at_prediction)
        self.pending_predictions = []
        self.prediction_results = {
            'total': 0,
            'correct': 0,
            'bullish_total': 0,
            'bullish_correct': 0,
            'bearish_total': 0,
            'bearish_correct': 0,
            'neutral': 0,  # predictions of 0
        }
        self.prediction_lookforward = 4  # Bars to look forward for validation

    def _get_lorentzian_distance(self, idx):
        """
        Calculate Lorentzian distance between current features and historical features.

        Lorentzian distance: sum of log(1 + |x_i - y_i|) for each feature

        This metric reduces the influence of outliers compared to Euclidean distance.
        """
        distance = 0.0
        for i, feature in enumerate(self.features):
            if idx < len(self.feature_arrays[i]):
                current_val = self._get_feature_value(feature)
                historical_val = self.feature_arrays[i][idx]
                distance += math.log(1 + abs(current_val - historical_val))
        return distance

    def _get_feature_value(self, feature):
        """Get the current value from a feature indicator."""
        if hasattr(feature, 'nrsi'):
            return feature.nrsi[0]
        elif hasattr(feature, 'nwt'):
            return feature.nwt[0]
        elif hasattr(feature, 'ncci'):
            return feature.ncci[0]
        elif hasattr(feature, 'nadx'):
            return feature.nadx[0]
        return 0.0

    def _store_features(self):
        """Store current feature values in arrays."""
        for i, feature in enumerate(self.features):
            val = self._get_feature_value(feature)
            self.feature_arrays[i].append(val)

    def _calculate_label(self):
        """
        Calculate training label based on price movement.

        Note: The original TradingView implementation uses backward-looking labels:
        - If price rose over past 4 bars -> SHORT label
        - If price fell over past 4 bars -> LONG label

        This creates a mean-reversion bias in the model.
        """
        if len(self) < 5:
            return 0

        current_price = self.data.close[0]
        past_price = self.data.close[-4]

        if past_price < current_price:
            return -1  # SHORT (price rose, expect reversal)
        elif past_price > current_price:
            return 1   # LONG (price fell, expect reversal)
        return 0

    def _run_knn(self):
        """
        Run Approximate Nearest Neighbors search with Lorentzian distance.

        Key optimizations from original:
        1. Only sample every 4th bar for chronological spacing
        2. Maintain sliding window of k neighbors
        3. Use 75th percentile distance reset to prevent runaway
        """
        if len(self.label_array) < self.p.neighbors_count:
            return 0

        distances = []
        predictions = []
        last_distance = -1.0

        size_loop = min(self.p.max_bars_back - 1, len(self.label_array) - 1)

        for i in range(size_loop):
            d = self._get_lorentzian_distance(i)

            # Only consider every 4th bar (chronological spacing)
            if d >= last_distance and (i % 4) != 0:
                last_distance = d
                distances.append(d)
                predictions.append(self.label_array[i])

                # Maintain k-nearest neighbors
                if len(predictions) > self.p.neighbors_count:
                    # Reset distance threshold to 75th percentile
                    sorted_dist = sorted(distances)
                    idx_75 = int(self.p.neighbors_count * 3 / 4)
                    if idx_75 < len(sorted_dist):
                        last_distance = sorted_dist[idx_75]
                    distances.pop(0)
                    predictions.pop(0)

        return sum(predictions) if predictions else 0

    def _check_filters(self):
        """Check all filter conditions."""
        # Volatility filter
        if self.p.use_volatility_filter:
            if self.volatility_filter.filter[0] <= 0:
                return False

        # Regime filter
        if self.p.use_regime_filter:
            if self.regime_filter.filter[0] <= 0:
                return False

        # ADX filter
        if self.p.use_adx_filter:
            if self.adx[0] < self.p.adx_threshold:
                return False

        return True

    def _check_ema_uptrend(self):
        """Check if price is above EMA."""
        if not self.p.use_ema_filter:
            return True
        return self.data.close[0] > self.ema[0]

    def _check_ema_downtrend(self):
        """Check if price is below EMA."""
        if not self.p.use_ema_filter:
            return True
        return self.data.close[0] < self.ema[0]

    def _check_sma_uptrend(self):
        """Check if price is above SMA."""
        if not self.p.use_sma_filter:
            return True
        return self.data.close[0] > self.sma[0]

    def _check_sma_downtrend(self):
        """Check if price is below SMA."""
        if not self.p.use_sma_filter:
            return True
        return self.data.close[0] < self.sma[0]

    def _check_kernel_bullish(self):
        """Check kernel regression for bullish signal."""
        if not self.p.use_kernel_filter:
            return True

        if self.p.use_kernel_smoothing:
            # Crossover-based: Gaussian above Rational Quadratic
            return self.kernel_gaussian.estimate[0] >= self.kernel_rq.estimate[0]
        else:
            # Rate-based: Kernel is rising
            if len(self.kernel_rq) < 2:
                return True
            return self.kernel_rq.estimate[0] > self.kernel_rq.estimate[-1]

    def _check_kernel_bearish(self):
        """Check kernel regression for bearish signal."""
        if not self.p.use_kernel_filter:
            return True

        if self.p.use_kernel_smoothing:
            # Crossover-based: Gaussian below Rational Quadratic
            return self.kernel_gaussian.estimate[0] <= self.kernel_rq.estimate[0]
        else:
            # Rate-based: Kernel is falling
            if len(self.kernel_rq) < 2:
                return True
            return self.kernel_rq.estimate[0] < self.kernel_rq.estimate[-1]

    def _update_signal(self):
        """Update trading signal based on ML prediction and filters."""
        old_signal = self.signal

        if self.prediction > 0 and self._check_filters():
            self.signal = 1  # Long
        elif self.prediction < 0 and self._check_filters():
            self.signal = -1  # Short
        # else keep previous signal

        # Track signal changes
        if old_signal != self.signal:
            self.bars_held = 0
        else:
            self.bars_held += 1

        return old_signal != self.signal

    def next(self):
        """Main strategy logic called on each bar."""
        # Skip if not enough data for indicators
        if len(self) < 50:
            return

        # Force exit on final bar
        if self._is_final_bar():
            self._force_close_position()
            return

        # Skip if order pending
        if self.order:
            return

        # Store features and labels (always do this to build training data)
        self._store_features()
        label = self._calculate_label()
        self.label_array.append(label)

        # Skip trading if before test period start
        # (still accumulate training data above, just don't trade)
        if self.p.test_start_idx > 0 and len(self) < self.p.test_start_idx:
            return

        # Run ML prediction
        self.prediction = self._run_knn()

        # === ML Prediction Accuracy Tracking ===
        # Validate old predictions that have matured
        self._validate_predictions()

        # Store new prediction for future validation
        if self.prediction != 0:
            self.pending_predictions.append({
                'bar_idx': len(self),
                'prediction': self.prediction,
                'price': self.data.close[0],
            })

        # Update signal
        signal_changed = self._update_signal()

        # Check for entries
        if not self.position:
            self._check_entry(signal_changed)
        else:
            self._check_exit(signal_changed)

    def _validate_predictions(self):
        """
        Validate predictions that are now old enough to check.
        A prediction is correct if:
        - Bullish (>0): price increased over lookforward period
        - Bearish (<0): price decreased over lookforward period
        """
        current_bar = len(self)
        current_price = self.data.close[0]

        # Check predictions that are old enough
        still_pending = []
        for pred in self.pending_predictions:
            bars_elapsed = current_bar - pred['bar_idx']

            if bars_elapsed >= self.prediction_lookforward:
                # Prediction is mature, validate it
                price_change = current_price - pred['price']
                prediction = pred['prediction']

                self.prediction_results['total'] += 1

                if prediction > 0:  # Bullish prediction
                    self.prediction_results['bullish_total'] += 1
                    if price_change > 0:  # Price went up - correct
                        self.prediction_results['correct'] += 1
                        self.prediction_results['bullish_correct'] += 1
                elif prediction < 0:  # Bearish prediction
                    self.prediction_results['bearish_total'] += 1
                    if price_change < 0:  # Price went down - correct
                        self.prediction_results['correct'] += 1
                        self.prediction_results['bearish_correct'] += 1
            else:
                # Keep for later validation
                still_pending.append(pred)

        self.pending_predictions = still_pending

    def get_prediction_stats(self):
        """
        Get ML prediction accuracy statistics.
        Returns dict with accuracy metrics.
        """
        stats = self.prediction_results.copy()

        # Calculate accuracy percentages
        if stats['total'] > 0:
            stats['accuracy_pct'] = (stats['correct'] / stats['total']) * 100
        else:
            stats['accuracy_pct'] = 0

        if stats['bullish_total'] > 0:
            stats['bullish_accuracy_pct'] = (stats['bullish_correct'] / stats['bullish_total']) * 100
        else:
            stats['bullish_accuracy_pct'] = 0

        if stats['bearish_total'] > 0:
            stats['bearish_accuracy_pct'] = (stats['bearish_correct'] / stats['bearish_total']) * 100
        else:
            stats['bearish_accuracy_pct'] = 0

        # Prediction bias (how often model predicts bullish vs bearish)
        total_directional = stats['bullish_total'] + stats['bearish_total']
        if total_directional > 0:
            stats['bullish_bias_pct'] = (stats['bullish_total'] / total_directional) * 100
        else:
            stats['bullish_bias_pct'] = 50

        return stats

    def _check_entry(self, signal_changed):
        """Check for entry conditions."""
        # Long entry
        is_new_buy = (
            signal_changed and
            self.signal == 1 and
            self._check_kernel_bullish() and
            self._check_ema_uptrend() and
            self._check_sma_uptrend()
        )

        # Short entry (only if long_only=False)
        is_new_short = (
            not self.p.long_only and
            signal_changed and
            self.signal == -1 and
            self._check_kernel_bearish() and
            self._check_ema_downtrend() and
            self._check_sma_downtrend()
        )

        if is_new_buy:
            self._execute_buy()
        elif is_new_short:
            self._execute_short()

    def _check_exit(self, signal_changed):
        """Check for exit conditions."""
        if self.p.use_dynamic_exits:
            self._check_dynamic_exit()
        else:
            self._check_strict_exit(signal_changed)

    def _check_strict_exit(self, signal_changed):
        """
        Check for strict exit conditions.
        Exit after bars_to_hold bars or on signal flip.
        """
        bars_since_entry = len(self) - self.entry_bar

        # Exit after holding period
        if bars_since_entry >= self.p.bars_to_hold:
            self._close_position("HOLDING PERIOD COMPLETE")
            return

        # Exit long on bearish signal flip
        if self.position.size > 0 and signal_changed and self.signal == -1:
            self._close_position("SIGNAL FLIP TO BEARISH")
            return

        # Exit short on bullish signal flip
        if self.position.size < 0 and signal_changed and self.signal == 1:
            self._close_position("SIGNAL FLIP TO BULLISH")
            return

        # RSI threshold exit
        if self.p.use_rsi_exit:
            rsi_val = self.rsi_exit[0]
            # Exit long when RSI crosses above overbought threshold
            if self.position.size > 0 and rsi_val >= self.p.rsi_overbought:
                self._close_position(f"RSI OVERBOUGHT ({rsi_val:.1f})")
                return
            # Exit short when RSI crosses below oversold threshold
            if self.position.size < 0 and rsi_val <= self.p.rsi_oversold:
                self._close_position(f"RSI OVERSOLD ({rsi_val:.1f})")
                return

        # Stop loss
        if self.p.use_stop_loss:
            if self.position.size > 0:
                # Long position stop loss
                current_pnl_pct = (self.data.close[0] - self.entry_price) / self.entry_price
            else:
                # Short position stop loss
                current_pnl_pct = (self.entry_price - self.data.close[0]) / self.entry_price

            if current_pnl_pct <= -float(self.p.stop_loss_pct):
                self._close_position("STOP LOSS HIT")

    def _check_dynamic_exit(self):
        """Check for dynamic exit based on kernel regression."""
        if len(self.kernel_rq) < 2:
            return

        # Exit long when kernel turns bearish
        if self.position.size > 0:
            was_bullish = self.kernel_rq.estimate[-2] < self.kernel_rq.estimate[-1]
            is_bearish = self.kernel_rq.estimate[-1] > self.kernel_rq.estimate[0]
            if was_bullish and is_bearish:
                self._close_position("KERNEL BEARISH CHANGE")

        # Exit short when kernel turns bullish
        elif self.position.size < 0:
            was_bearish = self.kernel_rq.estimate[-2] > self.kernel_rq.estimate[-1]
            is_bullish = self.kernel_rq.estimate[-1] < self.kernel_rq.estimate[0]
            if was_bearish and is_bullish:
                self._close_position("KERNEL BULLISH CHANGE")

    def _execute_buy(self):
        """Execute buy order (go long)."""
        size = self._calculate_position_size()
        if size > 0:
            self.order = self.buy(size=size)
            self.entry_bar = len(self)
            self.entry_price = self.data.close[0]

            if self.p.verbose:
                print(f"BUY SIGNAL: {self.data.datetime.date(0)} | "
                      f"Prediction: {self.prediction} | "
                      f"Price: ${self.data.close[0]:.2f}")

    def _execute_short(self):
        """Execute short sell order (go short)."""
        size = self._calculate_position_size()
        if size > 0:
            self.order = self.sell(size=size)
            self.entry_bar = len(self)
            self.entry_price = self.data.close[0]

            if self.p.verbose:
                print(f"SHORT SIGNAL: {self.data.datetime.date(0)} | "
                      f"Prediction: {self.prediction} | "
                      f"Price: ${self.data.close[0]:.2f}")

    def _close_position(self, reason):
        """Close current position (long or short)."""
        if self.position.size > 0:
            # Close long position
            pnl = (self.data.close[0] - self.entry_price) * self.position.size
            if self.p.verbose:
                print(f"CLOSE LONG: {reason} | {self.data.datetime.date(0)} | "
                      f"Entry: ${self.entry_price:.2f} | "
                      f"Exit: ${self.data.close[0]:.2f} | "
                      f"P&L: ${pnl:.2f}")
            self.order = self.sell(size=self.position.size)

        elif self.position.size < 0:
            # Close short position
            pnl = (self.entry_price - self.data.close[0]) * abs(self.position.size)
            if self.p.verbose:
                print(f"CLOSE SHORT: {reason} | {self.data.datetime.date(0)} | "
                      f"Entry: ${self.entry_price:.2f} | "
                      f"Exit: ${self.data.close[0]:.2f} | "
                      f"P&L: ${pnl:.2f}")
            self.order = self.buy(size=abs(self.position.size))

    def _calculate_position_size(self):
        """Calculate position size based on available cash."""
        cash = self.broker.getcash()
        position_value = cash * float(self.p.position_size_pct)
        size = int(position_value / self.data.close[0])
        return max(0, size)

    def _is_final_bar(self):
        """Check if current bar is the last available bar."""
        return len(self.data) == self.data.buflen()

    def _force_close_position(self):
        """Close position on final bar."""
        if self.position:
            if self.p.verbose:
                print(f"FINAL BAR - Closing position at ${self.data.close[0]:.2f}")
            if self.position.size > 0:
                self.order = self.sell(size=self.position.size)
            elif self.position.size < 0:
                self.order = self.buy(size=abs(self.position.size))

    def notify_order(self, order):
        """Handle order status notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if self.p.verbose and order.isbuy():
                print(f"BUY Executed: {self.data.datetime.date(0)}, "
                      f"size={order.executed.size}, "
                      f"price=${order.executed.price:.2f}")
            elif self.p.verbose and order.issell():
                print(f"SELL Executed: {self.data.datetime.date(0)}, "
                      f"size={order.executed.size}, "
                      f"price=${order.executed.price:.2f}")

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if self.p.verbose:
                print(f"ORDER FAILED: {self.data.datetime.date(0)}, "
                      f"status={order.getstatusname()}")

        self.order = None


# Alias for easier import
Strategy = LorentzianClassificationStrategy
