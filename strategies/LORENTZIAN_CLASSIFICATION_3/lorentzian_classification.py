"""
Machine Learning: Lorentzian Classification Strategy - Trend Features

A backtrader implementation using trend-focused features for better pattern matching.

This version uses a 5-feature vector optimized for trend detection:
1. RSI(14) - Momentum/overbought-oversold
2. ADX(14) - Trend strength
3. ATR Ratio - Normalized volatility (ATR/Close)
4. Price Position - Where price sits in recent range (0-1)
5. Efficiency Ratio - Kaufman's trend efficiency metric

These features provide more meaningful data for Lorentzian distance calculations
when identifying similar market conditions and trend behavior patterns.

Author: Backtrader implementation based on TradingView indicator by @jdehorty
Modified: Trend-focused feature vector
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


class NormalizedADX(bt.Indicator):
    """
    Normalized ADX indicator.
    Returns ADX rescaled to -1 to 1 range.
    """
    lines = ('nadx',)
    params = (
        ('period', 14),
    )

    def __init__(self):
        adx = bt.indicators.ADX(self.data, period=self.p.period)
        # Normalize: ADX ranges 0-100, scale to -1 to 1
        self.lines.nadx = adx / 50 - 1


class ATRRatio(bt.Indicator):
    """
    ATR Ratio - Normalized volatility indicator.

    Measures ATR relative to closing price, providing a normalized
    volatility measure that works across different price levels.

    Formula: (ATR / Close) * 100, then normalized to -1 to 1
    Typical range: 0.5% to 5% of price
    """
    lines = ('atr_ratio',)
    params = (
        ('period', 14),
    )

    def __init__(self):
        self.atr = bt.indicators.ATR(self.data, period=self.p.period)
        self.addminperiod(self.p.period)

    def next(self):
        if self.data.close[0] > 0:
            # ATR as percentage of price
            ratio_pct = (self.atr[0] / self.data.close[0]) * 100
            # Normalize: typical range 0.5-5%, center at 2.5%, scale to -1 to 1
            # (ratio - 2.5) / 2.5 gives roughly -1 to 1 for typical values
            self.lines.atr_ratio[0] = (ratio_pct - 2.5) / 2.5
        else:
            self.lines.atr_ratio[0] = 0


class PricePosition(bt.Indicator):
    """
    Price Position indicator.

    Shows where current price sits within the recent price range.
    Similar to Stochastic %K but normalized to -1 to 1.

    Formula: ((Close - Lowest Low) / (Highest High - Lowest Low)) * 2 - 1

    Values:
    - +1: Price at highest point of range
    - 0: Price at midpoint of range
    - -1: Price at lowest point of range
    """
    lines = ('position',)
    params = (
        ('period', 14),
    )

    def __init__(self):
        self.highest = bt.indicators.Highest(self.data.high, period=self.p.period)
        self.lowest = bt.indicators.Lowest(self.data.low, period=self.p.period)
        self.addminperiod(self.p.period)

    def next(self):
        high_low_range = self.highest[0] - self.lowest[0]
        if high_low_range > 0:
            # Position in range: 0 to 1
            pos = (self.data.close[0] - self.lowest[0]) / high_low_range
            # Scale to -1 to 1
            self.lines.position[0] = pos * 2 - 1
        else:
            self.lines.position[0] = 0


class EfficiencyRatio(bt.Indicator):
    """
    Kaufman's Efficiency Ratio (ER).

    Measures trend efficiency by comparing directional movement to total movement.
    Used in Kaufman's Adaptive Moving Average (KAMA).

    Formula: ER = abs(Close - Close[n]) / sum(abs(Close[i] - Close[i-1]))

    Values:
    - 1.0: Perfect trend (price moved in one direction only)
    - 0.0: No net movement (choppy/ranging market)

    Normalized to -1 to 1 range: (ER * 2) - 1
    """
    lines = ('er',)
    params = (
        ('period', 14),
    )

    def __init__(self):
        self.addminperiod(self.p.period + 1)

    def next(self):
        if len(self) < self.p.period + 1:
            self.lines.er[0] = 0
            return

        # Direction: net price change over period
        direction = abs(self.data.close[0] - self.data.close[-self.p.period])

        # Volatility: sum of absolute bar-to-bar changes
        volatility = 0.0
        for i in range(self.p.period):
            volatility += abs(self.data.close[-i] - self.data.close[-i-1])

        # Efficiency Ratio
        if volatility > 0:
            er = direction / volatility
        else:
            er = 0

        # Normalize to -1 to 1 (ER naturally ranges 0 to 1)
        self.lines.er[0] = er * 2 - 1


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
    Machine Learning Lorentzian Classification Strategy - Trend Features.

    Uses K-Nearest Neighbors with Lorentzian distance metric for
    price direction classification.

    This version uses trend-focused features:
    - RSI(14): Momentum
    - ADX(14): Trend strength
    - ATR Ratio: Normalized volatility
    - Price Position: Location in recent range
    - Efficiency Ratio: Trend quality
    """

    params = (
        # === General Settings ===
        ('neighbors_count', 9),          # Number of neighbors for KNN
        ('max_bars_back', 2000),         # Maximum lookback for training data
        ('feature_count', 5),            # Number of features (2-5)
        ('trend_following_labels', False),  # False=mean-reversion labels, True=trend-following labels
        ('allow_reentry', True),         # True=enter anytime signal favorable, False=only on signal flip
        ('min_prediction_strength', 4),  # Minimum |prediction| to generate signal (0=any, 4=half neighbors agree)

        # === Feature 1 (RSI) ===
        ('f1_type', 'RSI'),
        ('f1_param_a', 14),              # RSI period
        ('f1_param_b', 1),               # Smoothing (1 = none)

        # === Feature 2 (ADX) ===
        ('f2_type', 'ADX'),
        ('f2_param_a', 14),              # ADX period
        ('f2_param_b', 1),               # Not used

        # === Feature 3 (ATR Ratio) ===
        ('f3_type', 'ATRR'),
        ('f3_param_a', 14),              # ATR period
        ('f3_param_b', 1),               # Not used

        # === Feature 4 (Price Position) ===
        ('f4_type', 'PP'),
        ('f4_param_a', 14),              # Lookback period
        ('f4_param_b', 1),               # Not used

        # === Feature 5 (Efficiency Ratio) ===
        ('f5_type', 'ER'),
        ('f5_param_a', 14),              # ER period
        ('f5_param_b', 1),               # Not used

        # === Filters ===
        ('use_volatility_filter', True),
        ('use_regime_filter', True),
        ('regime_threshold', -0.1),
        ('use_adx_filter', False),
        ('adx_threshold', 20),
        ('use_ema_filter', False),
        ('ema_period', 200),
        ('use_sma_filter', False),
        ('sma_period', 200),

        # === Kernel Settings ===
        ('use_kernel_filter', False),
        ('use_kernel_smoothing', False),
        ('kernel_lookback', 8),
        ('kernel_rel_weight', 8.0),
        ('kernel_start_bar', 25),
        ('kernel_lag', 2),

        # === Exit Settings ===
        ('use_dynamic_exits', True),
        ('bars_to_hold', 100000),

        # === RSI Exit Settings ===
        ('use_rsi_exit', False),          # Enable RSI threshold exits
        ('rsi_exit_period', 14),         # RSI period for exit signals
        ('rsi_overbought', 70),          # Exit longs when RSI crosses above this
        ('rsi_oversold', 30),            # Exit shorts when RSI crosses below this

        # === Kernel Exit Settings ===
        ('use_kernel_exit', True),       # Enable kernel line exit (price crosses below kernel)

        # === Risk Management ===
        ('position_size_pct', Decimal('0.95')),
        ('stop_loss_pct', Decimal('0.05')),
        ('use_stop_loss', True),

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
        elif ftype == 'ADX':
            return NormalizedADX(self.data, period=param_a)
        elif ftype == 'ATRR':
            return ATRRatio(self.data, period=param_a)
        elif ftype == 'PP':
            return PricePosition(self.data, period=param_a)
        elif ftype == 'ER':
            return EfficiencyRatio(self.data, period=param_a)
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
        # Create kernel indicators if needed for entry filter OR exit
        if self.p.use_kernel_filter or self.p.use_kernel_exit:
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

        # Raw prediction diagnostics (tracks ALL predictions including 0)
        self.prediction_diagnostics = {
            'total_bars': 0,
            'bullish_predictions': 0,    # prediction > 0
            'bearish_predictions': 0,    # prediction < 0
            'neutral_predictions': 0,    # prediction == 0
            'strong_bullish': 0,         # prediction >= neighbors_count/2
            'strong_bearish': 0,         # prediction <= -neighbors_count/2
            'prediction_sum': 0,         # for calculating average
            'signal_changes': 0,         # how often signal flips
            'entry_attempts': 0,         # how often we tried to enter
            'entries_blocked_by_kernel': 0,
            'entries_blocked_by_ema': 0,
            'entries_blocked_by_sma': 0,
        }

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
        elif hasattr(feature, 'nadx'):
            return feature.nadx[0]
        elif hasattr(feature, 'atr_ratio'):
            return feature.atr_ratio[0]
        elif hasattr(feature, 'position'):
            return feature.position[0]
        elif hasattr(feature, 'er'):
            return feature.er[0]
        return 0.0

    def _store_features(self):
        """Store current feature values in arrays."""
        for i, feature in enumerate(self.features):
            val = self._get_feature_value(feature)
            self.feature_arrays[i].append(val)

    def _calculate_label(self):
        """
        Calculate training label based on price movement.

        Two modes available via `trend_following_labels` parameter:

        MEAN-REVERSION (trend_following_labels=False, default):
        - If price rose over past 4 bars -> SHORT label (expect reversal down)
        - If price fell over past 4 bars -> LONG label (expect reversal up)
        - Best with oscillator features: RSI, CCI, Stochastic

        TREND-FOLLOWING (trend_following_labels=True):
        - If price rose over past 4 bars -> LONG label (expect continuation up)
        - If price fell over past 4 bars -> SHORT label (expect continuation down)
        - Best with trend features: ADX, Efficiency Ratio, Price Position
        """
        if len(self) < 5:
            return 0

        current_price = self.data.close[0]
        past_price = self.data.close[-4]

        if self.p.trend_following_labels:
            # TREND-FOLLOWING: expect price to continue in same direction
            if current_price > past_price:
                return 1   # LONG (price rising, expect continuation)
            elif current_price < past_price:
                return -1  # SHORT (price falling, expect continuation)
        else:
            # MEAN-REVERSION: expect price to reverse (original behavior)
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

        # Check if prediction meets minimum strength requirement
        min_strength = self.p.min_prediction_strength
        prediction_strong_enough = abs(self.prediction) >= min_strength

        if self.prediction > 0 and prediction_strong_enough and self._check_filters():
            self.signal = 1  # Long
        elif self.prediction < 0 and prediction_strong_enough and self._check_filters():
            self.signal = -1  # Short
        elif not prediction_strong_enough:
            # Weak prediction - go neutral (exit existing positions on next check)
            self.signal = 0
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

        # === Raw Prediction Diagnostics ===
        self.prediction_diagnostics['total_bars'] += 1
        self.prediction_diagnostics['prediction_sum'] += self.prediction
        if self.prediction > 0:
            self.prediction_diagnostics['bullish_predictions'] += 1
            if self.prediction >= self.p.neighbors_count / 2:
                self.prediction_diagnostics['strong_bullish'] += 1
        elif self.prediction < 0:
            self.prediction_diagnostics['bearish_predictions'] += 1
            if self.prediction <= -self.p.neighbors_count / 2:
                self.prediction_diagnostics['strong_bearish'] += 1
        else:
            self.prediction_diagnostics['neutral_predictions'] += 1

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
        if signal_changed:
            self.prediction_diagnostics['signal_changes'] += 1

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

    def get_diagnostics(self):
        """
        Get raw prediction diagnostics to understand ML behavior.
        Returns dict with diagnostic metrics.
        """
        diag = self.prediction_diagnostics.copy()

        # Calculate percentages
        if diag['total_bars'] > 0:
            diag['bullish_pct'] = (diag['bullish_predictions'] / diag['total_bars']) * 100
            diag['bearish_pct'] = (diag['bearish_predictions'] / diag['total_bars']) * 100
            diag['neutral_pct'] = (diag['neutral_predictions'] / diag['total_bars']) * 100
            diag['avg_prediction'] = diag['prediction_sum'] / diag['total_bars']
        else:
            diag['bullish_pct'] = diag['bearish_pct'] = diag['neutral_pct'] = 0
            diag['avg_prediction'] = 0

        # Entry blocking percentages
        if diag['entry_attempts'] > 0:
            diag['kernel_block_pct'] = (diag['entries_blocked_by_kernel'] / diag['entry_attempts']) * 100
            diag['ema_block_pct'] = (diag['entries_blocked_by_ema'] / diag['entry_attempts']) * 100
            diag['sma_block_pct'] = (diag['entries_blocked_by_sma'] / diag['entry_attempts']) * 100
        else:
            diag['kernel_block_pct'] = diag['ema_block_pct'] = diag['sma_block_pct'] = 0

        return diag

    def _check_entry(self, signal_changed):
        """Check for entry conditions."""
        # Determine signal requirement based on allow_reentry setting
        # allow_reentry=True: enter anytime signal is favorable (don't require flip)
        # allow_reentry=False: only enter on signal flip (original behavior)
        signal_ok = signal_changed or self.p.allow_reentry

        # Track entry attempts for diagnostics (when signal is bullish and we're checking)
        if signal_ok and self.signal == 1:
            self.prediction_diagnostics['entry_attempts'] += 1
            # Track what's blocking
            if not self._check_kernel_bullish():
                self.prediction_diagnostics['entries_blocked_by_kernel'] += 1
            if not self._check_ema_uptrend():
                self.prediction_diagnostics['entries_blocked_by_ema'] += 1
            if not self._check_sma_uptrend():
                self.prediction_diagnostics['entries_blocked_by_sma'] += 1

        # Long entry
        is_new_buy = (
            signal_ok and
            self.signal == 1 and
            self._check_kernel_bullish() and
            self._check_ema_uptrend() and
            self._check_sma_uptrend()
        )

        # Short entry (only if long_only=False)
        is_new_short = (
            not self.p.long_only and
            signal_ok and
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

        # Kernel line exit (price crosses below kernel)
        if self.p.use_kernel_exit and hasattr(self, 'kernel_rq'):
            kernel_val = self.kernel_rq.estimate[0]
            # Exit long when price crosses below kernel line
            if self.position.size > 0 and self.data.close[0] < kernel_val:
                self._close_position(f"PRICE BELOW KERNEL ({self.data.close[0]:.2f} < {kernel_val:.2f})")
                return
            # Exit short when price crosses above kernel line
            if self.position.size < 0 and self.data.close[0] > kernel_val:
                self._close_position(f"PRICE ABOVE KERNEL ({self.data.close[0]:.2f} > {kernel_val:.2f})")
                return

        # Stop loss
        if self.p.use_stop_loss:
            if self.position.size > 0:
                current_pnl_pct = (self.data.close[0] - self.entry_price) / self.entry_price
                print(current_pnl_pct)
            elif self.position.size < 0:
                current_pnl_pct = (self.entry_price - self.data.close[0]) / self.entry_price
                print(current_pnl_pct)
            else:
                return

            stop = float(self.p.stop_loss_pct)
            if current_pnl_pct <= -stop:
                self._close_position("STOP LOSS HIT")

    def _check_dynamic_exit(self):
        # Stop loss
        if self.p.use_stop_loss:
            if self.position.size > 0:
                current_pnl_pct = (self.data.close[0] - self.entry_price) / self.entry_price
                #print(current_pnl_pct)
            elif self.position.size < 0:
                current_pnl_pct = (self.entry_price - self.data.close[0]) / self.entry_price
                #print(current_pnl_pct)
            else:
                return

            stop = float(self.p.stop_loss_pct)
            if current_pnl_pct <= -stop:
                self._close_position("STOP LOSS HIT")

        """Check for dynamic exit based on kernel regression."""
        # If kernel filter is disabled, fall back to signal-based exit
        if not self.p.use_kernel_filter:
            # Exit on signal flip when kernel not available
            if self.position.size > 0 and self.signal == -1:
                self._close_position("SIGNAL FLIP TO BEARISH")
            elif self.position.size < 0 and self.signal == 1:
                self._close_position("SIGNAL FLIP TO BULLISH")
            return

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
            pnl_percent = ((self.data.close[0] / self.entry_price) - 1) * 100
            if self.p.verbose:
                print(f"CLOSE LONG: {reason} | {self.data.datetime.date(0)} | "
                      f"Entry: ${self.entry_price:.2f} | "
                      f"Exit: ${self.data.close[0]:.2f} | "
                      f"P&L: ${pnl:.2f} | "
                      f"P&L%: {pnl_percent:.2f}%")
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
