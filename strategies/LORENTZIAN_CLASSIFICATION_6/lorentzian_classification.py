"""
Machine Learning: Lorentzian Classification Strategy - Diverse Feature Set

A backtrader implementation using diverse, information-rich features for
better pattern matching across multiple market dimensions.

This version uses a 5-feature vector targeting different predictive signals:
1. RSM(20,252) - Relative Strength Momentum (percentile rank of current return)
2. VA(20) - Volume Anomaly (log-scaled volume vs its moving average)
3. MTD(5,60) - Multi-Timeframe Divergence (short vs long ROC conflict)
4. ZS(50) - Mean Reversion Z-Score (distance from equilibrium)
5. VCR(20,100) - Volatility Contraction Ratio (BB width percentile)

These features capture momentum persistence, volume anomalies, timeframe
divergence, mean-reversion stretch, and volatility compression.

Author: Backtrader implementation based on TradingView indicator by @jdehorty
Modified: Diverse feature vector
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


class RelativeStrengthMomentum(bt.Indicator):
    """
    Relative Strength Momentum (RSM).

    Percentile rank of current momentum_period-day return vs trailing
    lookback-day distribution of momentum_period-day returns.

    High = stock running hotter than usual for itself.
    Low  = underperforming its own history.

    Output: (percentile_rank * 2) - 1  → range -1 to 1
    """
    lines = ('rsm',)
    params = (
        ('momentum_period', 20),
        ('lookback', 252),
    )

    def __init__(self):
        self.addminperiod(self.p.lookback + self.p.momentum_period)

    def next(self):
        mp = self.p.momentum_period
        lb = self.p.lookback

        # Current momentum_period-day return
        if self.data.close[-mp] == 0:
            self.lines.rsm[0] = 0.0
            return
        current_ret = (self.data.close[0] / self.data.close[-mp]) - 1

        # Build distribution of momentum_period-day returns over lookback window
        count_below = 0
        total = 0
        for i in range(1, lb):
            past_idx = -i
            older_idx = -i - mp
            if len(self) + older_idx < 0:
                break
            past_price = self.data.close[past_idx]
            older_price = self.data.close[older_idx]
            if older_price == 0:
                continue
            hist_ret = (past_price / older_price) - 1
            total += 1
            if hist_ret < current_ret:
                count_below += 1

        if total > 0:
            percentile = count_below / total
        else:
            percentile = 0.5

        self.lines.rsm[0] = percentile * 2 - 1


class VolumeAnomaly(bt.Indicator):
    """
    Volume Anomaly (VA).

    Current volume relative to its moving average, log-scaled and
    smoothly clamped to -1..1 via tanh.

    Output: tanh(log(volume / sma_volume)) → range -1 to 1
    """
    lines = ('va',)
    params = (
        ('period', 20),
    )

    def __init__(self):
        self.vol_sma = bt.indicators.SMA(self.data.volume, period=self.p.period)
        self.addminperiod(self.p.period)

    def next(self):
        avg_vol = self.vol_sma[0]
        cur_vol = self.data.volume[0]

        if avg_vol > 0 and cur_vol > 0:
            ratio = cur_vol / avg_vol
            self.lines.va[0] = math.tanh(math.log(ratio))
        else:
            self.lines.va[0] = 0.0


class MultiTimeframeDivergence(bt.Indicator):
    """
    Multi-Timeframe Divergence (MTD).

    Difference between short-term and long-term rate of change,
    normalized by ATR as a percentage of price.

    When short ROC is positive but long ROC is negative (or vice versa),
    a turning point may be forming.

    Output: tanh((ROC_short - ROC_long) / ATR_pct) → range -1 to 1
    """
    lines = ('mtd',)
    params = (
        ('short_period', 5),
        ('long_period', 60),
    )

    def __init__(self):
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.addminperiod(self.p.long_period + 1)

    def next(self):
        close = self.data.close[0]

        # Short-term ROC
        if self.data.close[-self.p.short_period] != 0:
            roc_short = (close / self.data.close[-self.p.short_period]) - 1
        else:
            roc_short = 0

        # Long-term ROC
        if self.data.close[-self.p.long_period] != 0:
            roc_long = (close / self.data.close[-self.p.long_period]) - 1
        else:
            roc_long = 0

        # ATR as percentage of price
        atr_pct = (self.atr[0] / close) if close > 0 else 1e-8
        atr_pct = max(atr_pct, 1e-8)

        divergence = (roc_short - roc_long) / atr_pct
        self.lines.mtd[0] = math.tanh(divergence)


class MeanReversionZScore(bt.Indicator):
    """
    Mean Reversion Z-Score (ZS).

    How many standard deviations price is from its period-SMA.
    Clamped to -1..1 by dividing by 3 and clipping.

    Output: clamp(z_score / 3, -1, 1)
    """
    lines = ('zscore',)
    params = (
        ('period', 50),
    )

    def __init__(self):
        self.sma = bt.indicators.SMA(self.data.close, period=self.p.period)
        self.addminperiod(self.p.period)

    def next(self):
        # Calculate standard deviation manually over the period
        mean = self.sma[0]
        sq_sum = 0.0
        for i in range(self.p.period):
            diff = self.data.close[-i] - mean
            sq_sum += diff * diff
        std = math.sqrt(sq_sum / self.p.period)

        if std > 0:
            z = (self.data.close[0] - mean) / std
            self.lines.zscore[0] = max(-1.0, min(1.0, z / 3.0))
        else:
            self.lines.zscore[0] = 0.0


class VolatilityContractionRatio(bt.Indicator):
    """
    Volatility Contraction Ratio (VCR).

    Current Bollinger Band width as a percentile of its recent history.
    Low = tight squeeze (pre-breakout). High = expanded (post-move).

    Output: (percentile_rank * 2) - 1 → range -1 to 1
    """
    lines = ('vcr',)
    params = (
        ('bb_period', 20),
        ('lookback', 100),
    )

    def __init__(self):
        self.bb = bt.indicators.BollingerBands(
            self.data.close, period=self.p.bb_period, devfactor=2.0
        )
        self.addminperiod(self.p.bb_period + self.p.lookback)

    def next(self):
        # Current BB width (normalized by midline)
        mid = self.bb.mid[0]
        if mid > 0:
            current_width = (self.bb.top[0] - self.bb.bot[0]) / mid
        else:
            self.lines.vcr[0] = 0.0
            return

        # Build distribution of BB widths over lookback
        count_below = 0
        total = 0
        for i in range(1, self.p.lookback):
            hist_mid = self.bb.mid[-i]
            if hist_mid > 0:
                hist_width = (self.bb.top[-i] - self.bb.bot[-i]) / hist_mid
                total += 1
                if hist_width < current_width:
                    count_below += 1

        if total > 0:
            percentile = count_below / total
        else:
            percentile = 0.5

        self.lines.vcr[0] = percentile * 2 - 1


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
    Machine Learning Lorentzian Classification Strategy - Diverse Features.

    Uses K-Nearest Neighbors with Lorentzian distance metric for
    price direction classification.

    This version uses diverse features targeting different information:
    - RSM(20,252): Relative Strength Momentum
    - VA(20): Volume Anomaly
    - MTD(5,60): Multi-Timeframe Divergence
    - ZS(50): Mean Reversion Z-Score
    - VCR(20,100): Volatility Contraction Ratio
    """

    params = (
        # === General Settings ===
        ('neighbors_count', 9),          # Number of neighbors for KNN
        ('max_bars_back', 2000),         # Maximum lookback for training data
        ('feature_count', 5),            # Number of features (2-5)
        ('trend_following_labels', False),  # False=mean-reversion labels, True=trend-following labels
        ('allow_reentry', True),         # True=enter anytime signal favorable, False=only on signal flip
        ('min_prediction_strength', 20),  # Minimum |prediction| to generate signal (normalized scale: 0-100)
        ('prediction_norm_window', 200),  # Rolling window for percentile rank normalization

        # === Label Settings ===
        ('label_lookahead', 4),          # Bars to look forward for label outcome
        ('label_dead_zone', 0.5),        # Min move in ATR multiples to get a label (0=disabled)
        ('use_forward_labels', True),    # True=forward-looking labels, False=backward (legacy)
        ('use_magnitude_labels', True),  # True=return/ATR continuous labels, False=binary +1/-1

        # === Feature 1 (Relative Strength Momentum) ===
        ('f1_type', 'RSM'),
        ('f1_param_a', 10),              # Momentum period
        ('f1_param_b', 126),             # Lookback for percentile distribution

        # === Feature 2 (Volume Anomaly) ===
        ('f2_type', 'VA'),
        ('f2_param_a', 20),              # Volume SMA period
        ('f2_param_b', 1),               # Not used

        # === Feature 3 (Multi-Timeframe Divergence) ===
        ('f3_type', 'MTD'),
        ('f3_param_a', 5),               # Short ROC period
        ('f3_param_b', 60),              # Long ROC period

        # === Feature 4 (Mean Reversion Z-Score) ===
        ('f4_type', 'ZS'),
        ('f4_param_a', 50),              # SMA period for z-score
        ('f4_param_b', 1),               # Not used

        # === Feature 5 (Efficiency Ratio) ===
        ('f5_type', 'ER'),
        ('f5_param_a', 10),              # ER period
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

        # === Loss Penalty (ML bearish feedback after losing trades) ===
        ('use_loss_penalty', True),        # Enable loss penalty on predictions
        ('loss_penalty_amount', 0),        # Penalty per loss (0 = auto: neighbors_count/2)
        ('loss_penalty_decay', 0.90),      # Decay rate per bar (0.90 = ~10 bars to halve)

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

        # === Cross-Symbol Training ===
        ('use_cross_symbol_training', False),
        ('cross_symbol_etfs', 'SPY,QQQ,IWM,TLT,GLD,XLE,EFA'),
        ('cross_symbol_lookback_years', 5),
        ('use_regime_balancing', False),
        ('cross_symbol_auto_peers', True),
        ('cross_symbol_target_symbol', ''),
        ('cross_symbol_max_peers', 7),
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

        # ATR for label normalization (dead zone + magnitude labels)
        self.label_atr = bt.indicators.ATR(self.data, period=14)

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
        elif ftype == 'RSM':
            return RelativeStrengthMomentum(self.data, momentum_period=param_a, lookback=param_b)
        elif ftype == 'VA':
            return VolumeAnomaly(self.data, period=param_a)
        elif ftype == 'MTD':
            return MultiTimeframeDivergence(self.data, short_period=param_a, long_period=param_b)
        elif ftype == 'ZS':
            return MeanReversionZScore(self.data, period=param_a)
        elif ftype == 'VCR':
            return VolatilityContractionRatio(self.data, bb_period=param_a, lookback=param_b)
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

        # Cross-symbol training data seeding
        if self.p.use_cross_symbol_training:
            try:
                from cross_symbol_preloader import seed_strategy
                n_seeded = seed_strategy(self)
                if self.p.verbose:
                    print(f"CROSS-SYMBOL: Seeded {n_seeded} training bars from {self.p.cross_symbol_etfs}")
            except ImportError:
                if self.p.verbose:
                    print("CROSS-SYMBOL: cross_symbol_preloader.py not found, skipping")
            except Exception as e:
                if self.p.verbose:
                    print(f"CROSS-SYMBOL: Error during seeding: {e}")

        # Deferred label queue for forward-looking labels
        # Stores (bar_index, price_at_bar, atr_at_bar) waiting for future outcome
        self.pending_labels = deque()

        # Trading state
        self.order = None
        self.signal = 0  # 1 = long, -1 = short, 0 = neutral
        self.bars_held = 0
        self.entry_bar = 0
        self.entry_price = 0
        self.prediction = 0
        self.raw_prediction = 0.0

        # Rolling prediction normalization (percentile rank -> [-100, +100])
        self.raw_prediction_history = deque(maxlen=self.p.prediction_norm_window)

        # Percentile band performance tracking
        self.prediction_band_stats = {
            '0-20': {'trades': 0, 'wins': 0, 'losses': 0, 'total_pnl_pct': 0.0},
            '20-40': {'trades': 0, 'wins': 0, 'losses': 0, 'total_pnl_pct': 0.0},
            '40-60': {'trades': 0, 'wins': 0, 'losses': 0, 'total_pnl_pct': 0.0},
            '60-80': {'trades': 0, 'wins': 0, 'losses': 0, 'total_pnl_pct': 0.0},
            '80-100': {'trades': 0, 'wins': 0, 'losses': 0, 'total_pnl_pct': 0.0},
        }
        self.entry_norm_prediction = 0.0

        # Loss penalty: accumulated bearish bias from recent losing trades
        self.loss_penalty = 0.0
        self.entry_prediction = 0  # prediction strength at time of entry

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
        elif hasattr(feature, 'rsm'):
            return feature.rsm[0]
        elif hasattr(feature, 'va'):
            return feature.va[0]
        elif hasattr(feature, 'mtd'):
            return feature.mtd[0]
        elif hasattr(feature, 'zscore'):
            return feature.zscore[0]
        elif hasattr(feature, 'vcr'):
            return feature.vcr[0]
        return 0.0

    def _store_features(self):
        """Store current feature values in arrays."""
        for i, feature in enumerate(self.features):
            val = self._get_feature_value(feature)
            self.feature_arrays[i].append(val)

    def _calculate_label_legacy(self):
        """
        Legacy backward-looking label calculation.

        Used when use_forward_labels=False.
        """
        if len(self) < 5:
            return 0

        current_price = self.data.close[0]
        past_price = self.data.close[-4]

        if self.p.trend_following_labels:
            if current_price > past_price:
                return 1
            elif current_price < past_price:
                return -1
        else:
            if past_price < current_price:
                return -1
            elif past_price > current_price:
                return 1
        return 0

    def _resolve_pending_labels(self):
        """
        Resolve forward-looking labels for bars whose outcome is now known.

        For each pending bar, we now know the price N bars later.
        Label = (future_price - bar_price) / bar_atr, clamped and filtered.

        Features were already stored at the pending bar's index, so we just
        append the label to label_array in order.
        """
        current_price = self.data.close[0]
        current_bar = len(self)
        lookahead = self.p.label_lookahead

        while self.pending_labels and (current_bar - self.pending_labels[0][0]) >= lookahead:
            bar_idx, bar_price, bar_atr = self.pending_labels.popleft()

            # Price change from the pending bar to N bars later
            price_change = current_price - bar_price
            atr = max(bar_atr, 1e-8)  # avoid division by zero

            # Normalized return in ATR units
            norm_return = price_change / atr

            # Dead zone: if move is too small relative to ATR, label as neutral
            if abs(norm_return) < self.p.label_dead_zone:
                self.label_array.append(0)
                continue

            if self.p.use_magnitude_labels:
                # Continuous label: return in ATR units, clamped to [-3, 3]
                label = max(-3.0, min(3.0, norm_return))
            else:
                # Binary label: just direction
                label = 1.0 if price_change > 0 else -1.0

            # Apply mean-reversion flip if configured
            if not self.p.trend_following_labels:
                label = -label

            self.label_array.append(label)

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

    def _normalize_prediction(self, raw):
        """
        Normalize raw KNN prediction to [-100, +100] via rolling percentile rank.

        During warmup (< 30 predictions), uses linear scaling based on
        theoretical max (K * 3 for magnitude labels).

        After warmup, computes percentile rank within rolling window,
        maps [0, 1] -> [-100, +100]. Naturally preserves sign since large
        positive raws rank high (positive normalized) and vice versa.
        """
        self.raw_prediction_history.append(raw)
        history = self.raw_prediction_history

        if len(history) < 30:
            # Warmup: linear scale based on theoretical max
            max_raw = self.p.neighbors_count * 3.0
            if max_raw > 0:
                return max(-100.0, min(100.0, (raw / max_raw) * 100.0))
            return 0.0

        # Percentile rank: fraction of history values below current
        count_below = sum(1 for h in history if h < raw)
        count_equal = sum(1 for h in history if h == raw)
        # Mid-rank method: ties get average rank
        percentile = (count_below + 0.5 * count_equal) / len(history)

        # Map [0, 1] -> [-100, +100]
        return (percentile * 2.0 - 1.0) * 100.0

    def _get_prediction_band(self, abs_norm_prediction):
        """Get the percentile band label for an absolute normalized prediction."""
        if abs_norm_prediction < 20:
            return '0-20'
        elif abs_norm_prediction < 40:
            return '20-40'
        elif abs_norm_prediction < 60:
            return '40-60'
        elif abs_norm_prediction < 80:
            return '60-80'
        else:
            return '80-100'

    def _track_band_result(self, pnl_percent):
        """Record a trade result in the appropriate percentile band."""
        band = self._get_prediction_band(self.entry_norm_prediction)
        self.prediction_band_stats[band]['trades'] += 1
        self.prediction_band_stats[band]['total_pnl_pct'] += pnl_percent
        if pnl_percent > 0:
            self.prediction_band_stats[band]['wins'] += 1
        else:
            self.prediction_band_stats[band]['losses'] += 1

    def get_percentile_band_stats(self):
        """
        Get trade performance breakdown by prediction strength band.

        Returns dict mapping band label -> {trades, wins, losses, win_rate,
        total_pnl_pct, avg_pnl_pct}.
        """
        stats = {}
        for band, data in self.prediction_band_stats.items():
            trades = data['trades']
            stats[band] = {
                'trades': trades,
                'wins': data['wins'],
                'losses': data['losses'],
                'win_rate': (data['wins'] / trades * 100) if trades > 0 else 0,
                'total_pnl_pct': data['total_pnl_pct'],
                'avg_pnl_pct': (data['total_pnl_pct'] / trades) if trades > 0 else 0,
            }
        return stats

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
        else:
            # Strong prediction but filters blocked - go neutral to prevent stale signals
            self.signal = 0

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

        # Store features (always do this to build training data)
        self._store_features()

        # Label generation
        if self.p.use_forward_labels:
            # Forward-looking: queue this bar for labeling once outcome is known
            atr_val = self.label_atr[0] if len(self.label_atr) > 0 else 0
            self.pending_labels.append((len(self), self.data.close[0], atr_val))
            # Resolve any pending labels whose lookahead has elapsed
            self._resolve_pending_labels()
        else:
            # Legacy backward-looking labels
            label = self._calculate_label_legacy()
            self.label_array.append(label)

        # Skip trading if before test period start
        # (still accumulate training data above, just don't trade)
        if self.p.test_start_idx > 0 and len(self) < self.p.test_start_idx:
            return

        # Run ML prediction
        self.prediction = self._run_knn()

        # Apply loss penalty: recent losses make the model more bearish
        if self.p.use_loss_penalty and self.loss_penalty > 0:
            if self.prediction > 0:
                self.prediction = self.prediction - self.loss_penalty
            elif self.prediction < 0:
                self.prediction = self.prediction + self.loss_penalty
            # Decay the penalty each bar
            self.loss_penalty *= self.p.loss_penalty_decay
            if self.loss_penalty < 0.1:
                self.loss_penalty = 0.0

        # Normalize prediction to [-100, +100] via rolling percentile rank
        self.raw_prediction = self.prediction
        self.prediction = self._normalize_prediction(self.raw_prediction)

        # === Raw Prediction Diagnostics (prediction_sum uses raw for meaningful average) ===
        self.prediction_diagnostics['total_bars'] += 1
        self.prediction_diagnostics['prediction_sum'] += self.raw_prediction
        if self.prediction > 0:
            self.prediction_diagnostics['bullish_predictions'] += 1
            if self.prediction >= 50:
                self.prediction_diagnostics['strong_bullish'] += 1
        elif self.prediction < 0:
            self.prediction_diagnostics['bearish_predictions'] += 1
            if self.prediction <= -50:
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
                self.signal = -1 if self.position.size > 0 else 1
                self._close_position("STOP LOSS HIT")
                return

        """Check for dynamic exit based on kernel regression."""
        # If no kernel indicators available, fall back to signal-based exit
        if not self.p.use_kernel_filter and not self.p.use_kernel_exit:
            # Exit on signal flip when kernel not available
            if self.position.size > 0 and self.signal == -1:
                self._close_position("SIGNAL FLIP TO BEARISH")
            elif self.position.size < 0 and self.signal == 1:
                self._close_position("SIGNAL FLIP TO BULLISH")
            return

        if not hasattr(self, 'kernel_rq') or len(self.kernel_rq) < 2:
            return

        # Kernel line exit: price crosses kernel (use_kernel_exit)
        # Force signal to opposite direction so re-entry requires a full signal flip
        if self.p.use_kernel_exit:
            kernel_val = self.kernel_rq.estimate[0]
            if self.position.size > 0 and self.data.close[0] < kernel_val:
                self.signal = -1
                self._close_position(f"PRICE BELOW KERNEL ({self.data.close[0]:.2f} < {kernel_val:.2f})")
                return
            if self.position.size < 0 and self.data.close[0] > kernel_val:
                self.signal = 1
                self._close_position(f"PRICE ABOVE KERNEL ({self.data.close[0]:.2f} > {kernel_val:.2f})")
                return

        # Kernel direction change exit (use_kernel_filter)
        if self.p.use_kernel_filter:
            if self.position.size > 0:
                was_bullish = self.kernel_rq.estimate[-2] < self.kernel_rq.estimate[-1]
                is_bearish = self.kernel_rq.estimate[-1] > self.kernel_rq.estimate[0]
                if was_bullish and is_bearish:
                    self._close_position("KERNEL BEARISH CHANGE")

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
            self.entry_prediction = abs(self.raw_prediction)
            self.entry_norm_prediction = abs(self.prediction)

            if self.p.verbose:
                band = self._get_prediction_band(self.entry_norm_prediction)
                print(f"BUY SIGNAL: {self.data.datetime.date(0)} | "
                      f"Prediction: {self.prediction:.1f} (raw: {self.raw_prediction:.2f}) [{band}] | "
                      f"Price: ${self.data.close[0]:.2f}")

    def _execute_short(self):
        """Execute short sell order (go short)."""
        size = self._calculate_position_size()
        if size > 0:
            self.order = self.sell(size=size)
            self.entry_bar = len(self)
            self.entry_price = self.data.close[0]
            self.entry_prediction = abs(self.raw_prediction)
            self.entry_norm_prediction = abs(self.prediction)

            if self.p.verbose:
                band = self._get_prediction_band(self.entry_norm_prediction)
                print(f"SHORT SIGNAL: {self.data.datetime.date(0)} | "
                      f"Prediction: {self.prediction:.1f} (raw: {self.raw_prediction:.2f}) [{band}] | "
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
            # Accumulate loss penalty scaled to entry prediction strength
            if pnl < 0 and self.p.use_loss_penalty:
                penalty = self.p.loss_penalty_amount if self.p.loss_penalty_amount > 0 else self.entry_prediction * 2.5
                self.loss_penalty += penalty
            self._track_band_result(pnl_percent)
            self.order = self.sell(size=self.position.size)

        elif self.position.size < 0:
            # Close short position
            pnl = (self.entry_price - self.data.close[0]) * abs(self.position.size)
            pnl_percent = ((self.entry_price - self.data.close[0]) / self.entry_price) * 100 if self.entry_price > 0 else 0
            if self.p.verbose:
                print(f"CLOSE SHORT: {reason} | {self.data.datetime.date(0)} | "
                      f"Entry: ${self.entry_price:.2f} | "
                      f"Exit: ${self.data.close[0]:.2f} | "
                      f"P&L: ${pnl:.2f} | "
                      f"P&L%: {pnl_percent:.2f}%")
            # Accumulate loss penalty scaled to entry prediction strength
            if pnl < 0 and self.p.use_loss_penalty:
                penalty = self.p.loss_penalty_amount if self.p.loss_penalty_amount > 0 else self.entry_prediction * 2.5
                self.loss_penalty += penalty
            self._track_band_result(pnl_percent)
            self.order = self.buy(size=abs(self.position.size))

    def _calculate_position_size(self):
        """Calculate position size based on available cash.
        Uses 95% of target to leave margin for gap-up between signal and fill."""
        cash = self.broker.getcash()
        position_value = cash * float(self.p.position_size_pct)
        size = int(position_value / self.data.close[0] * 0.95)
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
