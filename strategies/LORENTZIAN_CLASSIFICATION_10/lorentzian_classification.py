"""
Machine Learning: Lorentzian Classification Strategy - Temporal Feature Set

A backtrader implementation using diverse, information-rich features for
better pattern matching across multiple market dimensions.

This version adds 3 temporal derivative features (VCOMP, MPER, VMC) that
encode multi-bar sequential context into single values, capturing patterns
like "big swing followed by consolidation with high volume" that point-in-time
snapshots miss.

New temporal features:
12. VCOMP(4,16) - Volatility Compression (consolidation after a big move)
13. MPER(4,20) - Momentum Persistence (pullback within intact trend vs reversal)
14. VMC(5,40) - Volume-Momentum Coupling (elevated volume during consolidation)

Author: Backtrader implementation based on TradingView indicator by @jdehorty
Modified: Temporal derivative features added
"""

import math
from decimal import Decimal
from collections import deque
from datetime import timedelta
import backtrader as bt
import numpy as np
import pandas as pd


# =============================================================================
# SPY Market Regime Cache
# =============================================================================

# Module-level cache: persists across strategy instances within a process.
# In ProcessPoolExecutor, each worker downloads SPY once, then all subsequent
# strategy instances in that worker hit the cache.
_SPY_REGIME_CACHE = {}


def _compute_spy_regime(period='weekly', min_date=None, max_date=None):
    """
    Download SPY daily data, compute market regime using weekly/monthly
    high/low breakout logic (matching MarketRegimeDetector), and return
    a dict mapping date -> regime value (-1, 0, or 1).

    Uses module-level cache to avoid re-downloading within the same process.

    Args:
        period: 'weekly' or 'monthly' - resampling period for H/L reference
        min_date: earliest date needed (datetime.date or None for 10y lookback)
        max_date: latest date needed (datetime.date or None for today)

    Returns:
        dict[datetime.date, int]: mapping of trading dates to regime values
        Empty dict on download failure (filter becomes permissive).
    """
    import yfinance as yf
    from datetime import datetime

    if max_date is None:
        max_date = datetime.now().date()
    if min_date is None:
        min_date = max_date - timedelta(days=365 * 10)

    # Buffer: need at least one full period before min_date for prev H/L
    buffer_days = 40 if period == 'weekly' else 90
    download_start = min_date - timedelta(days=buffer_days)

    # Cache key: period + date range (rounded to month for cache hits)
    cache_key = (period, download_start.year, download_start.month,
                 max_date.year, max_date.month)

    global _SPY_REGIME_CACHE
    if cache_key in _SPY_REGIME_CACHE:
        return _SPY_REGIME_CACHE[cache_key]

    try:
        df = yf.download('SPY', start=str(download_start),
                         end=str(max_date + timedelta(days=5)),
                         interval='1d', progress=False)
        if df.empty or len(df) < 20:
            _SPY_REGIME_CACHE[cache_key] = {}
            return {}

        df.index = df.index.tz_localize(None)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.columns = ['open', 'high', 'low', 'close', 'volume']

    except Exception:
        _SPY_REGIME_CACHE[cache_key] = {}
        return {}

    # Resample to get period high/low
    rule = 'W' if period == 'weekly' else 'ME'
    period_agg = df.resample(rule).agg({'high': 'max', 'low': 'min'})
    period_agg = period_agg.dropna()

    # Shift by 1 to get PREVIOUS completed period's high/low
    period_agg['prev_high'] = period_agg['high'].shift(1)
    period_agg['prev_low'] = period_agg['low'].shift(1)
    period_agg = period_agg.dropna()

    # Forward-fill period-end values onto daily dates
    prev_hl = period_agg[['prev_high', 'prev_low']]
    daily_prev = prev_hl.reindex(df.index, method='ffill')

    # Compute regime for each daily bar
    regime_dict = {}
    for idx in df.index:
        if idx not in daily_prev.index:
            continue
        row = daily_prev.loc[idx]
        if pd.isna(row['prev_high']) or pd.isna(row['prev_low']):
            continue

        close = df.loc[idx, 'close']
        if close > row['prev_high']:
            regime = 1    # Bullish: close above previous period's high
        elif close < row['prev_low']:
            regime = -1   # Bearish: close below previous period's low
        else:
            regime = 0    # Reverting: between prev high and low

        regime_dict[idx.date()] = regime

    _SPY_REGIME_CACHE[cache_key] = regime_dict
    return regime_dict


# =============================================================================
# Custom Indicators
# =============================================================================

class SafeADX(bt.Indicator):
    """
    ADX indicator that handles division-by-zero when +DI and -DI are both 0.
    Backtrader's built-in ADX crashes on stocks with periods of zero directional
    movement because it computes DX = |+DI - -DI| / (+DI + -DI) using line
    operations that don't guard against zero denominators.
    """
    lines = ('adx',)
    params = (('period', 14),)

    def __init__(self):
        atr = bt.indicators.ATR(self.data, period=self.p.period)
        upmove = self.data.high - self.data.high(-1)
        downmove = self.data.low(-1) - self.data.low

        plus = bt.And(upmove > downmove, upmove > 0.0)
        plusDM = bt.If(plus, upmove, 0.0)
        self.plusDMav = bt.indicators.SmoothedMovingAverage(plusDM, period=self.p.period)

        minus = bt.And(downmove > upmove, downmove > 0.0)
        minusDM = bt.If(minus, downmove, 0.0)
        self.minusDMav = bt.indicators.SmoothedMovingAverage(minusDM, period=self.p.period)

        self.atr = atr
        self.dx_history = deque(maxlen=self.p.period)
        self.addminperiod(self.p.period * 2)

    def next(self):
        atr_val = self.atr[0]
        if atr_val > 0:
            plus_di = 100.0 * self.plusDMav[0] / atr_val
            minus_di = 100.0 * self.minusDMav[0] / atr_val
        else:
            plus_di = 0.0
            minus_di = 0.0

        di_sum = plus_di + minus_di
        if di_sum > 0:
            dx = 100.0 * abs(plus_di - minus_di) / di_sum
        else:
            dx = 0.0

        self.dx_history.append(dx)
        self.lines.adx[0] = sum(self.dx_history) / len(self.dx_history)


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
        adx = SafeADX(self.data, period=self.p.period)
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

        closes = np.array(self.data.close.get(size=self.p.period + 1))
        direction = abs(closes[-1] - closes[0])
        volatility = float(np.sum(np.abs(np.diff(closes))))

        if volatility > 0:
            er = direction / volatility
        else:
            er = 0

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

        if self.data.close[-mp] == 0:
            self.lines.rsm[0] = 0.0
            return
        current_ret = (self.data.close[0] / self.data.close[-mp]) - 1

        total_needed = lb + mp
        if len(self) < total_needed:
            self.lines.rsm[0] = 0.0
            return

        closes = np.array(self.data.close.get(size=total_needed))
        # closes[-1] = current bar, closes[0] = oldest
        # Historical mp-day returns: recent[j] / older[j] - 1
        # For i=1..lb-1: numerator=closes[-(i+1)], denominator=closes[-(i+1)-mp]
        # recent prices (excluding current bar): closes[mp:-1]
        # older prices: closes[:-mp-1]
        recent = closes[mp:-1]
        older = closes[:len(recent)]

        valid_mask = older != 0
        if not np.any(valid_mask):
            self.lines.rsm[0] = 0.0
            return

        hist_rets = np.empty(len(recent))
        hist_rets[valid_mask] = (recent[valid_mask] / older[valid_mask]) - 1
        hist_rets[~valid_mask] = np.nan

        valid_rets = hist_rets[~np.isnan(hist_rets)]
        total = len(valid_rets)
        if total > 0:
            count_below = int(np.sum(valid_rets < current_ret))
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
        mean = self.sma[0]
        closes = np.array(self.data.close.get(size=self.p.period))
        std = float(np.sqrt(np.mean((closes - mean) ** 2)))

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
        mid = self.bb.mid[0]
        if mid <= 0:
            self.lines.vcr[0] = 0.0
            return

        current_width = (self.bb.top[0] - self.bb.bot[0]) / mid
        lb = self.p.lookback

        tops = np.array(self.bb.top.get(size=lb))
        bots = np.array(self.bb.bot.get(size=lb))
        mids = np.array(self.bb.mid.get(size=lb))

        # Exclude current bar (last element), use historical only
        hist_mids = mids[:-1]
        valid = hist_mids > 0
        if not np.any(valid):
            self.lines.vcr[0] = 0.0
            return

        hist_widths = (tops[:-1][valid] - bots[:-1][valid]) / hist_mids[valid]
        count_below = int(np.sum(hist_widths < current_width))
        total = len(hist_widths)

        if total > 0:
            percentile = count_below / total
        else:
            percentile = 0.5

        self.lines.vcr[0] = percentile * 2 - 1


class VolumePriceDivergence(bt.Indicator):
    """
    Pearson correlation between price deltas and volume deltas over a window.
    Captures whether volume confirms or contradicts price direction.
    Output: [-1, 1] (naturally bounded by correlation).
    """
    lines = ('vpd',)
    params = (('period', 14), ('smoothing', 1))

    def __init__(self):
        self.addminperiod(self.p.period + 1)

    def next(self):
        n = self.p.period
        closes = np.array(self.data.close.get(size=n + 1))
        volumes = np.array(self.data.volume.get(size=n + 1))

        price_deltas = np.diff(closes)
        vol_deltas = np.diff(volumes)

        sum_xy = float(np.dot(price_deltas, vol_deltas))
        sum_x2 = float(np.dot(price_deltas, price_deltas))
        sum_y2 = float(np.dot(vol_deltas, vol_deltas))

        denom = math.sqrt(sum_x2) * math.sqrt(sum_y2)
        if denom > 0:
            self.lines.vpd[0] = sum_xy / denom
        else:
            self.lines.vpd[0] = 0.0


class CandleStructure(bt.Indicator):
    """
    Average signed body-to-range ratio over a window, tanh-scaled.
    Captures candle conviction (full bodies) vs indecision (dojis/wicks).
    Output: tanh(avg * sensitivity) → [-1, 1].
    """
    lines = ('cs',)
    params = (('period', 5), ('sensitivity', 2.0))

    def __init__(self):
        self.addminperiod(self.p.period)

    def next(self):
        n = self.p.period
        closes = np.array(self.data.close.get(size=n))
        opens = np.array(self.data.open.get(size=n))
        highs = np.array(self.data.high.get(size=n))
        lows = np.array(self.data.low.get(size=n))

        bodies = closes - opens
        ranges = highs - lows
        safe_ranges = np.where(ranges > 0, ranges, 1.0)
        ratios = np.where(ranges > 0, bodies / safe_ranges, 0.0)
        avg = float(np.mean(ratios))
        self.lines.cs[0] = math.tanh(avg * self.p.sensitivity)


class MomentumAcceleration(bt.Indicator):
    """
    Second derivative of price: ROC now vs ROC lagged, ATR-normalized.
    Captures whether momentum is accelerating or decelerating.
    Output: tanh(acceleration) → [-1, 1].
    """
    lines = ('macc',)
    params = (('roc_period', 5), ('lag', 5))

    def __init__(self):
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.addminperiod(self.p.roc_period + self.p.lag + 14)

    def next(self):
        rp = self.p.roc_period
        lag = self.p.lag
        c = self.data.close

        if c[-rp] != 0 and c[-lag - rp] != 0:
            roc_now = (c[0] / c[-rp]) - 1
            roc_prev = (c[-lag] / c[-lag - rp]) - 1
            atr_pct = self.atr[0] / c[0] if c[0] > 0 else 1e-8
            atr_pct = max(atr_pct, 1e-8)
            accel = (roc_now - roc_prev) / atr_pct
            self.lines.macc[0] = math.tanh(accel)
        else:
            self.lines.macc[0] = 0.0


class OBVTrend(bt.Indicator):
    """
    Z-score of cumulative On-Balance Volume vs its own SMA.
    Captures sustained money inflow/outflow relative to recent norms.
    Output: clamp(z / z_divisor, -1, 1).
    """
    lines = ('obvt',)
    params = (('period', 20), ('z_divisor', 3.0))

    def __init__(self):
        self.obv = 0.0
        self.obv_history = deque(maxlen=500)  # large buffer for warmup
        self.addminperiod(self.p.period + 1)

    def next(self):
        if len(self) > 1:
            if self.data.close[0] > self.data.close[-1]:
                self.obv += self.data.volume[0]
            elif self.data.close[0] < self.data.close[-1]:
                self.obv -= self.data.volume[0]

        self.obv_history.append(self.obv)

        if len(self.obv_history) >= self.p.period:
            start_idx = len(self.obv_history) - self.p.period
            arr = np.array(list(self.obv_history)[start_idx:])
            mean_obv = float(np.mean(arr))
            std_obv = float(np.sqrt(np.mean((arr - mean_obv) ** 2)))
            if std_obv > 0:
                z = (self.obv - mean_obv) / std_obv
                self.lines.obvt[0] = max(-1.0, min(1.0, z / self.p.z_divisor))
            else:
                self.lines.obvt[0] = 0.0
        else:
            self.lines.obvt[0] = 0.0


class StreakPattern(bt.Indicator):
    """
    Consecutive up/down close streak length × average bar magnitude.
    Captures behavioral serial correlation and streak exhaustion.
    Output: direction × norm_len × (0.5 + 0.5 × norm_mag) → [-1, 1].
    """
    lines = ('strk',)
    params = (('max_streak', 10), ('atr_mult', 1.5))

    def __init__(self):
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.addminperiod(self.p.max_streak + 14)

    def next(self):
        max_s = self.p.max_streak
        streak_len = 0
        streak_sum = 0.0
        streak_dir = 0

        for i in range(max_s):
            if self.data.close[-i] > self.data.close[-i - 1]:
                d = 1
            elif self.data.close[-i] < self.data.close[-i - 1]:
                d = -1
            else:
                d = 0

            if i == 0:
                streak_dir = d
                if d == 0:
                    break
            if d == streak_dir and d != 0:
                streak_len += 1
                streak_sum += abs(self.data.close[-i] / self.data.close[-i - 1] - 1)
            else:
                break

        if streak_len == 0:
            self.lines.strk[0] = 0.0
            return

        norm_len = streak_len / max_s
        avg_move = streak_sum / streak_len
        atr_pct = self.atr[0] / self.data.close[0] if self.data.close[0] > 0 else 1e-8
        atr_pct = max(atr_pct, 1e-8)
        norm_mag = min(1.0, avg_move / (atr_pct * self.p.atr_mult))

        self.lines.strk[0] = streak_dir * norm_len * (0.5 + 0.5 * norm_mag)


class ChoppinessIndex(bt.Indicator):
    """
    Choppiness Index measures whether the market is trending or choppy.

    Formula: 100 * LOG10(SUM(ATR, period) / (highest - lowest)) / LOG10(period)

    High values (near 100) = choppy/range-bound, low values (near 0) = trending.
    Normalized to [-1, 1]: (CHOP / 50 - 1) inverted so trending = +1, choppy = -1.

    param_a: ATR/range period (default 14)
    param_b: not used
    """
    lines = ('chop',)
    params = (('period', 14), ('unused', 1))

    def __init__(self):
        self.atr = bt.indicators.ATR(self.data, period=self.p.period)
        self.addminperiod(self.p.period + 1)

    def next(self):
        p = self.p.period

        atr_vals = np.array(self.atr.get(size=p))
        # Filter NaN
        valid_mask = ~np.isnan(atr_vals)
        if not np.any(valid_mask):
            self.lines.chop[0] = 0.0
            return
        atr_sum = float(np.sum(atr_vals[valid_mask]))

        highs = np.array(self.data.high.get(size=p))
        lows = np.array(self.data.low.get(size=p))
        highest = float(np.max(highs))
        lowest = float(np.min(lows))

        rng = highest - lowest
        if rng > 0 and atr_sum > 0:
            chop_raw = 100.0 * math.log10(atr_sum / rng) / math.log10(p)
            chop_raw = max(0.0, min(100.0, chop_raw))
            self.lines.chop[0] = -1.0 * (chop_raw / 50.0 - 1.0)
        else:
            self.lines.chop[0] = 0.0


class VolatilityCompression(bt.Indicator):
    """
    Volatility Compression (VCOMP).

    Ratio of recent to lookback true-range volatility, inverted so that
    compression (recent vol < lookback vol) yields positive values.

    Detects "consolidation after a big move" — a coiled spring before
    continuation. Differs from VCR (Bollinger percentile) by directly
    encoding the expansion-to-contraction transition via raw ATR windows.

    param_a: recent window (bars for recent volatility, e.g., 3-5)
    param_b: lookback window (bars for baseline volatility, e.g., 10-20)

    Output: tanh((1 - recent_vol/lookback_vol) * 2)
        +1 = extreme compression (recent vol << lookback vol)
         0 = no change in volatility regime
        -1 = extreme expansion (recent vol >> lookback vol)
    """
    lines = ('vcomp',)
    params = (('recent', 4), ('lookback', 16))

    def __init__(self):
        self.addminperiod(self.p.lookback + 1)

    def next(self):
        recent_n = self.p.recent
        lookback_n = self.p.lookback

        highs = np.array(self.data.high.get(size=lookback_n))
        lows = np.array(self.data.low.get(size=lookback_n))
        closes = np.array(self.data.close.get(size=lookback_n + 1))
        prev_closes = closes[:-1]

        # True Range = max(H-L, |H-prevC|, |L-prevC|)
        tr = np.maximum(
            highs - lows,
            np.maximum(np.abs(highs - prev_closes), np.abs(lows - prev_closes))
        )

        recent_vol = float(np.mean(tr[-recent_n:]))
        lookback_vol = float(np.mean(tr))

        if lookback_vol > 1e-10:
            ratio = recent_vol / lookback_vol
            raw = 1.0 - ratio
            self.lines.vcomp[0] = math.tanh(raw * 2.0)
        else:
            self.lines.vcomp[0] = 0.0


class MomentumPersistence(bt.Indicator):
    """
    Momentum Persistence (MPER).

    Measures whether the medium-term trend direction is being confirmed
    or contradicted by short-term price action. Captures "pullback within
    a trend" (still positive) vs "momentum has reversed" (goes to zero/negative).

    Unlike MTD (divergence magnitude), MPER encodes directional alignment:
    the medium-term trend's direction dominates, modulated by short-term
    confirmation or contradiction.

    param_a: short momentum period (e.g., 3-5 bars, captures recent candles)
    param_b: medium momentum period (e.g., 15-25 bars, captures the swing)

    Output:
        +1 = strong uptrend confirmed by short-term (bullish continuation)
        +0.3 to +0.5 = uptrend intact but short-term pulling back (the setup!)
         0 = no trend or trend fully contradicted by short-term
        -0.3 to -0.5 = downtrend intact with short-term bounce
        -1 = strong downtrend confirmed by short-term
    """
    lines = ('mper',)
    params = (('short_period', 4), ('medium_period', 20))

    def __init__(self):
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.addminperiod(self.p.medium_period + 14)

    def next(self):
        sp = self.p.short_period
        mp = self.p.medium_period
        c = self.data.close

        if c[-sp] == 0 or c[-mp] == 0 or c[0] == 0:
            self.lines.mper[0] = 0.0
            return

        short_roc = (c[0] / c[-sp]) - 1.0
        medium_roc = (c[0] / c[-mp]) - 1.0

        atr_pct = self.atr[0] / c[0]
        atr_pct = max(atr_pct, 1e-8)

        s = short_roc / atr_pct
        m = medium_roc / atr_pct

        trend_strength = math.tanh(m)

        if abs(m) > 1e-8:
            m_sign = 1.0 if m > 0 else -1.0
            alignment = math.tanh(s) * m_sign
        else:
            alignment = 0.0

        persistence = trend_strength * (0.5 + 0.5 * alignment)
        self.lines.mper[0] = max(-1.0, min(1.0, persistence))


class VolumeMomentumCoupling(bt.Indicator):
    """
    Volume-Momentum Coupling (VMC).

    Detects whether elevated volume is confirming price direction or
    occurring during consolidation (accumulation/distribution signal).

    Unlike VPD (correlation of deltas), VMC encodes the interaction between
    volume regime (elevated vs depressed) and momentum direction,
    specifically designed to be positive when volume is high during
    consolidation — "smart money still engaged."

    param_a: period for recent volume and momentum measurement (e.g., 5-10)
    param_b: period for baseline volume average (e.g., 30-50)

    Output:
        +0.7 to +1 = elevated volume + upward momentum (trend confirmation)
        +0.2 to +0.5 = elevated volume + flat momentum (accumulation setup!)
         0 = average volume or conflicting signals
        -0.2 to -0.5 = elevated volume + downward momentum (distribution)
        -0.5 to -1 = depressed volume (no participation)
    """
    lines = ('vmc',)
    params = (('recent', 5), ('baseline', 40))

    def __init__(self):
        self.vol_sma = bt.indicators.SMA(self.data.volume, period=self.p.baseline)
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.addminperiod(max(self.p.baseline, 14) + 1)

    def next(self):
        recent_n = self.p.recent
        baseline_avg = self.vol_sma[0]
        c = self.data.close

        if baseline_avg <= 0 or c[0] <= 0 or c[-recent_n] == 0:
            self.lines.vmc[0] = 0.0
            return

        vols = np.array(self.data.volume.get(size=recent_n))
        recent_vol_avg = float(np.mean(vols))

        if recent_vol_avg > 0:
            vol_regime = math.log(recent_vol_avg / baseline_avg)
        else:
            vol_regime = -2.0

        vol_signal = math.tanh(vol_regime * 2.0)

        roc = (c[0] / c[-recent_n]) - 1.0
        atr_pct = self.atr[0] / c[0]
        atr_pct = max(atr_pct, 1e-8)
        mom = roc / atr_pct
        mom_direction = math.tanh(mom)

        coupling = vol_signal * (0.5 + 0.5 * mom_direction)
        self.lines.vmc[0] = max(-1.0, min(1.0, coupling))


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


class MarketRegimeDetector(bt.Indicator):
    """
    Market Regime Detector based on previous period's high/low.

    Inspired by RWCS_LTD's "Market Regime Detector" (TradingView).
    Uses the previous period's high and low as reference levels:
      - Bullish:   close > previous period's high  → regime =  1
      - Bearish:   close < previous period's low   → regime = -1
      - Reverting: close between prev high and low  → regime =  0

    Also tracks:
      - direction: transition direction of the last regime change.
        +1 = improving (bear→revert or revert→bull)
        -1 = declining (bull→revert or revert→bear)
         0 = no change yet
        This distinguishes "reverting from bearish" (bullish recovery)
        from "reverting from bullish" (trend exhaustion).

      - stability: how many regime flips occurred in the last N bars.
        1.0 = stable (0 flips), 0.0 = max chop

    Period can be 'weekly' or 'monthly' (controlled by regime_period param).
    """
    lines = ('regime', 'direction', 'stability')
    params = (
        ('period', 'weekly'),      # 'weekly' or 'monthly'
        ('stability_window', 60),  # Bars to look back for flip counting
        ('max_flips', 18),         # Flips at which stability = 0 (18 = ~1 flip every 3-4 days is tolerable)
    )

    def __init__(self):
        self.prev_high = None
        self.prev_low = None
        self.curr_high = None
        self.curr_low = None
        self.curr_period = None
        self.regime_history = deque(maxlen=self.p.stability_window)
        self.last_regime = None
        self.prev_regime_state = None  # regime before the most recent change

    def _get_period_key(self, dt):
        if self.p.period == 'weekly':
            return dt.isocalendar()[:2]  # (year, week_number)
        else:
            return (dt.year, dt.month)

    def next(self):
        dt = self.data.datetime.date(0)
        period_key = self._get_period_key(dt)

        if self.curr_period is None:
            self.curr_period = period_key
            self.curr_high = self.data.high[0]
            self.curr_low = self.data.low[0]
        elif period_key != self.curr_period:
            # New period — previous period is now complete
            self.prev_high = self.curr_high
            self.prev_low = self.curr_low
            self.curr_period = period_key
            self.curr_high = self.data.high[0]
            self.curr_low = self.data.low[0]
        else:
            self.curr_high = max(self.curr_high, self.data.high[0])
            self.curr_low = min(self.curr_low, self.data.low[0])

        if self.prev_high is None:
            self.lines.regime[0] = 0.0
            self.lines.direction[0] = 0.0
            self.lines.stability[0] = 1.0
            return

        close = self.data.close[0]
        if close > self.prev_high:
            current_regime = 1.0    # Bullish
        elif close < self.prev_low:
            current_regime = -1.0   # Bearish
        else:
            current_regime = 0.0    # Reverting

        self.lines.regime[0] = current_regime

        # Track regime transitions for direction
        flipped = 0
        if self.last_regime is not None and current_regime != self.last_regime:
            flipped = 1
            self.prev_regime_state = self.last_regime
        self.last_regime = current_regime
        self.regime_history.append(flipped)

        # Direction: based on last transition
        # +1 = improving (came from lower regime), -1 = declining (came from higher)
        if self.prev_regime_state is not None:
            self.lines.direction[0] = current_regime - self.prev_regime_state
        else:
            self.lines.direction[0] = 0.0

        # Stability: 1.0 = no flips (stable), 0.0 = max_flips or more (choppy)
        flip_count = sum(self.regime_history)
        self.lines.stability[0] = max(0.0, 1.0 - flip_count / self.p.max_flips)


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
        ('min_raw_prediction', 0.0),      # Minimum expected return in ATR units (0=disabled). Requires neighbors
                                          # to collectively predict at least N * ATR of price movement.
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

        # === Feature 6 (Volume-Price Divergence) ===
        ('f6_type', 'VPD'),
        ('f6_param_a', 14),             # Correlation window
        ('f6_param_b', 1),              # Smoothing (1=none)

        # === Feature 7 (Candle Structure) ===
        ('f7_type', 'CS'),
        ('f7_param_a', 5),              # Averaging window
        ('f7_param_b', 2),              # Sensitivity (tanh scaling)

        # === Feature 8 (Momentum Acceleration) ===
        ('f8_type', 'MACC'),
        ('f8_param_a', 5),              # ROC period
        ('f8_param_b', 5),              # Lag for comparison

        # === Feature 9 (OBV Trend) ===
        ('f9_type', 'OBVT'),
        ('f9_param_a', 20),             # SMA/StdDev period
        ('f9_param_b', 3),              # Z-score divisor

        # === Feature 10 (Streak Pattern) ===
        ('f10_type', 'STRK'),
        ('f10_param_a', 10),            # Max streak length
        ('f10_param_b', 2),             # ATR multiplier for magnitude

        # === Feature 11 (Choppiness Index) ===
        ('f11_type', 'CHOP'),
        ('f11_param_a', 14),            # ATR/range period
        ('f11_param_b', 1),             # Not used

        # === Feature 12 (Volatility Compression) ===
        ('f12_type', 'VCOMP'),
        ('f12_param_a', 4),             # Recent volatility window
        ('f12_param_b', 16),            # Lookback volatility window

        # === Feature 13 (Momentum Persistence) ===
        ('f13_type', 'MPER'),
        ('f13_param_a', 4),             # Short momentum period
        ('f13_param_b', 20),            # Medium momentum period

        # === Feature 14 (Volume-Momentum Coupling) ===
        ('f14_type', 'VMC'),
        ('f14_param_a', 5),             # Recent volume/momentum window
        ('f14_param_b', 40),            # Baseline volume average period

        # === Filters ===
        ('use_volatility_filter', True),
        ('use_regime_filter', True),
        ('regime_threshold', 0),  # 0=block bearish, 1=require bullish
        ('regime_period', 'weekly'),  # 'weekly' or 'monthly'
        ('use_regime_direction', True),  # True=allow reverting-from-bearish when threshold=1
        ('regime_stability_min', 0.0),  # Min stability to trade (0=off, 0.5=moderate, 0.7=strict)
        ('regime_stability_window', 60),  # Bars to look back for regime flip counting
        ('regime_max_flips', 18),  # Number of flips at which stability = 0 (higher = more permissive)
        ('use_adx_filter', False),
        ('adx_threshold', 20),
        ('use_ema_filter', False),
        ('ema_period', 200),
        ('ema_slope_lookback', 5),  # Bars to measure EMA slope over
        ('use_sma_filter', False),
        ('sma_period', 200),
        ('sma_slope_lookback', 5),  # Bars to measure SMA slope over

        # === SPY Market Regime Filter ===
        ('use_spy_filter', False),           # Enable SPY market-wide regime filter
        ('spy_regime_threshold', 0),         # 0=block bearish, 1=require bullish
        ('spy_regime_period', 'weekly'),     # 'weekly' or 'monthly'

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

        # === Chandelier Exit (ATR Trailing Stop) ===
        ('use_trailing_atr_exit', False),           # Enable chandelier ATR trailing stop
        ('trailing_atr_warmup', 3),                 # Bars in trade before stop activates
        ('chandelier_start_mult', 3.0),             # Starting (wide) ATR multiplier — stop set from peak HIGH
        ('chandelier_end_mult', 1.5),               # Ending (tight) ATR multiplier — tightens over time/profit
        ('chandelier_tighten_bars', 20),            # Bars over which to tighten (time mode)
        ('chandelier_mult_mode', 'blend'),          # 'time', 'profit', or 'blend' (min of both)
        ('chandelier_profit_atr_threshold', 1.0),   # ATR multiples of profit to fully tighten (profit mode)
        ('chandelier_breakeven_atr', 0.0),          # Lock stop at entry price once profit >= N * ATR (0 = off)

        # === Loss Penalty (ML bearish feedback after losing trades) ===
        ('use_loss_penalty', True),        # Enable loss penalty on predictions
        ('loss_penalty_amount', 0),        # Penalty per loss (0 = auto: neighbors_count/2)
        ('loss_penalty_decay', 0.90),      # Decay rate per bar (0.90 = ~10 bars to halve)

        # === Prediction Validation Exit ===
        ('use_prediction_exit', False),       # Force-exit if ML wrong after label_lookahead bars
        ('prediction_exit_threshold', -0.5),  # Exit if P&L% below this at validation bar

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

        # === Fundamental / Earnings Settings ===
        ('use_fundamental_filter', False),          # Master switch for fundamental filtering
        ('fundamental_quality_weight', 0.4),        # Weight for quality score in combined
        ('fundamental_momentum_weight', 0.6),       # Weight for growth momentum in combined

        # Earnings Blackout
        ('earnings_blackout_before', 5),            # Days before earnings to block trades
        ('earnings_blackout_after', 2),             # Days after earnings to block trades
        ('close_before_earnings', True),            # Close positions before blackout starts

        # Score Thresholds
        ('min_trending_probability', 50),           # Minimum combined score to trade (0-100)
        ('full_position_threshold', 70),            # Score >= this gets full position
        ('reduced_position_pct', Decimal('0.75')),  # Position size when score 50-70

        # Staleness Handling
        ('max_days_since_earnings', 100),           # Max days before confidence decays significantly
        ('staleness_decay_rate', 0.005),            # Confidence decay per day after earnings

        # Individual Component Thresholds (optional hard filters)
        ('min_quality_score', 0),                   # Minimum quality score (0 = disabled)
        ('min_momentum_score', 0),                  # Minimum momentum score (0 = disabled)

        # Post-Earnings Reaction Filter
        ('use_earnings_reaction', False),            # Use price reaction after earnings as sentiment filter
        ('earnings_reaction_days', 5),               # Trading days after earnings to measure reaction
        ('min_earnings_reaction_pct', -3.0),         # Min % change to allow trades (below = blocked)

        # Fundamental data symbol override (empty = auto-detect from data feed)
        ('fundamental_symbol', ''),
    )

    def __init__(self):
        """Initialize indicators and state."""
        self._init_features()
        self._init_filters()
        self._init_kernels()
        self._init_state()
        self._init_fundamentals()

    def _init_features(self):
        """Initialize feature indicators."""
        self.features = []

        feature_configs = [
            (self.p.f1_type, self.p.f1_param_a, self.p.f1_param_b),
            (self.p.f2_type, self.p.f2_param_a, self.p.f2_param_b),
            (self.p.f3_type, self.p.f3_param_a, self.p.f3_param_b),
            (self.p.f4_type, self.p.f4_param_a, self.p.f4_param_b),
            (self.p.f5_type, self.p.f5_param_a, self.p.f5_param_b),
            (self.p.f6_type, self.p.f6_param_a, self.p.f6_param_b),
            (self.p.f7_type, self.p.f7_param_a, self.p.f7_param_b),
            (self.p.f8_type, self.p.f8_param_a, self.p.f8_param_b),
            (self.p.f9_type, self.p.f9_param_a, self.p.f9_param_b),
            (self.p.f10_type, self.p.f10_param_a, self.p.f10_param_b),
            (self.p.f11_type, self.p.f11_param_a, self.p.f11_param_b),
            (self.p.f12_type, self.p.f12_param_a, self.p.f12_param_b),
            (self.p.f13_type, self.p.f13_param_a, self.p.f13_param_b),
            (self.p.f14_type, self.p.f14_param_a, self.p.f14_param_b),
        ]

        actual_count = min(self.p.feature_count, len(feature_configs))
        if self.p.feature_count > len(feature_configs) and self.p.verbose:
            print(f"WARNING: feature_count={self.p.feature_count} but only {len(feature_configs)} feature slots defined. Using {len(feature_configs)}.")
        for i, (ftype, param_a, param_b) in enumerate(feature_configs[:actual_count]):
            feature = self._create_feature(ftype, param_a, param_b)
            self.features.append(feature)

        # ATR for label normalization (dead zone + magnitude labels)
        self.label_atr = bt.indicators.ATR(self.data, period=14)

    # Map feature type to its output line name for O(1) value retrieval
    _FEATURE_LINE_NAMES = {
        'RSI': 'nrsi', 'ADX': 'nadx', 'ATRR': 'atr_ratio', 'PP': 'position',
        'ER': 'er', 'RSM': 'rsm', 'VA': 'va', 'MTD': 'mtd', 'ZS': 'zscore',
        'VCR': 'vcr', 'VPD': 'vpd', 'CS': 'cs', 'MACC': 'macc',
        'OBVT': 'obvt', 'STRK': 'strk', 'CHOP': 'chop',
        'VCOMP': 'vcomp', 'MPER': 'mper', 'VMC': 'vmc',
    }

    def _create_feature(self, ftype, param_a, param_b):
        """Create a feature indicator based on type."""
        if ftype == 'RSI':
            feature = NormalizedRSI(self.data, period=param_a, smoothing=param_b)
        elif ftype == 'ADX':
            feature = NormalizedADX(self.data, period=param_a)
        elif ftype == 'ATRR':
            feature = ATRRatio(self.data, period=param_a)
        elif ftype == 'PP':
            feature = PricePosition(self.data, period=param_a)
        elif ftype == 'ER':
            feature = EfficiencyRatio(self.data, period=param_a)
        elif ftype == 'RSM':
            feature = RelativeStrengthMomentum(self.data, momentum_period=param_a, lookback=param_b)
        elif ftype == 'VA':
            feature = VolumeAnomaly(self.data, period=param_a)
        elif ftype == 'MTD':
            feature = MultiTimeframeDivergence(self.data, short_period=param_a, long_period=param_b)
        elif ftype == 'ZS':
            feature = MeanReversionZScore(self.data, period=param_a)
        elif ftype == 'VCR':
            feature = VolatilityContractionRatio(self.data, bb_period=param_a, lookback=param_b)
        elif ftype == 'VPD':
            feature = VolumePriceDivergence(self.data, period=param_a, smoothing=param_b)
        elif ftype == 'CS':
            feature = CandleStructure(self.data, period=param_a, sensitivity=param_b)
        elif ftype == 'MACC':
            feature = MomentumAcceleration(self.data, roc_period=param_a, lag=param_b)
        elif ftype == 'OBVT':
            feature = OBVTrend(self.data, period=param_a, z_divisor=param_b)
        elif ftype == 'STRK':
            feature = StreakPattern(self.data, max_streak=param_a, atr_mult=param_b)
        elif ftype == 'CHOP':
            feature = ChoppinessIndex(self.data, period=param_a, unused=param_b)
        elif ftype == 'VCOMP':
            feature = VolatilityCompression(self.data, recent=param_a, lookback=param_b)
        elif ftype == 'MPER':
            feature = MomentumPersistence(self.data, short_period=param_a, medium_period=param_b)
        elif ftype == 'VMC':
            feature = VolumeMomentumCoupling(self.data, recent=param_a, baseline=param_b)
        else:
            raise ValueError(f"Unknown feature type: {ftype}")

        feature._line_name = self._FEATURE_LINE_NAMES[ftype]
        return feature

    def _init_filters(self):
        """Initialize filter indicators."""
        # Volatility filter
        if self.p.use_volatility_filter:
            self.volatility_filter = VolatilityFilter(self.data)

        # Regime filter (previous period high/low breakout detector)
        if self.p.use_regime_filter:
            self.regime_filter = MarketRegimeDetector(
                self.data,
                period=self.p.regime_period,
                stability_window=self.p.regime_stability_window,
                max_flips=self.p.regime_max_flips
            )

        # ADX filter
        if self.p.use_adx_filter:
            self.adx = SafeADX(self.data, period=14)

        # EMA filter
        if self.p.use_ema_filter:
            self.ema = bt.indicators.EMA(self.data.close, period=self.p.ema_period)

        # SMA filter
        if self.p.use_sma_filter:
            self.sma = bt.indicators.SMA(self.data.close, period=self.p.sma_period)

        # SPY market-wide regime filter
        if self.p.use_spy_filter:
            self._spy_regime_lookup = _compute_spy_regime(
                period=self.p.spy_regime_period
            )
            if not self._spy_regime_lookup:
                if self.p.verbose:
                    print("WARNING: SPY data download failed, disabling SPY filter")
                self._spy_filter_active = False
            else:
                self._spy_filter_active = True
        else:
            self._spy_regime_lookup = {}
            self._spy_filter_active = False

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
        self._exit_reason = None
        self.prediction = 0
        self.raw_prediction = 0.0

        # ATR trailing stop state
        self._highest_since_entry = 0.0
        self._lowest_since_entry = 0.0
        self._trailing_stop_level = 0.0
        self._trailing_entry_price = 0.0

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

        # Regime diagnostics
        self.regime_diagnostics = {
            'bullish_bars': 0,
            'bearish_bars': 0,
            'reverting_bars': 0,
            'reverting_improving_bars': 0,   # reverting with direction > 0 (from bearish)
            'reverting_declining_bars': 0,   # reverting with direction < 0 (from bullish)
            'trades_in_bullish': 0,
            'trades_in_bearish': 0,
            'trades_in_reverting': 0,
            'trades_in_reverting_improving': 0,
            'trades_in_reverting_declining': 0,
        }

    def _init_fundamentals(self):
        """Initialize fundamental data provider if enabled."""
        self.fundamental_provider = None
        self._cached_fundamental_date = None
        self._cached_trending_prob = None
        self._cached_quality_score = None
        self._cached_momentum_score = None
        self._cached_confidence = None
        self._cached_earnings_reaction = None

        if not self.p.use_fundamental_filter:
            return

        try:
            from fundamental_data import FundamentalCache

            # Get symbol from params or data feed name
            symbol = self.p.fundamental_symbol
            if not symbol:
                # Try to extract from data feed name
                data_name = self.data._name if hasattr(self.data, '_name') else ''
                if data_name:
                    symbol = data_name.split('_')[0].upper()
                else:
                    if self.p.verbose:
                        print("FUNDAMENTAL: Could not determine symbol, disabling filter")
                    return

            self.fundamental_provider = FundamentalCache.get_provider(
                symbol,
                lookback_years=5,
                verbose=False
            )

            if self.p.verbose:
                summary = self.fundamental_provider.get_summary()
                print(f"FUNDAMENTAL: Loaded data for {symbol}")
                print(f"  Quality Score: {summary['quality_score']:.1f}")
                print(f"  Momentum Score: {summary['momentum_score']:.1f}")
                print(f"  Trending Probability: {summary['trending_probability']:.1f}")
                print(f"  Has Earnings Dates: {summary['has_earnings_dates']}")

        except ImportError:
            if self.p.verbose:
                print("FUNDAMENTAL: fundamental_data.py not found, disabling filter")
        except Exception as e:
            if self.p.verbose:
                print(f"FUNDAMENTAL: Error loading data: {e}")

    def _get_fundamental_scores(self, current_date):
        """Get fundamental scores with caching (scores only change on earnings dates)."""
        if self.fundamental_provider is None:
            return None, None, None, None

        # Check cache - fundamental scores don't change intra-day
        if self._cached_fundamental_date == current_date:
            return (self._cached_trending_prob, self._cached_quality_score,
                    self._cached_momentum_score, self._cached_confidence)

        # Calculate fresh scores
        self._cached_fundamental_date = current_date
        current_price = float(self.data.close[0])
        self._cached_quality_score = self.fundamental_provider.get_quality_score(current_date, price=current_price)
        self._cached_momentum_score = self.fundamental_provider.get_growth_momentum_score(current_date)
        self._cached_trending_prob = self.fundamental_provider.get_trending_probability(
            current_date,
            quality_weight=self.p.fundamental_quality_weight,
            momentum_weight=self.p.fundamental_momentum_weight,
            price=current_price
        )
        self._cached_confidence = self.fundamental_provider.get_confidence_multiplier(current_date)
        self._cached_earnings_reaction = self.fundamental_provider.get_earnings_reaction(
            current_date, self.p.earnings_reaction_days)

        return (self._cached_trending_prob, self._cached_quality_score,
                self._cached_momentum_score, self._cached_confidence)

    def _check_earnings_blackout(self, current_date):
        """Check if current date is in earnings blackout window."""
        if self.fundamental_provider is None:
            return False

        return self.fundamental_provider.is_in_earnings_blackout(
            current_date,
            self.p.earnings_blackout_before,
            self.p.earnings_blackout_after
        )

    def _check_earnings_exit(self):
        """Close position if we're about to enter earnings blackout window."""
        if not self.p.use_fundamental_filter or not self.p.close_before_earnings:
            return False

        if not self.position or self.fundamental_provider is None:
            return False

        current_date = self.data.datetime.date(0)
        days_until = self.fundamental_provider.days_until_earnings(current_date)

        if days_until is not None and 0 < days_until <= self.p.earnings_blackout_before:
            if self.p.verbose:
                print(f"EARNINGS EXIT: Closing position {days_until} days before earnings on {current_date}")
            self._close_position("EARNINGS BLACKOUT APPROACHING")
            return True

        return False

    def _get_lorentzian_distance(self, idx):
        """
        Calculate Lorentzian distance between current features and historical features.

        Lorentzian distance: sum of log(1 + |x_i - y_i|) for each feature

        This metric reduces the influence of outliers compared to Euclidean distance.

        Note: This per-sample method is kept for compatibility but the main KNN
        path uses vectorized batch computation via numpy in _run_knn().
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
        return getattr(feature.lines, feature._line_name)[0]

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

        Distance-weighted voting: each neighbor's label is weighted by
        1 / (distance + epsilon), so closer neighbors have more influence.
        This improves prediction accuracy over uniform voting because
        near-identical historical patterns are trusted more than distant ones.

        Performance: distances are computed in batch via numpy, then the
        sequential neighbor selection operates on pre-computed scalars.
        """
        if len(self.label_array) < self.p.neighbors_count:
            return 0

        n_features = len(self.features)
        size_loop = min(self.p.max_bars_back - 1, len(self.label_array) - 1)
        for fi in range(n_features):
            size_loop = min(size_loop, len(self.feature_arrays[fi]))
        if size_loop <= 0:
            return 0

        # Build historical feature matrix: shape (size_loop, n_features)
        hist_matrix = np.empty((size_loop, n_features))
        for fi in range(n_features):
            fa = self.feature_arrays[fi]
            hist_matrix[:, fi] = list(fa)[:size_loop]

        # Current feature vector: shape (n_features,)
        current_vec = np.empty(n_features)
        for fi, feature in enumerate(self.features):
            current_vec[fi] = self._get_feature_value(feature)

        # Vectorized Lorentzian distance: log(1 + |current - historical|) summed over features
        all_distances = np.sum(np.log1p(np.abs(current_vec - hist_matrix)), axis=1)

        # Labels for the same range
        labels_arr = list(self.label_array)

        # Sequential neighbor selection (preserves original algorithm behavior)
        distances = []
        predictions = []
        last_distance = -1.0

        for i in range(size_loop):
            d = all_distances[i]

            if d >= last_distance and (i % 4) != 0:
                last_distance = d
                distances.append(d)
                predictions.append(labels_arr[i])

                if len(predictions) > self.p.neighbors_count:
                    sorted_dist = sorted(distances)
                    idx_75 = int(self.p.neighbors_count * 3 / 4)
                    if idx_75 < len(sorted_dist):
                        last_distance = sorted_dist[idx_75]
                    distances.pop(0)
                    predictions.pop(0)

        if not predictions:
            return 0

        # Distance-weighted voting: weight = 1 / (distance + epsilon)
        # Closer neighbors (smaller distance) get higher weight.
        epsilon = 1e-8
        weighted_sum = 0.0
        weight_total = 0.0
        for d, label in zip(distances, predictions):
            w = 1.0 / (d + epsilon)
            weighted_sum += w * label
            weight_total += w

        # Scale to comparable magnitude as uniform voting (sum of K labels)
        # so downstream normalization thresholds remain valid.
        if weight_total > 0:
            avg_vote = weighted_sum / weight_total
            return avg_vote * len(predictions)
        return 0

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
        arr = np.fromiter(history, dtype=np.float64, count=len(history))
        count_below = int(np.sum(arr < raw))
        count_equal = int(np.sum(arr == raw))
        # Mid-rank method: ties get average rank
        percentile = (count_below + 0.5 * count_equal) / len(arr)

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

        # Regime filter (period high/low breakout)
        # regime: 1=bullish, 0=reverting, -1=bearish
        # direction: +1=improving (came from lower), -1=declining (came from higher)
        # threshold=0: blocks bearish only; threshold=1: requires bullish
        if self.p.use_regime_filter:
            regime = self.regime_filter.regime[0]
            if regime < self.p.regime_threshold:
                # Below threshold - but allow reverting-from-bearish if direction enabled
                # This catches recovery entries (bear→revert) while blocking
                # weakening entries (bull→revert)
                if (self.p.use_regime_direction
                        and regime == 0
                        and self.regime_filter.direction[0] > 0):
                    pass  # Allow: reverting with improving direction
                else:
                    return False
            # Regime stability: block trades when regime is flipping too often (choppy)
            if self.p.regime_stability_min > 0:
                if self.regime_filter.stability[0] < self.p.regime_stability_min:
                    return False

        # SPY market-wide regime filter
        if self.p.use_spy_filter and self._spy_filter_active:
            current_date = self.data.datetime.date(0)
            spy_regime = self._spy_regime_lookup.get(current_date, None)
            if spy_regime is not None:
                if spy_regime < self.p.spy_regime_threshold:
                    return False
            # If date not found (holiday mismatch), pass through permissively

        # ADX filter
        if self.p.use_adx_filter:
            if self.adx[0] < self.p.adx_threshold:
                return False

        # Fundamental filter
        if self.p.use_fundamental_filter and self.fundamental_provider is not None:
            current_date = self.data.datetime.date(0)

            # Check earnings blackout (in window today, or approaching within blackout_before days)
            if self._check_earnings_blackout(current_date):
                if self.p.verbose:
                    print(f"FILTER BLOCKED: Earnings blackout on {current_date}")
                return False
            if self.p.close_before_earnings:
                days_until = self.fundamental_provider.days_until_earnings(current_date)
                if days_until is not None and 0 < days_until <= self.p.earnings_blackout_before + 1:
                    if self.p.verbose:
                        print(f"FILTER BLOCKED: Earnings in {days_until} days on {current_date}")
                    return False

            # Check combined score threshold
            trending_prob, quality, momentum, _ = self._get_fundamental_scores(current_date)

            if trending_prob is not None and trending_prob < self.p.min_trending_probability:
                if self.p.verbose:
                    print(f"FILTER BLOCKED: Trending probability {trending_prob:.1f} < {self.p.min_trending_probability}")
                return False

            # Optional individual score checks
            if self.p.min_quality_score > 0 and quality is not None:
                if quality < self.p.min_quality_score:
                    if self.p.verbose:
                        print(f"FILTER BLOCKED: Quality score {quality:.1f} < {self.p.min_quality_score}")
                    return False

            if self.p.min_momentum_score > 0 and momentum is not None:
                if momentum < self.p.min_momentum_score:
                    if self.p.verbose:
                        print(f"FILTER BLOCKED: Momentum score {momentum:.1f} < {self.p.min_momentum_score}")
                    return False

            # Post-earnings reaction filter
            if self.p.use_earnings_reaction:
                reaction = self._cached_earnings_reaction
                if reaction is not None and reaction < self.p.min_earnings_reaction_pct:
                    if self.p.verbose:
                        print(f"FILTER BLOCKED: Earnings reaction {reaction:.1f}% < {self.p.min_earnings_reaction_pct}%")
                    return False

        return True

    def _check_ema_uptrend(self):
        """Check if EMA slope is rising."""
        if not self.p.use_ema_filter:
            return True
        lb = self.p.ema_slope_lookback
        if len(self.ema) <= lb:
            return True
        return self.ema[0] > self.ema[-lb]

    def _check_ema_downtrend(self):
        """Check if EMA slope is falling."""
        if not self.p.use_ema_filter:
            return True
        lb = self.p.ema_slope_lookback
        if len(self.ema) <= lb:
            return True
        return self.ema[0] < self.ema[-lb]

    def _check_sma_uptrend(self):
        """Check if SMA slope is rising."""
        if not self.p.use_sma_filter:
            return True
        lb = self.p.sma_slope_lookback
        if len(self.sma) <= lb:
            return True
        return self.sma[0] > self.sma[-lb]

    def _check_sma_downtrend(self):
        """Check if SMA slope is falling."""
        if not self.p.use_sma_filter:
            return True
        lb = self.p.sma_slope_lookback
        if len(self.sma) <= lb:
            return True
        return self.sma[0] < self.sma[-lb]

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

        # Check normalized prediction strength (relative: percentile rank 0-100)
        min_strength = self.p.min_prediction_strength
        prediction_strong_enough = abs(self.prediction) >= min_strength

        # Check raw prediction (absolute: expected return in ATR units)
        min_raw = self.p.min_raw_prediction
        raw_strong_enough = abs(self.raw_prediction) >= min_raw if min_raw > 0 else True

        if self.prediction > 0 and prediction_strong_enough and raw_strong_enough and self._check_filters():
            self.signal = 1  # Long
        elif self.prediction < 0 and prediction_strong_enough and raw_strong_enough and self._check_filters():
            self.signal = -1  # Short
        else:
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

        # Check for earnings exit (close position before earnings blackout)
        if self._check_earnings_exit():
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

        # === Regime Diagnostics ===
        if self.p.use_regime_filter:
            regime_val = self.regime_filter.regime[0]
            direction_val = self.regime_filter.direction[0]
            if regime_val > 0.5:
                self.regime_diagnostics['bullish_bars'] += 1
            elif regime_val < -0.5:
                self.regime_diagnostics['bearish_bars'] += 1
            else:
                self.regime_diagnostics['reverting_bars'] += 1
                if direction_val > 0:
                    self.regime_diagnostics['reverting_improving_bars'] += 1
                elif direction_val < 0:
                    self.regime_diagnostics['reverting_declining_bars'] += 1

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

    def get_regime_diagnostics(self):
        """Get market regime distribution and trade-regime breakdown."""
        rd = self.regime_diagnostics.copy()
        total = rd['bullish_bars'] + rd['bearish_bars'] + rd['reverting_bars']
        if total > 0:
            rd['bullish_pct'] = (rd['bullish_bars'] / total) * 100
            rd['bearish_pct'] = (rd['bearish_bars'] / total) * 100
            rd['reverting_pct'] = (rd['reverting_bars'] / total) * 100
        else:
            rd['bullish_pct'] = rd['bearish_pct'] = rd['reverting_pct'] = 0
        rev = rd['reverting_bars']
        if rev > 0:
            rd['reverting_improving_pct'] = (rd['reverting_improving_bars'] / rev) * 100
            rd['reverting_declining_pct'] = (rd['reverting_declining_bars'] / rev) * 100
        else:
            rd['reverting_improving_pct'] = rd['reverting_declining_pct'] = 0
        return rd

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

        # ATR trailing stop
        if self._update_and_check_trailing_stop():
            return

        # Prediction validation exit — force-exit if ML was wrong after label_lookahead bars
        if self.p.use_prediction_exit and bars_since_entry == self.p.label_lookahead:
            if self.position.size > 0:
                pnl_pct = ((self.data.close[0] / self.entry_price) - 1) * 100
            else:
                pnl_pct = ((self.entry_price / self.data.close[0]) - 1) * 100
            if pnl_pct < self.p.prediction_exit_threshold:
                self._close_position(f"ML PREDICTION WRONG ({pnl_pct:.2f}%)")
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
            elif self.position.size < 0:
                current_pnl_pct = (self.entry_price - self.data.close[0]) / self.entry_price
            else:
                return

            stop = float(self.p.stop_loss_pct)
            if current_pnl_pct <= -stop:
                self._close_position("STOP LOSS HIT")

    def _compute_dynamic_multiplier(self, bars_in_trade, atr_val):
        """Compute dynamic Chandelier Exit multiplier based on time and/or profit."""
        start = self.p.chandelier_start_mult
        end = self.p.chandelier_end_mult
        mode = self.p.chandelier_mult_mode

        # Time factor: linear interpolation over tighten_bars
        time_frac = min(1.0, bars_in_trade / max(1, self.p.chandelier_tighten_bars))
        time_mult = start + (end - start) * time_frac

        # Profit factor: tighten based on unrealized profit in ATR units
        if self.position.size > 0:
            unrealized_profit = self.data.close[0] - self._trailing_entry_price
        else:
            unrealized_profit = self._trailing_entry_price - self.data.close[0]

        if unrealized_profit > 0 and atr_val > 0:
            profit_atr = unrealized_profit / atr_val
            profit_frac = min(1.0, profit_atr / max(0.01, self.p.chandelier_profit_atr_threshold))
        else:
            profit_frac = 0.0
        profit_mult = start + (end - start) * profit_frac

        # Mode selection
        if mode == 'time':
            return time_mult
        elif mode == 'profit':
            return profit_mult
        else:  # 'blend' — use the tighter of the two
            return min(time_mult, profit_mult)

    def _update_and_check_trailing_stop(self):
        """
        Chandelier Exit: ATR trailing stop from highest HIGH (longs) or lowest LOW (shorts).

        The multiplier tightens dynamically from chandelier_start_mult to chandelier_end_mult
        over time and/or as profit grows. If chandelier_breakeven_atr is set, the stop
        is floored at entry price once unrealized profit reaches that many ATR multiples.

        Returns:
            bool: True if stop was triggered and position closed.
        """
        if not self.p.use_trailing_atr_exit:
            return False

        atr_val = self.label_atr[0]
        if atr_val <= 0:
            return False

        bars_in_trade = len(self) - self.entry_bar
        mult = self._compute_dynamic_multiplier(bars_in_trade, atr_val)

        if self.position.size > 0:
            # Long: track highest HIGH (classic Chandelier), ratchet stop upward
            if self.data.high[0] > self._highest_since_entry:
                self._highest_since_entry = self.data.high[0]

            new_stop = self._highest_since_entry - (mult * atr_val)

            # Break-even floor: once profit >= N * ATR, stop can't go below entry
            if self.p.chandelier_breakeven_atr > 0:
                if (self.data.close[0] - self.entry_price) >= self.p.chandelier_breakeven_atr * atr_val:
                    new_stop = max(new_stop, self.entry_price)

            if new_stop > self._trailing_stop_level:
                self._trailing_stop_level = new_stop

            if bars_in_trade >= self.p.trailing_atr_warmup:
                if self.data.close[0] < self._trailing_stop_level:
                    self.signal = -1
                    self._close_position(
                        f"CHANDELIER STOP ({self.data.close[0]:.2f} < "
                        f"{self._trailing_stop_level:.2f}, "
                        f"peak={self._highest_since_entry:.2f}, mult={mult:.2f}x)"
                    )
                    return True

        elif self.position.size < 0:
            # Short: track lowest LOW, ratchet stop downward
            if self.data.low[0] < self._lowest_since_entry:
                self._lowest_since_entry = self.data.low[0]

            new_stop = self._lowest_since_entry + (mult * atr_val)

            # Break-even floor: once profit >= N * ATR, stop can't go above entry
            if self.p.chandelier_breakeven_atr > 0:
                if (self.entry_price - self.data.close[0]) >= self.p.chandelier_breakeven_atr * atr_val:
                    new_stop = min(new_stop, self.entry_price)

            if new_stop < self._trailing_stop_level:
                self._trailing_stop_level = new_stop

            if bars_in_trade >= self.p.trailing_atr_warmup:
                if self.data.close[0] > self._trailing_stop_level:
                    self.signal = 1
                    self._close_position(
                        f"CHANDELIER STOP ({self.data.close[0]:.2f} > "
                        f"{self._trailing_stop_level:.2f}, "
                        f"trough={self._lowest_since_entry:.2f}, mult={mult:.2f}x)"
                    )
                    return True

        return False

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

        # ATR trailing stop (wider than kernel, survives normal pullbacks)
        if self._update_and_check_trailing_stop():
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

            # Initialize chandelier trailing stop (track HIGH from entry bar)
            self._highest_since_entry = self.data.high[0]
            self._trailing_entry_price = self.data.close[0]
            self._trailing_stop_level = self.data.close[0] - (self.p.chandelier_start_mult * self.label_atr[0])

            # Track regime at entry
            if self.p.use_regime_filter:
                r = self.regime_filter.regime[0]
                d = self.regime_filter.direction[0]
                if r > 0.5: self.regime_diagnostics['trades_in_bullish'] += 1
                elif r < -0.5: self.regime_diagnostics['trades_in_bearish'] += 1
                else:
                    self.regime_diagnostics['trades_in_reverting'] += 1
                    if d > 0: self.regime_diagnostics['trades_in_reverting_improving'] += 1
                    elif d < 0: self.regime_diagnostics['trades_in_reverting_declining'] += 1

            if self.p.verbose:
                band = self._get_prediction_band(self.entry_norm_prediction)
                regime_str = ""
                if self.p.use_regime_filter:
                    r = self.regime_filter.regime[0]
                    d = self.regime_filter.direction[0]
                    regime_label = 'BULL' if r > 0.5 else 'BEAR' if r < -0.5 else 'REVERT'
                    if r == 0:
                        regime_label += f"({'UP' if d > 0 else 'DN' if d < 0 else '?'})"
                    regime_str = f" | Regime: {regime_label}"
                print(f"BUY SIGNAL: {self.data.datetime.date(0)} | "
                      f"Prediction: {self.prediction:.1f} (raw: {self.raw_prediction:.2f}) [{band}] | "
                      f"Price: ${self.data.close[0]:.2f}{regime_str}")

    def _execute_short(self):
        """Execute short sell order (go short)."""
        size = self._calculate_position_size()
        if size > 0:
            self.order = self.sell(size=size)
            self.entry_bar = len(self)
            self.entry_price = self.data.close[0]
            self.entry_prediction = abs(self.raw_prediction)
            self.entry_norm_prediction = abs(self.prediction)

            # Initialize chandelier trailing stop (track LOW from entry bar)
            self._lowest_since_entry = self.data.low[0]
            self._trailing_entry_price = self.data.close[0]
            self._trailing_stop_level = self.data.close[0] + (self.p.chandelier_start_mult * self.label_atr[0])

            # Track regime at entry
            if self.p.use_regime_filter:
                r = self.regime_filter.regime[0]
                d = self.regime_filter.direction[0]
                if r > 0.5: self.regime_diagnostics['trades_in_bullish'] += 1
                elif r < -0.5: self.regime_diagnostics['trades_in_bearish'] += 1
                else:
                    self.regime_diagnostics['trades_in_reverting'] += 1
                    if d > 0: self.regime_diagnostics['trades_in_reverting_improving'] += 1
                    elif d < 0: self.regime_diagnostics['trades_in_reverting_declining'] += 1

            if self.p.verbose:
                band = self._get_prediction_band(self.entry_norm_prediction)
                regime_str = ""
                if self.p.use_regime_filter:
                    r = self.regime_filter.regime[0]
                    regime_str = f" | Regime: {'BULL' if r > 0.5 else 'BEAR' if r < -0.5 else 'REVERT'}"
                print(f"SHORT SIGNAL: {self.data.datetime.date(0)} | "
                      f"Prediction: {self.prediction:.1f} (raw: {self.raw_prediction:.2f}) [{band}] | "
                      f"Price: ${self.data.close[0]:.2f}{regime_str}")

    def _close_position(self, reason):
        """Close current position (long or short)."""
        self._exit_reason = reason
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
        """Calculate position size based on available cash and fundamental score.
        Uses 95% of target to leave margin for gap-up between signal and fill.
        Scales position based on fundamental trending probability if enabled."""
        cash = self.broker.getcash()
        base_position_value = cash * float(self.p.position_size_pct)

        # Apply fundamental score scaling
        position_multiplier = 1.0

        if self.p.use_fundamental_filter and self.fundamental_provider is not None:
            current_date = self.data.datetime.date(0)
            trending_prob, _, _, confidence = self._get_fundamental_scores(current_date)

            if trending_prob is not None:
                if trending_prob >= self.p.full_position_threshold:
                    # Full position for high-scoring stocks
                    position_multiplier = 1.0
                elif trending_prob >= self.p.min_trending_probability:
                    # Reduced position for acceptable but not stellar stocks
                    position_multiplier = float(self.p.reduced_position_pct)
                else:
                    # Should not reach here (filter would block), but safety
                    position_multiplier = 0.0

                # Apply confidence decay for stale data
                if confidence is not None:
                    position_multiplier *= confidence

                if self.p.verbose and position_multiplier < 1.0:
                    print(f"POSITION SIZING: Trending prob {trending_prob:.1f}, "
                          f"confidence {confidence:.2f}, multiplier {position_multiplier:.2f}")

        position_value = base_position_value * position_multiplier
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
                self._exit_reason = "FINAL BAR"
                self.order = self.sell(size=self.position.size)
            elif self.position.size < 0:
                self._exit_reason = "FINAL BAR"
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
                reason_str = f" [{self._exit_reason}]" if self._exit_reason else ""
                print(f"SELL Executed: {self.data.datetime.date(0)}, "
                      f"size={order.executed.size}, "
                      f"price=${order.executed.price:.2f}{reason_str}")
                self._exit_reason = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if self.p.verbose:
                print(f"ORDER FAILED: {self.data.datetime.date(0)}, "
                      f"status={order.getstatusname()}")

        self.order = None


# Alias for easier import
Strategy = LorentzianClassificationStrategy
