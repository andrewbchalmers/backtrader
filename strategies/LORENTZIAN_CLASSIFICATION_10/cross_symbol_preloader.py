"""
Cross-Symbol Training Data Preloader

Downloads OHLCV data from diverse ETFs (SPY, QQQ, IWM, TLT, GLD, XLE, EFA),
computes features + labels in pure numpy (no backtrader dependency), classifies
each bar into one of 4 market regimes, balances regime representation, and
seeds the strategy's deques so the KNN has diverse training data from bar 1.

Usage:
    Called automatically by LorentzianClassificationStrategy._init_state()
    when use_cross_symbol_training=True.
"""

import math
import os
import pickle
import time
from pathlib import Path

import numpy as np

# Module-level cache: prevents re-download within the SAME process
_PRECOMPUTED_CACHE = {}
_SECTOR_CACHE = {}    # symbol -> sector string
_DOWNLOAD_CACHE = {}  # symbol -> DataFrame (avoids re-downloading same peer across runs)

# Disk cache: shared across ALL processes (solves multiprocessing rate-limit problem).
# Each worker subprocess can read pre-downloaded ETF data written by the main process
# or an earlier worker, so yfinance is only hit once per symbol per day.
_DISK_CACHE_DIR = Path.home() / '.cache' / 'lorentzian_cross_symbol'
_DISK_CACHE_TTL_SECONDS = 86400  # 24 hours

_RETRY_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 5        # initial sleep for non-rate-limit errors; doubles each attempt
_RATE_LIMIT_COOLDOWN_SECONDS = 60  # full pause when yfinance returns 429
_PREFETCH_INTER_REQUEST_DELAY = 1.0  # seconds between yfinance calls during prefetch

# Sector lookups change rarely — cache for 7 days on disk so all worker subprocesses share them.
_SECTOR_DISK_CACHE_PATH = _DISK_CACHE_DIR / 'sectors.json'
_SECTOR_DISK_CACHE_TTL_SECONDS = 7 * 86400  # 7 days


def _load_sector_disk_cache() -> dict:
    """Load the sector disk cache. Returns {} if missing or stale."""
    if not _SECTOR_DISK_CACHE_PATH.exists():
        return {}
    if time.time() - _SECTOR_DISK_CACHE_PATH.stat().st_mtime > _SECTOR_DISK_CACHE_TTL_SECONDS:
        return {}
    try:
        import json
        with open(_SECTOR_DISK_CACHE_PATH, 'r') as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_sector_disk_cache(cache: dict) -> None:
    """Persist the sector cache to disk atomically."""
    try:
        import json
        _DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _SECTOR_DISK_CACHE_PATH.with_suffix('.tmp')
        with open(tmp, 'w') as fh:
            json.dump(cache, fh)
        tmp.replace(_SECTOR_DISK_CACHE_PATH)
    except Exception:
        pass  # non-critical


def _disk_cache_path(etf: str, start_date_str: str, end_date_str: str) -> Path:
    """Return the pickle path for this ETF + date range."""
    _DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{etf}_{start_date_str}_{end_date_str}.pkl"
    return _DISK_CACHE_DIR / filename


def _load_from_disk(etf: str, start_date_str: str, end_date_str: str):
    """Load cached DataFrame from disk. Returns None if missing or stale (>24 h)."""
    path = _disk_cache_path(etf, start_date_str, end_date_str)
    if not path.exists():
        return None
    if time.time() - path.stat().st_mtime > _DISK_CACHE_TTL_SECONDS:
        return None
    try:
        with open(path, 'rb') as fh:
            return pickle.load(fh)
    except Exception:
        return None


def _save_to_disk(etf: str, start_date_str: str, end_date_str: str, df) -> None:
    """Persist DataFrame to disk so other worker processes can reuse it."""
    path = _disk_cache_path(etf, start_date_str, end_date_str)
    try:
        tmp = path.with_suffix('.tmp')
        with open(tmp, 'wb') as fh:
            pickle.dump(df, fh)
        tmp.replace(path)   # atomic rename avoids corrupt reads by concurrent workers
    except Exception:
        pass  # non-critical - workers will just re-download if this fails


# =============================================================================
# Numpy Feature Implementations
# =============================================================================

def compute_atr_numpy(high, low, close, period=14):
    """
    Wilder-smoothed ATR matching backtrader's implementation.
    Returns 1D array with NaN for warmup bars.
    """
    n = len(close)
    atr = np.full(n, np.nan)

    # True Range
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))

    # Wilder smoothing: first value is SMA, then EMA with alpha=1/period
    if n < period:
        return atr

    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


def compute_feature_numpy(df, ftype, param_a, param_b):
    """
    Pure numpy reimplementation of each feature type.
    Returns 1D array with NaN for warmup bars.

    Supported types: RSM, VA, MTD, ZS, ER, RSI, ADX, ATRR, PP, VCR, VPD, CS, MACC, OBVT, STRK, CHOP
    """
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    volume = df['volume'].values.astype(float)
    n = len(close)
    out = np.full(n, np.nan)

    if ftype == 'RSM':
        # Relative Strength Momentum - percentile rank of current return
        momentum_period = param_a
        lookback = param_b
        warmup = lookback + momentum_period
        for i in range(warmup, n):
            if close[i - momentum_period] == 0:
                out[i] = 0.0
                continue
            current_ret = (close[i] / close[i - momentum_period]) - 1
            count_below = 0
            total = 0
            for j in range(1, lookback):
                past_idx = i - j
                older_idx = i - j - momentum_period
                if older_idx < 0:
                    break
                if close[older_idx] == 0:
                    continue
                hist_ret = (close[past_idx] / close[older_idx]) - 1
                total += 1
                if hist_ret < current_ret:
                    count_below += 1
            percentile = count_below / total if total > 0 else 0.5
            out[i] = percentile * 2 - 1

    elif ftype == 'VA':
        # Volume Anomaly - log-scaled volume ratio via tanh
        period = param_a
        for i in range(period, n):
            avg_vol = np.mean(volume[i - period:i])
            cur_vol = volume[i]
            if avg_vol > 0 and cur_vol > 0:
                ratio = cur_vol / avg_vol
                out[i] = math.tanh(math.log(ratio))
            else:
                out[i] = 0.0

    elif ftype == 'MTD':
        # Multi-Timeframe Divergence
        short_period = param_a
        long_period = param_b
        atr = compute_atr_numpy(high, low, close, 14)
        warmup = long_period + 1
        for i in range(warmup, n):
            c = close[i]
            if close[i - short_period] != 0:
                roc_short = (c / close[i - short_period]) - 1
            else:
                roc_short = 0
            if close[i - long_period] != 0:
                roc_long = (c / close[i - long_period]) - 1
            else:
                roc_long = 0
            atr_pct = (atr[i] / c) if c > 0 and not np.isnan(atr[i]) else 1e-8
            atr_pct = max(atr_pct, 1e-8)
            divergence = (roc_short - roc_long) / atr_pct
            out[i] = math.tanh(divergence)

    elif ftype == 'ZS':
        # Mean Reversion Z-Score
        period = param_a
        for i in range(period, n):
            window = close[i - period + 1:i + 1]
            mean = np.mean(window)
            std = np.std(window)
            if std > 0:
                z = (close[i] - mean) / std
                out[i] = max(-1.0, min(1.0, z / 3.0))
            else:
                out[i] = 0.0

    elif ftype == 'ER':
        # Efficiency Ratio
        period = param_a
        for i in range(period, n):
            direction = abs(close[i] - close[i - period])
            volatility = 0.0
            for j in range(period):
                volatility += abs(close[i - j] - close[i - j - 1])
            if volatility > 0:
                er = direction / volatility
            else:
                er = 0
            out[i] = er * 2 - 1

    elif ftype == 'RSI':
        # Normalized RSI
        period = param_a
        if n <= period:
            return out
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        for i in range(period, n - 1):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            out[i + 1] = (rsi - 50) / 50

    elif ftype == 'ADX':
        # Normalized ADX (simplified)
        period = param_a
        if n <= period * 2:
            return out
        # +DM / -DM
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        tr = np.zeros(n)
        for i in range(1, n):
            up = high[i] - high[i - 1]
            down = low[i - 1] - low[i]
            plus_dm[i] = up if (up > down and up > 0) else 0
            minus_dm[i] = down if (down > up and down > 0) else 0
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i] - close[i - 1]))
        # Wilder smoothing
        atr_s = np.full(n, np.nan)
        plus_di_s = np.full(n, np.nan)
        minus_di_s = np.full(n, np.nan)
        atr_s[period] = np.sum(tr[1:period + 1])
        plus_di_s[period] = np.sum(plus_dm[1:period + 1])
        minus_di_s[period] = np.sum(minus_dm[1:period + 1])
        for i in range(period + 1, n):
            atr_s[i] = atr_s[i - 1] - atr_s[i - 1] / period + tr[i]
            plus_di_s[i] = plus_di_s[i - 1] - plus_di_s[i - 1] / period + plus_dm[i]
            minus_di_s[i] = minus_di_s[i - 1] - minus_di_s[i - 1] / period + minus_dm[i]
        # DX and ADX
        dx = np.full(n, np.nan)
        for i in range(period, n):
            if np.isnan(atr_s[i]) or atr_s[i] == 0:
                continue
            pdi = 100 * plus_di_s[i] / atr_s[i]
            mdi = 100 * minus_di_s[i] / atr_s[i]
            dsum = pdi + mdi
            if dsum > 0:
                dx[i] = 100 * abs(pdi - mdi) / dsum
        # ADX = smoothed DX
        adx_start = period * 2
        if adx_start < n:
            valid_dx = dx[period:adx_start]
            valid_dx = valid_dx[~np.isnan(valid_dx)]
            if len(valid_dx) > 0:
                adx_val = np.mean(valid_dx)
                out[adx_start] = adx_val / 50 - 1
                for i in range(adx_start + 1, n):
                    if not np.isnan(dx[i]):
                        adx_val = (adx_val * (period - 1) + dx[i]) / period
                        out[i] = adx_val / 50 - 1

    elif ftype == 'ATRR':
        # ATR Ratio
        period = param_a
        atr = compute_atr_numpy(high, low, close, period)
        for i in range(period, n):
            if close[i] > 0 and not np.isnan(atr[i]):
                ratio_pct = (atr[i] / close[i]) * 100
                out[i] = (ratio_pct - 2.5) / 2.5
            else:
                out[i] = 0.0

    elif ftype == 'PP':
        # Price Position
        period = param_a
        for i in range(period - 1, n):
            h = np.max(high[max(0, i - period + 1):i + 1])
            l = np.min(low[max(0, i - period + 1):i + 1])
            rng = h - l
            if rng > 0:
                pos = (close[i] - l) / rng
                out[i] = pos * 2 - 1
            else:
                out[i] = 0.0

    elif ftype == 'VCR':
        # Volatility Contraction Ratio
        bb_period = param_a
        lookback = param_b
        warmup = bb_period + lookback
        for i in range(warmup, n):
            window = close[i - bb_period + 1:i + 1]
            mid = np.mean(window)
            std = np.std(window)
            if mid > 0:
                current_width = (4 * std) / mid  # 2*devfactor*std / mid
            else:
                out[i] = 0.0
                continue
            count_below = 0
            total = 0
            for j in range(1, lookback):
                idx = i - j
                if idx < bb_period:
                    break
                hw = close[idx - bb_period + 1:idx + 1]
                hmid = np.mean(hw)
                hstd = np.std(hw)
                if hmid > 0:
                    hist_width = (4 * hstd) / hmid
                    total += 1
                    if hist_width < current_width:
                        count_below += 1
            percentile = count_below / total if total > 0 else 0.5
            out[i] = percentile * 2 - 1

    elif ftype == 'VPD':
        # Volume-Price Divergence - Pearson correlation of price/volume deltas
        period = param_a
        warmup = period + 1
        for i in range(warmup, n):
            sum_xy = 0.0
            sum_x2 = 0.0
            sum_y2 = 0.0
            for j in range(period):
                idx = i - j
                pd_ = close[idx] - close[idx - 1]
                vd_ = volume[idx] - volume[idx - 1]
                sum_xy += pd_ * vd_
                sum_x2 += pd_ * pd_
                sum_y2 += vd_ * vd_
            denom = np.sqrt(sum_x2) * np.sqrt(sum_y2)
            if denom > 0:
                out[i] = sum_xy / denom
            else:
                out[i] = 0.0

    elif ftype == 'CS':
        # Candle Structure - avg signed body/range ratio, tanh-scaled
        period = param_a
        sensitivity = param_b if param_b > 0 else 2.0
        open_ = df['open'].values.astype(float)
        for i in range(period - 1, n):
            total = 0.0
            for j in range(period):
                idx = i - j
                body = close[idx] - open_[idx]
                rng = high[idx] - low[idx]
                if rng > 0:
                    total += body / rng
            avg = total / period
            out[i] = np.tanh(avg * sensitivity)

    elif ftype == 'MACC':
        # Momentum Acceleration - 2nd derivative of ROC, ATR-normalized
        roc_period = param_a
        lag = param_b
        atr = compute_atr_numpy(high, low, close, 14)
        warmup = roc_period + lag + 14
        for i in range(warmup, n):
            if close[i - roc_period] != 0 and close[i - lag - roc_period] != 0:
                roc_now = (close[i] / close[i - roc_period]) - 1
                roc_prev = (close[i - lag] / close[i - lag - roc_period]) - 1
                atr_pct = atr[i] / close[i] if close[i] > 0 and not np.isnan(atr[i]) else 1e-8
                atr_pct = max(atr_pct, 1e-8)
                accel = (roc_now - roc_prev) / atr_pct
                out[i] = np.tanh(accel)
            else:
                out[i] = 0.0

    elif ftype == 'OBVT':
        # OBV Trend - Z-score of cumulative OBV vs its SMA
        period = param_a
        z_divisor = param_b if param_b > 0 else 3.0
        # Compute cumulative OBV
        obv = np.zeros(n)
        for i in range(1, n):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]
        # Z-score of OBV over rolling window
        for i in range(period, n):
            window = obv[i - period + 1:i + 1]
            mean_obv = np.mean(window)
            std_obv = np.std(window)
            if std_obv > 0:
                z = (obv[i] - mean_obv) / std_obv
                out[i] = max(-1.0, min(1.0, z / z_divisor))
            else:
                out[i] = 0.0

    elif ftype == 'STRK':
        # Streak Pattern - consecutive up/down streak length × magnitude
        max_streak = param_a
        atr_mult = param_b if param_b > 0 else 1.5
        atr = compute_atr_numpy(high, low, close, 14)
        warmup = max_streak + 14
        for i in range(warmup, n):
            streak_len = 0
            streak_sum = 0.0
            streak_dir = 0
            for j in range(max_streak):
                idx = i - j
                if close[idx] > close[idx - 1]:
                    d = 1
                elif close[idx] < close[idx - 1]:
                    d = -1
                else:
                    d = 0
                if j == 0:
                    streak_dir = d
                    if d == 0:
                        break
                if d == streak_dir and d != 0:
                    streak_len += 1
                    streak_sum += abs(close[idx] / close[idx - 1] - 1)
                else:
                    break
            if streak_len == 0 or streak_dir == 0:
                out[i] = 0.0
                continue
            norm_len = streak_len / max_streak
            avg_move = streak_sum / streak_len
            atr_pct = atr[i] / close[i] if close[i] > 0 and not np.isnan(atr[i]) else 1e-8
            atr_pct = max(atr_pct, 1e-8)
            norm_mag = min(1.0, avg_move / (atr_pct * atr_mult))
            out[i] = streak_dir * norm_len * (0.5 + 0.5 * norm_mag)

    elif ftype == 'VCOMP':
        # Volatility Compression - tanh((1 - recent_vol/lookback_vol) * 2)
        recent_n = param_a
        lookback_n = param_b
        warmup = lookback_n + 1
        for i in range(warmup, n):
            h = high[i - lookback_n + 1:i + 1]
            l = low[i - lookback_n + 1:i + 1]
            pc = close[i - lookback_n:i]
            tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
            lookback_vol = np.mean(tr)
            if lookback_vol > 1e-10:
                recent_vol = np.mean(tr[-recent_n:])
                out[i] = math.tanh((1.0 - recent_vol / lookback_vol) * 2.0)
            else:
                out[i] = 0.0

    elif ftype == 'MPER':
        # Momentum Persistence - trend strength modulated by short-term alignment
        sp = param_a
        mp = param_b
        atr = compute_atr_numpy(high, low, close, 14)
        warmup = mp + 14
        for i in range(warmup, n):
            if close[i - sp] == 0 or close[i - mp] == 0 or close[i] == 0 or np.isnan(atr[i]):
                out[i] = 0.0
                continue
            atr_pct = max(atr[i] / close[i], 1e-8)
            s = ((close[i] / close[i - sp]) - 1.0) / atr_pct
            m = ((close[i] / close[i - mp]) - 1.0) / atr_pct
            trend_strength = math.tanh(m)
            alignment = math.tanh(s) * (1.0 if m > 0 else -1.0) if abs(m) > 1e-8 else 0.0
            out[i] = max(-1.0, min(1.0, trend_strength * (0.5 + 0.5 * alignment)))

    elif ftype == 'VMC':
        # Volume-Momentum Coupling - vol regime * momentum direction
        recent_n = param_a
        baseline_n = param_b
        atr = compute_atr_numpy(high, low, close, 14)
        warmup = max(baseline_n, 14) + 1
        for i in range(warmup, n):
            if close[i] <= 0 or close[i - recent_n] == 0 or np.isnan(atr[i]):
                out[i] = 0.0
                continue
            baseline_avg = np.mean(volume[i - baseline_n + 1:i + 1])
            if baseline_avg <= 0:
                out[i] = 0.0
                continue
            recent_vol_avg = np.mean(volume[i - recent_n + 1:i + 1])
            vol_regime = math.log(recent_vol_avg / baseline_avg) if recent_vol_avg > 0 else -2.0
            vol_signal = math.tanh(vol_regime * 2.0)
            atr_pct = max(atr[i] / close[i], 1e-8)
            mom = ((close[i] / close[i - recent_n]) - 1.0) / atr_pct
            out[i] = max(-1.0, min(1.0, vol_signal * (0.5 + 0.5 * math.tanh(mom))))

    elif ftype == 'CHOP':
        # Choppiness Index - measures trending vs choppy market
        # 100 * LOG10(SUM(ATR, period) / (HH - LL)) / LOG10(period)
        # Normalized: trending=+1, choppy=-1
        period = param_a
        atr = compute_atr_numpy(high, low, close, period)
        warmup = period + 14
        for i in range(warmup, n):
            # Sum of ATR over period
            atr_sum = 0.0
            valid_atr = True
            for j in range(period):
                idx = i - j
                if np.isnan(atr[idx]):
                    valid_atr = False
                    break
                atr_sum += atr[idx]

            if not valid_atr or atr_sum <= 0:
                out[i] = 0.0
                continue

            # Highest high and lowest low over period
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            rng = hh - ll

            if rng > 0:
                chop_raw = 100.0 * math.log10(atr_sum / rng) / math.log10(period)
                chop_raw = max(0.0, min(100.0, chop_raw))
                # Invert: trending=+1, choppy=-1
                out[i] = -1.0 * (chop_raw / 50.0 - 1.0)
            else:
                out[i] = 0.0

    return out


def compute_labels_numpy(df, lookahead, dead_zone, use_magnitude, trend_following):
    """
    Forward-looking labels using numpy.
    Returns 1D array with NaN for bars where label can't be computed.
    """
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    n = len(close)
    labels = np.full(n, np.nan)

    atr = compute_atr_numpy(high, low, close, 14)

    for i in range(n - lookahead):
        if np.isnan(atr[i]) or atr[i] < 1e-8:
            continue
        future_price = close[i + lookahead]
        price_change = future_price - close[i]
        norm_return = price_change / atr[i]

        # Dead zone filter
        if abs(norm_return) < dead_zone:
            labels[i] = 0.0
            continue

        if use_magnitude:
            label = max(-3.0, min(3.0, norm_return))
        else:
            label = 1.0 if price_change > 0 else -1.0

        # Mean-reversion flip
        if not trend_following:
            label = -label

        labels[i] = label

    return labels


# =============================================================================
# Regime Classification
# =============================================================================

def classify_regime(df, sma_period=50):
    """
    Per-symbol adaptive percentile classification into 4 regimes:
        0 = TREND_UP
        1 = TREND_DOWN
        2 = CHOP
        3 = HIGH_VOL

    Uses per-symbol percentiles so each ETF contributes proportionally
    to every regime bucket regardless of absolute volatility/trend magnitude.
    """
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    n = len(close)
    regimes = np.full(n, np.nan)

    atr = compute_atr_numpy(high, low, close, 14)

    # ATR as percentage of close
    atr_pct = np.full(n, np.nan)
    for i in range(n):
        if close[i] > 0 and not np.isnan(atr[i]):
            atr_pct[i] = atr[i] / close[i]

    # SMA and its slope
    sma = np.full(n, np.nan)
    sma_slope = np.full(n, np.nan)
    for i in range(sma_period - 1, n):
        sma[i] = np.mean(close[i - sma_period + 1:i + 1])
    for i in range(sma_period, n):
        if not np.isnan(sma[i]) and not np.isnan(sma[i - 1]) and sma[i - 1] > 0:
            sma_slope[i] = (sma[i] - sma[i - 1]) / sma[i - 1]

    # Compute per-symbol percentiles (only on valid values)
    valid_atr = atr_pct[~np.isnan(atr_pct)]
    valid_slope = sma_slope[~np.isnan(sma_slope)]

    if len(valid_atr) < 10 or len(valid_slope) < 10:
        # Not enough data - assign all to CHOP
        regimes[~np.isnan(atr_pct)] = 2
        return regimes

    atr_75 = np.percentile(valid_atr, 75)
    slope_67 = np.percentile(valid_slope, 67)
    slope_33 = np.percentile(valid_slope, 33)

    for i in range(sma_period, n):
        if np.isnan(atr_pct[i]) or np.isnan(sma_slope[i]):
            continue
        if atr_pct[i] > atr_75:
            regimes[i] = 3  # HIGH_VOL
        elif sma_slope[i] > slope_67:
            regimes[i] = 0  # TREND_UP
        elif sma_slope[i] < slope_33:
            regimes[i] = 1  # TREND_DOWN
        else:
            regimes[i] = 2  # CHOP

    return regimes


# =============================================================================
# Regime Balancing
# =============================================================================

def balance_regimes(features, labels, regimes, max_total, seed=42):
    """
    Downsample each regime to max_total // 4 samples.
    Deterministic via fixed seed.

    Args:
        features: list of 1D arrays (one per feature), each length N
        labels: 1D array length N
        regimes: 1D array length N (values 0-3)
        max_total: target total samples
        seed: random seed for reproducibility

    Returns:
        (balanced_features, balanced_labels) - lists of arrays / array
    """
    rng = np.random.RandomState(seed)
    per_regime = max_total // 4

    selected_indices = []
    for regime_id in range(4):
        mask = regimes == regime_id
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        if len(indices) > per_regime:
            chosen = rng.choice(indices, size=per_regime, replace=False)
        else:
            chosen = indices
        selected_indices.append(chosen)

    if not selected_indices:
        return features, labels

    all_indices = np.concatenate(selected_indices)
    # Sort to preserve chronological order
    all_indices.sort()

    balanced_features = [f[all_indices] for f in features]
    balanced_labels = labels[all_indices]

    return balanced_features, balanced_labels


# =============================================================================
# Sector-Based Peer Selection
# =============================================================================

def get_ticker_sector(symbol):
    """
    Look up the sector for a given ticker symbol using yfinance.
    Results are cached in _SECTOR_CACHE (in-process) and on disk (cross-process).
    Returns sector string or None on failure.
    """
    global _SECTOR_CACHE

    # Seed from disk cache on first call in this process
    if not _SECTOR_CACHE:
        _SECTOR_CACHE.update(_load_sector_disk_cache())

    if symbol in _SECTOR_CACHE:
        return _SECTOR_CACHE[symbol]

    import yfinance as yf
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            info = yf.Ticker(symbol).info
            sector = info.get('sector', None)
            _SECTOR_CACHE[symbol] = sector
            _save_sector_disk_cache(_SECTOR_CACHE)
            return sector
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = 'too many requests' in err_str or 'rate limit' in err_str
            if attempt < _RETRY_ATTEMPTS - 1:
                delay = _RATE_LIMIT_COOLDOWN_SECONDS if is_rate_limit else _RETRY_DELAY_SECONDS * (2 ** attempt)
                print(f"CROSS-SYMBOL: sector lookup for {symbol} failed (attempt {attempt + 1}/{_RETRY_ATTEMPTS}): {e} — retrying in {delay}s")
                time.sleep(delay)
            else:
                print(f"CROSS-SYMBOL: sector lookup for {symbol} failed after {_RETRY_ATTEMPTS} attempts (skipping)")
    _SECTOR_CACHE[symbol] = None
    return None


def prefetch_sectors(symbols):
    """
    Pre-fetch sector info for all symbols upfront.
    Populates _SECTOR_CACHE (and disk cache) so worker subprocesses hit cache only.

    Throttles to _PREFETCH_INTER_REQUEST_DELAY seconds between live yfinance calls
    to avoid triggering rate limits during bulk prefetch.

    Returns:
        dict mapping sector -> list of symbols in that sector
    """
    # Seed from disk first so we only fetch what's genuinely missing
    if not _SECTOR_CACHE:
        _SECTOR_CACHE.update(_load_sector_disk_cache())

    sector_map = {}
    fetched = 0
    for i, sym in enumerate(symbols):
        needs_fetch = sym not in _SECTOR_CACHE
        if needs_fetch and fetched > 0:
            time.sleep(_PREFETCH_INTER_REQUEST_DELAY)
        sector = get_ticker_sector(sym)
        if needs_fetch:
            fetched += 1
        if sector:
            sector_map.setdefault(sector, []).append(sym)
        if (i + 1) % 25 == 0:
            print(f"  Sector lookup progress: {i + 1}/{len(symbols)} ({fetched} fetched, {i + 1 - fetched} cached)")
    return sector_map


def find_sector_peers(target_symbol, universe, max_peers=7):
    """
    Find same-sector peers for target_symbol from the universe list.

    Args:
        target_symbol: The stock to find peers for
        universe: List of candidate symbols
        max_peers: Maximum number of peers to return

    Returns:
        List of peer symbols (same sector, excluding target).
        Falls back to first max_peers from universe if no sector match.
    """
    target_sector = get_ticker_sector(target_symbol)

    peers = []
    if target_sector:
        for sym in universe:
            if sym == target_symbol:
                continue
            sym_sector = get_ticker_sector(sym)
            if sym_sector == target_sector:
                peers.append(sym)
                if len(peers) >= max_peers:
                    break

    # Fallback: if no sector match or lookup failed, use first max_peers from universe
    if not peers:
        peers = [s for s in universe if s != target_symbol][:max_peers]

    return peers


# =============================================================================
# Main Entry Point
# =============================================================================

def seed_strategy(strategy):
    """
    Main entry point called by LorentzianClassificationStrategy._init_state().

    1. Parse ETF list from strategy params
    2. Check module-level cache
    3. Download OHLCV from yfinance for each ETF
    4. Compute features + labels + regimes per ETF
    5. Pool all data, filter NaN rows
    6. If use_regime_balancing: balance regimes
    7. Inject into strategy.feature_arrays and strategy.label_array deques
    8. Return count of seeded bars
    """
    import yfinance as yf  # Deferred import - keeps yfinance optional

    etf_str = str(strategy.p.cross_symbol_etfs)
    etfs = [s.strip() for s in etf_str.split(',') if s.strip()]
    lookback_years = int(strategy.p.cross_symbol_lookback_years)
    use_balancing = bool(strategy.p.use_regime_balancing)
    feature_count = int(strategy.p.feature_count)

    # Auto-peer selection: resolve same-sector peers from universe
    auto_peers = getattr(strategy.p, 'cross_symbol_auto_peers', False)
    target_symbol = str(getattr(strategy.p, 'cross_symbol_target_symbol', ''))
    max_peers = int(getattr(strategy.p, 'cross_symbol_max_peers', 7))

    if auto_peers and target_symbol:
        etfs = find_sector_peers(target_symbol, etfs, max_peers)
        if hasattr(strategy.p, 'verbose') and strategy.p.verbose:
            sector = get_ticker_sector(target_symbol) or 'Unknown'
            print(f"CROSS-SYMBOL: Target={target_symbol}, Sector={sector}, Peers={etfs}")

    # Feature configs from strategy params (max 11 feature slots defined)
    feature_configs = []
    for fi in range(1, min(feature_count, 11) + 1):
        ftype = str(getattr(strategy.p, f'f{fi}_type'))
        pa = int(getattr(strategy.p, f'f{fi}_param_a'))
        pb = int(getattr(strategy.p, f'f{fi}_param_b'))
        feature_configs.append((ftype, pa, pb))
    feature_count = len(feature_configs)

    # Label settings
    lookahead = int(strategy.p.label_lookahead)
    dead_zone = float(strategy.p.label_dead_zone)
    use_magnitude = bool(strategy.p.use_magnitude_labels)
    trend_following = bool(strategy.p.trend_following_labels)

    # Cache key (uses resolved etfs list, which may differ per target symbol)
    cache_key = (
        tuple(etfs), lookback_years, tuple(feature_configs),
        lookahead, dead_zone, use_magnitude, trend_following,
        use_balancing, int(strategy.p.max_bars_back)
    )

    global _PRECOMPUTED_CACHE
    if cache_key in _PRECOMPUTED_CACHE:
        cached = _PRECOMPUTED_CACHE[cache_key]
        _inject_into_deques(strategy, cached['features'], cached['labels'], feature_count)
        return len(cached['labels'])

    # Download and compute
    from datetime import datetime, timedelta
    import pandas as pd

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_years * 365)

    all_features = [[] for _ in range(feature_count)]
    all_labels = []
    all_regimes = []

    global _DOWNLOAD_CACHE

    for etf in etfs:
        try:
            start_str = str(start_date.date())
            end_str   = str(end_date.date())
            cache_dl_key = (etf, start_str, end_str)

            if cache_dl_key in _DOWNLOAD_CACHE:
                # In-process memory cache (fastest path)
                df = _DOWNLOAD_CACHE[cache_dl_key]
            else:
                # Disk cache: shared across all worker subprocesses
                df = _load_from_disk(etf, start_str, end_str)
                if df is None:
                    # Cache miss - actually download from yfinance (with retry)
                    df = None
                    for attempt in range(_RETRY_ATTEMPTS):
                        try:
                            df = yf.Ticker(etf).history(start=start_date, end=end_date, interval='1d')
                            break
                        except Exception as e:
                            if attempt < _RETRY_ATTEMPTS - 1:
                                delay = _RETRY_DELAY_SECONDS * (2 ** attempt)
                                print(f"CROSS-SYMBOL: {etf} download failed (attempt {attempt + 1}/{_RETRY_ATTEMPTS}): {e} — retrying in {delay}s")
                                time.sleep(delay)
                            else:
                                print(f"CROSS-SYMBOL: {etf} download failed after {_RETRY_ATTEMPTS} attempts: {e} (skipping)")
                    if df is None or df.empty or len(df) < 100:
                        print(f"CROSS-SYMBOL: {etf} returned {len(df) if df is not None else 0} bars (skipping)")
                        continue

                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    df.columns = ['open', 'high', 'low', 'close', 'volume']

                    # Persist to disk so other worker processes skip the download
                    _save_to_disk(etf, start_str, end_str, df)

                _DOWNLOAD_CACHE[cache_dl_key] = df

            if len(df) < 100:
                continue

            # Compute features
            feat_arrays = []
            for ftype, pa, pb in feature_configs:
                feat = compute_feature_numpy(df, ftype, pa, pb)
                feat_arrays.append(feat)

            # Compute labels
            labels = compute_labels_numpy(df, lookahead, dead_zone,
                                          use_magnitude, trend_following)

            # Compute regimes
            regimes = classify_regime(df)

            # Find valid mask (no NaN in any feature or label or regime)
            valid = ~np.isnan(labels)
            for fa in feat_arrays:
                valid &= ~np.isnan(fa)
            valid &= ~np.isnan(regimes)

            valid_indices = np.where(valid)[0]
            if len(valid_indices) == 0:
                continue

            for fi in range(feature_count):
                all_features[fi].append(feat_arrays[fi][valid_indices])
            all_labels.append(labels[valid_indices])
            all_regimes.append(regimes[valid_indices])

        except Exception as e:
            print(f"CROSS-SYMBOL: Failed to process {etf}: {e}")
            continue

    if not all_labels:
        return 0

    # Pool across all ETFs
    pooled_features = [np.concatenate(all_features[fi]) for fi in range(feature_count)]
    pooled_labels = np.concatenate(all_labels)
    pooled_regimes = np.concatenate(all_regimes)

    if strategy.p.verbose:
        # Report regime distribution before balancing
        for regime_id, regime_name in enumerate(['TREND_UP', 'TREND_DOWN', 'CHOP', 'HIGH_VOL']):
            count = np.sum(pooled_regimes == regime_id)
            pct = count / len(pooled_regimes) * 100 if len(pooled_regimes) > 0 else 0
            print(f"CROSS-SYMBOL: Pre-balance regime {regime_name}: {count} ({pct:.1f}%)")

    # Regime balancing
    if use_balancing:
        max_total = int(strategy.p.max_bars_back * 0.6)
        pooled_features, pooled_labels = balance_regimes(
            pooled_features, pooled_labels, pooled_regimes, max_total, seed=42
        )

    if strategy.p.verbose:
        print(f"CROSS-SYMBOL: Pooled {len(pooled_labels)} bars from {len(etfs)} ETFs"
              f" (balancing={'ON' if use_balancing else 'OFF'})")

    # Cache for future optimizer runs
    _PRECOMPUTED_CACHE[cache_key] = {
        'features': pooled_features,
        'labels': pooled_labels,
    }

    # Inject into strategy deques
    _inject_into_deques(strategy, pooled_features, pooled_labels, feature_count)

    return len(pooled_labels)


def _inject_into_deques(strategy, features, labels, feature_count):
    """Inject precomputed features and labels into strategy deques."""
    n = len(labels)
    for i in range(n):
        for fi in range(feature_count):
            strategy.feature_arrays[fi].append(float(features[fi][i]))
        strategy.label_array.append(float(labels[i]))
