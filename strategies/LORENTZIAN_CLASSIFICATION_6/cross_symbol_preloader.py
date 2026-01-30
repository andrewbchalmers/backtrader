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
import numpy as np

# Module-level cache: prevents re-download during optimizer runs
_PRECOMPUTED_CACHE = {}
_SECTOR_CACHE = {}    # symbol -> sector string
_DOWNLOAD_CACHE = {}  # symbol -> DataFrame (avoids re-downloading same peer across runs)


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

    Supported types: RSM, VA, MTD, ZS, ER, RSI, ADX, ATRR, PP, VCR
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
    Results are cached in _SECTOR_CACHE.
    Returns sector string or None on failure.
    """
    global _SECTOR_CACHE
    if symbol in _SECTOR_CACHE:
        return _SECTOR_CACHE[symbol]

    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info
        sector = info.get('sector', None)
        _SECTOR_CACHE[symbol] = sector
        return sector
    except Exception:
        _SECTOR_CACHE[symbol] = None
        return None


def prefetch_sectors(symbols):
    """
    Pre-fetch sector info for all symbols upfront.
    Populates _SECTOR_CACHE so that find_sector_peers() hits cache only.

    Returns:
        dict mapping sector -> list of symbols in that sector
    """
    sector_map = {}
    for i, sym in enumerate(symbols):
        sector = get_ticker_sector(sym)
        if sector:
            sector_map.setdefault(sector, []).append(sym)
        if (i + 1) % 25 == 0:
            print(f"  Sector lookup progress: {i + 1}/{len(symbols)}")
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

    # Feature configs from strategy params
    feature_configs = []
    for fi in range(1, feature_count + 1):
        ftype = str(getattr(strategy.p, f'f{fi}_type'))
        pa = int(getattr(strategy.p, f'f{fi}_param_a'))
        pb = int(getattr(strategy.p, f'f{fi}_param_b'))
        feature_configs.append((ftype, pa, pb))

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
            # Use download cache to avoid redundant downloads across runs
            cache_dl_key = (etf, str(start_date.date()), str(end_date.date()))
            if cache_dl_key in _DOWNLOAD_CACHE:
                df = _DOWNLOAD_CACHE[cache_dl_key]
            else:
                df = yf.download(etf, start=start_date, end=end_date,
                                 interval='1d', progress=False)
                if df.empty or len(df) < 100:
                    continue

                df.index = df.index.tz_localize(None)
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df.columns = ['open', 'high', 'low', 'close', 'volume']
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
            if hasattr(strategy.p, 'verbose') and strategy.p.verbose:
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
