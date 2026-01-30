"""
Walk-Forward Stock Classification for Lorentzian Classification Strategy

Identifies which stocks are best suited for the strategy via walk-forward analysis:
- Holds strategy parameters fixed
- Runs the strategy on every stock individually
- Ranks stocks by composite performance score
- Groups into tiers (BEST / GOOD / FAIR / POOR)
- Validates out-of-sample: do training-period winners keep winning?
- Collects stock characteristics (sector, market cap, beta, volatility)

Usage:
    python classify_stocks.py
"""

from decimal import Decimal
import os
import matplotlib
matplotlib.use('Agg')
import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timedelta
from collections import defaultdict
import sys

# Reuse functions from optimize_strategy.py
from optimize_strategy import (
    generate_walkforward_periods,
    print_walkforward_schedule,
    load_all_periods_data,
    backtest_single_config,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Input/Output
    'csv_file': '../sp500_2025.csv',
    'results_file': 'stock_classification_results.csv',
    'detail_file': 'stock_classification_detail.csv',

    # Walk-Forward Settings (same as optimize_strategy.py)
    'train_period_months': 18,
    'test_period_months': 6,
    'step_months': 3,
    'total_periods': 3,

    # Date range control
    'end_date': None,
    'lookback_years': 5,

    # COVID handling
    'exclude_covid': True,
    'covid_start': '2020-02-01',
    'covid_end': '2021-06-30',

    # Backtest settings
    'initial_cash': 10000,
    'commission': 0.0001,

    # Classification settings
    'num_tiers': 4,
    'tier_labels': {1: 'BEST', 2: 'GOOD', 3: 'FAIR', 4: 'POOR'},
    'ranking_metric': 'composite',

    # Multiprocessing
    'max_workers': None,
    'print_progress_every': 50,

    # Fixed strategy parameters (from optimization results)
    'strategy_params': {
        # ML Settings
        'neighbors_count': 9,
        'max_bars_back': 500,
        'feature_count': 5,
        'trend_following_labels': False,
        'allow_reentry': True,
        'min_prediction_strength': 15,          # Normalized scale: 0-100

        # Label Settings
        'label_lookahead': 4,
        'label_dead_zone': 0.5,
        'use_magnitude_labels': True,

        # Feature 1 (RSI - Momentum)
        'f1_type': 'RSI',
        'f1_param_a': 14,
        'f1_param_b': 1,

        # Feature 2 (ADX - Trend Strength)
        'f2_type': 'ADX',
        'f2_param_a': 14,
        'f2_param_b': 1,

        # Feature 3 (ATR Ratio - Volatility)
        'f3_type': 'ATRR',
        'f3_param_a': 14,
        'f3_param_b': 1,

        # Feature 4 (Price Position)
        'f4_type': 'PP',
        'f4_param_a': 14,
        'f4_param_b': 1,

        # Feature 5 (Efficiency Ratio)
        'f5_type': 'ER',
        'f5_param_a': 14,
        'f5_param_b': 1,

        # Filters
        'use_volatility_filter': True,
        'use_regime_filter': True,
        'regime_threshold': Decimal('-0.1'),
        'use_adx_filter': False,
        'adx_threshold': 20,
        'use_ema_filter': False,
        'ema_period': 200,
        'use_sma_filter': False,
        'sma_period': 200,

        # Kernel Settings
        'use_kernel_filter': False,
        'use_kernel_smoothing': False,
        'kernel_lookback': 8,
        'kernel_rel_weight': 8.0,
        'kernel_start_bar': 25,
        'kernel_lag': 2,

        # Exit Settings
        'use_dynamic_exits': True,
        'bars_to_hold': 100000,

        # RSI Exit Settings
        'use_rsi_exit': False,
        'rsi_exit_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,

        # Kernel Exit Settings
        'use_kernel_exit': True,

        # Risk Management
        'position_size_pct': Decimal('0.95'),
        'stop_loss_pct': Decimal('0.05'),
        'use_stop_loss': True,
        'long_only': True,

        # Other
        'verbose': False,
        'test_start_idx': 0,
    },
}


# ============================================================================
# Stock Characteristics
# ============================================================================

def get_stock_characteristics(symbols):
    """
    Fetch stock characteristics for each symbol using yfinance.

    Collects screener-ready attributes: sector, industry, market cap, beta,
    average volume, current price, 52-week range, trailing PE, dividend yield.

    Called once before the walk-forward loop since characteristics don't change
    between periods.
    """
    print(f"\n📊 Fetching stock characteristics for {len(symbols)} symbols...")
    characteristics = {}

    default_chars = {
        'sector': 'Unknown',
        'industry': 'Unknown',
        'market_cap': 0,
        'beta': 0,
        'avg_volume': 0,
        'price': 0,
        'fifty_two_week_high': 0,
        'fifty_two_week_low': 0,
        'trailing_pe': 0,
        'dividend_yield': 0,
    }

    for i, symbol in enumerate(symbols):
        try:
            info = yf.Ticker(symbol).info
            characteristics[symbol] = {
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0) or 0,
                'beta': info.get('beta', 0) or 0,
                'avg_volume': info.get('averageVolume', 0) or 0,
                'price': info.get('currentPrice') or info.get('previousClose', 0) or 0,
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0) or 0,
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0) or 0,
                'trailing_pe': info.get('trailingPE', 0) or 0,
                'dividend_yield': (info.get('dividendYield', 0) or 0) * 100,  # as pct
            }
        except Exception:
            characteristics[symbol] = dict(default_chars)

        if (i + 1) % 25 == 0:
            print(f"   Progress: {i + 1}/{len(symbols)} ({(i + 1) / len(symbols) * 100:.0f}%)")

    known = sum(1 for c in characteristics.values() if c['sector'] != 'Unknown')
    print(f"   ✓ Fetched characteristics for {known}/{len(symbols)} symbols")

    return characteristics


def compute_historical_volatility(data_cache, valid_symbols, periods):
    """
    Compute annualized historical volatility (std dev of daily returns * sqrt(252))
    for each symbol using the downloaded price data.

    Returns dict: symbol -> annualized_volatility_pct
    """
    volatilities = {}

    for symbol in valid_symbols:
        all_returns = []
        for period_idx in data_cache.get(symbol, {}):
            period_data = data_cache[symbol][period_idx]
            for phase in ('train', 'test'):
                df = period_data.get(phase)
                if df is not None and len(df) > 10:
                    daily_returns = df['close'].pct_change().dropna()
                    all_returns.extend(daily_returns.tolist())

        if all_returns:
            volatilities[symbol] = float(np.std(all_returns) * np.sqrt(252) * 100)
        else:
            volatilities[symbol] = 0.0

    return volatilities


# ============================================================================
# Backtesting All Stocks
# ============================================================================

def backtest_all_stocks(data_cache, valid_symbols, strategy_params, period_idx, phase, config):
    """
    Run backtest for all valid symbols with fixed strategy parameters.
    Returns dict: symbol -> backtest result dict
    """
    work_items = []

    for symbol in valid_symbols:
        if period_idx not in data_cache.get(symbol, {}):
            continue

        period_data = data_cache[symbol][period_idx]
        df = period_data[phase]
        test_start_idx = period_data.get(f'{phase}_test_start_idx', 0)

        if df is None or len(df) < 50:
            continue

        run_params = dict(strategy_params)
        run_params['test_start_idx'] = test_start_idx

        work_items.append((symbol, run_params, df, config['initial_cash'], config['commission']))

    total_backtests = len(work_items)
    max_workers = config.get('max_workers', None)

    print(f"   Running {total_backtests} backtests on {phase} data (workers: {max_workers or os.cpu_count()})...")

    results = {}
    completed = 0
    progress_interval = config['print_progress_every']

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {}
        for symbol, run_params, df, cash, commission in work_items:
            future = executor.submit(backtest_single_config, symbol, run_params, df, cash, commission)
            future_to_symbol[future] = symbol

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
            except Exception:
                result = None

            if result is not None:
                results[symbol] = result

            completed += 1
            if completed % progress_interval == 0:
                print(f"      Progress: {completed}/{total_backtests} ({completed / total_backtests * 100:.0f}%) | Successful: {len(results)}")

    print(f"   ✓ Completed {completed} backtests, {len(results)} successful")
    return results


# ============================================================================
# Stock Classification
# ============================================================================

def classify_stocks(per_stock_results, num_tiers, ranking_metric='composite'):
    """
    Calculate composite score per stock, rank, and assign tiers.

    Composite score = (return * max(sharpe, 0) * max(profit_factor, 0)) / max(max_drawdown, 0.01)

    Returns dict: symbol -> {tier, rank, score, result}
    """
    scored = []

    for symbol, result in per_stock_results.items():
        ret = result['return_pct']
        sharpe = max(result.get('sharpe', 0), 0)
        pf = max(result.get('profit_factor', 0), 0)
        dd = max(result.get('max_drawdown', 0), 0.01)

        if ranking_metric == 'composite':
            score = (ret * sharpe * pf) / dd
        elif ranking_metric == 'return':
            score = ret
        elif ranking_metric == 'sharpe':
            score = result.get('sharpe', 0)
        else:
            score = (ret * sharpe * pf) / dd

        scored.append((symbol, score, result))

    # Sort descending by score
    scored.sort(key=lambda x: x[1], reverse=True)

    n = len(scored)
    tier_size = max(1, n // num_tiers)
    classifications = {}

    for rank, (symbol, score, result) in enumerate(scored, 1):
        tier = min((rank - 1) // tier_size + 1, num_tiers)
        classifications[symbol] = {
            'tier': tier,
            'rank': rank,
            'score': score,
            'result': result,
        }

    return classifications


# ============================================================================
# Main Walk-Forward Classification
# ============================================================================

def run_stock_classification(config):
    """
    Main walk-forward stock classification loop.

    For each period:
    1. Train: backtest all stocks
    2. Classify: rank/tier stocks by training performance
    3. Test: backtest all stocks on out-of-sample data
    4. Compare: compute avg test-period metrics per tier
    5. Record: per-stock detail rows
    6. Track: tier history for consistency analysis
    """
    print("\n" + "=" * 70)
    print("WALK-FORWARD STOCK CLASSIFICATION - LORENTZIAN CLASSIFICATION")
    print("=" * 70)

    # Generate walk-forward periods
    periods = generate_walkforward_periods(config)

    if not periods:
        print("\n❌ No valid periods generated. Check configuration.")
        sys.exit(1)

    print_walkforward_schedule(periods)

    # Load symbols
    try:
        with open(config['csv_file'], 'r') as f:
            symbols = [row[0].strip().upper() for row in csv.reader(f) if row and row[0].strip()]
    except FileNotFoundError:
        print(f"\n❌ File '{config['csv_file']}' not found")
        sys.exit(1)

    print(f"\n📋 Classification Configuration")
    print(f"   Symbols: {len(symbols)} stocks")
    print(f"   Walk-forward periods: {len(periods)}")
    print(f"   Train period: {config['train_period_months']} months")
    print(f"   Test period: {config['test_period_months']} months")
    print(f"   Number of tiers: {config['num_tiers']}")
    print(f"   Ranking metric: {config['ranking_metric']}")
    print(f"   COVID exclusion: {'Yes' if config['exclude_covid'] else 'No'}")

    # Fetch stock characteristics (one-time)
    characteristics = get_stock_characteristics(symbols)

    # Download data for all periods
    lookback_bars = config['strategy_params'].get('max_bars_back', 500) + 100
    data_cache, valid_symbols = load_all_periods_data(symbols, periods, lookback_bars)

    if not valid_symbols:
        print("\n❌ No valid data. Exiting.")
        sys.exit(1)

    # Compute historical volatility from downloaded price data
    print(f"\n📈 Computing historical volatility...")
    volatilities = compute_historical_volatility(data_cache, valid_symbols, periods)
    print(f"   ✓ Computed volatility for {sum(1 for v in volatilities.values() if v > 0)} symbols")

    print(f"\n🚀 Starting Walk-Forward Stock Classification...\n")

    tier_labels = config['tier_labels']
    num_tiers = config['num_tiers']
    strategy_params = config['strategy_params']

    all_tier_rows = []     # tier-level aggregates per period
    all_detail_rows = []   # per-stock per-period detail
    tier_history = defaultdict(list)  # symbol -> [tier per period]

    for period_idx, period_info in enumerate(periods):
        train_start, train_end, test_start, test_end = period_info

        print(f"\n{'=' * 70}")
        print(f"PERIOD {period_idx + 1}/{len(periods)}")
        print(f"{'=' * 70}")
        print(f"Train: {train_start} to {train_end}")
        print(f"Test:  {test_start} to {test_end}")

        # --- Train phase: backtest all stocks ---
        print(f"\n1️⃣  TRAINING PHASE (In-Sample)")
        train_results = backtest_all_stocks(
            data_cache, valid_symbols, strategy_params, period_idx, 'train', config
        )

        if not train_results:
            print(f"   ❌ No valid training results for period {period_idx + 1}!")
            continue

        # --- Classify stocks based on training performance ---
        print(f"\n2️⃣  CLASSIFICATION PHASE")
        classifications = classify_stocks(train_results, num_tiers, config['ranking_metric'])

        # Print tier summary
        for tier_num in range(1, num_tiers + 1):
            tier_symbols = [s for s, c in classifications.items() if c['tier'] == tier_num]
            label = tier_labels.get(tier_num, f'TIER {tier_num}')
            print(f"   Tier {tier_num} ({label}): {len(tier_symbols)} stocks")

        # --- Test phase: backtest all stocks on out-of-sample data ---
        print(f"\n3️⃣  VALIDATION PHASE (Out-of-Sample)")
        test_results = backtest_all_stocks(
            data_cache, valid_symbols, strategy_params, period_idx, 'test', config
        )

        # --- Compare: compute avg test-period metrics per tier ---
        print(f"\n4️⃣  TIER COMPARISON")
        for tier_num in range(1, num_tiers + 1):
            tier_symbols = [s for s, c in classifications.items() if c['tier'] == tier_num]
            label = tier_labels.get(tier_num, f'TIER {tier_num}')

            # Training metrics for this tier
            tier_train = [train_results[s] for s in tier_symbols if s in train_results]
            # Test metrics for this tier
            tier_test = [test_results[s] for s in tier_symbols if s in test_results]

            if tier_train:
                train_avg_ret = np.mean([r['return_pct'] for r in tier_train])
                train_avg_sharpe = np.mean([r['sharpe'] for r in tier_train])
                train_avg_wr = np.mean([r['win_rate'] for r in tier_train])
            else:
                train_avg_ret = train_avg_sharpe = train_avg_wr = 0

            if tier_test:
                test_avg_ret = np.mean([r['return_pct'] for r in tier_test])
                test_avg_sharpe = np.mean([r['sharpe'] for r in tier_test])
                test_avg_wr = np.mean([r['win_rate'] for r in tier_test])
            else:
                test_avg_ret = test_avg_sharpe = test_avg_wr = 0

            ret_degradation = train_avg_ret - test_avg_ret

            print(f"\n   Tier {tier_num} ({label}) - {len(tier_symbols)} stocks:")
            print(f"     Train: Return={train_avg_ret:+.2f}%  Sharpe={train_avg_sharpe:.3f}  WinRate={train_avg_wr:.1f}%")
            print(f"     Test:  Return={test_avg_ret:+.2f}%  Sharpe={test_avg_sharpe:.3f}  WinRate={test_avg_wr:.1f}%")
            print(f"     Degradation: {ret_degradation:+.2f}%")

            # Record tier-level aggregate row
            all_tier_rows.append({
                'period': period_idx + 1,
                'tier': tier_num,
                'tier_label': label,
                'n_stocks': len(tier_symbols),
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_avg_return': train_avg_ret,
                'train_avg_sharpe': train_avg_sharpe,
                'train_avg_win_rate': train_avg_wr,
                'test_avg_return': test_avg_ret,
                'test_avg_sharpe': test_avg_sharpe,
                'test_avg_win_rate': test_avg_wr,
                'return_degradation': ret_degradation,
                'stocks': ','.join(sorted(tier_symbols)),
            })

        # --- Record per-stock detail rows ---
        for symbol, cls in classifications.items():
            tier_history[symbol].append(cls['tier'])

            train_r = train_results.get(symbol, {})
            test_r = test_results.get(symbol, {})
            chars = characteristics.get(symbol, {})

            all_detail_rows.append({
                'period': period_idx + 1,
                'symbol': symbol,
                'tier': cls['tier'],
                'rank': cls['rank'],
                'composite_score': cls['score'],
                'train_return': train_r.get('return_pct', 0),
                'train_sharpe': train_r.get('sharpe', 0),
                'train_trades': train_r.get('total_trades', 0),
                'train_win_rate': train_r.get('win_rate', 0),
                'train_profit_factor': train_r.get('profit_factor', 0),
                'train_max_drawdown': train_r.get('max_drawdown', 0),
                'test_return': test_r.get('return_pct', 0),
                'test_sharpe': test_r.get('sharpe', 0),
                'test_trades': test_r.get('total_trades', 0),
                'test_win_rate': test_r.get('win_rate', 0),
                'test_profit_factor': test_r.get('profit_factor', 0),
                'test_max_drawdown': test_r.get('max_drawdown', 0),
                'sector': chars.get('sector', 'Unknown'),
                'industry': chars.get('industry', 'Unknown'),
                'market_cap': chars.get('market_cap', 0),
                'beta': chars.get('beta', 0),
                'avg_volume': chars.get('avg_volume', 0),
                'price': chars.get('price', 0),
                'fifty_two_week_high': chars.get('fifty_two_week_high', 0),
                'fifty_two_week_low': chars.get('fifty_two_week_low', 0),
                'trailing_pe': chars.get('trailing_pe', 0),
                'dividend_yield': chars.get('dividend_yield', 0),
                'hist_volatility': volatilities.get(symbol, 0),
            })

    df_results = pd.DataFrame(all_tier_rows)
    df_detail = pd.DataFrame(all_detail_rows)

    return df_results, df_detail, dict(tier_history)


# ============================================================================
# Results Reporting
# ============================================================================

def print_classification_summary(df_results, df_detail, tier_history, config):
    """Print comprehensive classification summary."""
    num_tiers = config['num_tiers']
    tier_labels = config['tier_labels']
    total_periods = len(df_results['period'].unique()) if not df_results.empty else 0

    print("\n" + "=" * 70)
    print("STOCK CLASSIFICATION RESULTS")
    print("=" * 70)

    # --- 1. Aggregate by tier ---
    print("\n" + "-" * 70)
    print("1. AGGREGATE PERFORMANCE BY TIER (across all periods)")
    print("-" * 70)

    for tier_num in range(1, num_tiers + 1):
        tier_data = df_results[df_results['tier'] == tier_num]
        label = tier_labels.get(tier_num, f'TIER {tier_num}')

        if tier_data.empty:
            continue

        avg_n = tier_data['n_stocks'].mean()
        avg_train_ret = tier_data['train_avg_return'].mean()
        avg_train_sharpe = tier_data['train_avg_sharpe'].mean()
        avg_train_wr = tier_data['train_avg_win_rate'].mean()
        avg_test_ret = tier_data['test_avg_return'].mean()
        avg_test_sharpe = tier_data['test_avg_sharpe'].mean()
        avg_test_wr = tier_data['test_avg_win_rate'].mean()
        avg_degradation = tier_data['return_degradation'].mean()

        print(f"\n  Tier {tier_num} ({label}) - avg {avg_n:.0f} stocks/period:")
        print(f"    Train:  Return={avg_train_ret:+7.2f}%  Sharpe={avg_train_sharpe:7.3f}  WinRate={avg_train_wr:5.1f}%")
        print(f"    Test:   Return={avg_test_ret:+7.2f}%  Sharpe={avg_test_sharpe:7.3f}  WinRate={avg_test_wr:5.1f}%")
        print(f"    Degradation: {avg_degradation:+.2f}%")

    # --- 2. Tier 1 vs Tier N spread ---
    print("\n" + "-" * 70)
    print(f"2. TIER 1 vs TIER {num_tiers} SPREAD (Out-of-Sample)")
    print("-" * 70)

    tier1_data = df_results[df_results['tier'] == 1]
    tierN_data = df_results[df_results['tier'] == num_tiers]

    if not tier1_data.empty and not tierN_data.empty:
        t1_ret = tier1_data['test_avg_return'].mean()
        tN_ret = tierN_data['test_avg_return'].mean()
        t1_sharpe = tier1_data['test_avg_sharpe'].mean()
        tN_sharpe = tierN_data['test_avg_sharpe'].mean()
        t1_wr = tier1_data['test_avg_win_rate'].mean()
        tN_wr = tierN_data['test_avg_win_rate'].mean()

        print(f"\n  Tier 1 ({tier_labels.get(1, 'BEST')}) OOS:  Return={t1_ret:+7.2f}%  Sharpe={t1_sharpe:7.3f}  WinRate={t1_wr:5.1f}%")
        print(f"  Tier {num_tiers} ({tier_labels.get(num_tiers, 'POOR')}) OOS:  Return={tN_ret:+7.2f}%  Sharpe={tN_sharpe:7.3f}  WinRate={tN_wr:5.1f}%")
        print(f"\n  Spread (Tier 1 - Tier {num_tiers}):")
        print(f"    Return:   {t1_ret - tN_ret:+7.2f}%")
        print(f"    Sharpe:   {t1_sharpe - tN_sharpe:+7.3f}")
        print(f"    WinRate:  {t1_wr - tN_wr:+5.1f}%")

        if t1_ret > tN_ret:
            print(f"\n  ✓ Top tier outperforms bottom tier out-of-sample by {t1_ret - tN_ret:.2f}%")
        else:
            print(f"\n  ⚠️  Bottom tier outperforms top tier out-of-sample - classification may not be predictive")
    else:
        print("\n  Insufficient data for tier comparison.")

    # --- 3. Consistently top stocks ---
    print("\n" + "-" * 70)
    print("3. CONSISTENTLY TOP-PERFORMING STOCKS")
    print("-" * 70)

    if total_periods > 0:
        # Stocks in Tier 1 across ALL periods
        always_tier1 = [
            s for s, tiers in tier_history.items()
            if len(tiers) == total_periods and all(t == 1 for t in tiers)
        ]
        # Stocks in Tier 1-2 across ALL periods
        always_top2 = [
            s for s, tiers in tier_history.items()
            if len(tiers) == total_periods and all(t <= 2 for t in tiers)
        ]

        print(f"\n  Stocks in Tier 1 across ALL {total_periods} periods ({len(always_tier1)}):")
        if always_tier1:
            for s in sorted(always_tier1):
                chars = df_detail[df_detail['symbol'] == s].iloc[0] if len(df_detail[df_detail['symbol'] == s]) > 0 else None
                sector = chars['sector'] if chars is not None else 'Unknown'
                avg_test_ret = df_detail[df_detail['symbol'] == s]['test_return'].mean()
                print(f"    {s:8s}  Sector: {sector:25s}  Avg OOS Return: {avg_test_ret:+.2f}%")
        else:
            print("    (none)")

        print(f"\n  Stocks in Tier 1-2 across ALL {total_periods} periods ({len(always_top2)}):")
        if always_top2:
            for s in sorted(always_top2):
                tiers = tier_history[s]
                chars = df_detail[df_detail['symbol'] == s].iloc[0] if len(df_detail[df_detail['symbol'] == s]) > 0 else None
                sector = chars['sector'] if chars is not None else 'Unknown'
                avg_test_ret = df_detail[df_detail['symbol'] == s]['test_return'].mean()
                avg_tier = np.mean(tiers)
                print(f"    {s:8s}  Sector: {sector:25s}  Avg Tier: {avg_tier:.1f}  Avg OOS Return: {avg_test_ret:+.2f}%")
        else:
            print("    (none)")

    # --- 4. Sector analysis ---
    print("\n" + "-" * 70)
    print("4. SECTOR ANALYSIS")
    print("-" * 70)

    if not df_detail.empty and 'sector' in df_detail.columns:
        sector_stats = df_detail.groupby('sector').agg(
            n_entries=('symbol', 'count'),
            avg_tier=('tier', 'mean'),
            avg_oos_return=('test_return', 'mean'),
            avg_oos_sharpe=('test_sharpe', 'mean'),
            avg_oos_win_rate=('test_win_rate', 'mean'),
        ).sort_values('avg_tier')

        print(f"\n  {'Sector':<30s}  {'N':>4s}  {'AvgTier':>7s}  {'OOS Ret':>8s}  {'Sharpe':>7s}  {'WinRate':>7s}")
        print(f"  {'-' * 30}  {'----':>4s}  {'-------':>7s}  {'--------':>8s}  {'-------':>7s}  {'-------':>7s}")
        for sector, row in sector_stats.iterrows():
            print(f"  {sector:<30s}  {row['n_entries']:4.0f}  {row['avg_tier']:7.2f}  {row['avg_oos_return']:+7.2f}%  {row['avg_oos_sharpe']:7.3f}  {row['avg_oos_win_rate']:6.1f}%")

    # --- 5. Volatility profile (beta-based) ---
    print("\n" + "-" * 70)
    print("5. VOLATILITY PROFILE (by Beta)")
    print("-" * 70)

    if not df_detail.empty and 'beta' in df_detail.columns:
        df_with_beta = df_detail[df_detail['beta'] > 0].copy()
        if not df_with_beta.empty:
            beta_33 = df_with_beta['beta'].quantile(0.33)
            beta_66 = df_with_beta['beta'].quantile(0.66)

            def vol_group(beta):
                if beta <= beta_33:
                    return 'Low Beta'
                elif beta <= beta_66:
                    return 'Med Beta'
                else:
                    return 'High Beta'

            df_with_beta['vol_group'] = df_with_beta['beta'].apply(vol_group)

            vol_stats = df_with_beta.groupby('vol_group').agg(
                n_entries=('symbol', 'count'),
                avg_beta=('beta', 'mean'),
                avg_tier=('tier', 'mean'),
                avg_oos_return=('test_return', 'mean'),
                avg_oos_sharpe=('test_sharpe', 'mean'),
                avg_oos_win_rate=('test_win_rate', 'mean'),
            )

            # Sort by Low/Med/High
            sort_order = {'Low Beta': 0, 'Med Beta': 1, 'High Beta': 2}
            vol_stats = vol_stats.loc[sorted(vol_stats.index, key=lambda x: sort_order.get(x, 99))]

            print(f"\n  {'Group':<12s}  {'N':>4s}  {'AvgBeta':>7s}  {'AvgTier':>7s}  {'OOS Ret':>8s}  {'Sharpe':>7s}  {'WinRate':>7s}")
            print(f"  {'-' * 12}  {'----':>4s}  {'-------':>7s}  {'-------':>7s}  {'--------':>8s}  {'-------':>7s}  {'-------':>7s}")
            for group, row in vol_stats.iterrows():
                print(f"  {group:<12s}  {row['n_entries']:4.0f}  {row['avg_beta']:7.2f}  {row['avg_tier']:7.2f}  {row['avg_oos_return']:+7.2f}%  {row['avg_oos_sharpe']:7.3f}  {row['avg_oos_win_rate']:6.1f}%")

            best_vol = vol_stats['avg_tier'].idxmin()
            print(f"\n  ✓ Best volatility group: {best_vol}")
        else:
            print("\n  No beta data available.")

    # --- 6. Market cap analysis ---
    print("\n" + "-" * 70)
    print("6. MARKET CAP ANALYSIS")
    print("-" * 70)

    if not df_detail.empty and 'market_cap' in df_detail.columns:
        df_with_cap = df_detail[df_detail['market_cap'] > 0].copy()
        if not df_with_cap.empty:
            def cap_group(cap):
                if cap < 2e9:
                    return 'Small Cap (<$2B)'
                elif cap < 10e9:
                    return 'Mid Cap ($2-10B)'
                else:
                    return 'Large Cap (>$10B)'

            df_with_cap['cap_group'] = df_with_cap['market_cap'].apply(cap_group)

            cap_stats = df_with_cap.groupby('cap_group').agg(
                n_entries=('symbol', 'count'),
                avg_tier=('tier', 'mean'),
                avg_oos_return=('test_return', 'mean'),
                avg_oos_sharpe=('test_sharpe', 'mean'),
                avg_oos_win_rate=('test_win_rate', 'mean'),
            )

            sort_order = {'Small Cap (<$2B)': 0, 'Mid Cap ($2-10B)': 1, 'Large Cap (>$10B)': 2}
            cap_stats = cap_stats.loc[sorted(cap_stats.index, key=lambda x: sort_order.get(x, 99))]

            print(f"\n  {'Group':<20s}  {'N':>4s}  {'AvgTier':>7s}  {'OOS Ret':>8s}  {'Sharpe':>7s}  {'WinRate':>7s}")
            print(f"  {'-' * 20}  {'----':>4s}  {'-------':>7s}  {'--------':>8s}  {'-------':>7s}  {'-------':>7s}")
            for group, row in cap_stats.iterrows():
                print(f"  {group:<20s}  {row['n_entries']:4.0f}  {row['avg_tier']:7.2f}  {row['avg_oos_return']:+7.2f}%  {row['avg_oos_sharpe']:7.3f}  {row['avg_oos_win_rate']:6.1f}%")

            best_cap = cap_stats['avg_tier'].idxmin()
            print(f"\n  ✓ Best market cap group: {best_cap}")
        else:
            print("\n  No market cap data available.")

    # --- 7. Stock Screener Criteria ---
    print("\n" + "=" * 70)
    print("7. STOCK SCREENER CRITERIA (from Tier 1 stocks, 10th-90th percentile)")
    print("=" * 70)

    if not df_detail.empty:
        top_tier = df_detail[df_detail['tier'] == 1].copy()

        if not top_tier.empty:
            def fmt_cap(v):
                """Format market cap in human-readable form."""
                if v >= 1e12:
                    return f'${v/1e12:.1f}T'
                elif v >= 1e9:
                    return f'${v/1e9:.1f}B'
                elif v >= 1e6:
                    return f'${v/1e6:.0f}M'
                else:
                    return f'${v:,.0f}'

            def fmt_vol(v):
                """Format volume in human-readable form."""
                if v >= 1e6:
                    return f'{v/1e6:.1f}M'
                elif v >= 1e3:
                    return f'{v/1e3:.0f}K'
                else:
                    return f'{v:,.0f}'

            # (field, label, formatter_func)
            # Using lambdas for simple formats, named funcs for complex ones
            screener_fields = [
                ('price',              'Price ($)',            lambda v: f'${v:.2f}'),
                ('market_cap',         'Market Cap',           fmt_cap),
                ('beta',               'Beta',                 lambda v: f'{v:.2f}'),
                ('avg_volume',         'Avg Daily Volume',     fmt_vol),
                ('hist_volatility',    'Hist Volatility (%)',  lambda v: f'{v:.1f}%'),
                ('trailing_pe',        'Trailing P/E',         lambda v: f'{v:.1f}'),
                ('dividend_yield',     'Dividend Yield (%)',   lambda v: f'{v:.2f}%'),
            ]

            print(f"\n  Based on {len(top_tier)} Tier 1 stock-period entries:\n")
            print(f"  {'Characteristic':<22s}  {'10th Pctl':>14s}  {'Median':>14s}  {'90th Pctl':>14s}")
            print(f"  {'-'*22}  {'-'*14}  {'-'*14}  {'-'*14}")

            screener_ranges = {}
            for field, label, fmt in screener_fields:
                if field not in top_tier.columns:
                    continue
                vals = top_tier[top_tier[field] > 0][field]
                if vals.empty:
                    continue
                p10 = vals.quantile(0.10)
                p50 = vals.quantile(0.50)
                p90 = vals.quantile(0.90)
                screener_ranges[field] = (p10, p50, p90)
                print(f"  {label:<22s}  {fmt(p10):>14s}  {fmt(p50):>14s}  {fmt(p90):>14s}")

            # Sectors
            top_sectors = top_tier.groupby('sector')['symbol'].nunique().sort_values(ascending=False)
            total_unique = top_tier['symbol'].nunique()
            print(f"\n  Sectors (by unique stock count in Tier 1):")
            for sector, count in top_sectors.items():
                pct = count / total_unique * 100
                print(f"    {sector:<30s}  {count:3d} stocks  ({pct:5.1f}%)")

            # Industries (top 10)
            top_industries = top_tier.groupby('industry')['symbol'].nunique().sort_values(ascending=False)
            print(f"\n  Top Industries:")
            for industry, count in top_industries.head(10).items():
                pct = count / total_unique * 100
                print(f"    {industry:<40s}  {count:3d} stocks  ({pct:5.1f}%)")

            # Ready-to-use screener summary
            print(f"\n  " + "-" * 66)
            print(f"  COPY-PASTE SCREENER FILTERS (Tier 1 profile, 10th-90th pctl):")
            print(f"  " + "-" * 66)

            filter_labels = {
                'price':           ('Price',           lambda lo, hi: f'${lo:.2f} - ${hi:.2f}'),
                'market_cap':      ('Market Cap',      lambda lo, hi: f'{fmt_cap(lo)} - {fmt_cap(hi)}'),
                'beta':            ('Beta',            lambda lo, hi: f'{lo:.2f} - {hi:.2f}'),
                'avg_volume':      ('Avg Volume',      lambda lo, hi: f'{fmt_vol(lo)} - {fmt_vol(hi)}'),
                'hist_volatility': ('Hist Volatility', lambda lo, hi: f'{lo:.1f}% - {hi:.1f}% (annualized)'),
                'trailing_pe':     ('Trailing P/E',    lambda lo, hi: f'{lo:.1f} - {hi:.1f}'),
                'dividend_yield':  ('Dividend Yield',  lambda lo, hi: f'{lo:.2f}% - {hi:.2f}%'),
            }

            for field, (label, fmt_range) in filter_labels.items():
                if field in screener_ranges:
                    lo, _, hi = screener_ranges[field]
                    print(f"    {label + ':':<20s}{fmt_range(lo, hi)}")

            if not top_sectors.empty:
                included = [s for s, c in top_sectors.items() if s != 'Unknown']
                if included:
                    print(f"    {'Sectors:':<20s}{', '.join(included[:6])}")
        else:
            print("\n  No Tier 1 data available.")

    # --- 8. Conclusion ---
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if not df_results.empty:
        tier1 = df_results[df_results['tier'] == 1]
        tierN = df_results[df_results['tier'] == num_tiers]

        if not tier1.empty and not tierN.empty:
            t1_oos = tier1['test_avg_return'].mean()
            tN_oos = tierN['test_avg_return'].mean()

            if t1_oos > tN_oos:
                print(f"\n  ✓ Training-period classification IS predictive of out-of-sample performance.")
                print(f"    Tier 1 OOS return ({t1_oos:+.2f}%) > Tier {num_tiers} OOS return ({tN_oos:+.2f}%)")
            else:
                print(f"\n  ⚠️  Training-period classification is NOT predictive of out-of-sample performance.")
                print(f"    Tier 1 OOS return ({t1_oos:+.2f}%) <= Tier {num_tiers} OOS return ({tN_oos:+.2f}%)")

    always_tier1 = [
        s for s, tiers in tier_history.items()
        if len(tiers) == total_periods and total_periods > 0 and all(t == 1 for t in tiers)
    ]
    if always_tier1:
        print(f"\n  Consistently top stocks ({len(always_tier1)}): {', '.join(sorted(always_tier1))}")

    print("\n  ⚠️  Past performance does not guarantee future results.")
    print("=" * 70 + "\n")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main execution."""
    df_results, df_detail, tier_history = run_stock_classification(CONFIG)

    if df_results.empty:
        print("\n❌ No results obtained")
        return

    print_classification_summary(df_results, df_detail, tier_history, CONFIG)

    # Save results
    df_results.to_csv(CONFIG['results_file'], index=False)
    print(f"📄 Tier-level results saved to '{CONFIG['results_file']}'")

    df_detail.to_csv(CONFIG['detail_file'], index=False)
    print(f"📄 Per-stock detail saved to '{CONFIG['detail_file']}'")

    print("\n✅ Stock classification complete!\n")


if __name__ == "__main__":
    main()
