"""
Walk-Forward Optimization for Lorentzian Classification Strategy - Trend Features

This script implements proper walk-forward analysis with:
- Rolling optimization windows (train/test splits)
- Out-of-sample validation on unseen data
- Aggregation across multiple forward periods
- COVID period handling options

Uses trend-focused feature vector:
- RSI(14): Momentum
- ADX(14): Trend strength
- ATR Ratio: Normalized volatility
- Price Position: Location in recent range
- Efficiency Ratio: Trend quality

Usage:
    python optimize_strategy.py
"""

from decimal import Decimal
import matplotlib
matplotlib.use('Agg')
import backtrader as bt
import yfinance as yf
from lorentzian_classification import LorentzianClassificationStrategy
import pandas as pd
import numpy as np
from itertools import product
import csv
from datetime import datetime, timedelta
import sys


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Input/Output
    'csv_file': '../optimization_set.csv',
    'results_file': 'walkforward_optimization_results.csv',

    # Walk-Forward Settings
    'train_period_months': 18,      # Optimize on 18 months
    'test_period_months': 6,        # Validate on next 6 months
    'step_months': 3,               # Roll forward 6 months each iteration
    'total_periods': 3,             # Number of train/test cycles

    # Date range control (optional - leave None for automatic)
    'end_date': None,               # None = use most recent data
    'lookback_years': 5,            # How many years of history to use (need more for ML)

    # COVID handling
    'exclude_covid': True,          # Skip COVID period in training
    'covid_start': '2020-02-01',    # COVID crash start
    'covid_end': '2021-06-30',      # Recovery period end

    # Backtest settings
    'initial_cash': 10000,
    'commission': 0.0001,           # 0.05% commission per trade

    # Optimization settings
    'top_n_results': 10,
    'print_progress_every': 50,

    # Quality filters - RELAXED for walk-forward (short periods, will average out)
    'quality_filters': {
        'min_sharpe': -0.5,
        'min_calmar': -1.0,
        'min_win_rate': 20.0,
        'min_rr_ratio': 0.5,
        'min_profit_factor': 0.8,
        'min_total_trades': 2,
        'max_drawdown': 80.0,
        'min_stock_win_rate': 20.0,
        'min_expectancy': -50.0,
    },

    # Parameter grid for Lorentzian Classification - Trend Features
    'param_grid': {
        # ==================== ML SETTINGS ====================
        'neighbors_count': [6, 8, 10],
        'max_bars_back': [500],            # Keep fixed - needs lots of history
        'feature_count': [5],
        'trend_following_labels': [True, False],  # False=mean-reversion, True=trend-following
        'allow_reentry': [True],            # True=enter anytime signal favorable
        'min_prediction_strength': [4],     # Minimum |prediction| to generate signal

        # ==================== FEATURE 1 (RSI - Momentum) ====================
        'f1_type': ['RSI'],
        'f1_param_a': [14],
        'f1_param_b': [1],

        # ==================== FEATURE 2 (ADX - Trend Strength) ====================
        'f2_type': ['ADX'],
        'f2_param_a': [14],
        'f2_param_b': [1],

        # ==================== FEATURE 3 (ATR Ratio - Volatility) ====================
        'f3_type': ['ATRR'],
        'f3_param_a': [14],
        'f3_param_b': [1],

        # ==================== FEATURE 4 (Price Position) ====================
        'f4_type': ['PP'],
        'f4_param_a': [14],
        'f4_param_b': [1],

        # ==================== FEATURE 5 (Efficiency Ratio) ====================
        'f5_type': ['ER'],
        'f5_param_a': [14],
        'f5_param_b': [1],

        # ==================== FILTERS ====================
        'use_volatility_filter': [True, False],
        'use_regime_filter': [True],
        'regime_threshold': [Decimal('-0.1')],
        'use_adx_filter': [True, False],
        'adx_threshold': [20],
        'use_ema_filter': [False],
        'ema_period': [200],
        'use_sma_filter': [False],
        'sma_period': [200],

        # ==================== KERNEL SETTINGS ====================
        'use_kernel_filter': [False, True],
        'use_kernel_smoothing': [False],
        'kernel_lookback': [8],
        'kernel_rel_weight': [8.0],
        'kernel_start_bar': [25],
        'kernel_lag': [2],

        # ==================== EXIT SETTINGS ====================
        'use_dynamic_exits': [True, False],
        'bars_to_hold': [10000],

        # ==================== RSI EXIT SETTINGS ====================
        'use_rsi_exit': [False],            # Enable RSI threshold exits
        'rsi_exit_period': [14],            # RSI period for exit signals
        'rsi_overbought': [70],             # Exit longs when RSI crosses above
        'rsi_oversold': [30],               # Exit shorts when RSI crosses below

        # ==================== KERNEL EXIT SETTINGS ====================
        'use_kernel_exit': [True, False],         # Enable kernel line exit

        # ==================== RISK MANAGEMENT ====================
        'position_size_pct': [Decimal('0.95')],
        'stop_loss_pct': [Decimal('0.05')],
        'use_stop_loss': [True],
        'long_only': [True],

        # ==================== OTHER ====================
        'verbose': [False],
        'test_start_idx': [0],              # Bar index to start trading
    }
}


# ============================================================================
# Walk-Forward Period Generation
# ============================================================================

def generate_walkforward_periods(config):
    """
    Generate train/test period pairs for walk-forward analysis.
    Works backwards from present to ensure all dates are historical.
    """
    periods = []

    if config.get('end_date'):
        end_date = pd.to_datetime(config['end_date']).to_pydatetime()
    else:
        end_date = datetime.now() - timedelta(days=7)

    lookback_years = config.get('lookback_years', 5)

    total_span_needed = (
            config['train_period_months'] +
            (config['total_periods'] - 1) * config['step_months'] +
            config['test_period_months']
    )

    months_to_go_back = max(total_span_needed + 6, lookback_years * 12)
    start_date = end_date - timedelta(days=months_to_go_back * 30)

    print(f"\n📅 Generating Walk-Forward Periods")
    print(f"   Lookback period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"   Total span: {months_to_go_back} months")

    covid_start = pd.to_datetime(config['covid_start']) if config['exclude_covid'] else None
    covid_end = pd.to_datetime(config['covid_end']) if config['exclude_covid'] else None

    current_train_start = start_date
    periods_attempted = 0
    periods_kept = 0

    while periods_attempted < config['total_periods'] * 2:
        periods_attempted += 1

        train_start = current_train_start
        train_end = train_start + timedelta(days=config['train_period_months'] * 30)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=config['test_period_months'] * 30)

        if test_end > end_date:
            print(f"   Stopped: Period {periods_attempted} would extend beyond {end_date.strftime('%Y-%m-%d')}")
            break

        if periods_kept >= config['total_periods']:
            break

        skip_period = False
        skip_reason = None

        if config['exclude_covid'] and covid_start and covid_end:
            train_covid_overlap = (
                    train_start < covid_end.to_pydatetime() and
                    train_end > covid_start.to_pydatetime()
            )
            if train_covid_overlap:
                overlap_start = max(train_start, covid_start.to_pydatetime())
                overlap_end = min(train_end, covid_end.to_pydatetime())
                overlap_days = (overlap_end - overlap_start).days

                if overlap_days > 30:
                    skip_period = True
                    skip_reason = f"COVID overlap ({overlap_days} days)"

        if skip_period:
            print(f"   Skipped period: Train {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')} - {skip_reason}")
        else:
            periods.append((
                train_start.strftime('%Y-%m-%d'),
                train_end.strftime('%Y-%m-%d'),
                test_start.strftime('%Y-%m-%d'),
                test_end.strftime('%Y-%m-%d')
            ))
            periods_kept += 1
            print(f"   ✓ Period {periods_kept}: Train {train_start.strftime('%Y-%m-%d')}-{train_end.strftime('%Y-%m-%d')}, Test {test_start.strftime('%Y-%m-%d')}-{test_end.strftime('%Y-%m-%d')}")

        current_train_start += timedelta(days=config['step_months'] * 30)

    if not periods:
        print("\n⚠️  ERROR: No valid periods generated!")
        print("   Try one of these:")
        print("   1. Set 'exclude_covid': False")
        print("   2. Increase 'lookback_years'")
        print("   3. Reduce 'total_periods'")
    else:
        print(f"\n   ✓ Generated {len(periods)} valid walk-forward periods")

    return periods


def print_walkforward_schedule(periods):
    """Print the walk-forward testing schedule."""
    print("\n" + "="*70)
    print("WALK-FORWARD TESTING SCHEDULE")
    print("="*70)

    for i, (train_start, train_end, test_start, test_end) in enumerate(periods, 1):
        print(f"\nPeriod {i}:")
        print(f"  Train: {train_start} to {train_end}")
        print(f"  Test:  {test_start} to {test_end} (OUT-OF-SAMPLE)")

    print("\n" + "="*70)
    print("NOTE: Parameters optimized ONLY on train periods.")
    print("      Test results are truly out-of-sample (unseen data).")
    print("="*70)


# ============================================================================
# Data Management
# ============================================================================

def load_symbol_data(symbol, start_date, end_date):
    """Download data for a specific date range."""
    try:
        start = pd.to_datetime(start_date) - timedelta(days=5)
        end = pd.to_datetime(end_date) + timedelta(days=5)

        df = yf.download(symbol, start=start, end=end, progress=False)

        if df.empty:
            return None

        df.index = df.index.tz_localize(None)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']

        df = df[(df.index >= pd.to_datetime(start_date)) &
                (df.index <= pd.to_datetime(end_date))]

        return df if len(df) > 50 else None  # Need more bars for ML

    except Exception:
        return None


def load_all_periods_data(symbols, periods):
    """Load data for all symbols across all periods."""
    data_cache = {}

    print(f"\n📥 Downloading data for {len(symbols)} stocks across {len(periods)} periods...")

    total_downloads = len(symbols) * len(periods)
    download_count = 0

    for symbol in symbols:
        data_cache[symbol] = {}

        for period_idx, (train_start, train_end, test_start, test_end) in enumerate(periods):
            train_df = load_symbol_data(symbol, train_start, train_end)
            test_df = load_symbol_data(symbol, test_start, test_end)

            if train_df is not None and test_df is not None:
                data_cache[symbol][period_idx] = {
                    'train': train_df,
                    'test': test_df
                }

            download_count += 1
            if download_count % 50 == 0:
                print(f"   Progress: {download_count}/{total_downloads} ({download_count/total_downloads*100:.0f}%)")

    valid_symbols = [s for s in symbols if len(data_cache.get(s, {})) >= len(periods) * 0.7]

    print(f"✓ Valid symbols with sufficient data: {len(valid_symbols)}/{len(symbols)}\n")

    return data_cache, valid_symbols


# ============================================================================
# Core Backtesting
# ============================================================================

def backtest_single_config(symbol, params, df, initial_cash, commission):
    """Run backtest for a single symbol with given parameters."""
    try:
        cerebro = bt.Cerebro(stdstats=False)

        data = bt.feeds.PandasData(
            dataname=df,
            datetime=None,
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume'
        )
        cerebro.adddata(data)
        cerebro.addstrategy(LorentzianClassificationStrategy, **params)

        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=commission)

        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")

        results = cerebro.run()
        strat = results[0]

        sharpe = strat.analyzers.sharpe.get_analysis()
        dd = strat.analyzers.dd.get_analysis()
        trades = strat.analyzers.trades.get_analysis()
        sqn = strat.analyzers.sqn.get_analysis()

        final_value = cerebro.broker.getvalue()
        return_pct = (final_value / initial_cash - 1) * 100

        max_dd_pct = dd.get('max', {}).get('drawdown', 0)
        calmar_ratio = (return_pct / max_dd_pct) if max_dd_pct > 0 else 0

        total_trades = trades.get('total', {}).get('total', 0)

        if total_trades > 0:
            wins = trades.get('won', {}).get('total', 0)
            win_rate = (wins / total_trades * 100)

            avg_win = trades.get('won', {}).get('pnl', {}).get('average', 0)
            avg_loss = abs(trades.get('lost', {}).get('pnl', {}).get('average', 0))
            rr_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0

            total_win_pnl = trades.get('won', {}).get('pnl', {}).get('total', 0)
            total_loss_pnl = abs(trades.get('lost', {}).get('pnl', {}).get('total', 0))
            profit_factor = (total_win_pnl / total_loss_pnl) if total_loss_pnl > 0 else 0

            expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
        else:
            win_rate = rr_ratio = profit_factor = expectancy = 0

        return {
            'return_pct': return_pct,
            'sharpe': sharpe.get('sharperatio', 0) or 0,
            'calmar': calmar_ratio,
            'sqn': sqn.get('sqn', 0),
            'max_drawdown': max_dd_pct,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'rr_ratio': rr_ratio,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
        }

    except Exception as e:
        return None


def aggregate_results(stock_results, params):
    """Aggregate results across all stocks for a parameter set."""
    if not stock_results:
        return None

    n_stocks = len(stock_results)

    avg_return = sum(r['return_pct'] for r in stock_results) / n_stocks
    avg_sharpe = sum(r['sharpe'] for r in stock_results) / n_stocks
    avg_calmar = sum(r['calmar'] for r in stock_results) / n_stocks
    avg_sqn = sum(r['sqn'] for r in stock_results) / n_stocks
    avg_drawdown = sum(r['max_drawdown'] for r in stock_results) / n_stocks
    total_trades = sum(r['total_trades'] for r in stock_results)
    avg_win_rate = sum(r['win_rate'] for r in stock_results) / n_stocks
    avg_rr_ratio = sum(r['rr_ratio'] for r in stock_results) / n_stocks
    avg_profit_factor = sum(r['profit_factor'] for r in stock_results) / n_stocks
    avg_expectancy = sum(r['expectancy'] for r in stock_results) / n_stocks

    winning_stocks = sum(1 for r in stock_results if r['return_pct'] > 0)
    stock_win_rate = (winning_stocks / n_stocks) * 100

    if avg_sharpe > 0 and avg_return > 0 and avg_drawdown > 0:
        composite_score = (avg_return * avg_sharpe * (stock_win_rate / 100)) / avg_drawdown
    else:
        composite_score = 0

    result = {
        'avg_return_pct': avg_return,
        'avg_sharpe': avg_sharpe,
        'avg_calmar': avg_calmar,
        'avg_sqn': avg_sqn,
        'avg_max_drawdown': avg_drawdown,
        'total_trades': total_trades,
        'avg_win_rate': avg_win_rate,
        'avg_rr_ratio': avg_rr_ratio,
        'avg_profit_factor': avg_profit_factor,
        'avg_expectancy': avg_expectancy,
        'stocks_tested': n_stocks,
        'winning_stocks': winning_stocks,
        'stock_win_rate': stock_win_rate,
        'composite_score': composite_score,
    }

    return result


# ============================================================================
# Walk-Forward Optimization Logic
# ============================================================================

# Define parameter names for the strategy (must match param_grid order exactly)
PARAM_NAMES = [
    # ML Settings
    'neighbors_count', 'max_bars_back', 'feature_count',
    'trend_following_labels', 'allow_reentry', 'min_prediction_strength',
    # Features
    'f1_type', 'f1_param_a', 'f1_param_b',
    'f2_type', 'f2_param_a', 'f2_param_b',
    'f3_type', 'f3_param_a', 'f3_param_b',
    'f4_type', 'f4_param_a', 'f4_param_b',
    'f5_type', 'f5_param_a', 'f5_param_b',
    # Filters
    'use_volatility_filter', 'use_regime_filter', 'regime_threshold',
    'use_adx_filter', 'adx_threshold', 'use_ema_filter', 'ema_period',
    'use_sma_filter', 'sma_period',
    # Kernel Settings
    'use_kernel_filter', 'use_kernel_smoothing', 'kernel_lookback',
    'kernel_rel_weight', 'kernel_start_bar', 'kernel_lag',
    # Exit Settings
    'use_dynamic_exits', 'bars_to_hold',
    # RSI Exit Settings
    'use_rsi_exit', 'rsi_exit_period', 'rsi_overbought', 'rsi_oversold',
    # Kernel Exit Settings
    'use_kernel_exit',
    # Risk Management
    'position_size_pct', 'stop_loss_pct', 'use_stop_loss', 'long_only',
    # Other
    'verbose', 'test_start_idx',
]


def optimize_single_period(data_cache, valid_symbols, params_list, period_idx, phase, config):
    """Optimize or test on a single period."""
    results = {}
    total_backtests = len(params_list) * len(valid_symbols)
    backtest_count = 0
    successful_backtests = 0

    print(f"   Running {total_backtests:,} backtests on {phase} data...")
    print(f"   Parameter combinations: {len(params_list)}")
    print(f"   Symbols: {len(valid_symbols)}")

    for params_tuple in params_list:
        if hasattr(params_tuple, '_asdict'):
            params = params_tuple._asdict()
        else:
            params = dict(zip(PARAM_NAMES, params_tuple))

        stock_results = []

        for symbol in valid_symbols:
            if period_idx not in data_cache.get(symbol, {}):
                continue

            df = data_cache[symbol][period_idx][phase]

            if df is None or len(df) < 50:
                continue

            result = backtest_single_config(
                symbol,
                params,
                df,
                config['initial_cash'],
                config['commission']
            )

            if result:
                stock_results.append(result)
                successful_backtests += 1

            backtest_count += 1
            if backtest_count % config['print_progress_every'] == 0:
                print(f"      Progress: {backtest_count:,}/{total_backtests:,} ({backtest_count/total_backtests*100:.0f}%) | Successful: {successful_backtests}")

        aggregated = aggregate_results(stock_results, params)
        if aggregated:
            results[params_tuple] = {**aggregated, 'params': params}

    print(f"   ✓ Completed {backtest_count:,} backtests, {successful_backtests} successful")
    print(f"   ✓ Generated results for {len(results)} parameter combinations")

    if len(results) == 0:
        print(f"   ⚠️  WARNING: No valid results! Check if data is available for this period.")

    return results


def run_walkforward_optimization(config):
    """Main walk-forward optimization function."""
    print("\n" + "="*70)
    print("WALK-FORWARD OPTIMIZATION - LORENTZIAN CLASSIFICATION (TREND FEATURES)")
    print("="*70)

    periods = generate_walkforward_periods(config)

    if not periods:
        print("\n❌ No valid periods generated. Check configuration.")
        sys.exit(1)

    print_walkforward_schedule(periods)

    try:
        with open(config['csv_file'], 'r') as f:
            symbols = [row[0].strip().upper() for row in csv.reader(f) if row and row[0].strip()]
    except FileNotFoundError:
        print(f"\n❌ File '{config['csv_file']}' not found")
        sys.exit(1)

    print(f"\n📋 Configuration")
    print(f"   Symbols: {len(symbols)} stocks")
    print(f"   Walk-forward periods: {len(periods)}")
    print(f"   Train period: {config['train_period_months']} months")
    print(f"   Test period: {config['test_period_months']} months")
    print(f"   COVID exclusion: {'Yes' if config['exclude_covid'] else 'No'}")

    data_cache, valid_symbols = load_all_periods_data(symbols, periods)

    if not valid_symbols:
        print("\n❌ No valid data. Exiting.")
        sys.exit(1)

    # Generate parameter combinations
    param_names = list(config['param_grid'].keys())
    param_values = list(config['param_grid'].values())
    from collections import namedtuple
    ParamSet = namedtuple('ParamSet', param_names)
    params_list = [ParamSet(*combo) for combo in product(*param_values)]

    print(f"\n🔧 Parameter Grid")
    print(f"   Combinations to test: {len(params_list):,}")
    print(f"   Total backtests per period: {len(params_list) * len(valid_symbols):,}")

    print(f"\n🚀 Starting Walk-Forward Optimization...\n")

    all_results = []

    for period_idx, period_info in enumerate(periods):
        train_start, train_end, test_start, test_end = period_info

        print(f"\n{'='*70}")
        print(f"PERIOD {period_idx + 1}/{len(periods)}")
        print(f"{'='*70}")
        print(f"Train: {train_start} to {train_end}")
        print(f"Test:  {test_start} to {test_end}")

        print(f"\n1️⃣  OPTIMIZATION PHASE (In-Sample)")
        train_results = optimize_single_period(
            data_cache, valid_symbols, params_list, period_idx, 'train', config
        )

        if not train_results:
            print(f"   ❌ No valid results for training period!")
            continue

        best_params = max(train_results.items(), key=lambda x: x[1]['avg_return_pct'])
        best_params_tuple = best_params[0]
        best_train_perf = best_params[1]

        print(f"\n   ✓ Best In-Sample Parameters Found:")
        print(f"   " + "="*66)

        params_dict = best_train_perf['params']
        print(f"   ML Settings:")
        print(f"     Neighbors (K):         {params_dict['neighbors_count']}")
        print(f"     Feature Count:         {params_dict['feature_count']}")
        print(f"   Trend Features:")
        print(f"     F1: {params_dict['f1_type']}({params_dict['f1_param_a']}) - Momentum")
        print(f"     F2: {params_dict['f2_type']}({params_dict['f2_param_a']}) - Trend Strength")
        print(f"     F3: {params_dict['f3_type']}({params_dict['f3_param_a']}) - Normalized Volatility")
        print(f"     F4: {params_dict['f4_type']}({params_dict['f4_param_a']}) - Price Position")
        print(f"     F5: {params_dict['f5_type']}({params_dict['f5_param_a']}) - Efficiency Ratio")
        print(f"   Filters:")
        print(f"     Volatility:            {'ON' if params_dict['use_volatility_filter'] else 'OFF'}")
        print(f"     Regime:                {'ON' if params_dict['use_regime_filter'] else 'OFF'}")
        print(f"     Kernel:                {'ON' if params_dict['use_kernel_filter'] else 'OFF'}")
        print(f"   Exit:")
        print(f"     Bars to Hold:          {params_dict['bars_to_hold']}")

        print(f"\n   In-Sample Performance:")
        print(f"     Return:                {best_train_perf['avg_return_pct']:7.2f}%")
        print(f"     Sharpe:                {best_train_perf['avg_sharpe']:7.3f}")
        print(f"     Calmar:                {best_train_perf['avg_calmar']:7.3f}")
        print(f"     Max Drawdown:          {best_train_perf['avg_max_drawdown']:7.2f}%")
        print(f"     Total Trades:          {int(best_train_perf['total_trades'])} across {int(best_train_perf['stocks_tested'])} stocks")
        print(f"     Avg Trades/Stock:      {best_train_perf['total_trades']/best_train_perf['stocks_tested']:.1f}")
        print(f"     Win Rate:              {best_train_perf['avg_win_rate']:7.1f}%")
        print(f"     RR Ratio:              {best_train_perf['avg_rr_ratio']:7.2f}")
        print(f"     Profit Factor:         {best_train_perf['avg_profit_factor']:7.2f}")
        print(f"     Expectancy:            ${best_train_perf['avg_expectancy']:7.2f}")
        print(f"     Stock Win Rate:        {best_train_perf['stock_win_rate']:7.1f}% ({int(best_train_perf['winning_stocks'])}/{int(best_train_perf['stocks_tested'])})")
        print(f"   " + "="*66)

        print(f"\n2️⃣  VALIDATION PHASE (Out-of-Sample)")
        print(f"   Testing best parameters on unseen {test_start} to {test_end} data...")

        test_results = optimize_single_period(
            data_cache, valid_symbols, [best_params_tuple], period_idx, 'test', config
        )

        if best_params_tuple in test_results:
            test_performance = test_results[best_params_tuple]

            degradation = best_train_perf['avg_return_pct'] - test_performance['avg_return_pct']
            degradation_pct = (degradation / best_train_perf['avg_return_pct'] * 100) if best_train_perf['avg_return_pct'] != 0 else 0

            print(f"\n   Out-of-Sample Performance:")
            print(f"     Return:                {test_performance['avg_return_pct']:7.2f}%")
            print(f"     Sharpe:                {test_performance['avg_sharpe']:7.3f}")
            print(f"     Calmar:                {test_performance['avg_calmar']:7.3f}")
            print(f"     Max Drawdown:          {test_performance['avg_max_drawdown']:7.2f}%")
            print(f"     Total Trades:          {int(test_performance['total_trades'])} across {int(test_performance['stocks_tested'])} stocks")
            print(f"     Avg Trades/Stock:      {test_performance['total_trades']/test_performance['stocks_tested']:.1f}")
            print(f"     Win Rate:              {test_performance['avg_win_rate']:7.1f}%")
            print(f"     RR Ratio:              {test_performance['avg_rr_ratio']:7.2f}")
            print(f"     Profit Factor:         {test_performance['avg_profit_factor']:7.2f}")
            print(f"     Expectancy:            ${test_performance['avg_expectancy']:7.2f}")
            print(f"     Stock Win Rate:        {test_performance['stock_win_rate']:7.1f}% ({int(test_performance['winning_stocks'])}/{int(test_performance['stocks_tested'])})")

            print(f"\n   Performance Comparison:")
            print(f"     Return Degradation:    {degradation_pct:+.1f}% ({best_train_perf['avg_return_pct']:.2f}% → {test_performance['avg_return_pct']:.2f}%)")

            if test_performance['total_trades'] == 0:
                print(f"     ⚠️  WARNING: No trades in out-of-sample period!")
            elif test_performance['total_trades'] < 10:
                print(f"     ⚠️  WARNING: Very few trades ({int(test_performance['total_trades'])}) in out-of-sample.")

            if abs(degradation_pct) > 50:
                print(f"     ⚠️  WARNING: High degradation suggests overfitting!")
            elif abs(degradation_pct) < 20:
                print(f"     ✓ Good: Low degradation indicates robust parameters.")

            period_result = {
                'period': period_idx + 1,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'in_sample_return': best_train_perf['avg_return_pct'],
                'in_sample_sharpe': best_train_perf['avg_sharpe'],
                'in_sample_drawdown': best_train_perf['avg_max_drawdown'],
                'in_sample_trades': best_train_perf['total_trades'],
                'in_sample_win_rate': best_train_perf['avg_win_rate'],
                'in_sample_profit_factor': best_train_perf['avg_profit_factor'],
                'out_sample_return': test_performance['avg_return_pct'],
                'out_sample_sharpe': test_performance['avg_sharpe'],
                'out_sample_drawdown': test_performance['avg_max_drawdown'],
                'out_sample_trades': test_performance['total_trades'],
                'out_sample_win_rate': test_performance['avg_win_rate'],
                'out_sample_rr_ratio': test_performance['avg_rr_ratio'],
                'out_sample_profit_factor': test_performance['avg_profit_factor'],
                'out_sample_expectancy': test_performance['avg_expectancy'],
                'return_degradation_pct': degradation_pct,
                'trade_difference': int(test_performance['total_trades']) - int(best_train_perf['total_trades']),
                'params': best_train_perf['params'],
            }

            all_results.append(period_result)
        else:
            print(f"\n   ❌ No results for test period!")

    return pd.DataFrame(all_results)


# ============================================================================
# Results Reporting
# ============================================================================

def print_walkforward_results(df_results):
    """Print walk-forward results summary."""
    print("\n" + "="*70)
    print("WALK-FORWARD OPTIMIZATION RESULTS")
    print("="*70)

    print("\nPer-Period Out-of-Sample Performance:")
    print("-" * 70)

    for _, row in df_results.iterrows():
        print(f"\nPeriod {int(row['period'])}: {row['test_start']} to {row['test_end']}")
        print(f"  In-Sample (Train):")
        print(f"    Return:          {row['in_sample_return']:7.2f}%")
        print(f"    Sharpe:          {row['in_sample_sharpe']:7.3f}")
        print(f"    Drawdown:        {row['in_sample_drawdown']:7.2f}%")
        print(f"    Trades:          {int(row['in_sample_trades'])}")
        print(f"    Win Rate:        {row['in_sample_win_rate']:7.1f}%")

        print(f"  Out-of-Sample (Test):")
        print(f"    Return:          {row['out_sample_return']:7.2f}%")
        print(f"    Sharpe:          {row['out_sample_sharpe']:7.3f}")
        print(f"    Drawdown:        {row['out_sample_drawdown']:7.2f}%")
        print(f"    Trades:          {int(row['out_sample_trades'])}")
        print(f"    Win Rate:        {row['out_sample_win_rate']:7.1f}%")
        print(f"    Profit Factor:   {row['out_sample_profit_factor']:7.2f}")

        print(f"  Comparison:")
        print(f"    Return Change:   {row['return_degradation_pct']:+7.1f}%")
        print(f"    Trade Diff:      {int(row['trade_difference']):+d}")

    print("\n" + "="*70)
    print("AGGREGATE OUT-OF-SAMPLE PERFORMANCE")
    print("="*70)

    avg_oos_return = df_results['out_sample_return'].mean()
    avg_oos_sharpe = df_results['out_sample_sharpe'].mean()
    avg_oos_drawdown = df_results['out_sample_drawdown'].mean()
    avg_oos_trades = df_results['out_sample_trades'].mean()
    avg_oos_win_rate = df_results['out_sample_win_rate'].mean()
    avg_oos_pf = df_results['out_sample_profit_factor'].mean()
    avg_degradation = df_results['return_degradation_pct'].mean()

    print(f"\nAverage Out-of-Sample Performance:")
    print(f"  Return:          {avg_oos_return:7.2f}%")
    print(f"  Sharpe Ratio:    {avg_oos_sharpe:7.3f}")
    print(f"  Max Drawdown:    {avg_oos_drawdown:7.2f}%")
    print(f"  Avg Trades:      {avg_oos_trades:7.1f}")
    print(f"  Win Rate:        {avg_oos_win_rate:7.1f}%")
    print(f"  Profit Factor:   {avg_oos_pf:7.2f}")
    print(f"\nAverage Return Degradation: {avg_degradation:+.1f}%")

    if abs(avg_degradation) > 30:
        print("\n⚠️  WARNING: Average degradation > 30% indicates potential overfitting")
    elif abs(avg_degradation) < 15:
        print("\n✓ Good: Low degradation indicates robust parameters")
    else:
        print("\n→ Moderate degradation - parameters are reasonable")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    test_period_months = 6
    annualized_return = avg_oos_return * (12 / test_period_months)

    print(f"\nExpected Performance (based on out-of-sample results):")
    print(f"  Per Test Period ({test_period_months} months): ~{avg_oos_return:.2f}%")
    print(f"  Annualized (estimated):       ~{annualized_return:.2f}%")
    print(f"  Sharpe Ratio:                  {avg_oos_sharpe:.2f}")
    print(f"  Max Drawdown:                  {avg_oos_drawdown:.2f}%")

    if avg_oos_return > 0 and avg_oos_sharpe > 0.5:
        print(f"\n✓ Strategy shows positive risk-adjusted returns in out-of-sample testing")
    elif avg_oos_return > 0:
        print(f"\n→ Strategy shows positive returns but with moderate risk-adjusted performance")
    else:
        print(f"\n⚠️  Strategy shows negative returns in out-of-sample testing")

    print("\n⚠️  Past performance does not guarantee future results")
    print("="*70 + "\n")

    # Parameter summary
    print("\n" + "="*70)
    print("OPTIMAL PARAMETERS SUMMARY")
    print("="*70)

    most_recent = df_results.iloc[-1]
    params = most_recent['params']

    print(f"\n📋 BEST PARAMETERS (Period {int(most_recent['period'])}):")
    print("="*70)

    print("\n🔹 ML SETTINGS:")
    print(f"   neighbors_count:          {params['neighbors_count']}")
    print(f"   max_bars_back:            {params['max_bars_back']}")
    print(f"   feature_count:            {params['feature_count']}")
    print(f"   trend_following_labels:   {params['trend_following_labels']}")
    print(f"   allow_reentry:            {params['allow_reentry']}")
    print(f"   min_prediction_strength:  {params['min_prediction_strength']}")

    print("\n🔹 TREND FEATURES:")
    print(f"   f1_type:                  {params['f1_type']} (Momentum)")
    print(f"   f1_param_a:               {params['f1_param_a']}")
    print(f"   f2_type:                  {params['f2_type']} (Trend Strength)")
    print(f"   f2_param_a:               {params['f2_param_a']}")
    print(f"   f3_type:                  {params['f3_type']} (Normalized Volatility)")
    print(f"   f3_param_a:               {params['f3_param_a']}")
    print(f"   f4_type:                  {params['f4_type']} (Price Position)")
    print(f"   f4_param_a:               {params['f4_param_a']}")
    print(f"   f5_type:                  {params['f5_type']} (Efficiency Ratio)")
    print(f"   f5_param_a:               {params['f5_param_a']}")

    print("\n🔹 FILTERS:")
    print(f"   use_volatility_filter:    {params['use_volatility_filter']}")
    print(f"   use_regime_filter:        {params['use_regime_filter']}")
    print(f"   regime_threshold:         {params['regime_threshold']}")
    print(f"   use_kernel_filter:        {params['use_kernel_filter']}")
    print(f"   kernel_lookback:          {params['kernel_lookback']}")

    print("\n🔹 EXIT SETTINGS:")
    print(f"   use_dynamic_exits:        {params['use_dynamic_exits']}")
    print(f"   bars_to_hold:             {params['bars_to_hold']}")
    print(f"   use_rsi_exit:             {params['use_rsi_exit']}")
    print(f"   rsi_exit_period:          {params['rsi_exit_period']}")
    print(f"   rsi_overbought:           {params['rsi_overbought']}")
    print(f"   rsi_oversold:             {params['rsi_oversold']}")
    print(f"   use_kernel_exit:          {params['use_kernel_exit']}")

    print("\n" + "="*70)
    print("COPY-PASTE READY PARAMETER DICT")
    print("="*70)
    print("\nstrategy_params = {")
    print("    # ML Settings")
    print(f"    'neighbors_count': {params['neighbors_count']},")
    print(f"    'max_bars_back': {params['max_bars_back']},")
    print(f"    'feature_count': {params['feature_count']},")
    print(f"    'trend_following_labels': {params['trend_following_labels']},")
    print(f"    'allow_reentry': {params['allow_reentry']},")
    print(f"    'min_prediction_strength': {params['min_prediction_strength']},")
    print("\n    # Feature 1 (RSI - Momentum)")
    print(f"    'f1_type': '{params['f1_type']}',")
    print(f"    'f1_param_a': {params['f1_param_a']},")
    print(f"    'f1_param_b': {params['f1_param_b']},")
    print("\n    # Feature 2 (ADX - Trend Strength)")
    print(f"    'f2_type': '{params['f2_type']}',")
    print(f"    'f2_param_a': {params['f2_param_a']},")
    print(f"    'f2_param_b': {params['f2_param_b']},")
    print("\n    # Feature 3 (ATR Ratio - Volatility)")
    print(f"    'f3_type': '{params['f3_type']}',")
    print(f"    'f3_param_a': {params['f3_param_a']},")
    print(f"    'f3_param_b': {params['f3_param_b']},")
    print("\n    # Feature 4 (Price Position)")
    print(f"    'f4_type': '{params['f4_type']}',")
    print(f"    'f4_param_a': {params['f4_param_a']},")
    print(f"    'f4_param_b': {params['f4_param_b']},")
    print("\n    # Feature 5 (Efficiency Ratio)")
    print(f"    'f5_type': '{params['f5_type']}',")
    print(f"    'f5_param_a': {params['f5_param_a']},")
    print(f"    'f5_param_b': {params['f5_param_b']},")
    print("\n    # Filters")
    print(f"    'use_volatility_filter': {params['use_volatility_filter']},")
    print(f"    'use_regime_filter': {params['use_regime_filter']},")
    print(f"    'regime_threshold': Decimal('{params['regime_threshold']}'),")
    print(f"    'use_adx_filter': {params['use_adx_filter']},")
    print(f"    'adx_threshold': {params['adx_threshold']},")
    print(f"    'use_ema_filter': {params['use_ema_filter']},")
    print(f"    'ema_period': {params['ema_period']},")
    print(f"    'use_sma_filter': {params['use_sma_filter']},")
    print(f"    'sma_period': {params['sma_period']},")
    print("\n    # Kernel Settings")
    print(f"    'use_kernel_filter': {params['use_kernel_filter']},")
    print(f"    'use_kernel_smoothing': {params['use_kernel_smoothing']},")
    print(f"    'kernel_lookback': {params['kernel_lookback']},")
    print(f"    'kernel_rel_weight': {params['kernel_rel_weight']},")
    print(f"    'kernel_start_bar': {params['kernel_start_bar']},")
    print(f"    'kernel_lag': {params['kernel_lag']},")
    print("\n    # Exit Settings")
    print(f"    'use_dynamic_exits': {params['use_dynamic_exits']},")
    print(f"    'bars_to_hold': {params['bars_to_hold']},")
    print("\n    # RSI Exit Settings")
    print(f"    'use_rsi_exit': {params['use_rsi_exit']},")
    print(f"    'rsi_exit_period': {params['rsi_exit_period']},")
    print(f"    'rsi_overbought': {params['rsi_overbought']},")
    print(f"    'rsi_oversold': {params['rsi_oversold']},")
    print("\n    # Kernel Exit Settings")
    print(f"    'use_kernel_exit': {params['use_kernel_exit']},")
    print("\n    # Risk Management")
    print(f"    'position_size_pct': Decimal('{params['position_size_pct']}'),")
    print(f"    'stop_loss_pct': Decimal('{params['stop_loss_pct']}'),")
    print(f"    'use_stop_loss': {params['use_stop_loss']},")
    print(f"    'long_only': {params['long_only']},")
    print("\n    # Other")
    print(f"    'verbose': False,")
    print(f"    'test_start_idx': {params['test_start_idx']},")
    print("}")
    print("\n" + "="*70 + "\n")


def main():
    """Main execution."""
    results_df = run_walkforward_optimization(CONFIG)

    if results_df.empty:
        print("\n❌ No results obtained")
        return

    print_walkforward_results(results_df)

    results_df.to_csv(CONFIG['results_file'], index=False)
    print(f"📄 Results saved to '{CONFIG['results_file']}'")

    print("\n✅ Walk-forward optimization complete!\n")


if __name__ == "__main__":
    main()
