"""
Walk-Forward Optimization for Lorentzian Classification Strategy - Trend-Following Features

This script implements proper walk-forward analysis with:
- Rolling optimization windows (train/test splits)
- Out-of-sample validation on unseen data
- Aggregation across multiple forward periods
- COVID period handling options

Uses trend-following feature vector:
- RSM(40,252): Relative Strength Momentum
- ER(25,13): Efficiency Ratio (Trend Quality)
- MTD(8,252): Multi-Timeframe Divergence
- STRK(30,3): Streak Pattern
- VCOMP(4,16): Volatility Compression
- MPER(4,20): Momentum Persistence
- VMC(3,40): Volume-Momentum Coupling
- CS(5,2): Candle Structure

Usage:
    python optimize_strategy_trend.py
"""

from decimal import Decimal
import os
import matplotlib
matplotlib.use('Agg')
import backtrader as bt
import yfinance as yf
from lorentzian_classification import LorentzianClassificationStrategy
import pandas as pd
import numpy as np
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timedelta
import sys
from portfolio_simulator import simulate_portfolio, print_portfolio_summary, print_trade_log_summary, plot_portfolio


# Load peer universe from classification CSV (same as backtest.py/backtest_multi.py)
_peer_universe = 'SPY,QQQ,IWM,TLT,GLD,XLE,EFA'
try:
    with open('../classification_set.csv') as _f:
        _symbols = [line.strip() for line in _f if line.strip()]
        if _symbols:
            _peer_universe = ','.join(_symbols)
except FileNotFoundError:
    pass


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Input/Output
    'csv_file': '../sp500_2024.csv',
    'results_file': 'walkforward_optimization_results_trend.csv',

    # Walk-Forward Settings
    'train_period_months': 6,     # Optimize on 18 months
    'test_period_months': 12,       # Validate on next 9 months
    'step_months': 6,              # Roll forward 9 months each iteration
    'total_periods': 3,            # Number of train/test cycles

    # Date range control (optional - leave None for automatic)
    'end_date': None,               # None = use most recent data

    # COVID handling
    'exclude_covid': True,          # Skip COVID period in training
    'covid_start': '2020-02-01',    # COVID crash start
    'covid_end': '2021-06-30',      # Recovery period end

    # Backtest settings
    'initial_cash': 10000,
    'commission': 0.0,           # 0.05% commission per trade

    # Optimization settings
    'top_n_results': 10,
    'print_progress_every': 50,
    'max_workers': None,  # None = use all CPU cores

    # Portfolio simulation settings (list values for optimization, same pattern as param_grid)
    'portfolio_config': {
        'portfolio_capital': [100_000],
        'max_positions': [10],    # Lower = better utilization; 10 mirrors backtest_multi.py
        'kelly_fraction': [0.0],      # 0.0=equal weight (more stable with short IS periods)
        'use_position_replacement': [True],    # Replace worst positions when better candidates appear
        'replacement_threshold': [0.4],       # Min score advantage to trigger replacement (higher=fewer swaps)
        'replacement_cooldown_days': [10],     # Min days before position eligible for replacement
        'replacement_max_per_day': [2],        # Max replacements per calendar day
        'prediction_validation_bars': [8],     # Force-exit if ML wrong after N bars (0=off, matches label_lookahead)
        'prediction_validation_threshold': [-0.5],  # Force-exit if P&L% below this after validation bars
        # Entry mode: 'confirmation'=wait for up day (validates mean-reversion bounce),
        # 'dip'=wait for down day (validates momentum pullback), 'market'=immediate,
        # 'dip_or_confirmation'=fill on either condition
        'entry_mode': ['market'],
        'entry_limit_window': [1],             # Bars to wait for limit fill (ignored for 'market')
        'entry_dip_threshold': [-0.5],         # Daily return % to qualify as dip fill
        'entry_confirmation_threshold': [0.3], # Daily return % to qualify as confirmation fill
        'portfolio_debug': [False],   # Set True to enable detailed trade logging
    },

    # Ranking metric: 'composite_score' for trading performance, 'ml_accuracy' for ML prediction accuracy
    # Use 'ml_accuracy' when optimizing ML parameters (features, labels, neighbors) to avoid overfitting
    # to specific price movements. Trading filters/exits should use 'composite_score'.
    'ranking_metric': 'composite_score',

    # Stock Qualification Filter (uses IS per-stock metrics to filter OOS stocks)
    'use_stock_qualification': True,
    'stock_qual_min_profit_factor': 0.0,   # Min profit factor per stock in training
    'stock_qual_min_win_rate': 0.0,       # Min win rate % per stock in training
    'stock_qual_min_trades': 0,            # Min trades per stock in training (avoid flukes)
    'stock_qual_min_return_pct': 0.0,      # Min return % per stock in training
    'stock_qual_min_ml_bullish_accuracy': 50.0,  # Min ML bullish prediction accuracy % (long-only relevant)

    # Quality filters - RELAXED for walk-forward (short periods, will average out)
    'quality_filters': {
        'min_sharpe': -0.5,
        'min_calmar': -1.0,
        'min_win_rate': 20.0,
        'min_rr_ratio': 0.5,
        'min_profit_factor': 0.8,
        'min_total_trades': 10,
        'max_drawdown': 80.0,
        'min_stock_win_rate': 20.0,
        'min_expectancy': -50.0,
    },

    # Parameter grid for Lorentzian Classification - Trend-Following Features
    'param_grid': {
        # ==================== ML SETTINGS ====================
        'neighbors_count': [10],
        'max_bars_back': [7000],            # Keep fixed - needs lots of history
        'feature_count': [8],
        'trend_following_labels': [True],   # True=trend-following labels (ride continuation)
        'allow_reentry': [True],            # True=enter anytime signal favorable
        'min_prediction_strength': [20],    # Normalized scale: 0-100
        'min_raw_prediction': [0.0],  # Min expected return in ATR units (0=disabled)

        # ==================== LABEL SETTINGS ====================
        # Longer lookahead for trends — they unfold over more bars than reversions
        'label_lookahead': [8],       # Bars to look forward: 8=captures trend moves
        'label_dead_zone': [0.225],   # Min ATR move for label: higher=cleaner trend labels
        'use_magnitude_labels': [True],

        # ==================== FEATURE 1 (RSM - Relative Strength Momentum) ====================
        # Core trend feature — where is this stock's momentum ranked historically?
        'f1_type': ['RSM'],
        'f1_param_a': [40],          # Momentum period: 40=two months of momentum
        'f1_param_b': [252],         # Lookback: 252=full year percentile ranking

        # ==================== FEATURE 2 (ER - Efficiency Ratio) ====================
        # Trend quality — is price moving directionally or chopping around?
        'f2_type': ['ER'],
        'f2_param_a': [25],          # ER period: trend quality over ~1 month
        'f2_param_b': [13],

        # ==================== FEATURE 3 (MTD - Multi-Timeframe Divergence) ====================
        # Are short and long timeframes aligned? Alignment = strong trend
        'f3_type': ['MTD'],
        'f3_param_a': [8],           # Short ROC period
        'f3_param_b': [252],         # Long ROC period

        # ==================== FEATURE 4 (STRK - Streak Pattern) ====================
        # Consecutive directional moves = trend persistence / serial correlation
        'f4_type': ['STRK'],
        'f4_param_a': [30],          # Max streak length
        'f4_param_b': [3],           # ATR multiplier for magnitude

        # ==================== FEATURE 5 (VCOMP - Volatility Compression) ====================
        # Consolidation after a big move — the coiled spring before breakout
        'f5_type': ['VCOMP'],
        'f5_param_a': [4],           # Recent volatility window
        'f5_param_b': [16],          # Lookback volatility window

        # ==================== FEATURE 6 (MPER - Momentum Persistence) ====================
        # Pullback within intact trend vs reversal
        'f6_type': ['MPER'],
        'f6_param_a': [4],           # Short momentum period
        'f6_param_b': [20],          # Medium momentum period

        # ==================== FEATURE 7 (VMC - Volume-Momentum Coupling) ====================
        # Volume still engaged during consolidation — smart money signal
        'f7_type': ['VMC'],
        'f7_param_a': [3],           # Recent volume/momentum window
        'f7_param_b': [40],          # Baseline volume average period

        # ==================== FEATURE 8 (CS - Candle Structure) ====================
        # Trend-quality candles (strong bodies, small wicks)
        'f8_type': ['CS'],
        'f8_param_a': [5],           # Averaging window
        'f8_param_b': [2],           # Sensitivity (tanh scaling)

        # ==================== FEATURES 9-14 (REDUNDANT - below feature_count cutoff) ===========
        'f9_type': ['CS'],
        'f9_param_a': [5],
        'f9_param_b': [2],

        'f10_type': ['CS'],
        'f10_param_a': [5],
        'f10_param_b': [2],

        'f11_type': ['CS'],
        'f11_param_a': [5],
        'f11_param_b': [2],

        'f12_type': ['VCOMP'],
        'f12_param_a': [4],
        'f12_param_b': [16],

        'f13_type': ['MPER'],
        'f13_param_a': [4],
        'f13_param_b': [20],

        'f14_type': ['VMC'],
        'f14_param_a': [5],
        'f14_param_b': [40],

        # ==================== FILTERS ====================
        'use_volatility_filter': [True],
        'use_regime_filter': [True],
        'regime_threshold': [Decimal('0')],  # 1=require bullish (trend-aligned)
        'regime_period': ['weekly'],
        'use_regime_direction': [True],
        'regime_stability_min': [0.0],
        'regime_stability_window': [60],
        'regime_max_flips': [8],
        'use_adx_filter': [True],
        'adx_threshold': [14],
        'use_ema_filter': [False],
        'ema_period': [400],
        'ema_slope_lookback': [20],
        'use_sma_filter': [False],
        'sma_period': [400],
        'sma_slope_lookback': [20],

        # ==================== SPY MARKET REGIME FILTER ====================
        'use_spy_filter': [True],
        'spy_regime_threshold': [Decimal('0')],  # 0=block bearish
        'spy_regime_period': ['monthly'],

        # ==================== KERNEL SETTINGS ====================
        'use_kernel_filter': [False],
        'use_kernel_smoothing': [False],
        'kernel_lookback': [8],
        'kernel_rel_weight': [8.0],
        'kernel_start_bar': [25],
        'kernel_lag': [2],

        # ==================== EXIT SETTINGS ====================
        'use_dynamic_exits': [False],
        'bars_to_hold': [100000],

        # ==================== RSI EXIT SETTINGS ====================
        # Widened thresholds — RSI overbought exits cut winning trends short
        'use_rsi_exit': [False],
        'rsi_exit_period': [14],
        'rsi_overbought': [80],             # Widened threshold (less likely to trigger)
        'rsi_oversold': [20],

        # ==================== KERNEL EXIT SETTINGS ====================
        'use_kernel_exit': [False],

        # ==================== PREDICTION VALIDATION EXIT ====================
        'use_prediction_exit': [True],
        'prediction_exit_threshold': [-2.0],  # Wider threshold — trends need more time

        # ==================== CHANDELIER EXIT (ATR TRAILING STOP) ====================
        # Wider settings give trends room to breathe through normal pullbacks
        'use_trailing_atr_exit': [True],
        'trailing_atr_warmup': [5],
        'chandelier_start_mult': [3.5],        # Wider than mean-reversion (3.0)
        'chandelier_end_mult': [1],          # Wider final stop (mean-rev uses 1.5)
        'chandelier_tighten_bars': [40],       # Slower tightening (mean-rev uses 20)
        'chandelier_mult_mode': ['blend'],
        'chandelier_profit_atr_threshold': [1.5],
        'chandelier_breakeven_atr': [0.0],    # Requires more profit before locking break-even

        # ==================== RISK MANAGEMENT ====================
        'position_size_pct': [Decimal('0.95')],
        'stop_loss_pct': [Decimal('0.05')],
        'use_stop_loss': [False],
        'long_only': [True],

        # ==================== OTHER ====================
        'verbose': [False],
        'test_start_idx': [0],              # Bar index to start trading

        # ==================== CROSS-SYMBOL TRAINING ====================
        'use_cross_symbol_training': [True],
        'cross_symbol_etfs': ['SPY,QQQ,IWM,TLT,GLD,XLE,EFA'],
        'cross_symbol_lookback_years': [5],
        'use_regime_balancing': [False],
        'cross_symbol_auto_peers': [True],
        'cross_symbol_target_symbol': [''],
        'cross_symbol_max_peers': [7],

        # ==================== FUNDAMENTAL DATA FILTER ====================
        'use_fundamental_filter': [True],
        'fundamental_symbol': [''],
        'fundamental_quality_weight': [0.2],
        'fundamental_momentum_weight': [0.3],
        'earnings_blackout_before': [5],
        'earnings_blackout_after': [2],
        'close_before_earnings': [True],
        'full_position_threshold': [50],
        'reduced_position_pct': [Decimal('0.75')],
        'min_trending_probability': [20],
        'min_quality_score': [20],
        'min_momentum_score': [20],
        'use_earnings_reaction': [True],
        'earnings_reaction_days': [5],
        'min_earnings_reaction_pct': [-5.0],
    }
}


# ============================================================================
# Walk-Forward Period Generation
# ============================================================================

def generate_walkforward_periods(config):
    """
    Generate train/test period pairs for walk-forward analysis.
    Works backwards from end_date so periods are as recent as possible.
    ML training lookback is handled separately by load_symbol_data (lookback_bars).
    """
    periods = []

    if config.get('end_date'):
        end_date = pd.to_datetime(config['end_date']).to_pydatetime()
    else:
        end_date = datetime.now() - timedelta(days=7)

    train_months = config['train_period_months']
    test_months = config['test_period_months']
    step_months = config['step_months']
    total_periods = config['total_periods']

    # Total span from first train_start to last test_end
    total_span_months = train_months + (total_periods - 1) * step_months + test_months

    # Place first train_start so that the last test_end lands at end_date
    first_train_start = end_date - timedelta(days=total_span_months * 30)

    print(f"\n📅 Generating Walk-Forward Periods")
    print(f"   End date: {end_date.strftime('%Y-%m-%d')}")
    print(f"   Period span: {total_span_months} months ({first_train_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
    print(f"   ML lookback: handled by max_bars_back ({max(config['param_grid'].get('max_bars_back', [500]))} bars before each period)")

    covid_start = pd.to_datetime(config['covid_start']) if config['exclude_covid'] else None
    covid_end = pd.to_datetime(config['covid_end']) if config['exclude_covid'] else None

    current_train_start = first_train_start
    periods_kept = 0
    max_attempts = total_periods * 3  # allow extra attempts for COVID skips

    for attempt in range(max_attempts):
        if periods_kept >= total_periods:
            break

        train_start = current_train_start
        train_end = train_start + timedelta(days=train_months * 30)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_months * 30)

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
            # Push start back further to get past COVID
            current_train_start -= timedelta(days=step_months * 30)
            # Also push all remaining periods back by recalculating
            first_train_start = current_train_start
        else:
            periods.append((
                train_start.strftime('%Y-%m-%d'),
                train_end.strftime('%Y-%m-%d'),
                test_start.strftime('%Y-%m-%d'),
                test_end.strftime('%Y-%m-%d')
            ))
            periods_kept += 1
            print(f"   ✓ Period {periods_kept}: Train {train_start.strftime('%Y-%m-%d')}-{train_end.strftime('%Y-%m-%d')}, Test {test_start.strftime('%Y-%m-%d')}-{test_end.strftime('%Y-%m-%d')}")
            current_train_start += timedelta(days=step_months * 30)

    if not periods:
        print("\n⚠️  ERROR: No valid periods generated!")
        print("   Try one of these:")
        print("   1. Set 'exclude_covid': False")
        print("   2. Reduce 'total_periods'")
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

def load_symbol_data(symbol, start_date, end_date, lookback_bars=0):
    """
    Download data for a specific date range, optionally with extra lookback
    data before start_date for ML warmup.

    Returns:
        tuple: (df, test_start_idx) where test_start_idx is the bar index
               where the actual period begins (after lookback warmup).
               Returns (None, 0) on failure.
    """
    try:
        # Calculate how far back to go for lookback data
        if lookback_bars > 0:
            lookback_calendar_days = int(lookback_bars * 1.5) + 10
            data_start = pd.to_datetime(start_date) - timedelta(days=lookback_calendar_days)
        else:
            data_start = pd.to_datetime(start_date) - timedelta(days=5)

        end = pd.to_datetime(end_date) + timedelta(days=5)

        df = yf.download(symbol, start=data_start, end=end, progress=False)

        if df.empty:
            return None, 0

        df.index = df.index.tz_localize(None)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']

        # Trim end to the period boundary
        df = df[df.index <= pd.to_datetime(end_date)]

        # Find where the actual period starts (after lookback)
        period_mask = df.index >= pd.to_datetime(start_date)
        if period_mask.any():
            test_start_idx = int(period_mask.argmax())
        else:
            return None, 0

        # Need enough bars in the actual period
        period_bars = len(df) - test_start_idx
        if period_bars < 50:
            return None, 0

        return df, test_start_idx

    except Exception:
        return None, 0


def _bulk_download(symbols, start, end):
    """Download data for many symbols in one yfinance call, return dict of {symbol: df}."""
    try:
        raw = yf.download(symbols, start=start, end=end, progress=False,
                          group_by='ticker', threads=True)
        if raw.empty:
            return {}
        result = {}
        for symbol in symbols:
            try:
                if len(symbols) == 1:
                    df = raw.copy()
                else:
                    df = raw[symbol].copy()
                df = df.dropna(subset=['Close'])
                if df.empty:
                    continue
                df.index = df.index.tz_localize(None)
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                result[symbol] = df
            except Exception:
                continue
        return result
    except Exception:
        return {}


def load_all_periods_data(symbols, periods, lookback_bars=600):
    """Load data for all symbols across all periods, with ML lookback warmup.

    Uses bulk yfinance downloads (one call per period) for much faster I/O.
    """
    data_cache = {}

    print(f"\n📥 Downloading data for {len(symbols)} stocks across {len(periods)} periods...")
    print(f"   ML lookback: {lookback_bars} bars before each period for training data warmup")

    lookback_calendar_days = int(lookback_bars * 1.5) + 10

    for period_idx, (train_start, train_end, test_start, test_end) in enumerate(periods):
        # Compute the widest date range needed for this period (train lookback to test end)
        bulk_start = pd.to_datetime(train_start) - timedelta(days=lookback_calendar_days)
        bulk_end = pd.to_datetime(test_end) + timedelta(days=5)

        print(f"   Period {period_idx + 1}/{len(periods)}: bulk download "
              f"{bulk_start.strftime('%Y-%m-%d')} to {bulk_end.strftime('%Y-%m-%d')} "
              f"({len(symbols)} symbols)...", end=' ', flush=True)

        bulk_data = _bulk_download(symbols, bulk_start, bulk_end)
        print(f"got {len(bulk_data)} symbols")

        for symbol in symbols:
            if symbol not in data_cache:
                data_cache[symbol] = {}

            full_df = bulk_data.get(symbol)
            if full_df is None or full_df.empty:
                continue

            # Slice train period (with lookback)
            train_start_dt = pd.to_datetime(train_start)
            train_end_dt = pd.to_datetime(train_end)
            train_lookback_start = train_start_dt - timedelta(days=lookback_calendar_days)
            train_df = full_df[(full_df.index >= train_lookback_start) & (full_df.index <= train_end_dt)].copy()

            # Slice test period (with lookback)
            test_start_dt = pd.to_datetime(test_start)
            test_end_dt = pd.to_datetime(test_end)
            test_lookback_start = test_start_dt - timedelta(days=lookback_calendar_days)
            test_df = full_df[(full_df.index >= test_lookback_start) & (full_df.index <= test_end_dt)].copy()

            # Find period start indices
            train_mask = train_df.index >= train_start_dt
            test_mask = test_df.index >= test_start_dt

            if not train_mask.any() or not test_mask.any():
                continue

            train_tsi = int(train_mask.argmax())
            test_tsi = int(test_mask.argmax())

            # Need enough bars in actual period
            if (len(train_df) - train_tsi) < 50 or (len(test_df) - test_tsi) < 50:
                continue

            data_cache[symbol][period_idx] = {
                'train': train_df,
                'train_test_start_idx': train_tsi,
                'test': test_df,
                'test_test_start_idx': test_tsi,
            }

    valid_symbols = [s for s in symbols if len(data_cache.get(s, {})) >= len(periods) * 0.7]

    print(f"✓ Valid symbols with sufficient data: {len(valid_symbols)}/{len(symbols)}\n")

    return data_cache, valid_symbols


# ============================================================================
# Core Backtesting
# ============================================================================

def backtest_single_config(symbol, params, df, initial_cash, commission, capture_equity=False):
    """Run backtest for a single symbol with given parameters.

    When capture_equity=True, also captures daily equity curve and trade log
    for portfolio simulation (adds TradeRecorder analyzer).
    """
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
        cerebro.broker.set_coc(True)  # Fill signals at bar close (consistent with backtest_multi.py)

        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                            timeframe=bt.TimeFrame.Days, riskfreerate=0.0)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")

        if capture_equity:
            cerebro.addanalyzer(TradeRecorder, _name="trade_recorder")

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

        # Extract ML prediction accuracy
        try:
            ml_stats = strat.get_prediction_stats()
            ml_accuracy = ml_stats.get('accuracy_pct', 0)
            ml_total_predictions = ml_stats.get('total', 0)
            ml_bullish_accuracy = ml_stats.get('bullish_accuracy_pct', 0)
            ml_bearish_accuracy = ml_stats.get('bearish_accuracy_pct', 0)
        except Exception:
            ml_accuracy = ml_total_predictions = 0
            ml_bullish_accuracy = ml_bearish_accuracy = 0

        result = {
            'symbol': symbol,
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
            'ml_accuracy': ml_accuracy,
            'ml_total_predictions': ml_total_predictions,
            'ml_bullish_accuracy': ml_bullish_accuracy,
            'ml_bearish_accuracy': ml_bearish_accuracy,
        }

        if capture_equity:
            trade_analysis = strat.analyzers.trade_recorder.get_analysis()
            all_dates = trade_analysis['daily_dates']
            all_values = trade_analysis['daily_values']

            # Filter to test period only (by date, not index)
            test_start_idx = params.get('test_start_idx', 0)
            if test_start_idx > 0 and test_start_idx < len(df):
                test_start_date = df.index[test_start_idx].date()
            else:
                test_start_date = df.index[0].date()

            daily_dates = []
            daily_values = []
            for d, v in zip(all_dates, all_values):
                if d >= test_start_date:
                    daily_dates.append(d)
                    daily_values.append(v)

            # Sanity check: equity curve return should approximately match backtest return
            if len(daily_values) >= 2 and daily_values[0] > 0:
                equity_ret = (daily_values[-1] / daily_values[0] - 1) * 100
                if abs(equity_ret - return_pct) > 10:
                    print(f"  ⚠ {symbol}: equity curve return {equity_ret:.2f}% vs backtest {return_pct:.2f}% "
                          f"(bars: {len(all_dates)} total, {len(daily_dates)} test, "
                          f"start={daily_values[0]:.0f}, end={daily_values[-1]:.0f})")

            result['_equity'] = {
                'symbol': symbol,
                'dates': daily_dates,
                'values': daily_values,
                'initial_value': initial_cash,
                'trade_log': trade_analysis.get('trade_log', []),
            }

        return result

    except Exception as e:
        return None


class PortfolioValue(bt.Observer):
    """Observer to track portfolio value over time."""
    lines = ('value',)
    plotinfo = dict(plot=False, subplot=False)

    def next(self):
        self.lines.value[0] = self._owner.broker.getvalue()

    def prenext(self):
        self.lines.value[0] = self._owner.broker.getvalue()


class TradeRecorder(bt.Analyzer):
    """Records entry/exit dates, daily equity values for portfolio simulation.

    Records broker value at every bar in plain Python lists (no backtrader
    line buffer), ensuring reliable date-value alignment for equity extraction.
    """

    def start(self):
        self.trade_log = []
        self._open_trade_entry = None
        self._open_trade_price = None
        self._open_trade_size = None
        self.daily_dates = []
        self.daily_values = []

    def prenext(self):
        self._record_daily()

    def next(self):
        self._record_daily()

    def _record_daily(self):
        self.daily_dates.append(self.data.datetime.date(0))
        self.daily_values.append(self.strategy.broker.getvalue())

    def notify_trade(self, trade):
        if trade.justopened:
            self._open_trade_entry = bt.num2date(trade.dtopen).date()
            self._open_trade_price = trade.price
            self._open_trade_size = trade.size

        if trade.isclosed:
            entry = self._open_trade_entry or bt.num2date(trade.dtopen).date()
            self.trade_log.append({
                'entry_date': entry,
                'exit_date': bt.num2date(trade.dtclose).date(),
                'entry_price': self._open_trade_price,
                'exit_price': trade.price,
                'size': self._open_trade_size,
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm,
            })
            self._open_trade_entry = None
            self._open_trade_price = None
            self._open_trade_size = None

    def stop(self):
        if self._open_trade_entry is not None:
            self.trade_log.append({
                'entry_date': self._open_trade_entry,
                'exit_date': self.data.datetime.date(0),
                'entry_price': self._open_trade_price,
                'size': self._open_trade_size,
                'still_open': True,
            })

    def get_analysis(self):
        return {
            'trade_log': self.trade_log,
            'daily_dates': self.daily_dates,
            'daily_values': self.daily_values,
        }


def aggregate_results(stock_results, params):
    """Aggregate results across all stocks for a parameter set."""
    if not stock_results:
        return None

    n_stocks = len(stock_results)

    sum_return = sum(r['return_pct'] for r in stock_results)
    avg_return = sum_return / n_stocks
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
    stocks_traded = sum(1 for r in stock_results if r['return_pct'] != 0)
    stock_win_rate = (winning_stocks / n_stocks) * 100

    if avg_sharpe > 0 and avg_return > 0 and avg_drawdown > 0:
        composite_score = (avg_return * avg_sharpe * (stock_win_rate / 100)) / avg_drawdown
    else:
        composite_score = 0

    # ML prediction accuracy aggregation
    avg_ml_accuracy = sum(r.get('ml_accuracy', 0) for r in stock_results) / n_stocks
    total_ml_predictions = sum(r.get('ml_total_predictions', 0) for r in stock_results)
    avg_ml_bullish_accuracy = sum(r.get('ml_bullish_accuracy', 0) for r in stock_results) / n_stocks
    avg_ml_bearish_accuracy = sum(r.get('ml_bearish_accuracy', 0) for r in stock_results) / n_stocks

    result = {
        'sum_return_pct': sum_return,
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
        'stocks_traded': stocks_traded,
        'winning_stocks': winning_stocks,
        'stock_win_rate': stock_win_rate,
        'composite_score': composite_score,
        'ml_accuracy': avg_ml_accuracy,
        'total_ml_predictions': total_ml_predictions,
        'ml_bullish_accuracy': avg_ml_bullish_accuracy,
        'ml_bearish_accuracy': avg_ml_bearish_accuracy,
    }

    return result


# ============================================================================
# Walk-Forward Optimization Logic
# ============================================================================

# Define parameter names for the strategy (must match param_grid order exactly)
PARAM_NAMES = [
    # ML Settings
    'neighbors_count', 'max_bars_back', 'feature_count',
    'trend_following_labels', 'allow_reentry', 'min_prediction_strength', 'min_raw_prediction',
    # Label Settings
    'label_lookahead', 'label_dead_zone', 'use_magnitude_labels',
    # Features
    'f1_type', 'f1_param_a', 'f1_param_b',
    'f2_type', 'f2_param_a', 'f2_param_b',
    'f3_type', 'f3_param_a', 'f3_param_b',
    'f4_type', 'f4_param_a', 'f4_param_b',
    'f5_type', 'f5_param_a', 'f5_param_b',
    'f6_type', 'f6_param_a', 'f6_param_b',
    'f7_type', 'f7_param_a', 'f7_param_b',
    'f8_type', 'f8_param_a', 'f8_param_b',
    'f9_type', 'f9_param_a', 'f9_param_b',
    'f10_type', 'f10_param_a', 'f10_param_b',
    'f11_type', 'f11_param_a', 'f11_param_b',
    'f12_type', 'f12_param_a', 'f12_param_b',
    'f13_type', 'f13_param_a', 'f13_param_b',
    'f14_type', 'f14_param_a', 'f14_param_b',
    # Filters
    'use_volatility_filter', 'use_regime_filter', 'regime_threshold', 'regime_period',
    'use_regime_direction', 'regime_stability_min', 'regime_stability_window', 'regime_max_flips',
    'use_adx_filter', 'adx_threshold', 'use_ema_filter', 'ema_period', 'ema_slope_lookback',
    'use_sma_filter', 'sma_period', 'sma_slope_lookback',
    # SPY Market Regime Filter
    'use_spy_filter', 'spy_regime_threshold', 'spy_regime_period',
    # Kernel Settings
    'use_kernel_filter', 'use_kernel_smoothing', 'kernel_lookback',
    'kernel_rel_weight', 'kernel_start_bar', 'kernel_lag',
    # Exit Settings
    'use_dynamic_exits', 'bars_to_hold',
    # RSI Exit Settings
    'use_rsi_exit', 'rsi_exit_period', 'rsi_overbought', 'rsi_oversold',
    # Kernel Exit Settings
    'use_kernel_exit',
    # Prediction Validation Exit
    'use_prediction_exit', 'prediction_exit_threshold',
    # Chandelier Exit (ATR Trailing Stop)
    'use_trailing_atr_exit', 'trailing_atr_warmup',
    'chandelier_start_mult', 'chandelier_end_mult',
    'chandelier_tighten_bars', 'chandelier_mult_mode',
    'chandelier_profit_atr_threshold', 'chandelier_breakeven_atr',
    # Risk Management
    'position_size_pct', 'stop_loss_pct', 'use_stop_loss', 'long_only',
    # Other
    'verbose', 'test_start_idx',
    # Cross-Symbol Training
    'use_cross_symbol_training', 'cross_symbol_etfs',
    'cross_symbol_lookback_years', 'use_regime_balancing',
    'cross_symbol_auto_peers', 'cross_symbol_target_symbol', 'cross_symbol_max_peers',
    # Fundamental Data Filter
    'use_fundamental_filter', 'fundamental_symbol',
    'fundamental_quality_weight', 'fundamental_momentum_weight',
    'earnings_blackout_before', 'earnings_blackout_after',
    'close_before_earnings', 'min_trending_probability',
    'full_position_threshold', 'reduced_position_pct',
    'min_quality_score', 'min_momentum_score',
    'use_earnings_reaction', 'earnings_reaction_days', 'min_earnings_reaction_pct',
]


def optimize_single_period(data_cache, valid_symbols, params_list, period_idx, phase, config, capture_equity=False):
    """Optimize or test on a single period using multiprocessing.

    When capture_equity=True, also captures per-symbol equity curves and trade logs
    for portfolio simulation. Returns (results, equity_data) instead of just results.
    """
    results = {}
    successful_backtests = 0
    equity_data = [] if capture_equity else None

    # --- Build phase: flatten all work items ---
    work_items = []  # list of (params_tuple, params_dict, symbol, run_params, df, cash, commission)
    for params_tuple in params_list:
        if hasattr(params_tuple, '_asdict'):
            params = params_tuple._asdict()
        else:
            params = dict(zip(PARAM_NAMES, params_tuple))

        for symbol in valid_symbols:
            if period_idx not in data_cache.get(symbol, {}):
                continue

            period_data = data_cache[symbol][period_idx]
            df = period_data[phase]
            test_start_idx = period_data.get(f'{phase}_test_start_idx', 0)

            if df is None or len(df) < 50:
                continue

            run_params = dict(params)
            run_params['test_start_idx'] = test_start_idx
            run_params['cross_symbol_target_symbol'] = symbol

            work_items.append((params_tuple, params, symbol, run_params, df,
                               config['initial_cash'], config['commission']))

    total_backtests = len(work_items)
    max_workers = config.get('max_workers', None)

    # Pre-fetch sector info for all symbols in the main process before workers spawn.
    # This warms the on-disk sector cache so subprocesses read from disk instead of
    # all hitting yfinance simultaneously.
    if config['param_grid'].get('cross_symbol_auto_peers', [False])[0]:
        from cross_symbol_preloader import prefetch_sectors
        peer_universe_str = config['param_grid'].get('cross_symbol_etfs', [''])[0]
        peer_universe = [s.strip() for s in peer_universe_str.split(',') if s.strip()]
        all_symbols_to_prefetch = list(set(valid_symbols) | set(peer_universe))
        print(f"   Pre-fetching sector info for {len(all_symbols_to_prefetch)} symbols...")
        prefetch_sectors(all_symbols_to_prefetch)

    print(f"   Running {total_backtests:,} backtests on {phase} data (workers: {max_workers or os.cpu_count()})...")
    print(f"   Parameter combinations: {len(params_list)}")
    print(f"   Symbols: {len(valid_symbols)}")

    # --- Execute phase: submit all work to process pool ---
    # Map: params_tuple -> list of result dicts
    grouped_results = {}
    # Keep a reference to params dict per params_tuple for aggregation
    params_dict_map = {}

    backtest_count = 0
    progress_interval = config['print_progress_every']

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all futures
        future_to_key = {}
        for item in work_items:
            params_tuple, params_dict, symbol, run_params, df, cash, commission = item
            future = executor.submit(backtest_single_config, symbol, run_params, df, cash, commission, capture_equity)
            future_to_key[future] = params_tuple
            params_dict_map[params_tuple] = params_dict

        # Collect results as they complete
        for future in as_completed(future_to_key):
            params_tuple = future_to_key[future]
            try:
                result = future.result()
            except Exception:
                result = None

            if result:
                grouped_results.setdefault(params_tuple, []).append(result)
                successful_backtests += 1

                # Collect equity data for portfolio simulation
                if capture_equity and '_equity' in result:
                    eq = result['_equity']
                    if len(eq.get('dates', [])) > 1:
                        equity_data.append(eq)

            backtest_count += 1
            if backtest_count % progress_interval == 0:
                print(f"      Progress: {backtest_count:,}/{total_backtests:,} ({backtest_count/total_backtests*100:.0f}%) | Successful: {successful_backtests}")

    # --- Aggregate phase: group by params_tuple and aggregate ---
    for params_tuple, stock_results in grouped_results.items():
        params = params_dict_map[params_tuple]
        aggregated = aggregate_results(stock_results, params)
        if aggregated:
            results[params_tuple] = {**aggregated, 'params': params}

    print(f"   ✓ Completed {backtest_count:,} backtests, {successful_backtests} successful")
    print(f"   ✓ Generated results for {len(results)} parameter combinations")

    if len(results) == 0:
        print(f"   ⚠️  WARNING: No valid results! Check if data is available for this period.")

    if capture_equity:
        return results, equity_data, grouped_results
    return results, grouped_results


def run_walkforward_optimization(config):
    """Main walk-forward optimization function."""
    print("\n" + "="*70)
    print("WALK-FORWARD OPTIMIZATION - LORENTZIAN CLASSIFICATION (TREND-FOLLOWING FEATURES)")
    print("="*70)

    periods = generate_walkforward_periods(config)

    if not periods:
        print("\n❌ No valid periods generated. Check configuration.")
        sys.exit(1)

    print_walkforward_schedule(periods)

    try:
        with open(config['csv_file'], 'r') as f:
            raw_symbols = [row[0].strip().upper() for row in csv.reader(f) if row and row[0].strip()]
        # Deduplicate while preserving order
        seen = set()
        symbols = []
        for s in raw_symbols:
            if s not in seen:
                seen.add(s)
                symbols.append(s)
        if len(symbols) < len(raw_symbols):
            print(f"  ⚠ Removed {len(raw_symbols) - len(symbols)} duplicate symbols from {config['csv_file']}")
    except FileNotFoundError:
        print(f"\n❌ File '{config['csv_file']}' not found")
        sys.exit(1)

    print(f"\n📋 Configuration")
    print(f"   Symbols: {len(symbols)} stocks")
    print(f"   Walk-forward periods: {len(periods)}")
    print(f"   Train period: {config['train_period_months']} months")
    print(f"   Test period: {config['test_period_months']} months")
    print(f"   COVID exclusion: {'Yes' if config['exclude_covid'] else 'No'}")
    _pcfg = config.get('portfolio_config', {})
    print(f"   Portfolio capital: {_pcfg.get('portfolio_capital', [100_000])}")
    print(f"   Max positions: {_pcfg.get('max_positions', [10])}")
    print(f"   Kelly fraction: {_pcfg.get('kelly_fraction', [0.0])}")

    # Calculate lookback from the largest max_bars_back in the param grid
    max_bars_back_values = config['param_grid'].get('max_bars_back', [500])
    lookback_bars = max(max_bars_back_values) + 100  # buffer for indicator warmup

    data_cache, valid_symbols = load_all_periods_data(symbols, periods, lookback_bars)

    if not valid_symbols:
        print("\n❌ No valid data. Exiting.")
        sys.exit(1)

    # Generate parameter combinations
    param_names = list(config['param_grid'].keys())
    param_values = list(config['param_grid'].values())
    from collections import namedtuple
    ParamSet = namedtuple('ParamSet', param_names)
    params_list = [ParamSet(*combo) for combo in product(*param_values)]

    # Generate portfolio parameter combinations
    portfolio_cfg = config.get('portfolio_config', {
        'portfolio_capital': [100_000], 'max_positions': [10],
        'kelly_fraction': [0.0], 'use_position_replacement': [False],
        'replacement_threshold': [0.1], 'replacement_cooldown_days': [5],
        'replacement_max_per_day': [3], 'prediction_validation_bars': [0],
        'prediction_validation_threshold': [-0.5], 'entry_mode': ['market'],
        'entry_limit_window': [3], 'entry_dip_threshold': [-0.5],
        'entry_confirmation_threshold': [0.5], 'portfolio_debug': [False],
    })
    portfolio_param_names = list(portfolio_cfg.keys())
    portfolio_param_values = list(portfolio_cfg.values())
    PortfolioParamSet = namedtuple('PortfolioParamSet', portfolio_param_names)
    portfolio_params_list = [PortfolioParamSet(*combo) for combo in product(*portfolio_param_values)]

    print(f"\n🔧 Parameter Grid")
    print(f"   Strategy combinations: {len(params_list):,}")
    print(f"   Portfolio combinations: {len(portfolio_params_list):,}")
    print(f"   Total backtests per period: {len(params_list) * len(valid_symbols):,}")

    print(f"\n🚀 Starting Walk-Forward Optimization...\n")

    all_results = []
    all_pf_by_period = []  # accumulates per-period pf_all_results for cross-period winner selection

    for period_idx, period_info in enumerate(periods):
        train_start, train_end, test_start, test_end = period_info

        print(f"\n{'='*70}")
        print(f"PERIOD {period_idx + 1}/{len(periods)}")
        print(f"{'='*70}")
        print(f"Train: {train_start} to {train_end}")
        print(f"Test:  {test_start} to {test_end}")

        print(f"\n1️⃣  OPTIMIZATION PHASE (In-Sample)")
        train_results, train_grouped = optimize_single_period(
            data_cache, valid_symbols, params_list, period_idx, 'train', config
        )

        if not train_results:
            print(f"   ❌ No valid results for training period!")
            continue

        # Apply quality filters to eliminate low-quality / fluke results
        qf = config['quality_filters']
        filtered_results = {
            k: v for k, v in train_results.items()
            if v['total_trades'] >= qf['min_total_trades']
            and v['avg_sharpe'] >= qf['min_sharpe']
            and v['avg_calmar'] >= qf['min_calmar']
            and v['avg_win_rate'] >= qf['min_win_rate']
            and v['avg_rr_ratio'] >= qf['min_rr_ratio']
            and v['avg_profit_factor'] >= qf['min_profit_factor']
            and v['avg_max_drawdown'] <= qf['max_drawdown']
            and v['stock_win_rate'] >= qf['min_stock_win_rate']
            and v['avg_expectancy'] >= qf['min_expectancy']
        }

        if not filtered_results:
            print(f"   ⚠️  No results passed quality filters! Relaxing to min trades only...")
            filtered_results = {
                k: v for k, v in train_results.items()
                if v['total_trades'] >= qf['min_total_trades']
            }

        if not filtered_results:
            print(f"   ❌ No results with minimum trade count ({qf['min_total_trades']})!")
            continue

        print(f"   Filtered: {len(filtered_results)}/{len(train_results)} param sets passed quality filters")

        # Rank by chosen metric
        ranking_metric = config.get('ranking_metric', 'composite_score')
        if ranking_metric == 'ml_accuracy':
            print(f"   Ranking by: ML prediction accuracy (avoids overfitting to price data)")
        else:
            print(f"   Ranking by: {ranking_metric} (trading performance)")
        best_params = max(filtered_results.items(), key=lambda x: x[1][ranking_metric])
        best_params_tuple = best_params[0]
        best_train_perf = best_params[1]

        print(f"\n   ✓ Best In-Sample Parameters Found:")
        print(f"   " + "="*66)

        params_dict = best_train_perf['params']
        print(f"   ML Settings:")
        print(f"     Neighbors (K):         {params_dict['neighbors_count']}")
        print(f"     Feature Count:         {params_dict['feature_count']}")
        print(f"   Trend-Following Features:")
        _fn = {'RSM': 'Rel Strength Mom', 'ER': 'Efficiency Ratio', 'MTD': 'Multi-TF Divergence',
               'STRK': 'Streak', 'VCOMP': 'Vol Compression', 'MPER': 'Mom Persistence',
               'VMC': 'Vol-Mom Coupling', 'CS': 'Candle Struct', 'MACC': 'Mom Accel',
               'OBVT': 'OBV Trend', 'VPD': 'Vol-Price Div', 'CHOP': 'Choppiness',
               'ZS': 'Mean Rev Z-Score', 'VA': 'Volume Anomaly', 'ADX': 'Norm ADX'}
        for _fi in range(1, 15):
            _ft = params_dict.get(f'f{_fi}_type')
            if _ft:
                _pa = params_dict[f'f{_fi}_param_a']
                _pb = params_dict.get(f'f{_fi}_param_b', 1)
                if _pb != 1:
                    print(f"     F{_fi}: {_ft}({_pa},{_pb}) - {_fn.get(_ft, _ft)}")
                else:
                    print(f"     F{_fi}: {_ft}({_pa}) - {_fn.get(_ft, _ft)}")
        print(f"   Filters:")
        print(f"     Volatility:            {'ON' if params_dict['use_volatility_filter'] else 'OFF'}")
        r_thr = params_dict.get('regime_threshold', 0)
        r_desc = "block bearish" if r_thr == 0 else "require bullish" if r_thr >= 1 else f"thr={r_thr}"
        print(f"     Regime:                {'ON' if params_dict['use_regime_filter'] else 'OFF'} (monthly H/L, {r_desc})")
        spy_thr = params_dict.get('spy_regime_threshold', 0)
        spy_desc = "block bearish" if spy_thr == 0 else "require bullish" if spy_thr >= 1 else f"thr={spy_thr}"
        print(f"     SPY Regime:            {'ON' if params_dict.get('use_spy_filter', False) else 'OFF'} ({spy_desc})")
        print(f"     Kernel:                {'ON' if params_dict['use_kernel_filter'] else 'OFF'}")
        print(f"   Exit:")
        print(f"     Bars to Hold:          {params_dict['bars_to_hold']}")

        print(f"\n   In-Sample Performance:")
        print(f"     Total Return (sum):    {best_train_perf['sum_return_pct']:7.2f}%")
        print(f"     Avg Return per Stock:  {best_train_perf['avg_return_pct']:7.2f}%")
        print(f"     Sharpe:                {best_train_perf['avg_sharpe']:7.3f}")
        print(f"     Calmar:                {best_train_perf['avg_calmar']:7.3f}")
        print(f"     Max Drawdown:          {best_train_perf['avg_max_drawdown']:7.2f}%")
        print(f"     Total Trades:          {int(best_train_perf['total_trades'])} across {int(best_train_perf['stocks_tested'])} stocks ({int(best_train_perf['stocks_traded'])} traded)")
        print(f"     Avg Trades/Stock:      {best_train_perf['total_trades']/best_train_perf['stocks_tested']:.1f}")
        print(f"     Win Rate:              {best_train_perf['avg_win_rate']:7.1f}%")
        print(f"     RR Ratio:              {best_train_perf['avg_rr_ratio']:7.2f}")
        print(f"     Profit Factor:         {best_train_perf['avg_profit_factor']:7.2f}")
        print(f"     Expectancy:            ${best_train_perf['avg_expectancy']:7.2f}")
        print(f"     Stock Win Rate:        {best_train_perf['stock_win_rate']:7.1f}% ({int(best_train_perf['winning_stocks'])}/{int(best_train_perf['stocks_tested'])})")
        print(f"\n   ML Prediction Accuracy (In-Sample):")
        print(f"     Overall Accuracy:      {best_train_perf['ml_accuracy']:7.1f}%")
        print(f"     Bullish Accuracy:      {best_train_perf['ml_bullish_accuracy']:7.1f}%")
        print(f"     Bearish Accuracy:      {best_train_perf['ml_bearish_accuracy']:7.1f}%")
        print(f"     Total Predictions:     {int(best_train_perf['total_ml_predictions'])}")
        print(f"   " + "="*66)

        # Stock qualification: filter stocks based on IS per-stock performance
        test_symbols = valid_symbols
        if config.get('use_stock_qualification', False) and best_params_tuple in train_grouped:
            best_stock_results = train_grouped[best_params_tuple]
            qualified_symbols = []
            disqualified = []
            min_pf = config.get('stock_qual_min_profit_factor', 1.0)
            min_wr = config.get('stock_qual_min_win_rate', 40.0)
            min_tr = config.get('stock_qual_min_trades', 3)
            min_ret = config.get('stock_qual_min_return_pct', 0.0)
            min_ml_bull = config.get('stock_qual_min_ml_bullish_accuracy', 50.0)

            for sr in best_stock_results:
                sym = sr['symbol']
                ml_bull = sr.get('ml_bullish_accuracy', 0)
                reasons = []
                if sr['total_trades'] < min_tr:
                    reasons.append(f"trades={int(sr['total_trades'])}<{min_tr}")
                if sr['profit_factor'] < min_pf:
                    reasons.append(f"PF={sr['profit_factor']:.2f}<{min_pf}")
                if sr['win_rate'] < min_wr:
                    reasons.append(f"WR={sr['win_rate']:.1f}%<{min_wr}%")
                if sr['return_pct'] < min_ret:
                    reasons.append(f"ret={sr['return_pct']:.1f}%<{min_ret}%")
                if ml_bull < min_ml_bull:
                    reasons.append(f"MLBull={ml_bull:.1f}%<{min_ml_bull}%")

                if not reasons:
                    qualified_symbols.append(sym)
                else:
                    disqualified.append((sym, sr, reasons))

            print(f"\n   📋 Stock Qualification (IS performance filter):")
            print(f"   Thresholds: PF>={min_pf}, WR>={min_wr}%, Trades>={min_tr}, Return>={min_ret}%, MLBull>={min_ml_bull}%")
            print(f"   {'Symbol':<8} {'Return%':>8} {'WinRate':>8} {'PF':>6} {'Trades':>7} {'MLBull%':>8}  Status")
            print(f"   {'-'*65}")
            for sr in sorted(best_stock_results, key=lambda x: x['return_pct'], reverse=True):
                sym = sr['symbol']
                ml_bull = sr.get('ml_bullish_accuracy', 0)
                status = "✓" if sym in qualified_symbols else "✗"
                print(f"   {sym:<8} {sr['return_pct']:>+7.1f}% {sr['win_rate']:>7.1f}% {sr['profit_factor']:>5.2f} {int(sr['total_trades']):>7} {ml_bull:>7.1f}%  {status}")
            print(f"   {'-'*55}")
            print(f"   Qualified: {len(qualified_symbols)}/{len(best_stock_results)} stocks "
                  f"({len(qualified_symbols)/len(best_stock_results)*100:.1f}%)")

            if qualified_symbols:
                test_symbols = qualified_symbols
            else:
                print(f"   ⚠️  No stocks qualified! Using all stocks.")
                test_symbols = valid_symbols

        print(f"\n2️⃣  VALIDATION PHASE (Out-of-Sample)")
        print(f"   Testing best parameters on unseen {test_start} to {test_end} data...")
        print(f"   Testing {len(test_symbols)} stocks (capturing equity curves for portfolio simulation)")

        test_results, equity_data, _ = optimize_single_period(
            data_cache, test_symbols, [best_params_tuple], period_idx, 'test', config,
            capture_equity=True
        )

        if best_params_tuple in test_results:
            test_performance = test_results[best_params_tuple]

            degradation = best_train_perf['sum_return_pct'] - test_performance['sum_return_pct']
            degradation_pct = (degradation / best_train_perf['sum_return_pct'] * 100) if best_train_perf['sum_return_pct'] != 0 else 0

            print(f"\n   Out-of-Sample Performance:")
            print(f"     Total Return (sum):    {test_performance['sum_return_pct']:7.2f}%")
            print(f"     Avg Return per Stock:  {test_performance['avg_return_pct']:7.2f}%")
            print(f"     Sharpe:                {test_performance['avg_sharpe']:7.3f}")
            print(f"     Calmar:                {test_performance['avg_calmar']:7.3f}")
            print(f"     Max Drawdown:          {test_performance['avg_max_drawdown']:7.2f}%")
            print(f"     Total Trades:          {int(test_performance['total_trades'])} across {int(test_performance['stocks_tested'])} stocks ({int(test_performance['stocks_traded'])} traded)")
            print(f"     Avg Trades/Stock:      {test_performance['total_trades']/test_performance['stocks_tested']:.1f}")
            print(f"     Win Rate:              {test_performance['avg_win_rate']:7.1f}%")
            print(f"     RR Ratio:              {test_performance['avg_rr_ratio']:7.2f}")
            print(f"     Profit Factor:         {test_performance['avg_profit_factor']:7.2f}")
            print(f"     Expectancy:            ${test_performance['avg_expectancy']:7.2f}")
            print(f"     Stock Win Rate:        {test_performance['stock_win_rate']:7.1f}% ({int(test_performance['winning_stocks'])}/{int(test_performance['stocks_tested'])})")
            print(f"\n   ML Prediction Accuracy (Out-of-Sample):")
            print(f"     Overall Accuracy:      {test_performance['ml_accuracy']:7.1f}%")
            print(f"     Bullish Accuracy:      {test_performance['ml_bullish_accuracy']:7.1f}%")
            print(f"     Bearish Accuracy:      {test_performance['ml_bearish_accuracy']:7.1f}%")
            print(f"     Total Predictions:     {int(test_performance['total_ml_predictions'])}")

            ml_accuracy_degradation = best_train_perf['ml_accuracy'] - test_performance['ml_accuracy']

            print(f"\n   Performance Comparison:")
            print(f"     Return Degradation:    {degradation_pct:+.1f}% ({best_train_perf['sum_return_pct']:.2f}% → {test_performance['sum_return_pct']:.2f}%)")
            print(f"     ML Accuracy Change:    {-ml_accuracy_degradation:+.1f}pp ({best_train_perf['ml_accuracy']:.1f}% → {test_performance['ml_accuracy']:.1f}%)")

            if test_performance['total_trades'] == 0:
                print(f"     ⚠️  WARNING: No trades in out-of-sample period!")
            elif test_performance['total_trades'] < 10:
                print(f"     ⚠️  WARNING: Very few trades ({int(test_performance['total_trades'])}) in out-of-sample.")

            if abs(degradation_pct) > 50:
                print(f"     ⚠️  WARNING: High degradation suggests overfitting!")
            elif abs(degradation_pct) < 20:
                print(f"     ✓ Good: Low degradation indicates robust parameters.")

            # Portfolio simulation using equity data captured during OOS backtests
            print(f"\n3️⃣  PORTFOLIO SIMULATION (Out-of-Sample)")
            portfolio_results = None
            if equity_data:
                # Diagnostic: verify equity data sanity
                _n_with_trades = sum(1 for s in equity_data if len(s.get('trade_log', [])) > 0)
                _returns = []
                for s in equity_data:
                    if len(s['values']) >= 2 and s['values'][0] > 0:
                        _returns.append((s['values'][-1] / s['values'][0] - 1) * 100)
                _avg_ret = np.mean(_returns) if _returns else 0
                print(f"   Equity data: {len(equity_data)} stocks, {_n_with_trades} with trades")
                print(f"   Per-stock equity returns: avg={_avg_ret:.2f}%, "
                      f"min={min(_returns):.2f}%, max={max(_returns):.2f}%" if _returns else
                      "   No per-stock equity returns")
                spy_df = None
                try:
                    all_dates = sorted(set(d for s in equity_data for d in s['dates']))
                    if all_dates:
                        spy_start = pd.Timestamp(all_dates[0])
                        spy_end = pd.Timestamp(all_dates[-1]) + timedelta(days=1)
                        spy_df = yf.download('SPY', start=spy_start, end=spy_end, progress=False)
                        if not spy_df.empty:
                            spy_df.index = spy_df.index.tz_localize(None)
                except Exception:
                    pass

                # Build per-stock stats from in-sample training results
                _kelly_stats = None
                _ranking_stats = None
                if best_params_tuple in train_grouped:
                    _kelly_stats = {}
                    _ranking_stats = {}
                    for sr in train_grouped[best_params_tuple]:
                        _kelly_stats[sr['symbol']] = {
                            'win_rate': sr['win_rate'],
                            'rr_ratio': sr['rr_ratio'],
                        }
                        _ranking_stats[sr['symbol']] = {
                            'win_rate': sr['win_rate'],
                            'rr_ratio': sr['rr_ratio'],
                            'expectancy': sr['expectancy'],
                            'profit_factor': sr['profit_factor'],
                        }

                pf_all_results = []  # track all portfolio results for best-selection
                for pf_idx, pf_params in enumerate(portfolio_params_list):
                    _portfolio_debug = pf_params.portfolio_debug
                    _portfolio_capital = pf_params.portfolio_capital
                    _max_positions = pf_params.max_positions
                    _kelly_fraction = pf_params.kelly_fraction
                    _use_replacement = pf_params.use_position_replacement
                    _repl_threshold = pf_params.replacement_threshold
                    _repl_cooldown = pf_params.replacement_cooldown_days
                    _repl_max_day = pf_params.replacement_max_per_day
                    _pred_val_bars = pf_params.prediction_validation_bars
                    _pred_val_threshold = pf_params.prediction_validation_threshold
                    _entry_mode = pf_params.entry_mode
                    _entry_limit_window = pf_params.entry_limit_window
                    _entry_dip_threshold = pf_params.entry_dip_threshold
                    _entry_confirm_threshold = pf_params.entry_confirmation_threshold

                    # Debug: show per-stock equity curve stats vs trade log stats
                    if _portfolio_debug:
                        print(f"\n   [DEBUG] Per-stock equity curve vs trade log reconciliation (first 20):")
                        print(f"   {'Symbol':<8} {'EqRet%':>8} {'#Trades':>8} {'TradePnL':>10} {'EqStart':>10} {'EqEnd':>10} {'#Days':>6}")
                        _debug_stocks = sorted(equity_data, key=lambda s: s['symbol'])[:20]
                        for s in _debug_stocks:
                            eq_ret = (s['values'][-1] / s['values'][0] - 1) * 100 if s['values'][0] > 0 else 0
                            n_trades = len(s.get('trade_log', []))
                            trade_pnl = sum(t.get('pnl', 0) or 0 for t in s.get('trade_log', []))
                            print(f"   {s['symbol']:<8} {eq_ret:>7.2f}% {n_trades:>8} ${trade_pnl:>9,.2f} "
                                  f"${s['values'][0]:>9,.0f} ${s['values'][-1]:>9,.0f} {len(s['dates']):>6}")

                    _pf_suffix = f'_pf{pf_idx + 1}' if len(portfolio_params_list) > 1 else ''
                    trade_log_path = f'portfolio_trade_log_period_{period_idx + 1}{_pf_suffix}.csv' if _portfolio_debug else None
                    portfolio_results = simulate_portfolio(
                        equity_data,
                        initial_capital=_portfolio_capital,
                        max_positions=_max_positions,
                        spy_df=spy_df,
                        debug=_portfolio_debug,
                        trade_log_file=trade_log_path,
                        kelly_fraction=_kelly_fraction,
                        kelly_stats=_kelly_stats,
                        ranking_stats=_ranking_stats,
                        use_position_replacement=_use_replacement,
                        replacement_threshold=_repl_threshold,
                        replacement_cooldown_days=_repl_cooldown,
                        replacement_max_per_day=_repl_max_day,
                        prediction_validation_bars=_pred_val_bars,
                        prediction_validation_threshold=_pred_val_threshold,
                        entry_mode=_entry_mode,
                        entry_limit_window=_entry_limit_window,
                        entry_dip_threshold=_entry_dip_threshold,
                        entry_confirmation_threshold=_entry_confirm_threshold,
                    )
                    if portfolio_results:
                        pf_all_results.append({'pf_idx': pf_idx, 'params': pf_params, 'results': portfolio_results})
                        if len(portfolio_params_list) > 1:
                            pass  # per-config output suppressed; winner shown below
                        else:
                            print_portfolio_summary(portfolio_results)
                            if _portfolio_debug:
                                print_trade_log_summary(portfolio_results)
                            plot_portfolio(portfolio_results, f'portfolio_sim_period_{period_idx + 1}.png')

                # Store for cross-period winner selection (done after all periods complete)
                all_pf_by_period.append(pf_all_results)
                best_pf_results = None  # filled in post-loop pass
                best_pf_params = None
            else:
                print(f"   No equity data captured for portfolio simulation")

            period_result = {
                'period': period_idx + 1,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'in_sample_return': best_train_perf['sum_return_pct'],
                'in_sample_avg_return': best_train_perf['avg_return_pct'],
                'in_sample_sharpe': best_train_perf['avg_sharpe'],
                'in_sample_drawdown': best_train_perf['avg_max_drawdown'],
                'in_sample_trades': best_train_perf['total_trades'],
                'in_sample_win_rate': best_train_perf['avg_win_rate'],
                'in_sample_profit_factor': best_train_perf['avg_profit_factor'],
                'out_sample_return': test_performance['sum_return_pct'],
                'out_sample_avg_return': test_performance['avg_return_pct'],
                'out_sample_sharpe': test_performance['avg_sharpe'],
                'out_sample_drawdown': test_performance['avg_max_drawdown'],
                'out_sample_trades': test_performance['total_trades'],
                'out_sample_win_rate': test_performance['avg_win_rate'],
                'out_sample_rr_ratio': test_performance['avg_rr_ratio'],
                'out_sample_profit_factor': test_performance['avg_profit_factor'],
                'out_sample_expectancy': test_performance['avg_expectancy'],
                'in_sample_ml_accuracy': best_train_perf['ml_accuracy'],
                'in_sample_ml_bullish_accuracy': best_train_perf['ml_bullish_accuracy'],
                'in_sample_ml_bearish_accuracy': best_train_perf['ml_bearish_accuracy'],
                'in_sample_ml_predictions': best_train_perf['total_ml_predictions'],
                'out_sample_ml_accuracy': test_performance['ml_accuracy'],
                'out_sample_ml_bullish_accuracy': test_performance['ml_bullish_accuracy'],
                'out_sample_ml_bearish_accuracy': test_performance['ml_bearish_accuracy'],
                'out_sample_ml_predictions': test_performance['total_ml_predictions'],
                'ml_accuracy_degradation': ml_accuracy_degradation,
                'return_degradation_pct': degradation_pct,
                'trade_difference': int(test_performance['total_trades']) - int(best_train_perf['total_trades']),
                'params': best_train_perf['params'],
                # Portfolio simulation metrics (from best config by Sharpe)
                'portfolio_return': best_pf_results['total_return_pct'] if best_pf_results else None,
                'portfolio_sharpe': best_pf_results['sharpe'] if best_pf_results else None,
                'portfolio_max_dd': best_pf_results['max_drawdown_pct'] if best_pf_results else None,
                'portfolio_avg_positions': best_pf_results['avg_positions_held'] if best_pf_results else None,
                'portfolio_skipped': best_pf_results['skipped_entries'] if best_pf_results else None,
            }

            all_results.append(period_result)
        else:
            print(f"\n   ❌ No results for test period!")

    # ── Cross-period portfolio winner selection ───────────────────────────────
    # Pick the single portfolio config with the best *average* Sharpe across
    # all OOS periods instead of a different winner each period.
    if all_pf_by_period:
        config_sharpes = {}  # pf_idx -> [sharpe per period]
        config_params  = {}  # pf_idx -> params object
        for period_pf in all_pf_by_period:
            for entry in period_pf:
                idx = entry['pf_idx']
                config_sharpes.setdefault(idx, []).append(entry['results'].get('sharpe', -999))
                config_params[idx] = entry['params']

        best_idx = max(config_sharpes, key=lambda i: sum(config_sharpes[i]) / len(config_sharpes[i]))
        best_pf_params_global = config_params[best_idx]
        avg_sharpe = sum(config_sharpes[best_idx]) / len(config_sharpes[best_idx])
        n_seen = len(config_sharpes[best_idx])

        if len(portfolio_params_list) > 1:
            print(f"\n{'='*70}")
            print(f"BEST PORTFOLIO CONFIG  (avg Sharpe: {avg_sharpe:.3f} across {n_seen} periods)")
            print(f"{'='*70}")
            p = best_pf_params_global
            print(f"   'kelly_fraction':                {p.kelly_fraction},")
            print(f"   'use_position_replacement':      {p.use_position_replacement},")
            print(f"   'replacement_threshold':         {p.replacement_threshold},")
            print(f"   'replacement_cooldown_days':     {p.replacement_cooldown_days},")
            print(f"   'replacement_max_per_day':       {p.replacement_max_per_day},")
            print(f"   'prediction_validation_bars':    {p.prediction_validation_bars},")
            print(f"   'prediction_validation_threshold': {p.prediction_validation_threshold},")
            print(f"   'entry_mode':                    '{p.entry_mode}',")
            print(f"   'entry_limit_window':            {p.entry_limit_window},")
            print(f"   'entry_dip_threshold':           {p.entry_dip_threshold},")
            print(f"   'entry_confirmation_threshold':  {p.entry_confirmation_threshold},")
            print(f"{'='*70}")

        # Update per-period CSV results and (for multi-config) print + plot
        for p_idx, period_pf in enumerate(all_pf_by_period):
            winner = next((e for e in period_pf if e['pf_idx'] == best_idx), None)
            if winner and p_idx < len(all_results):
                r = winner['results']
                all_results[p_idx]['portfolio_return']        = r['total_return_pct']
                all_results[p_idx]['portfolio_sharpe']        = r['sharpe']
                all_results[p_idx]['portfolio_max_dd']        = r['max_drawdown_pct']
                all_results[p_idx]['portfolio_avg_positions'] = r['avg_positions_held']
                all_results[p_idx]['portfolio_skipped']       = r['skipped_entries']
                if len(portfolio_params_list) > 1:
                    print(f"\n   Period {p_idx + 1} — winning config results:")
                    print_portfolio_summary(r)
                    if best_pf_params_global.portfolio_debug:
                        print_trade_log_summary(r)
                    plot_portfolio(r, f'portfolio_sim_period_{p_idx + 1}.png')

    return pd.DataFrame(all_results)


# ============================================================================
# Results Reporting
# ============================================================================

def print_walkforward_results(df_results):
    """Print walk-forward results summary."""
    print("\n" + "="*70)
    print("WALK-FORWARD OPTIMIZATION RESULTS (TREND-FOLLOWING)")
    print("="*70)

    print("\nPer-Period Out-of-Sample Performance:")
    print("-" * 70)

    for _, row in df_results.iterrows():
        print(f"\nPeriod {int(row['period'])}: {row['test_start']} to {row['test_end']}")
        print(f"  In-Sample (Train):")
        print(f"    Total Return:    {row['in_sample_return']:7.2f}%")
        print(f"    Avg Return:      {row['in_sample_avg_return']:7.2f}%")
        print(f"    Sharpe:          {row['in_sample_sharpe']:7.3f}")
        print(f"    Drawdown:        {row['in_sample_drawdown']:7.2f}%")
        print(f"    Trades:          {int(row['in_sample_trades'])}")
        print(f"    Win Rate:        {row['in_sample_win_rate']:7.1f}%")
        print(f"    ML Accuracy:     {row['in_sample_ml_accuracy']:7.1f}% ({int(row['in_sample_ml_predictions'])} predictions)")

        print(f"  Out-of-Sample (Test):")
        print(f"    Total Return:    {row['out_sample_return']:7.2f}%")
        print(f"    Avg Return:      {row['out_sample_avg_return']:7.2f}%")
        print(f"    Sharpe:          {row['out_sample_sharpe']:7.3f}")
        print(f"    Drawdown:        {row['out_sample_drawdown']:7.2f}%")
        print(f"    Trades:          {int(row['out_sample_trades'])}")
        print(f"    Win Rate:        {row['out_sample_win_rate']:7.1f}%")
        print(f"    Profit Factor:   {row['out_sample_profit_factor']:7.2f}")
        print(f"    ML Accuracy:     {row['out_sample_ml_accuracy']:7.1f}% ({int(row['out_sample_ml_predictions'])} predictions)")

        print(f"  Comparison:")
        print(f"    Return Change:   {row['return_degradation_pct']:+7.1f}%")
        print(f"    ML Acc. Change:  {-row['ml_accuracy_degradation']:+7.1f}pp")
        print(f"    Trade Diff:      {int(row['trade_difference']):+d}")

        if pd.notna(row.get('portfolio_return')):
            print(f"  Portfolio Simulation:")
            print(f"    Portfolio Return: {row['portfolio_return']:7.2f}%")
            print(f"    Portfolio Sharpe: {row['portfolio_sharpe']:7.3f}")
            print(f"    Portfolio Max DD: {row['portfolio_max_dd']:7.2f}%")
            print(f"    Avg Positions:   {row['portfolio_avg_positions']:7.1f}")
            print(f"    Skipped Entries: {int(row['portfolio_skipped'])}")

    print("\n" + "="*70)
    print("AGGREGATE OUT-OF-SAMPLE PERFORMANCE")
    print("="*70)

    sum_oos_return = df_results['out_sample_return'].sum()
    avg_oos_return = df_results['out_sample_return'].mean()
    avg_oos_sharpe = df_results['out_sample_sharpe'].mean()
    avg_oos_drawdown = df_results['out_sample_drawdown'].mean()
    avg_oos_trades = df_results['out_sample_trades'].mean()
    avg_oos_win_rate = df_results['out_sample_win_rate'].mean()
    avg_oos_pf = df_results['out_sample_profit_factor'].mean()
    avg_degradation = df_results['return_degradation_pct'].mean()
    avg_oos_ml_accuracy = df_results['out_sample_ml_accuracy'].mean()
    avg_is_ml_accuracy = df_results['in_sample_ml_accuracy'].mean()
    avg_ml_degradation = df_results['ml_accuracy_degradation'].mean()

    print(f"\nOut-of-Sample Performance (across {len(df_results)} periods):")
    print(f"  Total Return (sum):  {sum_oos_return:7.2f}%")
    print(f"  Avg Return/Period:   {avg_oos_return:7.2f}%")
    print(f"  Sharpe Ratio:        {avg_oos_sharpe:7.3f}")
    print(f"  Max Drawdown:        {avg_oos_drawdown:7.2f}%")
    print(f"  Avg Trades:          {avg_oos_trades:7.1f}")
    print(f"  Win Rate:            {avg_oos_win_rate:7.1f}%")
    print(f"  Profit Factor:       {avg_oos_pf:7.2f}")
    print(f"  ML Accuracy (OOS):   {avg_oos_ml_accuracy:7.1f}%")
    print(f"  ML Accuracy (IS):    {avg_is_ml_accuracy:7.1f}%")
    print(f"\nAverage Return Degradation: {avg_degradation:+.1f}%")
    print(f"Average ML Accuracy Change: {-avg_ml_degradation:+.1f}pp")

    # Portfolio simulation aggregate
    port_returns = df_results['portfolio_return'].dropna()
    if len(port_returns) > 0:
        print(f"\nPortfolio Simulation (across {len(port_returns)} periods):")
        print(f"  Avg Portfolio Return:  {port_returns.mean():7.2f}%")
        print(f"  Avg Portfolio Sharpe:  {df_results['portfolio_sharpe'].dropna().mean():7.3f}")
        print(f"  Avg Portfolio Max DD:  {df_results['portfolio_max_dd'].dropna().mean():7.2f}%")
        print(f"  Avg Positions Held:    {df_results['portfolio_avg_positions'].dropna().mean():7.1f}")
        print(f"  Total Skipped Entries: {int(df_results['portfolio_skipped'].dropna().sum())}")

    if abs(avg_degradation) > 30:
        print("\n⚠️  WARNING: Average degradation > 30% indicates potential overfitting")
    elif abs(avg_degradation) < 15:
        print("\n✓ Good: Low degradation indicates robust parameters")
    else:
        print("\n→ Moderate degradation - parameters are reasonable")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    test_period_months = 9
    annualized_return = avg_oos_return * (12 / test_period_months)

    print(f"\nExpected Performance (based on out-of-sample results):")
    print(f"  Total Return (all periods):    {sum_oos_return:.2f}%")
    print(f"  Per Test Period ({test_period_months} months): ~{avg_oos_return:.2f}%")
    print(f"  Annualized (estimated):       ~{annualized_return:.2f}%")
    print(f"  Sharpe Ratio:                  {avg_oos_sharpe:.2f}")
    print(f"  Max Drawdown:                  {avg_oos_drawdown:.2f}%")
    print(f"  ML Accuracy (OOS):             {avg_oos_ml_accuracy:.1f}%")

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
    print(f"   min_raw_prediction:       {params.get('min_raw_prediction', 0.0)} ATR")

    print(f"\n🔹 LABEL SETTINGS:")
    print(f"   label_lookahead:          {params['label_lookahead']}")
    print(f"   label_dead_zone:          {params['label_dead_zone']}")
    print(f"   use_magnitude_labels:     {params['use_magnitude_labels']}")

    print("\n🔹 TREND-FOLLOWING FEATURES:")
    fnames = {'RSM': 'Relative Strength Momentum', 'ER': 'Efficiency Ratio',
              'MTD': 'Multi-Timeframe Divergence', 'STRK': 'Streak Pattern',
              'VCOMP': 'Volatility Compression', 'MPER': 'Momentum Persistence',
              'VMC': 'Volume-Momentum Coupling', 'CS': 'Candle Structure',
              'MACC': 'Momentum Acceleration', 'OBVT': 'OBV Trend', 'VPD': 'Vol-Price Div',
              'CHOP': 'Choppiness Index', 'ZS': 'Mean Rev Z-Score', 'VA': 'Volume Anomaly'}
    for fi in range(1, 15):
        ft_key = f'f{fi}_type'
        if ft_key in params:
            print(f"   {ft_key}:{'  ' if fi < 10 else ' '}               {params[ft_key]} ({fnames.get(params[ft_key], params[ft_key])})")
            print(f"   f{fi}_param_a:{'  ' if fi < 10 else ' '}            {params[f'f{fi}_param_a']}")
            print(f"   f{fi}_param_b:{'  ' if fi < 10 else ' '}            {params[f'f{fi}_param_b']}")

    print("\n🔹 FILTERS:")
    print(f"   use_volatility_filter:    {params['use_volatility_filter']}")
    print(f"   use_regime_filter:        {params['use_regime_filter']}")
    print(f"   regime_threshold:         {params['regime_threshold']}")
    print(f"   regime_period:            {params['regime_period']}")
    print(f"   use_regime_direction:      {params.get('use_regime_direction', True)}")
    print(f"   regime_stability_min:     {params.get('regime_stability_min', 0.0)}")
    print(f"   regime_stability_window:  {params.get('regime_stability_window', 60)}")
    print(f"   regime_max_flips:         {params.get('regime_max_flips', 18)}")
    print(f"   use_spy_filter:           {params.get('use_spy_filter', False)}")
    print(f"   spy_regime_threshold:     {params.get('spy_regime_threshold', 0)}")
    print(f"   spy_regime_period:        {params.get('spy_regime_period', 'weekly')}")
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
    print(f"   use_prediction_exit:      {params.get('use_prediction_exit', False)}")
    if params.get('use_prediction_exit', False):
        print(f"   prediction_exit_threshold:{params.get('prediction_exit_threshold', -0.5)}%")
    print(f"   use_trailing_atr_exit:    {params.get('use_trailing_atr_exit', False)}")
    if params.get('use_trailing_atr_exit', False):
        print(f"   trailing_atr_warmup:      {params['trailing_atr_warmup']}")
        print(f"   chandelier_start_mult:    {params['chandelier_start_mult']}")
        print(f"   chandelier_end_mult:      {params['chandelier_end_mult']}")
        print(f"   chandelier_tighten_bars:  {params['chandelier_tighten_bars']}")
        print(f"   chandelier_mult_mode:     {params['chandelier_mult_mode']}")
        print(f"   chandelier_profit_atr_threshold: {params['chandelier_profit_atr_threshold']}")
        print(f"   chandelier_breakeven_atr: {params.get('chandelier_breakeven_atr', 0.0)}")

    print("\n🔹 CROSS-SYMBOL TRAINING:")
    print(f"   use_cross_symbol_training: {params.get('use_cross_symbol_training', False)}")
    if params.get('use_cross_symbol_training', False):
        print(f"   cross_symbol_etfs:         {params.get('cross_symbol_etfs', '')}")
        print(f"   cross_symbol_lookback_years: {params.get('cross_symbol_lookback_years', 5)}")
        print(f"   use_regime_balancing:      {params.get('use_regime_balancing', False)}")
        print(f"   cross_symbol_auto_peers:   {params.get('cross_symbol_auto_peers', False)}")
        print(f"   cross_symbol_max_peers:    {params.get('cross_symbol_max_peers', 7)}")

    print("\n🔹 FUNDAMENTAL DATA FILTER:")
    print(f"   use_fundamental_filter:   {params.get('use_fundamental_filter', False)}")
    if params.get('use_fundamental_filter', False):
        print(f"   fundamental_quality_weight:  {params.get('fundamental_quality_weight', 0.4)}")
        print(f"   fundamental_momentum_weight: {params.get('fundamental_momentum_weight', 0.6)}")
        print(f"   earnings_blackout_before: {params.get('earnings_blackout_before', 5)}")
        print(f"   earnings_blackout_after:  {params.get('earnings_blackout_after', 2)}")
        print(f"   close_before_earnings:    {params.get('close_before_earnings', False)}")
        print(f"   min_trending_probability: {params.get('min_trending_probability', 50)}")
        print(f"   full_position_threshold:  {params.get('full_position_threshold', 70)}")
        print(f"   reduced_position_pct:     {params.get('reduced_position_pct', Decimal('0.75'))}")
        if params.get('use_earnings_reaction', False):
            print(f"   earnings_reaction_days:   {params['earnings_reaction_days']}")
            print(f"   min_earnings_reaction_pct: {params['min_earnings_reaction_pct']}%")

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
    print(f"    'min_raw_prediction': {params.get('min_raw_prediction', 0.0)},")
    print("\n    # Label Settings")
    print(f"    'label_lookahead': {params['label_lookahead']},")
    print(f"    'label_dead_zone': {params['label_dead_zone']},")
    print(f"    'use_magnitude_labels': {params['use_magnitude_labels']},")
    print("\n    # Feature 1 (RSM - Relative Strength Momentum)")
    print(f"    'f1_type': '{params['f1_type']}',")
    print(f"    'f1_param_a': {params['f1_param_a']},")
    print(f"    'f1_param_b': {params['f1_param_b']},")
    _cpnames = {1: 'Relative Strength Momentum', 2: 'Efficiency Ratio', 3: 'Multi-Timeframe Divergence',
                4: 'Streak Pattern', 5: 'Volatility Compression', 6: 'Momentum Persistence',
                7: 'Volume-Momentum Coupling', 8: 'Candle Structure', 9: 'Candle Structure',
                10: 'Candle Structure', 11: 'Candle Structure', 12: 'Volatility Compression',
                13: 'Momentum Persistence', 14: 'Volume-Momentum Coupling'}
    for _fi in range(2, 15):
        if f'f{_fi}_type' in params:
            print(f"\n    # Feature {_fi} ({_cpnames.get(_fi, '')})")
            print(f"    'f{_fi}_type': '{params[f'f{_fi}_type']}',")
            print(f"    'f{_fi}_param_a': {params[f'f{_fi}_param_a']},")
            print(f"    'f{_fi}_param_b': {params[f'f{_fi}_param_b']},")
    print("\n    # Filters")
    print(f"    'use_volatility_filter': {params['use_volatility_filter']},")
    print(f"    'use_regime_filter': {params['use_regime_filter']},")
    print(f"    'regime_threshold': Decimal('{params['regime_threshold']}'),")
    print(f"    'regime_period': '{params['regime_period']}',")
    print(f"    'use_regime_direction': {params.get('use_regime_direction', True)},")
    print(f"    'regime_stability_min': {params.get('regime_stability_min', 0.0)},")
    print(f"    'regime_stability_window': {params.get('regime_stability_window', 60)},")
    print(f"    'regime_max_flips': {params.get('regime_max_flips', 18)},")
    print(f"    'use_adx_filter': {params['use_adx_filter']},")
    print(f"    'adx_threshold': {params['adx_threshold']},")
    print(f"    'use_ema_filter': {params['use_ema_filter']},")
    print(f"    'ema_period': {params['ema_period']},")
    print(f"    'ema_slope_lookback': {params['ema_slope_lookback']},")
    print(f"    'use_sma_filter': {params['use_sma_filter']},")
    print(f"    'sma_period': {params['sma_period']},")
    print(f"    'sma_slope_lookback': {params['sma_slope_lookback']},")
    print(f"    'use_spy_filter': {params.get('use_spy_filter', False)},")
    print(f"    'spy_regime_threshold': {params.get('spy_regime_threshold', 0)},")
    print(f"    'spy_regime_period': '{params.get('spy_regime_period', 'weekly')}',")
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
    print("\n    # Prediction Validation Exit")
    print(f"    'use_prediction_exit': {params.get('use_prediction_exit', False)},")
    print(f"    'prediction_exit_threshold': {params.get('prediction_exit_threshold', -0.5)},")
    print("\n    # Chandelier Exit (ATR Trailing Stop)")
    print(f"    'use_trailing_atr_exit': {params.get('use_trailing_atr_exit', False)},")
    print(f"    'trailing_atr_warmup': {params.get('trailing_atr_warmup', 3)},")
    print(f"    'chandelier_start_mult': {params.get('chandelier_start_mult', 3.0)},")
    print(f"    'chandelier_end_mult': {params.get('chandelier_end_mult', 1.5)},")
    print(f"    'chandelier_tighten_bars': {params.get('chandelier_tighten_bars', 20)},")
    print(f"    'chandelier_mult_mode': '{params.get('chandelier_mult_mode', 'blend')}',")
    print(f"    'chandelier_profit_atr_threshold': {params.get('chandelier_profit_atr_threshold', 1.0)},")
    print(f"    'chandelier_breakeven_atr': {params.get('chandelier_breakeven_atr', 0.0)},")
    print("\n    # Risk Management")
    print(f"    'position_size_pct': Decimal('{params['position_size_pct']}'),")
    print(f"    'stop_loss_pct': Decimal('{params['stop_loss_pct']}'),")
    print(f"    'use_stop_loss': {params['use_stop_loss']},")
    print(f"    'long_only': {params['long_only']},")
    print("\n    # Other")
    print(f"    'verbose': False,")
    print(f"    'test_start_idx': {params['test_start_idx']},")
    print("\n    # Cross-Symbol Training")
    print(f"    'use_cross_symbol_training': {params.get('use_cross_symbol_training', False)},")
    print(f"    'cross_symbol_etfs': '{params.get('cross_symbol_etfs', '')}',")
    print(f"    'cross_symbol_lookback_years': {params.get('cross_symbol_lookback_years', 5)},")
    print(f"    'use_regime_balancing': {params.get('use_regime_balancing', False)},")
    print(f"    'cross_symbol_auto_peers': {params.get('cross_symbol_auto_peers', False)},")
    print(f"    'cross_symbol_target_symbol': '',")
    print(f"    'cross_symbol_max_peers': {params.get('cross_symbol_max_peers', 7)},")
    print("\n    # Fundamental Data Filter")
    print(f"    'use_fundamental_filter': {params.get('use_fundamental_filter', False)},")
    if params.get('use_fundamental_filter', False):
        print(f"    'fundamental_quality_weight': {params.get('fundamental_quality_weight', 0.4)},")
        print(f"    'fundamental_momentum_weight': {params.get('fundamental_momentum_weight', 0.6)},")
        print(f"    'earnings_blackout_before': {params.get('earnings_blackout_before', 5)},")
        print(f"    'earnings_blackout_after': {params.get('earnings_blackout_after', 2)},")
        print(f"    'close_before_earnings': {params.get('close_before_earnings', False)},")
        print(f"    'min_trending_probability': {params.get('min_trending_probability', 50)},")
        print(f"    'full_position_threshold': {params.get('full_position_threshold', 70)},")
        print(f"    'reduced_position_pct': Decimal('{params.get('reduced_position_pct', Decimal('0.75'))}'),")
        print(f"    'min_quality_score': {params.get('min_quality_score', 0)},")
        print(f"    'min_momentum_score': {params.get('min_momentum_score', 0)},")
        print(f"    'use_earnings_reaction': {params.get('use_earnings_reaction', False)},")
        print(f"    'earnings_reaction_days': {params.get('earnings_reaction_days', 5)},")
        print(f"    'min_earnings_reaction_pct': {params.get('min_earnings_reaction_pct', -3.0)},")
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
