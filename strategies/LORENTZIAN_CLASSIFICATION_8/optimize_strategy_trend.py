"""
Walk-Forward Optimization for Lorentzian Classification Strategy - Trend-Following Features

This script implements proper walk-forward analysis with:
- Rolling optimization windows (train/test splits)
- Out-of-sample validation on unseen data
- Aggregation across multiple forward periods
- COVID period handling options

Uses trend-following feature vector (momentum/trend quality prioritized):
- RSM(20,252): Relative Strength Momentum (core trend signal)
- ER(20): Efficiency Ratio (trend quality / directional movement)
- MACC(5,5): Momentum Acceleration (trend strengthening/weakening)
- MTD(5,60): Multi-Timeframe Divergence (timeframe alignment)
- OBVT(20,3): OBV Trend (volume confirming trend direction)
- STRK(15,1): Streak Pattern (consecutive directional moves)
- CHOP(14): Choppiness Index (trending vs ranging regime)
- VPD(22,1): Volume-Price Divergence (trend exhaustion/continuation)
- VA(20): Volume Anomaly (volume conviction)
- CS(5,2): Candle Structure (trend bar quality)

Key differences from mean-reversion version:
- trend_following_labels = True (labels aligned with continuation, not fade)
- Longer label_lookahead (8 bars — trends unfold over more time)
- Feature order prioritizes momentum/trend features, ZS dropped
- RSI exit disabled (overbought exits cut trends short)
- Wider trailing ATR stop (give trends room to breathe)
- Kernel exit enabled (exit on trend reversal detection)

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


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Input/Output
    'csv_file': '../sp500_2024.csv',
    'results_file': 'walkforward_optimization_results_trend.csv',

    # Walk-Forward Settings
    'train_period_months': 18,      # Optimize on 18 months
    'test_period_months': 9,        # Validate on next 9 months
    'step_months': 9,               # Roll forward 9 months each iteration
    'total_periods': 2,             # Number of train/test cycles

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
    'max_workers': None,  # None = use all CPU cores

    # Ranking metric: 'composite_score' for trading performance, 'ml_accuracy' for ML prediction accuracy
    # Use 'ml_accuracy' when optimizing ML parameters (features, labels, neighbors) to avoid overfitting
    # to specific price movements. Trading filters/exits should use 'composite_score'.
    'ranking_metric': 'composite_score',

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
        'neighbors_count': [8],
        'max_bars_back': [7000],            # Keep fixed - needs lots of history
        'feature_count': [10],
        'trend_following_labels': [True],   # True=trend-following labels (ride continuation)
        'allow_reentry': [True],            # True=enter anytime signal favorable
        'min_prediction_strength': [20],    # Normalized scale: 0-100

        # ==================== LABEL SETTINGS ====================
        # Longer lookahead for trends — they unfold over more bars than reversions
        'label_lookahead': [8],       # Bars to look forward: 8=captures trend moves
        'label_dead_zone': [0.225],   # Min ATR move for label: same threshold
        'use_magnitude_labels': [True],

        # ==================== FEATURE 1 (RSM - Relative Strength Momentum) ====================
        # Core trend feature — where is this stock's momentum ranked historically?
        'f1_type': ['RSM'],
        'f1_param_a': [20],          # Momentum period: 20=monthly momentum
        'f1_param_b': [252],         # Lookback: 252=full year percentile ranking

        # ==================== FEATURE 2 (ER - Efficiency Ratio) ====================
        # Trend quality — is price moving directionally or chopping around?
        'f2_type': ['ER'],
        'f2_param_a': [20],          # ER period: 20=trend quality over ~1 month
        'f2_param_b': [1],

        # ==================== FEATURE 3 (MACC - Momentum Acceleration) ====================
        # Is the trend accelerating or decelerating?
        'f3_type': ['MACC'],
        'f3_param_a': [5],           # ROC period
        'f3_param_b': [5],           # Lag for comparison

        # ==================== FEATURE 4 (MTD - Multi-Timeframe Divergence) ====================
        # Are short and long timeframes aligned? Alignment = strong trend
        'f4_type': ['MTD'],
        'f4_param_a': [5],           # Short ROC period
        'f4_param_b': [60],          # Long ROC period

        # ==================== FEATURE 5 (OBVT - OBV Trend) ====================
        # Is volume confirming the price trend?
        'f5_type': ['OBVT'],
        'f5_param_a': [20],          # SMA/StdDev period
        'f5_param_b': [3],           # Z-score divisor

        # ==================== FEATURE 6 (STRK - Streak Pattern) ====================
        # Consecutive directional moves = trend persistence
        'f6_type': ['STRK'],
        'f6_param_a': [15],          # Max streak length
        'f6_param_b': [1],           # ATR multiplier for magnitude

        # ==================== FEATURE 7 (CHOP - Choppiness Index) ====================
        # Low choppiness = trending market, high = ranging (avoid)
        'f7_type': ['CHOP'],
        'f7_param_a': [14],          # ATR/range period
        'f7_param_b': [1],

        # ==================== FEATURE 8 (VPD - Volume-Price Divergence) ====================
        # Divergence can signal trend exhaustion or continuation
        'f8_type': ['VPD'],
        'f8_param_a': [22],          # Correlation window
        'f8_param_b': [1],           # Smoothing (1=none)

        # ==================== FEATURE 9 (VA - Volume Anomaly) ====================
        # Unusual volume during trend = conviction
        'f9_type': ['VA'],
        'f9_param_a': [20],
        'f9_param_b': [1],

        # ==================== FEATURE 10 (CS - Candle Structure) ====================
        # Trend-quality candles (strong bodies, small wicks)
        'f10_type': ['CS'],
        'f10_param_a': [5],          # Averaging window
        'f10_param_b': [2],          # Sensitivity (tanh scaling)

        # ==================== FEATURE 11 (ADX - Normalized ADX) ====================
        # ADX measures trend strength directly — backup feature if feature_count increased
        'f11_type': ['ADX'],
        'f11_param_a': [14],
        'f11_param_b': [1],

        # ==================== FILTERS ====================
        'use_volatility_filter': [True],
        'use_regime_filter': [True],
        'regime_threshold': [Decimal('1')],  # 1=require bullish (trend-aligned)
        'regime_period': ['weekly'],  # 'weekly' or 'monthly'
        'use_regime_direction': [True],  # True=allow reverting-from-bearish when threshold=1
        'regime_stability_min': [0.0],  # Min stability to trade (0=off, 0.5=moderate, 0.7=strict)
        'regime_stability_window': [60],  # Bars to look back for regime flip counting
        'regime_max_flips': [8],  # Number of flips at which stability = 0 (higher = more permissive)
        'use_adx_filter': [True],
        'adx_threshold': [14],
        'use_ema_filter': [False],
        'ema_period': [25],
        'use_sma_filter': [False],
        'sma_period': [100],

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
        # DISABLED for trend-following — RSI overbought exits cut winning trends short
        'use_rsi_exit': [False],
        'rsi_exit_period': [14],
        'rsi_overbought': [80],             # Widened threshold (less likely to trigger)
        'rsi_oversold': [20],               # Widened threshold

        # ==================== KERNEL EXIT SETTINGS ====================
        # Kernel exit detects trend reversal — useful for trend-following
        'use_kernel_exit': [True],

        # ==================== ATR TRAILING STOP EXIT SETTINGS ====================
        # Wider multiplier gives trends room to breathe through normal pullbacks
        'use_trailing_atr_exit': [True],
        'trailing_atr_mult': [2.5],         # Wider than mean-reversion's 2.0
        'trailing_atr_warmup': [3],

        # ==================== RISK MANAGEMENT ====================
        'position_size_pct': [Decimal('0.95')],
        'stop_loss_pct': [Decimal('0.05')],
        'use_stop_loss': [True],
        'long_only': [True],

        # ==================== OTHER ====================
        'verbose': [False],
        'test_start_idx': [0],              # Bar index to start trading

        # ==================== CROSS-SYMBOL TRAINING ====================
        'use_cross_symbol_training': [True],
        'cross_symbol_etfs': ['SPY,QQQ,IWM,TLT,GLD,XLE,EFA'],
        'cross_symbol_lookback_years': [5],
        'use_regime_balancing': [True],
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


def load_all_periods_data(symbols, periods, lookback_bars=600):
    """Load data for all symbols across all periods, with ML lookback warmup."""
    data_cache = {}

    print(f"\n📥 Downloading data for {len(symbols)} stocks across {len(periods)} periods...")
    print(f"   ML lookback: {lookback_bars} bars before each period for training data warmup")

    total_downloads = len(symbols) * len(periods)
    download_count = 0

    for symbol in symbols:
        data_cache[symbol] = {}

        for period_idx, (train_start, train_end, test_start, test_end) in enumerate(periods):
            train_df, train_tsi = load_symbol_data(symbol, train_start, train_end, lookback_bars)
            test_df, test_tsi = load_symbol_data(symbol, test_start, test_end, lookback_bars)

            if train_df is not None and test_df is not None:
                data_cache[symbol][period_idx] = {
                    'train': train_df,
                    'train_test_start_idx': train_tsi,
                    'test': test_df,
                    'test_test_start_idx': test_tsi,
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

        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                            timeframe=bt.TimeFrame.Days, riskfreerate=0.0)
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
            'ml_accuracy': ml_accuracy,
            'ml_total_predictions': ml_total_predictions,
            'ml_bullish_accuracy': ml_bullish_accuracy,
            'ml_bearish_accuracy': ml_bearish_accuracy,
        }

    except Exception as e:
        return None


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
    'trend_following_labels', 'allow_reentry', 'min_prediction_strength',
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
    # Filters
    'use_volatility_filter', 'use_regime_filter', 'regime_threshold', 'regime_period',
    'use_regime_direction', 'regime_stability_min', 'regime_stability_window', 'regime_max_flips',
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
    # ATR Trailing Stop Exit Settings
    'use_trailing_atr_exit', 'trailing_atr_mult', 'trailing_atr_warmup',
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
]


def optimize_single_period(data_cache, valid_symbols, params_list, period_idx, phase, config):
    """Optimize or test on a single period using multiprocessing."""
    results = {}
    successful_backtests = 0

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
            future = executor.submit(backtest_single_config, symbol, run_params, df, cash, commission)
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

    return results


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
        _fn = {'RSM': 'Rel Strength Mom', 'ER': 'Efficiency Ratio', 'MACC': 'Mom Accel',
               'MTD': 'Multi-TF Divergence', 'OBVT': 'OBV Trend', 'STRK': 'Streak',
               'CHOP': 'Choppiness', 'VPD': 'Vol-Price Div', 'VA': 'Volume Anomaly',
               'CS': 'Candle Struct', 'ADX': 'Norm ADX'}
        for _fi in range(1, 12):
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

        print(f"\n2️⃣  VALIDATION PHASE (Out-of-Sample)")
        print(f"   Testing best parameters on unseen {test_start} to {test_end} data...")

        test_results = optimize_single_period(
            data_cache, valid_symbols, [best_params_tuple], period_idx, 'test', config
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
    print("OPTIMAL PARAMETERS SUMMARY (TREND-FOLLOWING)")
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

    print(f"\n🔹 LABEL SETTINGS:")
    print(f"   label_lookahead:          {params['label_lookahead']}")
    print(f"   label_dead_zone:          {params['label_dead_zone']}")
    print(f"   use_magnitude_labels:     {params['use_magnitude_labels']}")

    print("\n🔹 TREND-FOLLOWING FEATURES:")
    fnames = {'RSM': 'Relative Strength Momentum', 'ER': 'Efficiency Ratio',
              'MACC': 'Momentum Acceleration', 'MTD': 'Multi-Timeframe Divergence',
              'OBVT': 'OBV Trend', 'STRK': 'Streak Pattern', 'CHOP': 'Choppiness Index',
              'VPD': 'Volume-Price Divergence', 'VA': 'Volume Anomaly',
              'CS': 'Candle Structure', 'ADX': 'Normalized ADX'}
    for fi in range(1, 12):
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
    print(f"   use_trailing_atr_exit:    {params.get('use_trailing_atr_exit', False)}")
    if params.get('use_trailing_atr_exit', False):
        print(f"   trailing_atr_mult:        {params['trailing_atr_mult']}")
        print(f"   trailing_atr_warmup:      {params['trailing_atr_warmup']}")

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
    print("\n    # Label Settings")
    print(f"    'label_lookahead': {params['label_lookahead']},")
    print(f"    'label_dead_zone': {params['label_dead_zone']},")
    print(f"    'use_magnitude_labels': {params['use_magnitude_labels']},")
    _cpnames = {
        'RSM': 'Relative Strength Momentum', 'ER': 'Efficiency Ratio',
        'MACC': 'Momentum Acceleration', 'MTD': 'Multi-Timeframe Divergence',
        'OBVT': 'OBV Trend', 'STRK': 'Streak Pattern', 'CHOP': 'Choppiness Index',
        'VPD': 'Volume-Price Divergence', 'VA': 'Volume Anomaly',
        'CS': 'Candle Structure', 'ADX': 'Normalized ADX',
    }
    for _fi in range(1, 12):
        if f'f{_fi}_type' in params:
            print(f"\n    # Feature {_fi} ({_cpnames.get(params[f'f{_fi}_type'], '')})")
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
    print("\n    # ATR Trailing Stop Exit Settings")
    print(f"    'use_trailing_atr_exit': {params.get('use_trailing_atr_exit', False)},")
    print(f"    'trailing_atr_mult': {params.get('trailing_atr_mult', 2.5)},")
    print(f"    'trailing_atr_warmup': {params.get('trailing_atr_warmup', 3)},")
    print("\n    # Risk Management")
    print(f"    'position_size_pct': Decimal('{params['position_size_pct']}'),")
    print(f"    'stop_loss_pct': Decimal('{params['stop_loss_pct']}'),")
    print(f"    'use_stop_loss': {params['use_stop_loss']},")
    print(f"    'long_only': {params['long_only']},")
    print("\n    # Other")
    print(f"    'verbose': False,")
    print(f"    'test_start_idx': {params['test_start_idx']},")
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
