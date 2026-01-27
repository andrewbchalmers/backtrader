#!/usr/bin/env python3
"""
Backtest script for Lorentzian Classification Strategy - Trend Features

Usage:
    source bt/bin/activate
    cd strategies/LORENTZIAN_CLASSIFICATION_3/
    python backtest.py

This script runs a backtest of the ML-based Lorentzian Classification strategy
using the trend-focused feature vector:
- RSI(14): Momentum indicator
- ADX(14): Trend strength
- ATR Ratio: Normalized volatility
- Price Position: Location in recent range
- Efficiency Ratio: Trend quality/efficiency
"""

import sys
from decimal import Decimal
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import backtrader as bt
import yfinance as yf
from math import isnan
import matplotlib.pyplot as plt
from lorentzian_classification import Strategy
import pandas as pd
import numpy as np


class BuySellArrows(bt.observers.BuySell):
    """Custom observer for buy/sell arrow markers on chart."""
    plotlines = dict(
        buy=dict(marker='^', markersize=8, color='lime', fillstyle='full', ls=''),
        sell=dict(marker='v', markersize=8, color='red', fillstyle='full', ls='')
    )

    def next(self):
        super(BuySellArrows, self).next()
        if self.lines.buy[0] and not isnan(self.lines.buy[0]):
            self.lines.buy[0] = self.data.low[0] * 0.97
        if self.lines.sell[0] and not isnan(self.lines.sell[0]):
            self.lines.sell[0] = self.data.high[0] * 1.03


class PortfolioValue(bt.Observer):
    """Observer to track portfolio value over time."""
    lines = ('value',)
    plotinfo = dict(plot=False, subplot=False)

    def next(self):
        self.lines.value[0] = self._owner.broker.getvalue()

    def prenext(self):
        self.lines.value[0] = self._owner.broker.getvalue()


def calculate_lookback(strategy_class, strategy_params=None):
    """
    Calculate required lookback period from strategy parameters.
    """
    params_dict = {}
    for param_name in dir(strategy_class.params):
        if not param_name.startswith('_'):
            param_value = getattr(strategy_class.params, param_name)
            params_dict[param_name] = param_value

    if strategy_params:
        params_dict.update(strategy_params)

    lookback_candidates = []
    for param_name, param_value in params_dict.items():
        if isinstance(param_value, Decimal):
            try:
                param_value = int(param_value)
            except (ValueError, TypeError):
                continue
        if isinstance(param_value, int) and param_value > 0:
            exclude_patterns = ['verbose', 'plot', 'print', 'count', 'feature']
            if not any(pattern in param_name.lower() for pattern in exclude_patterns):
                lookback_candidates.append(param_value)

    # Need extra warmup for ML model training data
    max_lookback = (max(lookback_candidates) if lookback_candidates else 50) + 100

    print(f"\n📊 Calculated Lookback: {max_lookback} bars")
    print(f"   Found period parameters: {sorted(lookback_candidates, reverse=True)[:5]}")

    return max_lookback


# =============================================================================
# Configuration
# =============================================================================

symbol = "USAR"
initial_cash = 10_000

# Backtest date range (test period - trades will only occur within this range)
start_date = "2025-01-01"  # Start of test period
end_date = "2026-01-01"    # End of test period
timeframe = "1d"           # Bar timeframe: 1m, 5m, 15m, 30m, 1h, 4h, 1d

# Strategy parameters - Trend Features configuration
strategy_params = {
    # General Settings
    'neighbors_count': 10,
    'max_bars_back': 2000,
    'feature_count': 5,

    # Label Mode: False=mean-reversion (expect reversals), True=trend-following (expect continuation)
    # Mean-reversion works better with oscillators (RSI, CCI), trend-following with trend features (ADX, ER)
    'trend_following_labels': False,

    # Re-entry Mode: True=enter anytime signal is favorable, False=only enter on signal flip
    # True gives more trades and catches moves you'd otherwise miss
    'allow_reentry': True,

    # Minimum prediction strength to trade (filters out weak/ranging signals)
    # 0=any signal, 4=half neighbors agree, 6=strong consensus, 8=unanimous
    'min_prediction_strength': 4,

    # Feature 1: RSI (Momentum)
    'f1_type': 'RSI',
    'f1_param_a': 14,
    'f1_param_b': 1,

    # Feature 2: ADX (Trend Strength)
    'f2_type': 'ADX',
    'f2_param_a': 14,
    'f2_param_b': 1,

    # Feature 3: ATR Ratio (Normalized Volatility)
    'f3_type': 'ATRR',
    'f3_param_a': 14,
    'f3_param_b': 1,

    # Feature 4: Price Position (Range Location)
    'f4_type': 'PP',
    'f4_param_a': 20,
    'f4_param_b': 1,

    # Feature 5: Efficiency Ratio (Trend Quality)
    'f5_type': 'ER',
    'f5_param_a': 10,
    'f5_param_b': 1,

    # Filters
    'use_volatility_filter': True,
    'use_regime_filter': True,
    'regime_threshold': -0.1,
    'use_adx_filter': False,
    'adx_threshold': 20,
    'use_ema_filter': False,
    'ema_period': 50,
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
    'bars_to_hold': 10000,

    # RSI Exit Settings
    'use_rsi_exit': False,        # Enable RSI threshold exits
    'rsi_exit_period': 14,        # RSI period for exit signals
    'rsi_overbought': 70,         # Exit longs when RSI >= this
    'rsi_oversold': 30,           # Exit shorts when RSI <= this

    # Kernel Exit Settings
    'use_kernel_exit': True,      # Exit when price crosses below kernel line

    # Risk Management
    'position_size_pct': Decimal('0.95'),
    'stop_loss_pct': Decimal('0.05'),
    'use_stop_loss': True,

    # Trade Direction
    'long_only': True,  # Set to False to enable short selling

    # Display
    'verbose': True,

    # Backtest control (set by script - do not modify)
    'test_start_idx': 0,  # Will be set automatically
}

# =============================================================================
# Data Download
# =============================================================================

lookback_bars = calculate_lookback(Strategy, strategy_params)

# Bars per trading day for each timeframe
bars_per_day = {
    '1m': 390,   # 6.5 hours * 60 minutes
    '5m': 78,    # 6.5 hours * 12
    '15m': 26,   # 6.5 hours * 4
    '30m': 13,   # 6.5 hours * 2
    '1h': 7,     # ~7 trading hours
    '4h': 2,     # ~2 bars per day
    '1d': 1,     # 1 bar per day
}
bpd = bars_per_day.get(timeframe, 1)

# Yahoo Finance data limits (calendar days from today)
yf_max_days = {
    '1m': 7,
    '5m': 60,
    '15m': 60,
    '30m': 60,
    '1h': 730,
    '4h': 730,
    '1d': 99999,  # No practical limit
}
max_days = yf_max_days.get(timeframe, 99999)

# Calculate max available bars for this timeframe
# Trading days ≈ calendar days * 5/7 (weekdays only)
max_trading_days = int(max_days * 5 / 7)
max_available_bars = max_trading_days * bpd

# Reserve some bars for test period (at least 20% or 100 bars minimum)
min_test_bars = max(100, int(max_available_bars * 0.2))
max_lookback_bars = max_available_bars - min_test_bars

# Adjust lookback if it exceeds what's available
original_lookback = lookback_bars
if lookback_bars > max_lookback_bars:
    lookback_bars = max_lookback_bars
    print(f"⚠️  Adjusting lookback from {original_lookback} to {lookback_bars} bars (Yahoo {timeframe} limit: {max_days} days)")
    # Also update strategy params so the strategy knows about reduced lookback
    if 'max_bars_back' in strategy_params and strategy_params['max_bars_back'] > lookback_bars:
        strategy_params['max_bars_back'] = lookback_bars - 100  # Leave buffer

# For intraday with limited data, download all available and use most recent
if timeframe != '1d' and max_days < 365:
    # Download all available data for this timeframe
    data_start = datetime.now() - timedelta(days=max_days - 1)
    data_end = datetime.now()
    print(f"   Timeframe {timeframe}: downloading last {max_days} days of available data")
    print(f"   Lookback: {lookback_bars} bars, Test period: remaining bars")
else:
    # Parse user-specified dates for daily timeframe
    test_start = datetime.strptime(start_date, "%Y-%m-%d")
    test_end = datetime.strptime(end_date, "%Y-%m-%d")

    # Calculate lookback start date
    lookback_trading_days = lookback_bars / bpd
    lookback_calendar_days = int(lookback_trading_days * 1.5) + 10
    data_start = test_start - timedelta(days=lookback_calendar_days)
    data_end = test_end
    print(f"   Test period: {start_date} to {end_date} ({timeframe} bars)")
print(f"   Downloading data from {data_start.date()} to {data_end.date()}...")

# Download data
df = yf.download(symbol, start=data_start, end=data_end, interval=timeframe)

if df.empty:
    print(f"❌ Error: No data available for {symbol} with {timeframe} timeframe")
    sys.exit(1)

df.index = df.index.tz_localize(None)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.columns = ['open', 'high', 'low', 'close', 'volume']

# For intraday with limited data, use lookback_bars as the split point
if timeframe != '1d' and max_days < 365:
    # Use first lookback_bars for warmup, rest for testing
    if len(df) <= lookback_bars:
        print(f"❌ Error: Only {len(df)} bars available, need at least {lookback_bars} for lookback + test data")
        print(f"   Try reducing max_bars_back in strategy_params")
        sys.exit(1)
    actual_test_start_idx = lookback_bars
    print(f"   Downloaded {len(df)} bars: {lookback_bars} lookback + {len(df) - lookback_bars} test bars")
else:
    # For daily data, use date-based split
    test_start_mask = df.index >= pd.Timestamp(start_date)
    if test_start_mask.any():
        actual_test_start_idx = int(test_start_mask.argmax())
    else:
        print(f"⚠️  Warning: No data found on or after {start_date}")
        actual_test_start_idx = lookback_bars  # Fallback to bar-based split

    # Verify we have enough lookback data
    if actual_test_start_idx < lookback_bars:
        print(f"⚠️  Warning: Only {actual_test_start_idx} bars for lookback (need {lookback_bars})")
        print(f"   Strategy will use available data but results may be less accurate")

# Split for reporting
lookback_df = df.iloc[:actual_test_start_idx]
test_df = df.iloc[actual_test_start_idx:]

if len(lookback_df) > 0:
    print(f"   Lookback period: {lookback_df.index[0].date()} to {lookback_df.index[-1].date()} ({len(lookback_df)} bars)")
print(f"   Test period: {test_df.index[0].date()} to {test_df.index[-1].date()} ({len(test_df)} bars)")

# Set test_start_idx in strategy params so trading only starts in test period
strategy_params['test_start_idx'] = actual_test_start_idx

# Download SPY for benchmark comparison (use actual test period dates)
spy_start = test_df.index[0]
spy_end = test_df.index[-1]
spy_df = yf.download('SPY', start=spy_start, end=spy_end + timedelta(days=1), interval=timeframe, progress=False)
spy_df.index = spy_df.index.tz_localize(None)

# Calculate SPY buy-and-hold return
spy_initial_price = float(spy_df['Close'].iloc[0].iloc[0]) if isinstance(spy_df['Close'].iloc[0], pd.Series) else float(spy_df['Close'].iloc[0])
spy_final_price = float(spy_df['Close'].iloc[-1].iloc[0]) if isinstance(spy_df['Close'].iloc[-1], pd.Series) else float(spy_df['Close'].iloc[-1])
spy_shares = initial_cash / spy_initial_price
spy_final_value = spy_shares * spy_final_price
spy_return = (spy_final_value / initial_cash - 1) * 100

# =============================================================================
# Backtrader Setup
# =============================================================================

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
cerebro.addstrategy(Strategy, **strategy_params)

# Broker settings
cerebro.broker.setcash(initial_cash)
cerebro.broker.setcommission(commission=0.0)

# Analyzers
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")
cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")

# Observers
cerebro.addobserver(BuySellArrows, plot=True, subplot=False)
cerebro.addobserver(bt.observers.Trades, plot=True, subplot=False)
cerebro.addobserver(PortfolioValue)

# =============================================================================
# Run Backtest
# =============================================================================

print(f"\n🚀 Running Lorentzian Classification backtest (Trend Features)...")
print(f"\nStrategy Configuration:")
print(f"  ML Model:")
print(f"    - Neighbors (K): {strategy_params['neighbors_count']}")
print(f"    - Max Bars Back: {strategy_params['max_bars_back']}")
print(f"    - Features: {strategy_params['feature_count']}")
label_mode = "TREND-FOLLOWING" if strategy_params.get('trend_following_labels', False) else "MEAN-REVERSION"
print(f"    - Label Mode: {label_mode}")
reentry_mode = "ANYTIME" if strategy_params.get('allow_reentry', True) else "SIGNAL FLIP ONLY"
print(f"    - Entry Mode: {reentry_mode}")
min_strength = strategy_params.get('min_prediction_strength', 0)
print(f"    - Min Signal Strength: {min_strength} (of {strategy_params['neighbors_count']} neighbors)")
print(f"  Trend Features:")
print(f"    - F1: {strategy_params['f1_type']}({strategy_params['f1_param_a']}) - Momentum")
print(f"    - F2: {strategy_params['f2_type']}({strategy_params['f2_param_a']}) - Trend Strength")
print(f"    - F3: {strategy_params['f3_type']}({strategy_params['f3_param_a']}) - Normalized Volatility")
print(f"    - F4: {strategy_params['f4_type']}({strategy_params['f4_param_a']}) - Price Position")
print(f"    - F5: {strategy_params['f5_type']}({strategy_params['f5_param_a']}) - Efficiency Ratio")
print(f"  Filters:")
print(f"    - Volatility: {'ON' if strategy_params['use_volatility_filter'] else 'OFF'}")
print(f"    - Regime: {'ON' if strategy_params['use_regime_filter'] else 'OFF'} (threshold: {strategy_params['regime_threshold']})")
print(f"    - ADX: {'ON' if strategy_params['use_adx_filter'] else 'OFF'}")
print(f"    - EMA({strategy_params['ema_period']}): {'ON' if strategy_params['use_ema_filter'] else 'OFF'}")
print(f"    - SMA({strategy_params['sma_period']}): {'ON' if strategy_params['use_sma_filter'] else 'OFF'}")
print(f"  Kernel:")
print(f"    - Use Kernel Filter: {'ON' if strategy_params['use_kernel_filter'] else 'OFF'}")
print(f"    - Kernel Smoothing: {'ON' if strategy_params['use_kernel_smoothing'] else 'OFF'}")
print(f"    - Lookback: {strategy_params['kernel_lookback']}, Weight: {strategy_params['kernel_rel_weight']}")
print(f"  Exit:")
print(f"    - Dynamic Exits: {'ON' if strategy_params['use_dynamic_exits'] else 'OFF'}")
print(f"    - RSI Exit: {'ON' if strategy_params['use_rsi_exit'] else 'OFF'}")
print(f"    - Kernel Exit: {'ON' if strategy_params.get('use_kernel_exit', False) else 'OFF'}")
print(f"    - Holding Period: {strategy_params['bars_to_hold']} bars")
print()

results = cerebro.run()
strat = results[0]

# =============================================================================
# Results Analysis (calculated from TEST PERIOD ONLY)
# =============================================================================

# Get trade statistics from analyzer (these are correct since no trades during lookback)
trades = strat.analyzers.trades.get_analysis()
sqn = strat.analyzers.sqn.get_analysis()

# Extract portfolio values for test period only
test_portfolio_values = []
observer = strat.observers.portfoliovalue
for i in range(len(observer.lines.value)):
    if i >= actual_test_start_idx:
        try:
            val = observer.lines.value.array[i]
            if not np.isnan(val) and val > 0:
                test_portfolio_values.append(val)
        except (IndexError, AttributeError):
            break

# Calculate returns from test period only
if len(test_portfolio_values) >= 2:
    test_start_value = test_portfolio_values[0]
    test_final_value = test_portfolio_values[-1]

    # Daily returns for test period
    daily_returns = []
    for i in range(1, len(test_portfolio_values)):
        daily_ret = (test_portfolio_values[i] / test_portfolio_values[i-1]) - 1
        daily_returns.append(daily_ret)

    # Calculate Sharpe ratio from test period returns
    if len(daily_returns) > 1:
        avg_daily_return = np.mean(daily_returns)
        std_daily_return = np.std(daily_returns, ddof=1)
        sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0
    else:
        sharpe_ratio = 0

    # Calculate max drawdown from test period only
    peak = test_portfolio_values[0]
    max_dd_pct = 0
    max_dd_money = 0
    for val in test_portfolio_values:
        if val > peak:
            peak = val
        dd_pct = (peak - val) / peak * 100
        dd_money = peak - val
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            max_dd_money = dd_money
else:
    test_start_value = initial_cash
    test_final_value = cerebro.broker.getvalue()
    sharpe_ratio = 0
    max_dd_pct = 0
    max_dd_money = 0
    daily_returns = []

# Calculate metrics
total_trades = trades.get('total', {}).get('total', 0)
if total_trades > 0:
    avg_win = trades.get('won', {}).get('pnl', {}).get('average', 0)
    avg_loss = abs(trades.get('lost', {}).get('pnl', {}).get('average', 0))
    rr_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0

    total_win_pnl = trades.get('won', {}).get('pnl', {}).get('total', 0)
    total_loss_pnl = abs(trades.get('lost', {}).get('pnl', {}).get('total', 0))
    profit_factor = (total_win_pnl / total_loss_pnl) if total_loss_pnl > 0 else 0

    win_count = trades.get('won', {}).get('total', 0)
    loss_count = trades.get('lost', {}).get('total', 0)
    win_rate = win_count / total_trades * 100 if total_trades > 0 else 0
    loss_rate = loss_count / total_trades * 100 if total_trades > 0 else 0
    expectancy = (win_rate/100 * avg_win) - (loss_rate/100 * avg_loss)

    max_win_streak = trades.get('streak', {}).get('won', {}).get('longest', 0)
    max_loss_streak = trades.get('streak', {}).get('lost', {}).get('longest', 0)

    best_trade = trades.get('won', {}).get('pnl', {}).get('max', 0)
    worst_trade = trades.get('lost', {}).get('pnl', {}).get('max', 0)

    avg_trade_len = trades.get('len', {}).get('average', 0)
    max_trade_len = trades.get('len', {}).get('max', 0)
    min_trade_len = trades.get('len', {}).get('min', 0)
else:
    avg_win = avg_loss = rr_ratio = profit_factor = expectancy = 0
    max_win_streak = max_loss_streak = 0
    best_trade = worst_trade = 0
    avg_trade_len = max_trade_len = min_trade_len = 0
    win_rate = loss_rate = win_count = loss_count = 0

# Use test period values for final calculations
final_value = test_final_value
total_return = final_value - initial_cash
total_return_pct = (final_value / initial_cash - 1) * 100

calmar_ratio = (total_return_pct / max_dd_pct) if max_dd_pct > 0 else 0
recovery_factor = (total_return / max_dd_money) if max_dd_money > 0 else 0
sqn_score = sqn.get('sqn', 0)

# Days traded is test period only
days_traded = len(test_df)
years = days_traded / 252
annualized_return = ((final_value / initial_cash) ** (1 / years) - 1) * 100 if years > 0 else 0

if total_trades > 0:
    total_bars_in_trades = trades.get('len', {}).get('total', 0)
    time_in_market = (total_bars_in_trades / days_traded) * 100
else:
    time_in_market = 0

# Get ML prediction accuracy stats
ml_stats = strat.get_prediction_stats()

# =============================================================================
# Print Results
# =============================================================================

print("\n" + "="*70)
print(f"LORENTZIAN CLASSIFICATION (TREND FEATURES) - {symbol}")
print(f"Test Period: {test_df.index[0].date()} to {test_df.index[-1].date()} ({len(test_df)} bars)")
print("="*70)

print(f"\n💰 Portfolio Performance:")
print(f"   Starting Value:     ${initial_cash:,.2f}")
print(f"   Final Value:        ${final_value:,.2f}")
print(f"   Total Return:       ${total_return:,.2f} ({total_return_pct:.2f}%)")
print(f"   Annualized Return:  {annualized_return:.2f}%")
print(f"\n   📊 Benchmark Comparison:")
print(f"   SPY Buy & Hold:     ${spy_final_value:,.2f} ({spy_return:.2f}%)")
print(f"   Outperformance:     {total_return_pct - spy_return:.2f}%")

print(f"\n📉 Risk Metrics:")
print(f"   Sharpe Ratio:       {sharpe_ratio:.3f}" if sharpe_ratio != 0 else "   Sharpe Ratio:       N/A")
print(f"   Calmar Ratio:       {calmar_ratio:.3f}")
print(f"   SQN (Quality):      {sqn_score:.2f}")
print(f"   Max Drawdown:       {max_dd_pct:.2f}%")
print(f"   Max Drawdown ($):   ${max_dd_money:,.2f}")
print(f"   Recovery Factor:    {recovery_factor:.2f}")

print(f"\n📈 Trade Statistics:")
print(f"   Total Trades:       {total_trades}")
if total_trades > 0:
    print(f"   Wins:               {win_count} ({win_rate:.1f}%)")
    print(f"   Losses:             {loss_count} ({loss_rate:.1f}%)")
    print(f"\n   💵 Profit Analysis:")
    print(f"   Total Wins:         ${total_win_pnl:,.2f}")
    print(f"   Total Losses:       ${total_loss_pnl:,.2f}")
    print(f"   Net P&L:            ${total_return:,.2f}")
    print(f"   Avg Win:            ${avg_win:,.2f}")
    print(f"   Avg Loss:           ${avg_loss:,.2f}")
    print(f"   Best Trade:         ${best_trade:,.2f}")
    print(f"   Worst Trade:        ${worst_trade:,.2f}")
    print(f"\n   📊 Performance Ratios:")
    print(f"   RR Ratio:           {rr_ratio:.2f}")
    print(f"   Profit Factor:      {profit_factor:.2f}")
    print(f"   Expectancy:         ${expectancy:.2f}")
    print(f"\n   ⏱️  Trade Duration:")
    print(f"   Avg Duration:       {avg_trade_len:.1f} bars")
    print(f"   Longest Trade:      {max_trade_len} bars")
    print(f"   Shortest Trade:     {min_trade_len} bars")
    print(f"   Time in Market:     {time_in_market:.1f}%")
    print(f"\n   🔥 Streaks:")
    print(f"   Max Win Streak:     {max_win_streak}")
    print(f"   Max Loss Streak:    {max_loss_streak}")

# ML Prediction Accuracy Section
print(f"\n🤖 ML Model Accuracy:")
print(f"   Total Predictions:  {ml_stats['total']}")
if ml_stats['total'] > 0:
    print(f"   Overall Accuracy:   {ml_stats['accuracy_pct']:.1f}% ({ml_stats['correct']}/{ml_stats['total']})")
    print(f"\n   📈 Bullish Predictions:")
    print(f"   Total Bullish:      {ml_stats['bullish_total']}")
    if ml_stats['bullish_total'] > 0:
        print(f"   Bullish Accuracy:   {ml_stats['bullish_accuracy_pct']:.1f}% ({ml_stats['bullish_correct']}/{ml_stats['bullish_total']})")
    print(f"\n   📉 Bearish Predictions:")
    print(f"   Total Bearish:      {ml_stats['bearish_total']}")
    if ml_stats['bearish_total'] > 0:
        print(f"   Bearish Accuracy:   {ml_stats['bearish_accuracy_pct']:.1f}% ({ml_stats['bearish_correct']}/{ml_stats['bearish_total']})")
    print(f"\n   ⚖️  Model Bias:")
    print(f"   Bullish Bias:       {ml_stats['bullish_bias_pct']:.1f}%")
    print(f"   (50% = balanced, >50% = bullish bias, <50% = bearish bias)")
else:
    print(f"   No predictions made during test period")

# ML Diagnostics Section - Understanding why trades aren't happening
diag = strat.get_diagnostics()
print(f"\n🔬 ML Diagnostics (Raw Prediction Breakdown):")
print(f"   Total Bars Analyzed: {diag['total_bars']}")
if diag['total_bars'] > 0:
    print(f"\n   📊 Prediction Distribution:")
    print(f"   Bullish (>0):       {diag['bullish_predictions']} ({diag['bullish_pct']:.1f}%)")
    print(f"   Bearish (<0):       {diag['bearish_predictions']} ({diag['bearish_pct']:.1f}%)")
    print(f"   Neutral (=0):       {diag['neutral_predictions']} ({diag['neutral_pct']:.1f}%)")
    print(f"   Avg Prediction:     {diag['avg_prediction']:.2f}")
    print(f"\n   💪 Strong Signals (±{strategy_params['neighbors_count']//2}+ neighbors agree):")
    print(f"   Strong Bullish:     {diag['strong_bullish']}")
    print(f"   Strong Bearish:     {diag['strong_bearish']}")
    print(f"\n   🔄 Signal Activity:")
    print(f"   Signal Changes:     {diag['signal_changes']}")
    print(f"   Entry Attempts:     {diag['entry_attempts']} (times signal was bullish & not in position)")
    if diag['entry_attempts'] > 0:
        print(f"\n   🚫 Entry Blockers (what prevented entries):")
        print(f"   Kernel Filter:      {diag['entries_blocked_by_kernel']} ({diag['kernel_block_pct']:.1f}%)")
        print(f"   EMA Filter:         {diag['entries_blocked_by_ema']} ({diag['ema_block_pct']:.1f}%)")
        print(f"   SMA Filter:         {diag['entries_blocked_by_sma']} ({diag['sma_block_pct']:.1f}%)")

print("="*70 + "\n")

# =============================================================================
# Generate Charts
# =============================================================================

plt.style.use('dark_background')

print("📊 Creating backtrader plot...")
figs = cerebro.plot(
    style='candlestick',
    iplot=False,
    barup='#597D35',
    bardown='#FF7171',
    volume=False,
)

for fig in figs:
    fig[0].savefig(f"{symbol}_lorentzian_backtest_full.png", dpi=150, bbox_inches='tight')
print(f"✓ Saved backtrader plot to {symbol}_lorentzian_backtest_full.png")

# Create custom performance chart
print("📊 Creating custom performance chart...")

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#1a1a1a')
gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)

# Subplot 1: Portfolio value comparison
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor('#2a2a2a')

# Get portfolio values
portfolio_values = []
dates = []
observer = strat.observers.portfoliovalue
for i in range(len(observer.lines.value)):
    if i >= actual_test_start_idx:
        try:
            val = observer.lines.value.array[i]
            if not np.isnan(val) and val > 0:
                portfolio_values.append(val)
                dates.append(df.index[i].date())
        except (IndexError, AttributeError):
            break

if len(portfolio_values) >= 2:
    # Calculate SPY values
    spy_values = []
    for date in dates:
        try:
            matching_rows = spy_df.loc[spy_df.index.date == date, 'Close']
            if len(matching_rows) > 0:
                spy_price = float(matching_rows.iloc[0])
            else:
                valid_dates = spy_df.index[spy_df.index.date <= date]
                if len(valid_dates) > 0:
                    spy_price = float(spy_df.loc[valid_dates[-1], 'Close'].iloc[0])
                else:
                    spy_price = spy_initial_price
            spy_values.append(spy_shares * spy_price)
        except:
            spy_values.append(spy_values[-1] if spy_values else initial_cash)

    dates = pd.to_datetime(dates)

    ax1.plot(dates, portfolio_values, label=f'{symbol} Lorentzian ML (Trend)', linewidth=2, color='#00ff88')
    ax1.plot(dates, spy_values, label='SPY Buy & Hold', linewidth=2, color='#ff6b6b', linestyle='--')
    ax1.axhline(y=initial_cash, color='gray', linestyle=':', alpha=0.5, label='Initial Capital')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12, color='white')
    ax1.set_title(f'{symbol} Lorentzian Classification (Trend Features) vs SPY Buy & Hold', fontsize=14, fontweight='bold', color='white')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(colors='white')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Subplot 2: Cumulative returns
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#2a2a2a')
    strategy_returns = [(v / initial_cash - 1) * 100 for v in portfolio_values]
    spy_returns_pct = [(v / initial_cash - 1) * 100 for v in spy_values]

    ax2.plot(dates, strategy_returns, label=f'{symbol} Lorentzian ML (Trend)', linewidth=2, color='#00ff88')
    ax2.plot(dates, spy_returns_pct, label='SPY Buy & Hold', linewidth=2, color='#ff6b6b', linestyle='--')
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax2.fill_between(dates, strategy_returns, 0, alpha=0.3, color='#00ff88')
    ax2.set_xlabel('Date', fontsize=12, color='white')
    ax2.set_ylabel('Cumulative Return (%)', fontsize=12, color='white')
    ax2.set_title('Cumulative Returns Over Time', fontsize=14, fontweight='bold', color='white')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(colors='white')

    # Add final return annotations
    if strategy_returns:
        ax2.annotate(f'{strategy_returns[-1]:.1f}%',
                     xy=(dates[-1], strategy_returns[-1]),
                     xytext=(10, 0), textcoords='offset points',
                     fontsize=10, color='#00ff88', fontweight='bold')
    if spy_returns_pct:
        ax2.annotate(f'{spy_returns_pct[-1]:.1f}%',
                     xy=(dates[-1], spy_returns_pct[-1]),
                     xytext=(10, 0), textcoords='offset points',
                     fontsize=10, color='#ff6b6b', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{symbol}_lorentzian_backtest.png", dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    print(f"✓ Saved performance chart to {symbol}_lorentzian_backtest.png")
else:
    print("⚠️  Insufficient data for performance chart")

print("\n✅ Backtest complete!")
