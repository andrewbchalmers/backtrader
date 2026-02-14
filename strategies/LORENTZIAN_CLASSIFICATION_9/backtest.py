#!/usr/bin/env python3
"""
Backtest script for Lorentzian Classification Strategy - Diverse Features

Usage:
    source bt/bin/activate
    cd strategies/LORENTZIAN_CLASSIFICATION_5/
    python backtest.py

This script runs a backtest of the ML-based Lorentzian Classification strategy
using a diverse feature vector:
- RSM(20,252): Relative Strength Momentum
- VA(20): Volume Anomaly
- MTD(5,60): Multi-Timeframe Divergence
- ZS(50): Mean Reversion Z-Score
- VCR(20,100): Volatility Contraction Ratio
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

symbol = "nsc"
initial_cash = 10_000

# Load peer universe from classification CSV (fallback to default ETFs)
_peer_universe = 'SPY,QQQ,IWM,TLT,GLD,XLE,EFA'
try:
    with open('../classification_set.csv') as _f:
        _symbols = [line.strip() for line in _f if line.strip()]
        if _symbols:
            _peer_universe = ','.join(_symbols)
except FileNotFoundError:
    pass

# Backtest date range (test period - trades will only occur within this range)
start_date = "2025-02-06"  # Start of test period
end_date = "2026-02-06"    # End of test period
timeframe = "1d"           # Bar timeframe: 1m, 5m, 15m, 30m, 1h, 4h, 1d

# Strategy parameters - Trend-Following Features configuration
strategy_params = {
    # ML Settings
    'neighbors_count': 9,
    'max_bars_back': 7000,
    'feature_count': 8,
    'trend_following_labels': True,    # Trend-following labels (ride continuation)
    'allow_reentry': True,             # Enter anytime signal is favorable
    'min_prediction_strength': 20,     # Normalized scale: 0-100

    # Label Settings
    'label_lookahead': 8,              # Longer lookahead for trends
    'label_dead_zone': 0.225,          # Min ATR move for label
    'use_magnitude_labels': True,      # True=continuous, False=binary +1/-1

    # Feature 1 (RSM - Relative Strength Momentum)
    'f1_type': 'RSM',
    'f1_param_a': 40,
    'f1_param_b': 252,

    # Feature 2 (ER - Efficiency Ratio)
    'f2_type': 'ER',
    'f2_param_a': 25,
    'f2_param_b': 13,

    # Feature 3 (MTD - Multi-Timeframe Divergence)
    'f3_type': 'MTD',
    'f3_param_a': 8,
    'f3_param_b': 252,

    # Feature 4 (STRK - Streak Pattern)
    'f4_type': 'STRK',
    'f4_param_a': 30,
    'f4_param_b': 3,

    # Feature 5 (VCOMP - Volatility Compression)
    'f5_type': 'VCOMP',
    'f5_param_a': 4,
    'f5_param_b': 16,

    # Feature 6 (MPER - Momentum Persistence)
    'f6_type': 'MPER',
    'f6_param_a': 4,
    'f6_param_b': 20,

    # Feature 7 (VMC - Volume-Momentum Coupling)
    'f7_type': 'VMC',
    'f7_param_a': 3,
    'f7_param_b': 40,

    # Feature 8 (CS - Candle Structure)
    'f8_type': 'CS',
    'f8_param_a': 5,
    'f8_param_b': 2,

    # Unused slots (available for optimization)
    'f9_type': 'CS',
    'f9_param_a': 5,
    'f9_param_b': 2,

    'f10_type': 'CS',
    'f10_param_a': 5,
    'f10_param_b': 2,

    'f11_type': 'CS',
    'f11_param_a': 5,
    'f11_param_b': 2,

    'f12_type': 'VCOMP',
    'f12_param_a': 4,
    'f12_param_b': 16,

    'f13_type': 'MPER',
    'f13_param_a': 4,
    'f13_param_b': 20,

    'f14_type': 'VMC',
    'f14_param_a': 5,
    'f14_param_b': 40,

    # Filters
    'use_volatility_filter': True,
    'use_regime_filter': True,
    'regime_threshold': 1,
    'regime_period': 'weekly',
    'use_regime_direction': True,
    'regime_stability_min': 0.0,
    'regime_stability_window': 60,
    'regime_max_flips': 8,
    'use_adx_filter': True,
    'adx_threshold': 14,
    'use_ema_filter': False,
    'ema_period': 25,
    'ema_slope_lookback': 5,
    'use_sma_filter': False,
    'sma_period': 100,
    'sma_slope_lookback': 5,

    # Kernel Settings
    'use_kernel_filter': False,
    'use_kernel_smoothing': False,
    'kernel_lookback': 8,
    'kernel_rel_weight': 8.0,
    'kernel_start_bar': 25,
    'kernel_lag': 2,

    # Exit Settings
    'use_dynamic_exits': False,
    'bars_to_hold': 100000,

    # RSI Exit Settings (widened thresholds for trend-following)
    'use_rsi_exit': True,
    'rsi_exit_period': 14,
    'rsi_overbought': 80,             # Widened — don't cut trends short
    'rsi_oversold': 20,               # Widened

    # Kernel Exit Settings
    'use_kernel_exit': False,

    # ATR Trailing Stop Exit Settings (wider for trends)
    'use_trailing_atr_exit': True,
    'trailing_atr_mult': 2.5,         # Wider than mean-reversion's 2.0
    'trailing_atr_warmup': 3,

    # Risk Management
    'position_size_pct': Decimal('0.95'),
    'stop_loss_pct': Decimal('0.05'),
    'use_stop_loss': True,

    # Trade Direction
    'long_only': True,

    # Display
    'verbose': True,

    # Cross-Symbol Training
    'use_cross_symbol_training': True,
    'cross_symbol_etfs': _peer_universe,
    'cross_symbol_lookback_years': 5,
    'use_regime_balancing': True,
    'cross_symbol_auto_peers': True,
    'cross_symbol_target_symbol': symbol,
    'cross_symbol_max_peers': 7,

    # Fundamental / Earnings Settings
    'use_fundamental_filter': True,
    'fundamental_symbol': symbol,
    'fundamental_quality_weight': 0.2,
    'fundamental_momentum_weight': 0.3,
    'earnings_blackout_before': 5,
    'earnings_blackout_after': 2,
    'close_before_earnings': True,
    'min_trending_probability': 20,
    'full_position_threshold': 50,
    'reduced_position_pct': Decimal('0.75'),
    'min_quality_score': 20,
    'min_momentum_score': 20,

    # Backtest control (set by script - do not modify)
    'test_start_idx': 0,
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
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                    timeframe=bt.TimeFrame.Days, riskfreerate=0.0)
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

print(f"\n🚀 Running Lorentzian Classification backtest (Diverse Features)...")
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
print(f"  Labels:")
print(f"    - Lookahead: {strategy_params.get('label_lookahead', 4)} bars")
print(f"    - Dead Zone: {strategy_params.get('label_dead_zone', 0.5)} ATR")
print(f"    - Magnitude Labels: {'ON' if strategy_params.get('use_magnitude_labels', True) else 'OFF (binary)'}")
print(f"  Diverse Features:")
print(f"    - F1: {strategy_params['f1_type']}({strategy_params['f1_param_a']},{strategy_params['f1_param_b']}) - Relative Strength Momentum")
print(f"    - F2: {strategy_params['f2_type']}({strategy_params['f2_param_a']}) - Volume Anomaly")
print(f"    - F3: {strategy_params['f3_type']}({strategy_params['f3_param_a']},{strategy_params['f3_param_b']}) - Multi-Timeframe Divergence")
print(f"    - F4: {strategy_params['f4_type']}({strategy_params['f4_param_a']}) - Mean Reversion Z-Score")
print(f"    - F5: {strategy_params['f5_type']}({strategy_params['f5_param_a']}) - Efficiency Ratio")
fc = strategy_params['feature_count']
if fc > 5:
    feature_names = {'VPD': 'Volume-Price Divergence', 'CS': 'Candle Structure',
                     'MACC': 'Momentum Acceleration', 'OBVT': 'OBV Trend', 'STRK': 'Streak Pattern',
                     'CHOP': 'Choppiness Index',
                     'RSM': 'Relative Strength Momentum', 'VA': 'Volume Anomaly',
                     'MTD': 'Multi-Timeframe Divergence', 'ZS': 'Z-Score', 'ER': 'Efficiency Ratio',
                     'RSI': 'RSI', 'ADX': 'ADX', 'ATRR': 'ATR Ratio', 'PP': 'Price Position', 'VCR': 'Volatility Contraction',
                     'VCOMP': 'Volatility Compression', 'MPER': 'Momentum Persistence', 'VMC': 'Vol-Mom Coupling'}
    for fi in range(6, min(fc + 1, 15)):
        ft = strategy_params.get(f'f{fi}_type', '?')
        pa = strategy_params.get(f'f{fi}_param_a', 0)
        pb = strategy_params.get(f'f{fi}_param_b', 0)
        print(f"    - F{fi}: {ft}({pa},{pb}) - {feature_names.get(ft, ft)}")
print(f"  Filters:")
print(f"    - Volatility: {'ON' if strategy_params['use_volatility_filter'] else 'OFF'}")
regime_thr = strategy_params['regime_threshold']
regime_per = strategy_params.get('regime_period', 'weekly')
regime_desc = "block bearish" if regime_thr == 0 else "require bullish" if regime_thr >= 1 else f"threshold={regime_thr}"
direction_desc = "+direction" if strategy_params.get('use_regime_direction', True) else ""
stability_min = strategy_params.get('regime_stability_min', 0.0)
stability_desc = f", stability>={stability_min}" if stability_min > 0 else ""
print(f"    - Regime: {'ON' if strategy_params['use_regime_filter'] else 'OFF'} ({regime_per} H/L, {regime_desc}{', ' + direction_desc if direction_desc else ''}{stability_desc})")
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
trailing = strategy_params.get('use_trailing_atr_exit', False)
print(f"    - ATR Trailing Stop: {'ON' if trailing else 'OFF'}", end="")
if trailing:
    print(f" ({strategy_params.get('trailing_atr_mult', 2.5)}x ATR, warmup={strategy_params.get('trailing_atr_warmup', 3)})")
else:
    print()
print(f"    - Holding Period: {strategy_params['bars_to_hold']} bars")
print(f"  Cross-Symbol Training:")
print(f"    - Enabled: {'ON' if strategy_params.get('use_cross_symbol_training', False) else 'OFF'}")
if strategy_params.get('use_cross_symbol_training', False):
    if strategy_params.get('cross_symbol_auto_peers', False):
        print(f"    - Auto Peers: ON (target={strategy_params.get('cross_symbol_target_symbol', '')}, max={strategy_params.get('cross_symbol_max_peers', 7)})")
        print(f"    - Universe: {len(strategy_params.get('cross_symbol_etfs', '').split(','))} symbols")
    else:
        print(f"    - ETFs: {strategy_params.get('cross_symbol_etfs', 'SPY,QQQ,IWM,TLT,GLD,XLE,EFA')}")
    print(f"    - Lookback Years: {strategy_params.get('cross_symbol_lookback_years', 5)}")
    print(f"    - Regime Balancing: {'ON' if strategy_params.get('use_regime_balancing', False) else 'OFF'}")
print(f"  Fundamental / Earnings:")
print(f"    - Enabled: {'ON' if strategy_params.get('use_fundamental_filter', False) else 'OFF'}")
if strategy_params.get('use_fundamental_filter', False):
    print(f"    - Symbol: {strategy_params.get('fundamental_symbol', symbol)}")
    print(f"    - Earnings Blackout: -{strategy_params.get('earnings_blackout_before', 5)}/+{strategy_params.get('earnings_blackout_after', 2)} days")
    print(f"    - Close Before Earnings: {'ON' if strategy_params.get('close_before_earnings', True) else 'OFF'}")
    print(f"    - Min Trending Prob: {strategy_params.get('min_trending_probability', 50)}")
    print(f"    - Full Position At: {strategy_params.get('full_position_threshold', 70)}")
    print(f"    - Reduced Position: {float(strategy_params.get('reduced_position_pct', Decimal('0.75')))*100:.0f}%")
    if strategy_params.get('min_quality_score', 0) > 0:
        print(f"    - Min Quality Score: {strategy_params.get('min_quality_score')}")
    if strategy_params.get('min_momentum_score', 0) > 0:
        print(f"    - Min Momentum Score: {strategy_params.get('min_momentum_score')}")
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
print(f"LORENTZIAN CLASSIFICATION (DIVERSE FEATURES) - {symbol}")
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
    print(f"\n   💪 Strong Signals (±50+ normalized strength):")
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

# Percentile Band Performance
if hasattr(strat, 'get_percentile_band_stats'):
    band_stats = strat.get_percentile_band_stats()
    total_band_trades = sum(b['trades'] for b in band_stats.values())
    if total_band_trades > 0:
        print(f"\n📊 Prediction Strength Band Performance (normalized |prediction| bands):")
        print(f"   {'Band':>8s}  {'Trades':>6s}  {'Win%':>6s}  {'AvgP&L':>8s}  {'TotalP&L':>9s}")
        print(f"   {'-'*8}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*9}")
        for band in ['0-20', '20-40', '40-60', '60-80', '80-100']:
            b = band_stats[band]
            if b['trades'] > 0:
                print(f"   {band:>8s}  {b['trades']:6d}  {b['win_rate']:5.1f}%  {b['avg_pnl_pct']:+7.2f}%  {b['total_pnl_pct']:+8.2f}%")
            else:
                print(f"   {band:>8s}  {0:6d}      -         -          -")

# Regime Diagnostics
if hasattr(strat, 'get_regime_diagnostics') and strategy_params.get('use_regime_filter', False):
    rd = strat.get_regime_diagnostics()
    total_regime_bars = rd['bullish_bars'] + rd['bearish_bars'] + rd['reverting_bars']
    if total_regime_bars > 0:
        rp = strategy_params.get('regime_period', 'weekly')
        print(f"\n🌍 Market Regime Distribution (previous {rp[:-2] if rp.endswith('ly') else rp}'s high/low):")
        print(f"   Bullish  (close > prev {rp} high): {rd['bullish_bars']:5d} bars ({rd['bullish_pct']:5.1f}%)")
        print(f"   Bearish  (close < prev {rp} low):  {rd['bearish_bars']:5d} bars ({rd['bearish_pct']:5.1f}%)")
        print(f"   Reverting (between):               {rd['reverting_bars']:5d} bars ({rd['reverting_pct']:5.1f}%)")
        if rd['reverting_bars'] > 0:
            print(f"     - Improving (from bearish):       {rd['reverting_improving_bars']:5d} bars ({rd['reverting_improving_pct']:5.1f}% of reverting)")
            print(f"     - Declining (from bullish):       {rd['reverting_declining_bars']:5d} bars ({rd['reverting_declining_pct']:5.1f}% of reverting)")
        total_trades = rd['trades_in_bullish'] + rd['trades_in_bearish'] + rd['trades_in_reverting']
        if total_trades > 0:
            print(f"\n   Trades by Regime:")
            print(f"   Bullish:              {rd['trades_in_bullish']}")
            print(f"   Bearish:              {rd['trades_in_bearish']}")
            print(f"   Reverting:            {rd['trades_in_reverting']}")
            if rd['trades_in_reverting'] > 0:
                print(f"     - Improving (buy):  {rd['trades_in_reverting_improving']}")
                print(f"     - Declining (buy):  {rd['trades_in_reverting_declining']}")

# Fundamental Analysis Summary
if strategy_params.get('use_fundamental_filter', False) and hasattr(strat, 'fundamental_provider') and strat.fundamental_provider is not None:
    print(f"\n📊 Fundamental Analysis Summary:")
    fund = strat.fundamental_provider
    # Use last close price and today's date for current scores
    from datetime import date
    last_price = float(strat.data.close[0]) if len(strat.data) > 0 else None
    summary = fund.get_summary(as_of_date=date.today(), price=last_price)
    print(f"   Symbol: {summary['symbol']}")
    print(f"   Quality Score:     {summary['quality_score']:.1f}/100")
    print(f"   Momentum Score:    {summary['momentum_score']:.1f}/100")
    print(f"   Trending Prob:     {summary['trending_probability']:.1f}/100")
    print(f"   Quarters Mapped:   {summary['quarters_mapped']}")

    # Show next earnings date
    next_earnings = fund.get_next_earnings_date(date.today())
    if next_earnings:
        days_until = fund.days_until_earnings(date.today())
        print(f"   Next Earnings:     {next_earnings.date()} ({days_until} days)")
    else:
        print(f"   Next Earnings:     Unknown")

    days_since = fund.days_since_earnings(date.today())
    if days_since is not None:
        confidence = fund.get_confidence_multiplier(date.today())
        print(f"   Days Since Last:   {days_since} (confidence: {confidence:.2f})")

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

    ax1.plot(dates, portfolio_values, label=f'{symbol} Lorentzian ML (Diverse)', linewidth=2, color='#00ff88')
    ax1.plot(dates, spy_values, label='SPY Buy & Hold', linewidth=2, color='#ff6b6b', linestyle='--')
    ax1.axhline(y=initial_cash, color='gray', linestyle=':', alpha=0.5, label='Initial Capital')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12, color='white')
    ax1.set_title(f'{symbol} Lorentzian Classification (Diverse Features) vs SPY Buy & Hold', fontsize=14, fontweight='bold', color='white')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(colors='white')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Subplot 2: Cumulative returns
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#2a2a2a')
    strategy_returns = [(v / initial_cash - 1) * 100 for v in portfolio_values]
    spy_returns_pct = [(v / initial_cash - 1) * 100 for v in spy_values]

    ax2.plot(dates, strategy_returns, label=f'{symbol} Lorentzian ML (Diverse)', linewidth=2, color='#00ff88')
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
