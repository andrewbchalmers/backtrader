# source bt/bin/activate
# cd strategies/SMA_ATR/
# python backtest.py

from decimal import Decimal
import matplotlib
matplotlib.use('Agg')
import backtrader as bt
import yfinance as yf
from math import isnan
import matplotlib.pyplot as plt
from sma_atr import Strategy
import pandas as pd
import numpy as np


class BuySellArrows(bt.observers.BuySell):
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
    """Observer to track portfolio value over time"""
    lines = ('value',)
    plotinfo = dict(plot=False, subplot=False)  # Don't plot in backtrader chart

    def next(self):
        self.lines.value[0] = self._owner.broker.getvalue()

    def prenext(self):
        self.lines.value[0] = self._owner.broker.getvalue()


def CalculateLookback(strategy_class, strategy_params=None):
    """
    Generic function to calculate lookback period from strategy parameters.
    Finds the highest integer value in the strategy's params that likely represents
    a period/length parameter.

    Args:
        strategy_class: The strategy class (not instance)
        strategy_params: Optional dict of parameter overrides

    Returns:
        int: Maximum lookback period needed
    """
    # Get default params from strategy class
    # backtrader params can be accessed as attributes
    params_dict = {}

    # Get all parameter names and their default values
    for param_name in dir(strategy_class.params):
        if not param_name.startswith('_'):
            param_value = getattr(strategy_class.params, param_name)
            params_dict[param_name] = param_value

    # Merge with any overrides
    if strategy_params:
        params_dict.update(strategy_params)

    # Find all integer/Decimal parameters that represent periods
    # Common naming patterns: *_len, *_period, *_window, trend_*, fast_*, slow_*
    lookback_candidates = []

    for param_name, param_value in params_dict.items():
        # Convert Decimal to int if needed
        if isinstance(param_value, Decimal):
            try:
                param_value = int(param_value)
            except (ValueError, TypeError):
                continue

        # Check if it's an integer and likely a period parameter
        if isinstance(param_value, int) and param_value > 0:
            # Exclude parameters that are clearly not periods
            exclude_patterns = ['verbose', 'plot', 'print']
            if not any(pattern in param_name.lower() for pattern in exclude_patterns):
                lookback_candidates.append(param_value)

    # Return the maximum value, or 0 if none found
    max_lookback = max(lookback_candidates) if lookback_candidates else 0

    print(f"\n📊 Calculated Lookback: {max_lookback} bars")
    print(f"   Found period parameters: {sorted(lookback_candidates, reverse=True)}")

    return max_lookback


# 1️⃣ Download data with lookback
symbol = "SOFI"

# Define strategy parameters
strategy_params = {
    'fast_len': 7,
    'slow_len': 50,
    'atr_len': 10,
    'atr_mult': Decimal("3.0"),
    'stop_loss_pct': Decimal("0.1")
}

# Calculate required lookback for the strategy
lookback_bars = CalculateLookback(Strategy, strategy_params)

# Download extra data for lookback period
# Add 50% buffer to ensure we have enough data even with market closures
total_bars_needed = 252 + lookback_bars  # 1 year + lookback
days_to_download = int(total_bars_needed * 1.5)  # Add buffer for weekends/holidays

print(f"   Downloading approximately {days_to_download} days of data...")

df = yf.download(symbol, period=f"{days_to_download}d", interval="1d")
df.index = df.index.tz_localize(None)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.columns = ['open', 'high', 'low', 'close', 'volume']

# Split data into lookback period and test period
if len(df) > lookback_bars:
    lookback_df = df.iloc[:lookback_bars]
    test_df = df.iloc[lookback_bars:]
    print(f"   Lookback period: {lookback_df.index[0].date()} to {lookback_df.index[-1].date()} ({len(lookback_df)} bars)")
    print(f"   Test period: {test_df.index[0].date()} to {test_df.index[-1].date()} ({len(test_df)} bars)")
    print(f"   Note: Strategy will use prenext() to skip trading until all indicators are ready")

    # Use the full dataframe for backtesting - backtrader will handle the warmup
    # The strategy's prenext() method ensures no trades during indicator warmup
    actual_test_start_date = test_df.index[0]
    actual_test_start_idx = lookback_bars
else:
    print(f"⚠️  Warning: Downloaded data ({len(df)} bars) is less than lookback requirement ({lookback_bars} bars)")
    actual_test_start_date = df.index[0]
    actual_test_start_idx = 0

# Download SPY for benchmark comparison (matching the test period only)
if len(df) > lookback_bars:
    spy_start_date = test_df.index[0]
    spy_end_date = test_df.index[-1]
else:
    spy_start_date = df.index[0]
    spy_end_date = df.index[-1]

spy_df = yf.download('SPY', start=spy_start_date, end=spy_end_date, progress=False)
spy_df.index = spy_df.index.tz_localize(None)

# Calculate SPY buy-and-hold return
initial_cash = 10_000
spy_initial_price = float(spy_df['Close'].iloc[0])
spy_final_price = float(spy_df['Close'].iloc[-1])
spy_shares = initial_cash / spy_initial_price
spy_final_value = spy_shares * spy_final_price
spy_return = (spy_final_value / initial_cash - 1) * 100

# 2️⃣ Backtrader setup
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
cerebro.addstrategy(Strategy,
                    fast_len=strategy_params['fast_len'],
                    slow_len=strategy_params['slow_len'],
                    atr_len=strategy_params['atr_len'],
                    atr_mult=strategy_params['atr_mult'],
                    stop_loss_pct=strategy_params['stop_loss_pct']
                    )

# Broker
cerebro.broker.setcash(initial_cash)
cerebro.broker.setcommission(commission=0.0)

# Analyzers
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")  # System Quality Number
cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")

# Observers
cerebro.addobserver(BuySellArrows, plot=True, subplot=False)
cerebro.addobserver(bt.observers.Trades, plot=True, subplot=False)
cerebro.addobserver(PortfolioValue)

# 3️⃣ Run
print(f"\n🚀 Running backtest...")
results = cerebro.run()
strat = results[0]

# Get analyzer results
sharpe = strat.analyzers.sharpe.get_analysis()
dd = strat.analyzers.dd.get_analysis()
trades = strat.analyzers.trades.get_analysis()
returns = strat.analyzers.returns.get_analysis()
sqn = strat.analyzers.sqn.get_analysis()
time_return = strat.analyzers.time_return.get_analysis()

# Calculate RR Ratio and other metrics
total_trades = trades.get('total', {}).get('total', 0)
if total_trades > 0:
    avg_win = trades.get('won', {}).get('pnl', {}).get('average', 0)
    avg_loss = abs(trades.get('lost', {}).get('pnl', {}).get('average', 0))
    rr_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0

    # Trade statistics
    total_win_pnl = trades.get('won', {}).get('pnl', {}).get('total', 0)
    total_loss_pnl = abs(trades.get('lost', {}).get('pnl', {}).get('total', 0))
    profit_factor = (total_win_pnl / total_loss_pnl) if total_loss_pnl > 0 else 0

    win_rate = trades['won']['total'] / total_trades * 100
    loss_rate = trades['lost']['total'] / total_trades * 100
    expectancy = (win_rate/100 * avg_win) - (loss_rate/100 * avg_loss)

    # Streaks
    max_win_streak = trades.get('streak', {}).get('won', {}).get('longest', 0)
    max_loss_streak = trades.get('streak', {}).get('lost', {}).get('longest', 0)
    current_streak = trades.get('streak', {}).get('won', {}).get('current', 0)

    # Best and worst trades
    best_trade = trades.get('won', {}).get('pnl', {}).get('max', 0)
    worst_trade = trades.get('lost', {}).get('pnl', {}).get('max', 0)

    # Trade duration
    avg_trade_len = trades.get('len', {}).get('average', 0)
    max_trade_len = trades.get('len', {}).get('max', 0)
    min_trade_len = trades.get('len', {}).get('min', 0)

    # Long stats (since we're long-only)
    long_trades = trades.get('long', {}).get('total', 0)
    long_won = trades.get('long', {}).get('won', 0)
    long_lost = trades.get('long', {}).get('lost', 0)
else:
    rr_ratio = profit_factor = expectancy = 0
    max_win_streak = max_loss_streak = current_streak = 0
    best_trade = worst_trade = 0
    avg_trade_len = max_trade_len = min_trade_len = 0
    long_trades = long_won = long_lost = 0
    win_rate = loss_rate = 0

# Calculate additional performance metrics
final_value = cerebro.broker.getvalue()
total_return = final_value - initial_cash
total_return_pct = (final_value / initial_cash - 1) * 100

# Calmar Ratio: Annual return / Max Drawdown
max_dd_pct = dd['max']['drawdown']
calmar_ratio = (total_return_pct / max_dd_pct) if max_dd_pct > 0 else 0

# Recovery Factor: Net Profit / Max Drawdown $
recovery_factor = (total_return / dd['max']['moneydown']) if dd['max']['moneydown'] > 0 else 0

# System Quality Number (Van Tharp)
sqn_score = sqn.get('sqn', 0)

# Annualized return (approximate based on TEST PERIOD, not including lookback)
days_traded = len(test_df) if len(df) > lookback_bars else len(df)
years = days_traded / 252  # Approximate trading days per year
annualized_return = ((final_value / initial_cash) ** (1 / years) - 1) * 100 if years > 0 else 0

# Calculate time in market
if total_trades > 0:
    total_bars_in_trades = trades.get('len', {}).get('total', 0)
    time_in_market = (total_bars_in_trades / days_traded) * 100
else:
    time_in_market = 0

# Print formatted results
print("\n" + "="*60)
print(f"BACKTEST RESULTS - {symbol}")
print("="*60)

print(f"\n💰 Portfolio Performance:")
print(f"   Starting Value:     ${initial_cash:,.2f}")
print(f"   Final Value:        ${final_value:,.2f}")
print(f"   Total Return:       ${total_return:,.2f} ({total_return_pct:.2f}%)")
print(f"   Annualized Return:  {annualized_return:.2f}%")
print(f"\n   📊 Benchmark Comparison:")
print(f"   SPY Buy & Hold:     ${spy_final_value:,.2f} ({spy_return:.2f}%)")
print(f"   Outperformance:     {total_return_pct - spy_return:.2f}%")

print(f"\n📉 Risk Metrics:")
print(f"   Sharpe Ratio:       {sharpe.get('sharperatio', 0):.3f}")
print(f"   Calmar Ratio:       {calmar_ratio:.3f}")
print(f"   SQN (Quality):      {sqn_score:.2f}")
print(f"   Max Drawdown:       {max_dd_pct:.2f}%")
print(f"   Max Drawdown ($):   ${dd['max']['moneydown']:,.2f}")
print(f"   Recovery Factor:    {recovery_factor:.2f}")
print(f"   Drawdown Duration:  {dd.get('len', 0)} days")

print(f"\n📈 Trade Statistics:")
print(f"   Total Trades:       {total_trades}")
if total_trades > 0:
    print(f"   Long Trades:        {long_trades} (Won: {long_won}, Lost: {long_lost})")
    print(f"   Wins:               {trades['won']['total']} ({win_rate:.1f}%)")
    print(f"   Losses:             {trades['lost']['total']} ({loss_rate:.1f}%)")
    print(f"\n   💵 Profit Analysis:")
    print(f"   Total Wins:         ${total_win_pnl:,.2f}")
    print(f"   Total Losses:       ${total_loss_pnl:,.2f}")
    print(f"   Net P&L:            ${total_return:,.2f}")
    print(f"   Avg Win:            ${avg_win:,.2f}")
    print(f"   Avg Loss:           ${avg_loss:,.2f}")
    print(f"   Best Trade:         ${best_trade:,.2f}")
    print(f"   Worst Trade:        ${worst_trade:,.2f}")
    print(f"   Avg Trade P&L:      ${trades['pnl']['net']['average']:,.2f}")
    print(f"\n   📊 Performance Ratios:")
    print(f"   RR Ratio:           {rr_ratio:.2f}")
    print(f"   Profit Factor:      {profit_factor:.2f}")
    print(f"   Expectancy:         ${expectancy:.2f}")
    print(f"   Win Rate:           {win_rate:.1f}%")
    print(f"\n   ⏱️  Trade Duration:")
    print(f"   Avg Duration:       {avg_trade_len:.1f} days")
    print(f"   Longest Trade:      {max_trade_len} days")
    print(f"   Shortest Trade:     {min_trade_len} days")
    print(f"   Time in Market:     {time_in_market:.1f}%")
    print(f"\n   🔥 Streaks:")
    print(f"   Max Win Streak:     {max_win_streak}")
    print(f"   Max Loss Streak:    {max_loss_streak}")
    print(f"   Current Streak:     {current_streak} wins")

print("="*60 + "\n")

plt.style.use('dark_background')

# 4️⃣ Generate backtrader's built-in plot (includes warmup period)
print("\n📊 Creating backtrader plot...")
figs = cerebro.plot(
    style='candlestick',
    iplot=False,
    barup='#597D35',
    bardown='#FF7171',
    volume=False,
)

# Save backtrader plot
for fig in figs:
    fig[0].savefig(f"{symbol}_backtest_full.png", dpi=150, bbox_inches='tight')
print(f"✓ Saved backtrader plot (with warmup period) to {symbol}_backtest_full.png")

# 5️⃣ Create custom plots (test period only - indicators fully loaded)
print("📊 Creating custom charts (test period only)...")

# Create figure with 3 subplots
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#1a1a1a')
gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)

# Subplot 1: Price chart with indicators
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor('#2a2a2a')

# Get data for test period only
test_start_idx = actual_test_start_idx
test_data = df.iloc[test_start_idx:]
test_dates = test_data.index

# Plot candlesticks (simplified as OHLC lines)
ax1.plot(test_dates, test_data['close'], color='white', linewidth=1, alpha=0.8, label='Close')

# Calculate and plot SMAs for test period
# We need to calculate from the beginning of df to get correct values, then slice
from_start_fast = df['close'].rolling(window=strategy_params['fast_len']).mean()
from_start_slow = df['close'].rolling(window=strategy_params['slow_len']).mean()

fast_sma_test = from_start_fast.iloc[test_start_idx:]
slow_sma_test = from_start_slow.iloc[test_start_idx:]

ax1.plot(test_dates, fast_sma_test, color='#00ff88', linewidth=1.5, label=f'SMA {strategy_params["fast_len"]}', alpha=0.8)
ax1.plot(test_dates, slow_sma_test, color='#ff6b6b', linewidth=1.5, label=f'SMA {strategy_params["slow_len"]}', alpha=0.8)

# Add buy/sell markers from trade history
# We'll extract this from the strategy's order execution
ax1.set_ylabel('Price ($)', fontsize=12, color='white')
ax1.set_title(f'{symbol} - Price Chart with Indicators (Test Period Only - Indicators Fully Loaded)', fontsize=14, fontweight='bold', color='white')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.2)
ax1.tick_params(colors='white')

# Subplot 2: Portfolio value comparison
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor('#2a2a2a')

# Get portfolio values and dates using the observer line buffer
# BUT only for the test period (after lookback)
portfolio_values = []
dates = []

# Access the line buffer array directly
observer = strat.observers.portfoliovalue
for i in range(len(observer.lines.value)):
    # Only include values from the test period
    if i >= actual_test_start_idx:
        try:
            val = observer.lines.value.array[i]
            if not np.isnan(val) and val > 0:
                portfolio_values.append(val)
                dates.append(df.index[i].date())
        except (IndexError, AttributeError):
            break

# Make sure we have data
if len(portfolio_values) < 2:
    print("❌ Insufficient portfolio value data for plotting")
    exit(0)

print(f"Debug: Collected {len(portfolio_values)} portfolio values (test period only)")

# Calculate SPY values over time - must match dates exactly
spy_values = []
for date in dates:
    # Find SPY price for this date
    try:
        # Try exact date match first
        matching_rows = spy_df.loc[spy_df.index.date == date, 'Close']
        if len(matching_rows) > 0:
            spy_price = float(matching_rows.iloc[0])
        else:
            # Find closest previous date
            valid_dates = spy_df.index[spy_df.index.date <= date]
            if len(valid_dates) > 0:
                spy_price = float(spy_df.loc[valid_dates[-1], 'Close'].iloc[0])
            else:
                spy_price = spy_initial_price

        spy_value = spy_shares * spy_price
        spy_values.append(spy_value)
    except Exception as e:
        # Fallback to last known value or initial
        if spy_values:
            spy_values.append(spy_values[-1])
        else:
            spy_values.append(initial_cash)

print(f"Debug: portfolio_values length: {len(portfolio_values)}")
print(f"Debug: spy_values length: {len(spy_values)}")
print(f"Debug: dates length: {len(dates)}")
print(f"Debug: Portfolio range: ${min(portfolio_values):,.2f} to ${max(portfolio_values):,.2f}")
print(f"Debug: SPY range: ${min(spy_values):,.2f} to ${max(spy_values):,.2f}")

# Convert dates to proper datetime for plotting
dates = pd.to_datetime(dates)

# Plot portfolio value comparison on ax2
ax2.plot(dates, portfolio_values, label=f'{symbol} Strategy', linewidth=2, color='#00ff88')
ax2.plot(dates, spy_values, label='SPY Buy & Hold', linewidth=2, color='#ff6b6b', linestyle='--')
ax2.axhline(y=initial_cash, color='gray', linestyle=':', alpha=0.5, label='Initial Capital')
ax2.set_ylabel('Portfolio Value ($)', fontsize=12, color='white')
ax2.set_title(f'{symbol} Strategy vs SPY Buy & Hold', fontsize=14, fontweight='bold', color='white')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.2)
ax2.tick_params(colors='white')

# Format y-axis as currency
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Subplot 3: Cumulative returns %
ax3 = fig.add_subplot(gs[2])
ax3.set_facecolor('#2a2a2a')
strategy_returns = [(v / initial_cash - 1) * 100 for v in portfolio_values]
spy_returns = [(v / initial_cash - 1) * 100 for v in spy_values]

ax3.plot(dates, strategy_returns, label=f'{symbol} Strategy', linewidth=2, color='#00ff88')
ax3.plot(dates, spy_returns, label='SPY Buy & Hold', linewidth=2, color='#ff6b6b', linestyle='--')
ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax3.fill_between(dates, strategy_returns, 0, alpha=0.3, color='#00ff88')
ax3.set_xlabel('Date', fontsize=12, color='white')
ax3.set_ylabel('Cumulative Return (%)', fontsize=12, color='white')
ax3.set_title('Cumulative Returns Over Time', fontsize=14, fontweight='bold', color='white')
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(True, alpha=0.2)
ax3.tick_params(colors='white')

# Add final return annotations
final_strategy_return = strategy_returns[-1]
final_spy_return = spy_returns[-1]
ax3.annotate(f'{final_strategy_return:.1f}%',
             xy=(dates[-1], final_strategy_return),
             xytext=(10, 0), textcoords='offset points',
             fontsize=10, color='#00ff88', fontweight='bold')
ax3.annotate(f'{final_spy_return:.1f}%',
             xy=(dates[-1], final_spy_return),
             xytext=(10, 0), textcoords='offset points',
             fontsize=10, color='#ff6b6b', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{symbol}_backtest.png", dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
print(f"✓ Saved complete backtest chart to {symbol}_backtest.png")

exit(0)