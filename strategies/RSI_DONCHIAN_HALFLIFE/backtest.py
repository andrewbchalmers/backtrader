# source bt/bin/activate
# cd strategies/RSI_DONCHIAN_HALFLIFE/
# python backtest.py

from decimal import Decimal
import matplotlib
matplotlib.use('Agg')
import backtrader as bt
import yfinance as yf
from math import isnan
import matplotlib.pyplot as plt
from rsi_donchian_halflife import Strategy
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
    plotinfo = dict(plot=False, subplot=False)

    def next(self):
        self.lines.value[0] = self._owner.broker.getvalue()

    def prenext(self):
        self.lines.value[0] = self._owner.broker.getvalue()


def CalculateLookback(strategy_class, strategy_params=None):
    """Calculate lookback period from strategy parameters."""
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
            exclude_patterns = ['verbose', 'plot', 'print', 'threshold']
            if not any(pattern in param_name.lower() for pattern in exclude_patterns):
                lookback_candidates.append(param_value)

    max_lookback = (max(lookback_candidates) if lookback_candidates else 0) + 20

    print(f"\n📊 Calculated Lookback: {max_lookback} bars")
    print(f"   Found period parameters: {sorted(lookback_candidates, reverse=True)}")

    return max_lookback


# ============================================================================
# CONFIGURATION
# ============================================================================
symbol = "COTY"  # Mean-reverting stock

strategy_params = {
    'rsi_period': 14,
    'rsi_threshold': 30,  # Buy when RSI < 30
    'donchian_period': 20,
    'use_donchian_filter': True,
    'halflife_period': 50,
    'halflife_exit_threshold': 50,  # Exit when halflife signal > 50
    'stop_loss_pct': Decimal("0.05"),  # 5% stop loss
    'take_profit_pct': Decimal("0.08"),  # 8% take profit
    'use_take_profit': True,
    'verbose': True
}

initial_cash = 10_000

# ============================================================================
# DOWNLOAD DATA
# ============================================================================
lookback_bars = CalculateLookback(Strategy, strategy_params)

total_bars_needed = 252 + lookback_bars
days_to_download = int(total_bars_needed * 1.5)

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
    actual_test_start_date = test_df.index[0]
    actual_test_start_idx = lookback_bars
else:
    print(f"⚠️  Warning: Downloaded data ({len(df)} bars) is less than lookback requirement ({lookback_bars} bars)")
    actual_test_start_date = df.index[0]
    actual_test_start_idx = 0

# Download SPY for benchmark
if len(df) > lookback_bars:
    spy_start_date = test_df.index[0]
    spy_end_date = test_df.index[-1]
else:
    spy_start_date = df.index[0]
    spy_end_date = df.index[-1]

spy_df = yf.download('SPY', start=spy_start_date, end=spy_end_date, progress=False)
spy_df.index = spy_df.index.tz_localize(None)

spy_initial_price = float(spy_df['Close'].iloc[0])
spy_final_price = float(spy_df['Close'].iloc[-1])
spy_shares = initial_cash / spy_initial_price
spy_final_value = spy_shares * spy_final_price
spy_return = (spy_final_value / initial_cash - 1) * 100

# ============================================================================
# BACKTEST SETUP
# ============================================================================
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

# ============================================================================
# RUN BACKTEST
# ============================================================================
print(f"\n🚀 Running backtest...")
print(f"Strategy Configuration:")
print(f"  Entry: RSI({strategy_params['rsi_period']}) < {strategy_params['rsi_threshold']} when price > DONCHIAN({strategy_params['donchian_period']})")
print(f"  Exit: HALFLIFE({strategy_params['halflife_period']}) > {strategy_params['halflife_exit_threshold']}")
print(f"        + {float(strategy_params['stop_loss_pct'])*100}% SL + {float(strategy_params['take_profit_pct'])*100}% TP")
print()

results = cerebro.run()
strat = results[0]

# ============================================================================
# EXTRACT RESULTS
# ============================================================================
sharpe = strat.analyzers.sharpe.get_analysis()
dd = strat.analyzers.dd.get_analysis()
trades = strat.analyzers.trades.get_analysis()
returns = strat.analyzers.returns.get_analysis()
sqn = strat.analyzers.sqn.get_analysis()

total_trades = trades.get('total', {}).get('total', 0)
if total_trades > 0:
    avg_win = trades.get('won', {}).get('pnl', {}).get('average', 0)
    avg_loss = abs(trades.get('lost', {}).get('pnl', {}).get('average', 0))
    rr_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0

    total_win_pnl = trades.get('won', {}).get('pnl', {}).get('total', 0)
    total_loss_pnl = abs(trades.get('lost', {}).get('pnl', {}).get('total', 0))
    profit_factor = (total_win_pnl / total_loss_pnl) if total_loss_pnl > 0 else 0

    win_rate = trades['won']['total'] / total_trades * 100
    loss_rate = trades['lost']['total'] / total_trades * 100
    expectancy = (win_rate/100 * avg_win) - (loss_rate/100 * avg_loss)

    max_win_streak = trades.get('streak', {}).get('won', {}).get('longest', 0)
    max_loss_streak = trades.get('streak', {}).get('lost', {}).get('longest', 0)

    best_trade = trades.get('won', {}).get('pnl', {}).get('max', 0)
    worst_trade = trades.get('lost', {}).get('pnl', {}).get('max', 0)

    avg_trade_len = trades.get('len', {}).get('average', 0)
    max_trade_len = trades.get('len', {}).get('max', 0)
    min_trade_len = trades.get('len', {}).get('min', 0)
else:
    rr_ratio = profit_factor = expectancy = 0
    max_win_streak = max_loss_streak = 0
    best_trade = worst_trade = avg_win = avg_loss = 0
    avg_trade_len = max_trade_len = min_trade_len = 0
    win_rate = loss_rate = 0

final_value = cerebro.broker.getvalue()
total_return = final_value - initial_cash
total_return_pct = (final_value / initial_cash - 1) * 100

max_dd_pct = dd['max']['drawdown']
calmar_ratio = (total_return_pct / max_dd_pct) if max_dd_pct > 0 else 0
recovery_factor = (total_return / dd['max']['moneydown']) if dd['max']['moneydown'] > 0 else 0
sqn_score = sqn.get('sqn', 0)

days_traded = len(test_df) if len(df) > lookback_bars else len(df)
years = days_traded / 252
annualized_return = ((final_value / initial_cash) ** (1 / years) - 1) * 100 if years > 0 else 0

if total_trades > 0:
    total_bars_in_trades = trades.get('len', {}).get('total', 0)
    time_in_market = (total_bars_in_trades / days_traded) * 100
else:
    time_in_market = 0

# ============================================================================
# PRINT RESULTS
# ============================================================================
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
sharpe_ratio = sharpe.get('sharperatio', None)
print(f"   Sharpe Ratio:       {sharpe_ratio:.3f}" if sharpe_ratio is not None else "   Sharpe Ratio:       N/A")
print(f"   Calmar Ratio:       {calmar_ratio:.3f}")
print(f"   SQN (Quality):      {sqn_score:.2f}")
print(f"   Max Drawdown:       {max_dd_pct:.2f}%")
print(f"   Max Drawdown ($):   ${dd['max']['moneydown']:,.2f}")
print(f"   Recovery Factor:    {recovery_factor:.2f}")

print(f"\n📈 Trade Statistics:")
print(f"   Total Trades:       {total_trades}")
if total_trades > 0:
    print(f"   Wins:               {trades['won']['total']} ({win_rate:.1f}%)")
    print(f"   Losses:             {trades['lost']['total']} ({loss_rate:.1f}%)")
    print(f"\n   💵 Profit Analysis:")
    print(f"   Avg Win:            ${avg_win:,.2f}")
    print(f"   Avg Loss:           ${avg_loss:,.2f}")
    print(f"   Best Trade:         ${best_trade:,.2f}")
    print(f"   Worst Trade:        ${worst_trade:,.2f}")
    print(f"\n   📊 Performance Ratios:")
    print(f"   RR Ratio:           {rr_ratio:.2f}")
    print(f"   Profit Factor:      {profit_factor:.2f}")
    print(f"   Expectancy:         ${expectancy:.2f}")
    print(f"\n   ⏱️  Trade Duration:")
    print(f"   Avg Duration:       {avg_trade_len:.1f} days")
    print(f"   Time in Market:     {time_in_market:.1f}%")
    print(f"\n   🔥 Streaks:")
    print(f"   Max Win Streak:     {max_win_streak}")
    print(f"   Max Loss Streak:    {max_loss_streak}")

print("="*60 + "\n")

# ============================================================================
# GENERATE PLOTS
# ============================================================================
plt.style.use('dark_background')

print("\n📊 Creating backtrader plot...")
figs = cerebro.plot(
    style='candlestick',
    iplot=False,
    barup='#597D35',
    bardown='#FF7171',
    volume=False,
)

for fig in figs:
    fig[0].savefig(f"{symbol}_backtest_full.png", dpi=150, bbox_inches='tight')
print(f"✓ Saved backtrader plot to {symbol}_backtest_full.png")

# Create custom comparison plot
print("📊 Creating comparison chart...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
fig.patch.set_facecolor('#1a1a1a')

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

    # Plot 1: Portfolio value
    ax1.plot(dates, portfolio_values, label=f'{symbol} Strategy', linewidth=2, color='#00ff88')
    ax1.plot(dates, spy_values, label='SPY Buy & Hold', linewidth=2, color='#ff6b6b', linestyle='--')
    ax1.axhline(y=initial_cash, color='gray', linestyle=':', alpha=0.5)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12, color='white')
    ax1.set_title(f'{symbol} RSI/Donchian/HalfLife Strategy vs SPY', fontsize=14, fontweight='bold', color='white')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(colors='white')
    ax1.set_facecolor('#2a2a2a')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Plot 2: Cumulative returns
    strategy_returns = [(v / initial_cash - 1) * 100 for v in portfolio_values]
    spy_returns_plot = [(v / initial_cash - 1) * 100 for v in spy_values]

    ax2.plot(dates, strategy_returns, label=f'{symbol} Strategy', linewidth=2, color='#00ff88')
    ax2.plot(dates, spy_returns_plot, label='SPY Buy & Hold', linewidth=2, color='#ff6b6b', linestyle='--')
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax2.fill_between(dates, strategy_returns, 0, alpha=0.3, color='#00ff88')
    ax2.set_xlabel('Date', fontsize=12, color='white')
    ax2.set_ylabel('Cumulative Return (%)', fontsize=12, color='white')
    ax2.set_title('Cumulative Returns', fontsize=14, fontweight='bold', color='white')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(colors='white')
    ax2.set_facecolor('#2a2a2a')

    plt.tight_layout()
    plt.savefig(f"{symbol}_backtest.png", dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    print(f"✓ Saved comparison chart to {symbol}_backtest.png")

print("\nDone!")
