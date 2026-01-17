"""
Backtrader Backtesting Script for Aroon Multi-Filter Strategy

This script performs comprehensive backtesting with:
- Automatic lookback period calculation
- SPY benchmark comparison
- Detailed performance metrics
- Professional visualizations
- Proper data handling with warmup period

Usage:
    python backtest.py
"""

from decimal import Decimal
import matplotlib
matplotlib.use('Agg')
import backtrader as bt
import yfinance as yf
from math import isnan
import matplotlib.pyplot as plt
from aroon_atr import AroonMultiFilterStrategy
import pandas as pd
import numpy as np
from datetime import datetime
import sys


# ============================================================================
# Custom Observers and Indicators
# ============================================================================

class BuySellArrows(bt.observers.BuySell):
    """Custom buy/sell markers with improved visibility."""

    plotlines = dict(
        buy=dict(marker='^', markersize=8, color='lime', fillstyle='full', ls=''),
        sell=dict(marker='v', markersize=8, color='red', fillstyle='full', ls='')
    )

    def next(self):
        super(BuySellArrows, self).next()
        # Position markers slightly away from candles for visibility
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


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_lookback(strategy_class, strategy_params=None):
    """
    Calculate the required lookback period for strategy indicators.

    Args:
        strategy_class: The strategy class to analyze
        strategy_params: Optional dict of parameter overrides

    Returns:
        int: Maximum lookback period needed
    """
    params_dict = {}

    # Extract default parameters from strategy class
    for param_name in dir(strategy_class.params):
        if not param_name.startswith('_'):
            param_value = getattr(strategy_class.params, param_name)
            params_dict[param_name] = param_value

    # Apply any overrides
    if strategy_params:
        params_dict.update(strategy_params)

    # Find all period-related parameters
    lookback_candidates = []
    exclude_patterns = ['verbose', 'plot', 'print', 'enable', 'use']

    for param_name, param_value in params_dict.items():
        # Convert Decimal to int if needed
        if isinstance(param_value, Decimal):
            try:
                param_value = int(param_value)
            except (ValueError, TypeError):
                continue

        # Check if it's a positive integer parameter
        if isinstance(param_value, int) and param_value > 0:
            # Skip non-period parameters
            if not any(pattern in param_name.lower() for pattern in exclude_patterns):
                lookback_candidates.append(param_value)

    # Add buffer for moving averages and warmup
    max_lookback = (max(lookback_candidates) if lookback_candidates else 50) + 20

    print(f"\n📊 Calculated Lookback Period")
    print(f"   Maximum lookback: {max_lookback} bars")
    print(f"   Period parameters found: {sorted(lookback_candidates, reverse=True)}")

    return max_lookback


def download_data(symbol, lookback_bars, test_period_days=252):
    """
    Download historical data with proper lookback period.

    Args:
        symbol: Stock ticker symbol
        lookback_bars: Required lookback period in bars
        test_period_days: Desired test period length in trading days

    Returns:
        tuple: (full_df, test_start_index, test_df)
    """
    # Calculate total bars needed (test period + lookback + buffer)
    total_bars_needed = test_period_days + lookback_bars
    days_to_download = int(total_bars_needed * 1.5)  # Add 50% buffer for weekends/holidays

    print(f"\n📥 Downloading Data")
    print(f"   Symbol: {symbol}")
    print(f"   Approximate days: {days_to_download}")
    print(f"   Lookback bars: {lookback_bars}")
    print(f"   Test period target: {test_period_days} trading days")

    try:
        df = yf.download(symbol, period=f"{days_to_download}d", interval="1d", progress=False)
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        sys.exit(1)

    if df.empty:
        print(f"❌ No data downloaded for {symbol}")
        sys.exit(1)

    # Clean data
    df.index = df.index.tz_localize(None)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = ['open', 'high', 'low', 'close', 'volume']

    print(f"   Downloaded: {len(df)} bars")
    print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")

    # Split into lookback and test periods
    if len(df) > lookback_bars:
        test_start_idx = lookback_bars
        test_df = df.iloc[lookback_bars:]

        print(f"\n   Lookback period: {df.index[0].date()} to {df.index[lookback_bars-1].date()} ({lookback_bars} bars)")
        print(f"   Test period: {test_df.index[0].date()} to {test_df.index[-1].date()} ({len(test_df)} bars)")
    else:
        print(f"\n   ⚠️  Warning: Insufficient data. Using all {len(df)} bars without lookback.")
        test_start_idx = 0
        test_df = df

    return df, test_start_idx, test_df


def download_benchmark(test_df):
    """
    Download SPY benchmark data for comparison.

    Args:
        test_df: Test period dataframe for date alignment

    Returns:
        pd.DataFrame: SPY data aligned to test period
    """
    print(f"\n📥 Downloading Benchmark (SPY)")

    try:
        # Add 1 day to end because yfinance 'end' parameter is EXCLUSIVE
        end_date = test_df.index[-1] + pd.Timedelta(days=1)

        spy_df = yf.download(
            'SPY',
            start=test_df.index[0],
            end=end_date,
            progress=False
        )
        spy_df.index = spy_df.index.tz_localize(None)

        # Verify we got the full date range
        print(f"   Downloaded: {len(spy_df)} bars")
        print(f"   SPY range: {spy_df.index[0].date()} to {spy_df.index[-1].date()}")
        print(f"   Test range: {test_df.index[0].date()} to {test_df.index[-1].date()}")

        return spy_df
    except Exception as e:
        print(f"   ⚠️  Warning: Could not download SPY data: {e}")
        return None


def calculate_benchmark_returns(spy_df, initial_cash):
    """
    Calculate SPY buy-and-hold returns.

    Args:
        spy_df: SPY dataframe
        initial_cash: Initial capital

    Returns:
        tuple: (final_value, return_pct, shares)
    """
    if spy_df is None or spy_df.empty:
        return None, None, None

    spy_initial_price = float(spy_df['Close'].iloc[0])
    spy_final_price = float(spy_df['Close'].iloc[-1])
    spy_shares = initial_cash / spy_initial_price
    spy_final_value = spy_shares * spy_final_price
    spy_return = (spy_final_value / initial_cash - 1) * 100

    return spy_final_value, spy_return, spy_shares


def setup_cerebro(df, strategy_params, initial_cash=10000):
    """
    Set up backtrader Cerebro engine with strategy and analyzers.

    Args:
        df: Price data dataframe
        strategy_params: Strategy parameter dict
        initial_cash: Starting capital

    Returns:
        bt.Cerebro: Configured Cerebro instance
    """
    print(f"\n⚙️  Setting Up Backtrader")

    cerebro = bt.Cerebro(stdstats=False)

    # Add data feed
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

    # Add strategy with parameters
    cerebro.addstrategy(AroonMultiFilterStrategy, **strategy_params)

    # Configure broker
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.0)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")

    # Add observers
    cerebro.addobserver(BuySellArrows, plot=True, subplot=False)
    cerebro.addobserver(bt.observers.Trades, plot=True, subplot=False)
    cerebro.addobserver(PortfolioValue)

    print(f"   Initial cash: ${initial_cash:,.2f}")
    print(f"   Commission: 0.0%")

    return cerebro


def print_results(strat, symbol, initial_cash, spy_final_value, spy_return, test_df):
    """
    Print comprehensive backtest results.

    Args:
        strat: Strategy instance with analyzers
        symbol: Stock ticker
        initial_cash: Starting capital
        spy_final_value: SPY final portfolio value
        spy_return: SPY return percentage
        test_df: Test period dataframe
    """
    # Extract analyzer results
    sharpe = strat.analyzers.sharpe.get_analysis()
    dd = strat.analyzers.dd.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    sqn = strat.analyzers.sqn.get_analysis()

    final_value = strat.broker.getvalue()
    total_return = final_value - initial_cash
    total_return_pct = (final_value / initial_cash - 1) * 100

    # Calculate additional metrics
    max_dd_pct = dd['max']['drawdown']
    calmar_ratio = (total_return_pct / max_dd_pct) if max_dd_pct > 0 else 0
    recovery_factor = (total_return / dd['max']['moneydown']) if dd['max']['moneydown'] > 0 else 0
    sqn_score = sqn.get('sqn', 0)

    # Annualized return
    days_traded = len(test_df)
    years = days_traded / 252
    annualized_return = ((final_value / initial_cash) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Trade statistics
    total_trades = trades.get('total', {}).get('total', 0)

    print("\n" + "="*70)
    print(f"BACKTEST RESULTS - {symbol}")
    print("="*70)

    print(f"\n💰 Portfolio Performance")
    print(f"   Starting Value:        ${initial_cash:,.2f}")
    print(f"   Final Value:           ${final_value:,.2f}")
    print(f"   Total Return:          ${total_return:,.2f} ({total_return_pct:.2f}%)")
    print(f"   Annualized Return:     {annualized_return:.2f}%")

    if spy_final_value is not None:
        print(f"\n   📊 Benchmark Comparison (SPY)")
        print(f"   SPY Buy & Hold:        ${spy_final_value:,.2f} ({spy_return:.2f}%)")
        print(f"   Outperformance:        {total_return_pct - spy_return:+.2f}%")

    print(f"\n📉 Risk Metrics")
    sharpe_ratio = sharpe.get('sharperatio', None)
    if sharpe_ratio is not None:
        print(f"   Sharpe Ratio:          {sharpe_ratio:.3f}")
    else:
        print(f"   Sharpe Ratio:          N/A")
    print(f"   Calmar Ratio:          {calmar_ratio:.3f}")
    print(f"   SQN (Quality):         {sqn_score:.2f}")
    print(f"   Max Drawdown:          {max_dd_pct:.2f}%")
    print(f"   Max Drawdown ($):      ${dd['max']['moneydown']:,.2f}")
    print(f"   Recovery Factor:       {recovery_factor:.2f}")
    print(f"   Drawdown Duration:     {dd.get('len', 0)} days")

    print(f"\n📈 Trade Statistics")
    print(f"   Total Trades:          {total_trades}")

    if total_trades > 0:
        _print_detailed_trade_stats(trades, total_trades, days_traded)
    else:
        print("   No trades executed during test period")

    print("="*70 + "\n")


def _print_detailed_trade_stats(trades, total_trades, days_traded):
    """Print detailed trade statistics (helper function)."""
    won = trades.get('won', {})
    lost = trades.get('lost', {})
    long_stats = trades.get('long', {})

    # Win/Loss stats
    wins = won.get('total', 0)
    losses = lost.get('total', 0)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    loss_rate = (losses / total_trades * 100) if total_trades > 0 else 0

    print(f"   Long Trades:           {long_stats.get('total', 0)} (Won: {long_stats.get('won', 0)}, Lost: {long_stats.get('lost', 0)})")
    print(f"   Wins:                  {wins} ({win_rate:.1f}%)")
    print(f"   Losses:                {losses} ({loss_rate:.1f}%)")

    # P&L stats
    avg_win = won.get('pnl', {}).get('average', 0)
    avg_loss = abs(lost.get('pnl', {}).get('average', 0))
    total_win_pnl = won.get('pnl', {}).get('total', 0)
    total_loss_pnl = abs(lost.get('pnl', {}).get('total', 0))

    print(f"\n   💵 Profit Analysis")
    print(f"   Total Wins:            ${total_win_pnl:,.2f}")
    print(f"   Total Losses:          ${total_loss_pnl:,.2f}")
    print(f"   Net P&L:               ${total_win_pnl - total_loss_pnl:,.2f}")
    print(f"   Avg Win:               ${avg_win:,.2f}")
    print(f"   Avg Loss:              ${avg_loss:,.2f}")
    print(f"   Best Trade:            ${won.get('pnl', {}).get('max', 0):,.2f}")
    print(f"   Worst Trade:           ${lost.get('pnl', {}).get('max', 0):,.2f}")
    print(f"   Avg Trade P&L:         ${trades.get('pnl', {}).get('net', {}).get('average', 0):,.2f}")

    # Performance ratios
    rr_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0
    profit_factor = (total_win_pnl / total_loss_pnl) if total_loss_pnl > 0 else 0
    expectancy = (win_rate/100 * avg_win) - (loss_rate/100 * avg_loss)

    print(f"\n   📊 Performance Ratios")
    print(f"   RR Ratio:              {rr_ratio:.2f}")
    print(f"   Profit Factor:         {profit_factor:.2f}")
    print(f"   Expectancy:            ${expectancy:.2f}")
    print(f"   Win Rate:              {win_rate:.1f}%")

    # Trade duration
    avg_trade_len = trades.get('len', {}).get('average', 0)
    max_trade_len = trades.get('len', {}).get('max', 0)
    min_trade_len = trades.get('len', {}).get('min', 0)
    total_bars_in_trades = trades.get('len', {}).get('total', 0)
    time_in_market = (total_bars_in_trades / days_traded) * 100 if days_traded > 0 else 0

    print(f"\n   ⏱️  Trade Duration")
    print(f"   Avg Duration:          {avg_trade_len:.1f} days")
    print(f"   Longest Trade:         {max_trade_len} days")
    print(f"   Shortest Trade:        {min_trade_len} days")
    print(f"   Time in Market:        {time_in_market:.1f}%")

    # Streaks
    max_win_streak = trades.get('streak', {}).get('won', {}).get('longest', 0)
    max_loss_streak = trades.get('streak', {}).get('lost', {}).get('longest', 0)
    current_streak = trades.get('streak', {}).get('won', {}).get('current', 0)

    print(f"\n   🔥 Streaks")
    print(f"   Max Win Streak:        {max_win_streak}")
    print(f"   Max Loss Streak:       {max_loss_streak}")
    print(f"   Current Streak:        {current_streak} wins")


def create_plots(cerebro, strat, symbol, df, test_start_idx, spy_df, spy_shares, initial_cash, strategy_params):
    """
    Create comprehensive visualization plots.

    Args:
        cerebro: Cerebro instance
        strat: Strategy instance
        symbol: Stock ticker
        df: Full dataframe
        test_start_idx: Index where test period starts
        spy_df: SPY benchmark data
        spy_shares: Number of SPY shares
        initial_cash: Starting capital
        strategy_params: Strategy parameters dict
    """
    print("\n📊 Creating Visualizations")

    # 1. Backtrader built-in plot
    print("   Generating backtrader plot...")
    try:
        figs = cerebro.plot(
            style='candlestick',
            iplot=False,
            barup='#597D35',
            bardown='#FF7171',
            volume=False,
        )

        for fig in figs:
            fig[0].savefig(f"{symbol}_backtest_full.png", dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: {symbol}_backtest_full.png")
    except Exception as e:
        print(f"   ⚠️  Warning: Could not create backtrader plot: {e}")

    # 2. Custom plots
    print("   Generating custom plots...")
    try:
        _create_custom_plots(strat, symbol, df, test_start_idx, spy_df, spy_shares, initial_cash, strategy_params)
        print(f"   ✓ Saved: {symbol}_backtest.png")
    except Exception as e:
        print(f"   ⚠️  Warning: Could not create custom plots: {e}")


def _create_custom_plots(strat, symbol, df, test_start_idx, spy_df, spy_shares, initial_cash, strategy_params):
    """Create custom matplotlib plots (helper function)."""
    plt.style.use('dark_background')

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#1a1a1a')
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)

    # Get test period data
    test_data = df.iloc[test_start_idx:]
    test_dates = test_data.index

    # Subplot 1: Price and Aroon
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#2a2a2a')

    ax1.plot(test_dates, test_data['close'], color='white', linewidth=1, alpha=0.8, label='Close Price')

    # Calculate Aroon for full dataset, then slice for test period
    aroon_up, aroon_down = _calculate_aroon_indicator(df['high'].values, df['low'].values, strategy_params['aroon_len'])
    aroon_up_test = pd.Series(aroon_up, index=df.index).iloc[test_start_idx:]
    aroon_down_test = pd.Series(aroon_down, index=df.index).iloc[test_start_idx:]

    # Second y-axis for Aroon
    ax1_aroon = ax1.twinx()
    ax1_aroon.plot(test_dates, aroon_up_test, color='#00ff88', linewidth=1.5, label=f'Aroon Up ({strategy_params["aroon_len"]})', alpha=0.8)
    ax1_aroon.plot(test_dates, aroon_down_test, color='#ff6b6b', linewidth=1.5, label=f'Aroon Down ({strategy_params["aroon_len"]})', alpha=0.8)
    ax1_aroon.axhline(y=70, color='gray', linestyle='--', alpha=0.3)
    ax1_aroon.axhline(y=30, color='gray', linestyle='--', alpha=0.3)
    ax1_aroon.set_ylabel('Aroon (%)', fontsize=12, color='white')
    ax1_aroon.tick_params(colors='white')
    ax1_aroon.set_ylim(0, 100)

    ax1.set_ylabel('Price ($)', fontsize=12, color='white')
    ax1.set_title(f'{symbol} - Price Chart with Aroon Indicator', fontsize=14, fontweight='bold', color='white')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_aroon.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(colors='white')

    # Subplot 2: Portfolio value comparison
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#2a2a2a')

    portfolio_values, dates = _extract_portfolio_values(strat, df, test_start_idx)

    if len(portfolio_values) >= 2:
        spy_values = _calculate_spy_values(dates, spy_df, spy_shares, initial_cash)

        ax2.plot(dates, portfolio_values, label=f'{symbol} Strategy', linewidth=2, color='#00ff88')
        if spy_values:
            ax2.plot(dates, spy_values, label='SPY Buy & Hold', linewidth=2, color='#ff6b6b', linestyle='--')
        ax2.axhline(y=initial_cash, color='gray', linestyle=':', alpha=0.5, label='Initial Capital')
        ax2.set_ylabel('Portfolio Value ($)', fontsize=12, color='white')
        ax2.set_title(f'{symbol} Strategy vs SPY Buy & Hold', fontsize=14, fontweight='bold', color='white')
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.2)
        ax2.tick_params(colors='white')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Subplot 3: Cumulative returns
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor('#2a2a2a')

    if len(portfolio_values) >= 2:
        strategy_returns = [(v / initial_cash - 1) * 100 for v in portfolio_values]

        ax3.plot(dates, strategy_returns, label=f'{symbol} Strategy', linewidth=2, color='#00ff88')

        if spy_values:
            spy_returns = [(v / initial_cash - 1) * 100 for v in spy_values]
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
        ax3.annotate(f'{final_strategy_return:.1f}%',
                     xy=(dates[-1], final_strategy_return),
                     xytext=(10, 0), textcoords='offset points',
                     fontsize=10, color='#00ff88', fontweight='bold')

        if spy_values:
            final_spy_return = spy_returns[-1]
            ax3.annotate(f'{final_spy_return:.1f}%',
                         xy=(dates[-1], final_spy_return),
                         xytext=(10, 0), textcoords='offset points',
                         fontsize=10, color='#ff6b6b', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{symbol}_backtest.png", dpi=150, bbox_inches='tight', facecolor='#1a1a1a')


def _calculate_aroon_indicator(high, low, period):
    """Calculate Aroon Up and Aroon Down indicators."""
    aroon_up = []
    aroon_down = []

    for i in range(len(high)):
        if i < period - 1:
            aroon_up.append(np.nan)
            aroon_down.append(np.nan)
        else:
            window_high = high[i-period+1:i+1]
            window_low = low[i-period+1:i+1]

            days_since_high = period - 1 - np.argmax(window_high)
            days_since_low = period - 1 - np.argmin(window_low)

            aroon_up.append(((period - days_since_high) / period) * 100)
            aroon_down.append(((period - days_since_low) / period) * 100)

    return aroon_up, aroon_down


def _extract_portfolio_values(strat, df, test_start_idx):
    """Extract portfolio values from strategy observer."""
    portfolio_values = []
    dates = []

    observer = strat.observers.portfoliovalue
    for i in range(len(observer.lines.value)):
        if i >= test_start_idx:
            try:
                val = observer.lines.value.array[i]
                if not np.isnan(val) and val > 0:
                    portfolio_values.append(val)
                    dates.append(df.index[i])
            except (IndexError, AttributeError):
                break

    dates = pd.to_datetime(dates)
    return portfolio_values, dates


def _calculate_spy_values(dates, spy_df, spy_shares, initial_cash):
    """Calculate SPY portfolio values over time."""
    if spy_df is None or spy_shares is None:
        return None

    spy_values = []
    for date in dates:
        try:
            matching_rows = spy_df.loc[spy_df.index.date == date.date(), 'Close']
            if len(matching_rows) > 0:
                spy_price = float(matching_rows.iloc[0])
            else:
                valid_dates = spy_df.index[spy_df.index.date <= date.date()]
                if len(valid_dates) > 0:
                    spy_price = float(spy_df.loc[valid_dates[-1], 'Close'])
                else:
                    spy_price = float(spy_df['Close'].iloc[0])

            spy_values.append(spy_shares * spy_price)
        except Exception:
            if spy_values:
                spy_values.append(spy_values[-1])
            else:
                spy_values.append(initial_cash)

    return spy_values


def print_strategy_config(strategy_params):
    """Print strategy configuration summary."""
    print("\n🔧 Strategy Configuration")
    print("="*70)
    print(f"   Aroon Length:              {strategy_params['aroon_len']}")
    print(f"   ATR Filter Length:         {strategy_params['atr_filter_len']}")
    print(f"   ATR Filter Baseline:       {strategy_params.get('atr_filter_baseline_len', 'N/A')}")
    print(f"   ATR Filter Multiplier:     {float(strategy_params['atr_filter_mult']):.1f}x")
    print(f"   ATR Stop Length:           {strategy_params['atr_stop_len']}")
    print(f"   ATR Stop Multiplier:       {float(strategy_params['atr_stop_mult']):.1f}x")
    print(f"   Stop Loss:                 {float(strategy_params['stop_loss_pct'])*100:.1f}%")
    print(f"   Take Profit:               {float(strategy_params['take_profit_pct'])*100:.1f}%")
    print(f"\n   Sideways Market Filters:")
    print(f"   Min Trend Strength:        {float(strategy_params['min_trend_strength']):.0f}")
    print(f"   Reject Both Middle:        {strategy_params['max_both_middle']}")
    print(f"   Stability Bars:            {strategy_params['stability_bars']}")
    print(f"   Use ADX Filter:            {strategy_params['use_adx_filter']}")
    if strategy_params['use_adx_filter']:
        print(f"   ADX Length:                {strategy_params['adx_length']}")
        print(f"   ADX Threshold:             {float(strategy_params['adx_threshold']):.1f}")
    print(f"\n   Dynamic Peak Exit (ATR-based):")
    print(f"   Enable Peak Exit:          {strategy_params['enable_peak_exit']}")
    if strategy_params['enable_peak_exit']:
        print(f"   Peak ATR Multiplier:       {float(strategy_params['peak_atr_mult']):.1f}x (adapts to volatility)")
        print(f"   Peak ATR Period:           {strategy_params['peak_atr_period']}")
        print(f"   Min Profit to Activate:    {float(strategy_params['min_profit_pct_to_activate'])*100:.1f}%")
    print("="*70)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main backtesting workflow."""

    # Configuration
    symbol = "LMT"
    initial_cash = 10000

    # Strategy parameters (matching TradingView defaults)
    strategy_params = {
        # ==================== ENTRY PARAMETERS ====================
        'aroon_len': 24,
        'atr_filter_len': 14,
        'atr_filter_baseline_len': 40,
        'atr_filter_mult': Decimal('2.5'),
        'min_trend_strength': Decimal('25.0'),
        'max_both_middle': True,
        'stability_bars': 3,
        'use_adx_filter': True,
        'adx_length': 14,
        'adx_threshold': Decimal('25.0'),

        # ==================== EXIT PARAMETERS ====================
        'atr_stop_len': 5,
        'atr_stop_mult': Decimal('3.0'),
        'stop_loss_pct': Decimal('0.10'),        # 10%
        'take_profit_pct': Decimal('0.13'),      # 13%

        # Dynamic ATR-based peak exit (adapts to stock volatility)
        'enable_peak_exit': False,
        'peak_atr_mult': Decimal('2.0'),          # Peak - (ATR * 2.0)
        'peak_atr_period': 14,                    # ATR period for calculation
        'min_profit_pct_to_activate': Decimal('0.03'),  # 3% min profit

        # ==================== OTHER PARAMETERS ====================
        'position_size_pct': Decimal('1.0'),     # 100%
        'verbose': True
    }

    print("\n" + "="*70)
    print("AROON MULTI-FILTER STRATEGY BACKTESTING")
    print("="*70)

    # Calculate lookback and download data
    lookback_bars = calculate_lookback(AroonMultiFilterStrategy, strategy_params)
    df, test_start_idx, test_df = download_data(symbol, lookback_bars, test_period_days=252)

    # Download benchmark
    spy_df = download_benchmark(test_df)
    spy_final_value, spy_return, spy_shares = calculate_benchmark_returns(spy_df, initial_cash)

    # Print strategy configuration
    print_strategy_config(strategy_params)

    # Setup and run backtest
    cerebro = setup_cerebro(df, strategy_params, initial_cash)

    print(f"\n🚀 Running Backtest...")
    results = cerebro.run()
    strat = results[0]

    # Print results
    print_results(strat, symbol, initial_cash, spy_final_value, spy_return, test_df)

    # Create visualizations
    create_plots(cerebro, strat, symbol, df, test_start_idx, spy_df, spy_shares, initial_cash, strategy_params)

    print("\n✅ Backtest Complete!\n")


if __name__ == "__main__":
    main()
