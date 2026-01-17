"""
Multi-Ticker Backtest Script for Aroon Multi-Filter Strategy

This script backtests the enhanced Aroon strategy across multiple tickers,
providing comprehensive comparative analysis and performance metrics.

Usage:
    python backtest_multi.py
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
import csv
import sys
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data configuration
    'csv_file': '../optimization_set.csv',
    'period': '2y',
    'initial_cash': 10000,

    # Output configuration
    'plot_best': True,
    'plot_worst': False,
    'save_individual_results': True,
    'results_csv': 'backtest_results.csv',

    # Strategy parameters (matching TradingView defaults)
    'strategy_params': {
        # ==================== ENTRY PARAMETERS ====================
        'aroon_len': 24,
        'atr_filter_len': 14,
        'atr_filter_baseline_len': 40,  # Longer baseline for volatility comparison
        'atr_filter_mult': Decimal('2.5'),
        'min_trend_strength': Decimal('25.0'),
        'max_both_middle': True,
        'stability_bars': 3,
        'use_adx_filter': False,
        'adx_length': 14,
        'adx_threshold': Decimal('25.0'),

        # ==================== EXIT PARAMETERS ====================
        'atr_stop_len': 5,
        'atr_stop_mult': Decimal('4.0'),
        'stop_loss_pct': Decimal('0.05'),        # 10%
        'take_profit_pct': Decimal('0.13'),      # 13%

        # Dynamic ATR-based peak exit (adapts to stock volatility)
        'enable_peak_exit': False,
        'peak_atr_mult': Decimal('2.0'),          # Peak - (ATR * 2.0)
        'peak_atr_period': 14,                    # ATR period for calculation
        'min_profit_pct_to_activate': Decimal('0.03'),  # 3% min profit

        # ==================== OTHER PARAMETERS ====================
        'position_size_pct': Decimal('0.95'),     # 100%
        'verbose': False  # Disable for multi-ticker to reduce output
    }
}


# ============================================================================
# Custom Observers
# ============================================================================

class BuySellArrows(bt.observers.BuySell):
    """Custom buy/sell markers with improved visibility."""

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


# ============================================================================
# Core Backtesting Functions
# ============================================================================

def load_symbols(csv_file):
    """
    Load ticker symbols from CSV file.

    Args:
        csv_file: Path to CSV file containing symbols

    Returns:
        list: List of ticker symbols
    """
    symbols = []

    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0].strip():
                    symbol = row[0].strip().upper()
                    if symbol and not symbol.startswith('#'):  # Skip comments
                        symbols.append(symbol)
    except FileNotFoundError:
        print(f"❌ Error: CSV file '{csv_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        sys.exit(1)

    if not symbols:
        print(f"❌ Error: No symbols found in '{csv_file}'")
        sys.exit(1)

    return symbols


def download_symbol_data(symbol, period):
    """
    Download historical data for a symbol.

    Args:
        symbol: Ticker symbol
        period: Time period (e.g., '2y', '5y')

    Returns:
        pd.DataFrame or None: Price data
    """
    try:
        df = yf.download(symbol, period=period, interval="1d", progress=False)

        if df.empty:
            return None

        df.index = df.index.tz_localize(None)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']

        return df

    except Exception as e:
        print(f"      Error downloading data: {e}")
        return None


def calculate_spy_benchmark(df, initial_cash):
    """
    Calculate SPY buy-and-hold benchmark returns.

    Args:
        df: Price dataframe with date range
        initial_cash: Starting capital

    Returns:
        tuple: (spy_df, spy_final_value, spy_return, spy_shares) or (None, None, None, None)
    """
    try:
        start_date = df.index[0]
        end_date = df.index[-1] + pd.Timedelta(days=1)  # Add 1 day because 'end' is EXCLUSIVE

        spy_df = yf.download('SPY', start=start_date, end=end_date, progress=False)

        if spy_df.empty:
            return None, None, None, None

        spy_df.index = spy_df.index.tz_localize(None)

        # Extract scalar values properly
        spy_initial_price = float(spy_df['Close'].iloc[0])
        spy_final_price = float(spy_df['Close'].iloc[-1])

        spy_shares = initial_cash / spy_initial_price
        spy_final_value = spy_shares * spy_final_price
        spy_return = (spy_final_value / initial_cash - 1) * 100

        return spy_df, spy_final_value, spy_return, spy_shares

    except Exception as e:
        print(f"      Warning: Could not calculate SPY benchmark: {e}")
        return None, None, None, None


def setup_cerebro_for_symbol(df, strategy_params, initial_cash, enable_plotting=False):
    """
    Set up Cerebro instance for backtesting.

    Args:
        df: Price dataframe
        strategy_params: Strategy parameters dict
        initial_cash: Starting capital
        enable_plotting: Whether to add observers for plotting

    Returns:
        bt.Cerebro: Configured Cerebro instance
    """
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

    # Add strategy
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

    # Add observers if plotting
    if enable_plotting:
        cerebro.addobserver(BuySellArrows, plot=True, subplot=False)
        cerebro.addobserver(bt.observers.Trades, plot=True, subplot=False)
        cerebro.addobserver(PortfolioValue)

    return cerebro


def extract_results(strat, symbol, df, initial_cash, spy_return):
    """
    Extract comprehensive results from completed backtest.

    Args:
        strat: Strategy instance with analyzers
        symbol: Ticker symbol
        df: Price dataframe
        initial_cash: Starting capital
        spy_return: SPY benchmark return percentage

    Returns:
        dict: Comprehensive results dictionary
    """
    # Get analyzer results
    sharpe = strat.analyzers.sharpe.get_analysis()
    dd = strat.analyzers.dd.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    sqn = strat.analyzers.sqn.get_analysis()

    # Basic performance metrics
    final_value = strat.broker.getvalue()
    total_return = final_value - initial_cash
    return_pct = (final_value / initial_cash - 1) * 100

    # Annualized return
    days_traded = len(df)
    years = days_traded / 252
    annualized_return = ((final_value / initial_cash) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Risk metrics
    max_dd_pct = dd.get('max', {}).get('drawdown', 0)
    calmar_ratio = (return_pct / max_dd_pct) if max_dd_pct > 0 else 0
    sqn_score = sqn.get('sqn', 0)

    # Trade statistics
    total_trades = trades.get('total', {}).get('total', 0)

    if total_trades > 0:
        trade_stats = _extract_trade_statistics(trades)
    else:
        trade_stats = _get_empty_trade_stats()

    # Combine all results
    result = {
        'symbol': symbol,
        'initial_value': initial_cash,
        'final_value': final_value,
        'total_return': total_return,
        'return_pct': return_pct,
        'annualized_return': annualized_return,
        'spy_return': spy_return if spy_return is not None else 0,
        'outperformance': return_pct - spy_return if spy_return is not None else 0,
        'sharpe': sharpe.get('sharperatio', 0) or 0,
        'calmar': calmar_ratio,
        'sqn': sqn_score,
        'max_drawdown': max_dd_pct,
        'max_drawdown_money': dd.get('max', {}).get('moneydown', 0),
        **trade_stats
    }

    return result


def _extract_trade_statistics(trades):
    """Extract detailed trade statistics (helper function)."""
    total_trades = trades.get('total', {}).get('total', 0)
    wins = trades.get('won', {}).get('total', 0)
    losses = trades.get('lost', {}).get('total', 0)

    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    avg_win = trades.get('won', {}).get('pnl', {}).get('average', 0)
    avg_loss = abs(trades.get('lost', {}).get('pnl', {}).get('average', 0))
    rr_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0

    total_win_pnl = trades.get('won', {}).get('pnl', {}).get('total', 0)
    total_loss_pnl = abs(trades.get('lost', {}).get('pnl', {}).get('total', 0))
    profit_factor = (total_win_pnl / total_loss_pnl) if total_loss_pnl > 0 else 0

    expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)

    return {
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'rr_ratio': rr_ratio,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'best_trade': trades.get('won', {}).get('pnl', {}).get('max', 0),
        'worst_trade': trades.get('lost', {}).get('pnl', {}).get('max', 0),
        'avg_trade_len': trades.get('len', {}).get('average', 0),
        'max_trade_len': trades.get('len', {}).get('max', 0),
        'max_win_streak': trades.get('streak', {}).get('won', {}).get('longest', 0),
        'max_loss_streak': trades.get('streak', {}).get('lost', {}).get('longest', 0),
    }


def _get_empty_trade_stats():
    """Return empty trade statistics dict."""
    return {
        'total_trades': 0,
        'wins': 0,
        'losses': 0,
        'win_rate': 0,
        'avg_win': 0,
        'avg_loss': 0,
        'rr_ratio': 0,
        'profit_factor': 0,
        'expectancy': 0,
        'best_trade': 0,
        'worst_trade': 0,
        'avg_trade_len': 0,
        'max_trade_len': 0,
        'max_win_streak': 0,
        'max_loss_streak': 0,
    }


def backtest_single_symbol(symbol, period, initial_cash, strategy_params, enable_plotting=False):
    """
    Run backtest for a single symbol.

    Args:
        symbol: Ticker symbol
        period: Time period for data
        initial_cash: Starting capital
        strategy_params: Strategy parameters dict
        enable_plotting: Whether to enable plotting

    Returns:
        dict or None: Results dictionary or None if failed
    """
    # Download data
    df = download_symbol_data(symbol, period)
    if df is None:
        return None

    # Calculate benchmark
    spy_df, spy_final_value, spy_return, spy_shares = calculate_spy_benchmark(df, initial_cash)

    # Setup and run backtest
    try:
        cerebro = setup_cerebro_for_symbol(df, strategy_params, initial_cash, enable_plotting)
        results = cerebro.run()
        strat = results[0]

        # Extract results
        result = extract_results(strat, symbol, df, initial_cash, spy_return)

        # Generate plots if requested
        if enable_plotting:
            _create_symbol_plots(cerebro, strat, symbol, df, spy_df, spy_shares, initial_cash)

        return result

    except Exception as e:
        print(f"      Error during backtest: {e}")
        return None


def _create_symbol_plots(cerebro, strat, symbol, df, spy_df, spy_shares, initial_cash):
    """Create plots for individual symbol (helper function)."""
    plt.style.use('dark_background')

    try:
        # Backtrader built-in plot
        figs = cerebro.plot(
            style='candlestick',
            iplot=False,
            barup='#597D35',
            bardown='#FF7171',
            volume=False,
        )

        for fig in figs:
            fig[0].savefig(f"{symbol}_backtest.png", dpi=150, bbox_inches='tight')

        print(f"      ✓ Saved backtrader plot: {symbol}_backtest.png")
    except Exception as e:
        print(f"      Warning: Could not create backtrader plot: {e}")

    # Custom comparison plot
    if spy_df is not None and spy_shares is not None:
        try:
            _create_comparison_plot(strat, df, spy_df, spy_shares, initial_cash, symbol)
            print(f"      ✓ Saved comparison plot: {symbol}_performance_comparison.png")
        except Exception as e:
            print(f"      Warning: Could not create comparison plot: {e}")


def _create_comparison_plot(strat, df, spy_df, spy_shares, initial_cash, symbol):
    """Create portfolio vs SPY comparison plot."""
    # Extract portfolio values
    portfolio_values = []
    dates = []

    observer = strat.observers.portfoliovalue
    for i in range(len(observer.lines.value)):
        try:
            val = observer.lines.value.array[i]
            if not np.isnan(val) and val > 0:
                portfolio_values.append(val)
                dates.append(df.index[i])
        except (IndexError, AttributeError):
            break

    if len(portfolio_values) < 2:
        return

    # Calculate SPY values
    spy_initial_price = float(spy_df['Close'].iloc[0])
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
                    spy_price = spy_initial_price

            spy_values.append(spy_shares * spy_price)
        except Exception:
            if spy_values:
                spy_values.append(spy_values[-1])
            else:
                spy_values.append(initial_cash)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.patch.set_facecolor('#1a1a1a')

    # Plot 1: Portfolio value comparison
    ax1.plot(dates, portfolio_values, label=f'{symbol} Strategy', linewidth=2, color='#00ff88')
    ax1.plot(dates, spy_values, label='SPY Buy & Hold', linewidth=2, color='#ff6b6b', linestyle='--')
    ax1.axhline(y=initial_cash, color='gray', linestyle=':', alpha=0.5, label='Initial Capital')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12, color='white')
    ax1.set_title(f'{symbol} Strategy vs SPY Buy & Hold', fontsize=14, fontweight='bold', color='white')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(colors='white')
    ax1.set_facecolor('#2a2a2a')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Plot 2: Cumulative returns
    strategy_returns = [(v / initial_cash - 1) * 100 for v in portfolio_values]
    spy_returns = [(v / initial_cash - 1) * 100 for v in spy_values]

    ax2.plot(dates, strategy_returns, label=f'{symbol} Strategy', linewidth=2, color='#00ff88')
    ax2.plot(dates, spy_returns, label='SPY Buy & Hold', linewidth=2, color='#ff6b6b', linestyle='--')
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax2.fill_between(dates, strategy_returns, 0, alpha=0.3, color='#00ff88')
    ax2.set_xlabel('Date', fontsize=12, color='white')
    ax2.set_ylabel('Cumulative Return (%)', fontsize=12, color='white')
    ax2.set_title('Cumulative Returns Over Time', fontsize=14, fontweight='bold', color='white')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(colors='white')
    ax2.set_facecolor('#2a2a2a')

    # Add final return annotations
    final_strategy_return = strategy_returns[-1]
    final_spy_return = spy_returns[-1]

    ax2.annotate(f'{final_strategy_return:.1f}%',
                 xy=(dates[-1], final_strategy_return),
                 xytext=(10, 0), textcoords='offset points',
                 fontsize=10, color='#00ff88', fontweight='bold')
    ax2.annotate(f'{final_spy_return:.1f}%',
                 xy=(dates[-1], final_spy_return),
                 xytext=(10, 0), textcoords='offset points',
                 fontsize=10, color='#ff6b6b', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{symbol}_performance_comparison.png", dpi=150, bbox_inches='tight', facecolor='#1a1a1a')


# ============================================================================
# Results Analysis and Reporting
# ============================================================================

def print_strategy_config(strategy_params):
    """Print strategy configuration."""
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


def print_aggregate_results(df_results, initial_cash):
    """Print comprehensive aggregate results."""
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")

    # Portfolio performance
    total_initial = len(df_results) * initial_cash
    total_final = df_results['final_value'].sum()
    total_return = total_final - total_initial
    avg_return_pct = df_results['return_pct'].mean()
    avg_spy_return = df_results['spy_return'].mean()

    print(f"\n💰 Portfolio Performance")
    print(f"   Stocks Tested:             {len(df_results)}")
    print(f"   Total Initial Value:       ${total_initial:,.2f}")
    print(f"   Total Final Value:         ${total_final:,.2f}")
    print(f"   Total Return:              ${total_return:,.2f} ({(total_return/total_initial)*100:.2f}%)")
    print(f"   Avg Return per Stock:      {avg_return_pct:.2f}%")
    print(f"   Avg SPY Return:            {avg_spy_return:.2f}%")
    print(f"   Avg Outperformance:        {df_results['outperformance'].mean():.2f}%")

    outperformed = len(df_results[df_results['outperformance'] > 0])
    print(f"   Stocks Beat SPY:           {outperformed} ({outperformed/len(df_results)*100:.1f}%)")

    # Risk metrics
    print(f"\n📉 Risk Metrics")
    avg_sharpe = df_results['sharpe'].mean()
    if not pd.isna(avg_sharpe):
        print(f"   Avg Sharpe Ratio:          {avg_sharpe:.3f}")
    else:
        print(f"   Avg Sharpe Ratio:          N/A")
    print(f"   Avg Calmar Ratio:          {df_results['calmar'].mean():.3f}")
    print(f"   Avg SQN:                   {df_results['sqn'].mean():.2f}")
    print(f"   Avg Max Drawdown:          {df_results['max_drawdown'].mean():.2f}%")
    print(f"   Worst Drawdown:            {df_results['max_drawdown'].max():.2f}%")

    # Trade statistics
    print(f"\n📈 Trade Statistics")
    total_trades = int(df_results['total_trades'].sum())
    total_wins = int(df_results['wins'].sum())
    total_losses = int(df_results['losses'].sum())

    print(f"   Total Trades:              {total_trades}")
    print(f"   Total Wins:                {total_wins}")
    print(f"   Total Losses:              {total_losses}")

    if total_trades > 0:
        overall_win_rate = (total_wins / total_trades * 100)
        print(f"   Overall Win Rate:          {overall_win_rate:.1f}%")
        print(f"   Avg RR Ratio:              {df_results['rr_ratio'].mean():.2f}")
        print(f"   Avg Profit Factor:         {df_results['profit_factor'].mean():.2f}")
        print(f"   Avg Expectancy:            ${df_results['expectancy'].mean():.2f}")
        print(f"   Avg Trade Duration:        {df_results['avg_trade_len'].mean():.1f} days")

    # Top and bottom performers
    _print_top_performers(df_results)
    _print_bottom_performers(df_results)
    _print_best_risk_adjusted(df_results)

    print(f"{'='*70}\n")


def _print_top_performers(df_results, n=5):
    """Print top N performers by return."""
    print(f"\n🏆 Top {n} Performers (by Return %)")
    top_n = df_results.nlargest(n, 'return_pct')[
        ['symbol', 'return_pct', 'outperformance', 'sharpe', 'total_trades']
    ]

    for idx, row in top_n.iterrows():
        sharpe_str = f"{row['sharpe']:5.2f}" if pd.notna(row['sharpe']) else "  N/A"
        print(
            f"   {row['symbol']:6s}: {row['return_pct']:7.2f}% | "
            f"vs SPY: {row['outperformance']:+7.2f}% | "
            f"Sharpe: {sharpe_str} | "
            f"Trades: {int(row['total_trades'])}"
        )


def _print_bottom_performers(df_results, n=5):
    """Print bottom N performers by return."""
    print(f"\n📉 Bottom {n} Performers")
    bottom_n = df_results.nsmallest(n, 'return_pct')[
        ['symbol', 'return_pct', 'outperformance', 'sharpe', 'total_trades']
    ]

    for idx, row in bottom_n.iterrows():
        sharpe_str = f"{row['sharpe']:5.2f}" if pd.notna(row['sharpe']) else "  N/A"
        print(
            f"   {row['symbol']:6s}: {row['return_pct']:7.2f}% | "
            f"vs SPY: {row['outperformance']:+7.2f}% | "
            f"Sharpe: {sharpe_str} | "
            f"Trades: {int(row['total_trades'])}"
        )


def _print_best_risk_adjusted(df_results, n=5):
    """Print best N risk-adjusted performers by Sharpe ratio."""
    print(f"\n🎯 Best {n} Risk-Adjusted (by Sharpe Ratio)")

    valid_sharpe = df_results[df_results['sharpe'].notna()]

    if len(valid_sharpe) > 0:
        best_sharpe = valid_sharpe.nlargest(min(n, len(valid_sharpe)), 'sharpe')[
            ['symbol', 'sharpe', 'return_pct', 'max_drawdown']
        ]

        for idx, row in best_sharpe.iterrows():
            print(
                f"   {row['symbol']:6s}: Sharpe {row['sharpe']:5.2f} | "
                f"Return: {row['return_pct']:7.2f}% | "
                f"MaxDD: {row['max_drawdown']:5.2f}%"
            )
    else:
        print("   No valid Sharpe ratios calculated")


# ============================================================================
# Main Execution
# ============================================================================

def run_multi_backtest(config):
    """
    Run backtest across multiple symbols.

    Args:
        config: Configuration dictionary
    """
    print("\n" + "="*70)
    print("AROON MULTI-FILTER STRATEGY - MULTI-TICKER BACKTEST")
    print("="*70)

    # Load symbols
    symbols = load_symbols(config['csv_file'])
    print(f"\n📋 Loaded {len(symbols)} symbols from '{config['csv_file']}'")

    # Print configuration
    print_strategy_config(config['strategy_params'])

    print(f"\n🚀 Running Backtests...")
    print(f"   Period: {config['period']}")
    print(f"   Initial cash per stock: ${config['initial_cash']:,.2f}")
    print()

    # Run backtests
    results = []
    for i, symbol in enumerate(symbols, 1):
        print(f"   [{i}/{len(symbols)}] Testing {symbol}...", end=" ")

        result = backtest_single_symbol(
            symbol,
            config['period'],
            config['initial_cash'],
            config['strategy_params'],
            enable_plotting=False
        )

        if result:
            results.append(result)
            print(
                f"✓ Return: {result['return_pct']:7.2f}% | "
                f"vs SPY: {result['outperformance']:+7.2f}%"
            )
        else:
            print("❌ Failed")

    if not results:
        print("\n❌ No valid results obtained!")
        return

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Print aggregate results
    print_aggregate_results(df_results, config['initial_cash'])

    # Save results
    if config['save_individual_results']:
        df_results.to_csv(config['results_csv'], index=False)
        print(f"📄 Detailed results saved to '{config['results_csv']}'")

    # Plot best and/or worst performers
    if config['plot_best'] and len(results) > 0:
        best_symbol = df_results.loc[df_results['return_pct'].idxmax(), 'symbol']
        print(f"\n📊 Generating detailed plots for best performer: {best_symbol}")
        backtest_single_symbol(
            best_symbol,
            config['period'],
            config['initial_cash'],
            config['strategy_params'],
            enable_plotting=True
        )

    if config.get('plot_worst', False) and len(results) > 0:
        worst_symbol = df_results.loc[df_results['return_pct'].idxmin(), 'symbol']
        print(f"\n📊 Generating detailed plots for worst performer: {worst_symbol}")
        backtest_single_symbol(
            worst_symbol,
            config['period'],
            config['initial_cash'],
            config['strategy_params'],
            enable_plotting=True
        )

    print("\n✅ Multi-ticker backtest complete!\n")


def main():
    """Main entry point."""
    run_multi_backtest(CONFIG)


if __name__ == "__main__":
    main()
