# source bt/bin/activate
# cd strategies/AROON_ATR/
# python backtest_multi.py

import matplotlib
matplotlib.use('Agg')
import backtrader as bt
import yfinance as yf
from math import isnan
import matplotlib.pyplot as plt
from aroon_atr import Strategy
import pandas as pd
import numpy as np
import csv
from decimal import Decimal


# ============================================================================
# CONFIGURATION - MODIFY THESE PARAMETERS
# ============================================================================
STRATEGY_PARAMS = {
    'aroon_len': 20,
    'atr_entry_len': 10,
    'atr_entry_mult': Decimal("2.0"),  # From ATR(10,2) - not used in filter
    'atr_entry_sma_period': 20,  # ATR is "low" when < its 20-period SMA
    'atr_filter_mult': Decimal("1.2"),  # Threshold: 1.0 = exact, 1.2 = 20% looser
    'use_atr_filter': True,  # Keep enabled - filters out noisy markets
    'atr_exit_len': 14,
    'atr_exit_mult': Decimal("3.0"),   # Exit: ATR trailing stop
    'stop_loss_pct': Decimal("0.02"),  # 2% stop loss
    'take_profit_pct': Decimal("0.13"), # 13% take profit
    'use_take_profit': True,  # Set to False to disable take profit
    'verbose': False
}

CSV_FILE = '../optimization_set.csv'
PERIOD = "2y"
INITIAL_CASH = 10_000
PLOT_BEST = True
# ============================================================================


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


def backtest_symbol(symbol, period="2y", initial_cash=10_000, strategy_params=None, plot=False):
    """Run backtest for a single symbol and return results"""
    try:
        # Download data
        df = yf.download(symbol, period=period, interval="1d", progress=False)

        if df.empty:
            print(f"❌ {symbol}: No data available")
            return None

        df.index = df.index.tz_localize(None)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']

        # Download SPY for benchmark
        start_date = df.index[0]
        end_date = df.index[-1]
        spy_df = yf.download('SPY', start=start_date, end=end_date, progress=False)
        spy_df.index = spy_df.index.tz_localize(None)

        # Calculate SPY buy-and-hold return
        spy_initial_price = spy_df['Close'].iloc[0]
        spy_final_price = spy_df['Close'].iloc[-1]

        # Ensure we have scalar values
        if hasattr(spy_initial_price, 'item'):
            spy_initial_price = spy_initial_price.item()
        if hasattr(spy_final_price, 'item'):
            spy_final_price = spy_final_price.item()

        spy_shares = initial_cash / spy_initial_price
        spy_final_value = spy_shares * spy_final_price
        spy_return = (spy_final_value / initial_cash - 1) * 100

        # Setup cerebro
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

        # Add strategy with custom parameters
        if strategy_params:
            cerebro.addstrategy(Strategy, **strategy_params)
        else:
            cerebro.addstrategy(Strategy)

        # Broker settings
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=0.0)

        # Analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")

        if plot:
            cerebro.addobserver(BuySellArrows, plot=True, subplot=False)
            cerebro.addobserver(bt.observers.Trades, plot=True, subplot=False)
            cerebro.addobserver(PortfolioValue)

        # Run
        results = cerebro.run()
        strat = results[0]

        # Extract results
        sharpe = strat.analyzers.sharpe.get_analysis()
        dd = strat.analyzers.dd.get_analysis()
        trades = strat.analyzers.trades.get_analysis()
        sqn = strat.analyzers.sqn.get_analysis()

        final_value = cerebro.broker.getvalue()
        total_return = final_value - initial_cash
        return_pct = (final_value / initial_cash - 1) * 100

        # Calculate detailed trade statistics
        total_trades = trades.get('total', {}).get('total', 0)

        if total_trades > 0:
            wins = trades.get('won', {}).get('total', 0)
            losses = trades.get('lost', {}).get('total', 0)
            win_rate = (wins / total_trades * 100)

            avg_win = trades.get('won', {}).get('pnl', {}).get('average', 0)
            avg_loss = abs(trades.get('lost', {}).get('pnl', {}).get('average', 0))

            rr_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0

            total_win_pnl = trades.get('won', {}).get('pnl', {}).get('total', 0)
            total_loss_pnl = abs(trades.get('lost', {}).get('pnl', {}).get('total', 0))
            profit_factor = (total_win_pnl / total_loss_pnl) if total_loss_pnl > 0 else 0

            expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)

            best_trade = trades.get('won', {}).get('pnl', {}).get('max', 0)
            worst_trade = trades.get('lost', {}).get('pnl', {}).get('max', 0)

            avg_trade_len = trades.get('len', {}).get('average', 0)
            max_trade_len = trades.get('len', {}).get('max', 0)

            max_win_streak = trades.get('streak', {}).get('won', {}).get('longest', 0)
            max_loss_streak = trades.get('streak', {}).get('lost', {}).get('longest', 0)
        else:
            win_rate = rr_ratio = profit_factor = expectancy = 0
            best_trade = worst_trade = avg_win = avg_loss = 0
            avg_trade_len = max_trade_len = 0
            max_win_streak = max_loss_streak = 0
            wins = losses = 0

        # Calculate additional metrics
        max_dd_pct = dd.get('max', {}).get('drawdown', 0)
        calmar_ratio = (return_pct / max_dd_pct) if max_dd_pct > 0 else 0
        sqn_score = sqn.get('sqn', 0)

        # Annualized return
        days_traded = len(df)
        years = days_traded / 252
        annualized_return = ((final_value / initial_cash) ** (1 / years) - 1) * 100 if years > 0 else 0

        result = {
            'symbol': symbol,
            'initial_value': initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'return_pct': return_pct,
            'annualized_return': annualized_return,
            'spy_return': spy_return,
            'outperformance': return_pct - spy_return,
            'sharpe': sharpe.get('sharperatio', 0) or 0,
            'calmar': calmar_ratio,
            'sqn': sqn_score,
            'max_drawdown': max_dd_pct,
            'max_drawdown_money': dd.get('max', {}).get('moneydown', 0),
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'rr_ratio': rr_ratio,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'avg_trade_len': avg_trade_len,
            'max_trade_len': max_trade_len,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak
        }

        # Plot if requested
        if plot:
            # Backtrader plot
            plt.style.use('dark_background')
            figs = cerebro.plot(
                style='candlestick',
                iplot=False,
                barup='#597D35',
                bardown='#FF7171',
                volume=False,
            )
            for fig in figs:
                fig[0].savefig(f"{symbol}_backtest.png", dpi=150, bbox_inches='tight')

            # Create comparison plot
            create_comparison_plot(strat, df, spy_df, spy_shares, initial_cash, symbol)

        return result

    except Exception as e:
        print(f"❌ {symbol}: Error - {str(e)}")
        return None


def create_comparison_plot(strat, df, spy_df, spy_shares, initial_cash, symbol):
    """Create portfolio vs SPY comparison plot"""

    # Get SPY initial price for reference
    spy_initial_price = spy_df['Close'].iloc[0]
    if hasattr(spy_initial_price, 'item'):
        spy_initial_price = spy_initial_price.item()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.patch.set_facecolor('#1a1a1a')

    # Get portfolio values from observer
    portfolio_values = []
    dates = []

    observer = strat.observers.portfoliovalue
    for i in range(len(observer.lines.value)):
        try:
            val = observer.lines.value.array[i]
            if not np.isnan(val) and val > 0:
                portfolio_values.append(val)
                dates.append(df.index[i].date())
        except (IndexError, AttributeError):
            break

    if len(portfolio_values) < 2:
        return

    # Calculate SPY values
    spy_values = []
    for date in dates:
        try:
            matching_rows = spy_df.loc[spy_df.index.date == date, 'Close']
            if len(matching_rows) > 0:
                spy_price = matching_rows.iloc[0]
                if hasattr(spy_price, 'item'):
                    spy_price = spy_price.item()
            else:
                valid_dates = spy_df.index[spy_df.index.date <= date]
                if len(valid_dates) > 0:
                    spy_price = spy_df.loc[valid_dates[-1], 'Close']
                    if hasattr(spy_price, 'item'):
                        spy_price = spy_price.item()
                else:
                    spy_price = spy_initial_price

            spy_value = spy_shares * spy_price
            spy_values.append(spy_value)
        except Exception as e:
            if spy_values:
                spy_values.append(spy_values[-1])
            else:
                spy_values.append(initial_cash)

    dates = pd.to_datetime(dates)

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

    # Plot 2: Cumulative returns %
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

    final_strategy_return = strategy_returns[-1]
    final_spy_return = spy_returns[-1]
    ax2.annotate(f'{final_strategy_return:.1f}%',
                 xy=(dates.iloc[-1], final_strategy_return),
                 xytext=(10, 0), textcoords='offset points',
                 fontsize=10, color='#00ff88', fontweight='bold')
    ax2.annotate(f'{final_spy_return:.1f}%',
                 xy=(dates.iloc[-1], final_spy_return),
                 xytext=(10, 0), textcoords='offset points',
                 fontsize=10, color='#ff6b6b', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{symbol}_performance_comparison.png", dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    print(f"✓ Saved performance comparison to {symbol}_performance_comparison.png")


def run_multi_backtest(csv_file='stocks.csv', period="2y", initial_cash=10_000,
                       strategy_params=None, plot_best=True):
    """Run backtest on multiple stocks from CSV file"""

    # Read symbols from CSV
    symbols = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                symbols.append(row[0].strip())

    print(f"\n{'='*70}")
    print(f"MULTI-STOCK BACKTEST - AROON/ATR STRATEGY")
    print(f"{'='*70}\n")
    print(f"Stocks to test: {len(symbols)}")
    if strategy_params:
        print(f"\nStrategy Configuration:")
        if strategy_params.get('use_atr_filter', True):
            sma_period = strategy_params.get('atr_entry_sma_period', 20)
            print(f"  Entry: AROON({strategy_params.get('aroon_len', 20)}) crossover when ATR({strategy_params.get('atr_entry_len', 10)}) < SMA({sma_period}) (calm)")
        else:
            print(f"  Entry: AROON({strategy_params.get('aroon_len', 20)}) crossover (ATR filter DISABLED)")
        print(f"  Exit: ATR({strategy_params.get('atr_exit_len', 14)}, {strategy_params.get('atr_exit_mult', 3.0)}) trailing stop")
        print(f"        + {float(strategy_params.get('stop_loss_pct', 0.02))*100}% stop loss")
        if strategy_params.get('use_take_profit', True):
            print(f"        + {float(strategy_params.get('take_profit_pct', 0.13))*100}% take profit")
        else:
            print(f"        (Take profit disabled)")
    print()

    results = []
    for symbol in symbols:
        print(f"Testing {symbol}...", end=" ")
        result = backtest_symbol(symbol, period, initial_cash, strategy_params, plot=False)
        if result:
            results.append(result)
            print(f"✓ Return: {result['return_pct']:7.2f}% | vs SPY: {result['outperformance']:+7.2f}%")
        else:
            print()

    if not results:
        print("No valid results!")
        return

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Print comprehensive summary
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")

    total_initial = len(results) * initial_cash
    total_final = df_results['final_value'].sum()
    total_return = total_final - total_initial
    avg_return_pct = df_results['return_pct'].mean()
    avg_spy_return = df_results['spy_return'].mean()

    print(f"\n💰 Portfolio Performance:")
    print(f"   Stocks Tested:         {len(results)}")
    print(f"   Total Initial Value:   ${total_initial:,.2f}")
    print(f"   Total Final Value:     ${total_final:,.2f}")
    print(f"   Total Return:          ${total_return:,.2f} ({(total_return/total_initial)*100:.2f}%)")
    print(f"   Avg Return per Stock:  {avg_return_pct:.2f}%")
    print(f"   Avg SPY Return:        {avg_spy_return:.2f}%")
    print(f"   Avg Outperformance:    {df_results['outperformance'].mean():.2f}%")
    print(f"   Stocks Beat SPY:       {len(df_results[df_results['outperformance'] > 0])} ({len(df_results[df_results['outperformance'] > 0])/len(results)*100:.1f}%)")

    print(f"\n📉 Risk Metrics:")
    avg_sharpe = df_results['sharpe'].mean()
    print(f"   Avg Sharpe Ratio:      {avg_sharpe:.3f}" if not pd.isna(avg_sharpe) else "   Avg Sharpe Ratio:      N/A")
    print(f"   Avg Calmar Ratio:      {df_results['calmar'].mean():.3f}")
    print(f"   Avg SQN:               {df_results['sqn'].mean():.2f}")
    print(f"   Avg Max Drawdown:      {df_results['max_drawdown'].mean():.2f}%")
    print(f"   Worst Drawdown:        {df_results['max_drawdown'].max():.2f}%")

    print(f"\n📈 Trade Statistics:")
    print(f"   Total Trades:          {int(df_results['total_trades'].sum())}")
    print(f"   Total Wins:            {int(df_results['wins'].sum())}")
    print(f"   Total Losses:          {int(df_results['losses'].sum())}")
    print(f"   Overall Win Rate:      {(df_results['wins'].sum() / df_results['total_trades'].sum() * 100):.1f}%")
    print(f"   Avg RR Ratio:          {df_results['rr_ratio'].mean():.2f}")
    print(f"   Avg Profit Factor:     {df_results['profit_factor'].mean():.2f}")
    print(f"   Avg Expectancy:        ${df_results['expectancy'].mean():.2f}")
    print(f"   Avg Trade Duration:    {df_results['avg_trade_len'].mean():.1f} days")

    print(f"\n🏆 Top 5 Performers (by Return %):")
    top5 = df_results.nlargest(5, 'return_pct')[['symbol', 'return_pct', 'outperformance', 'sharpe', 'total_trades']]
    for idx, row in top5.iterrows():
        sharpe_str = f"{row['sharpe']:5.2f}" if pd.notna(row['sharpe']) else "  N/A"
        print(f"   {row['symbol']:6s}: {row['return_pct']:7.2f}% | vs SPY: {row['outperformance']:+7.2f}% | Sharpe: {sharpe_str} | Trades: {int(row['total_trades'])}")

    print(f"\n📉 Bottom 5 Performers:")
    bottom5 = df_results.nsmallest(5, 'return_pct')[['symbol', 'return_pct', 'outperformance', 'sharpe', 'total_trades']]
    for idx, row in bottom5.iterrows():
        sharpe_str = f"{row['sharpe']:5.2f}" if pd.notna(row['sharpe']) else "  N/A"
        print(f"   {row['symbol']:6s}: {row['return_pct']:7.2f}% | vs SPY: {row['outperformance']:+7.2f}% | Sharpe: {sharpe_str} | Trades: {int(row['total_trades'])}")

    print(f"\n🎯 Best Risk-Adjusted (by Sharpe Ratio):")
    # Filter out NaN sharpe values before sorting
    valid_sharpe = df_results[df_results['sharpe'].notna()]
    if len(valid_sharpe) > 0:
        best_sharpe = valid_sharpe.nlargest(5, 'sharpe')[['symbol', 'sharpe', 'return_pct', 'max_drawdown']]
        for idx, row in best_sharpe.iterrows():
            print(f"   {row['symbol']:6s}: Sharpe {row['sharpe']:5.2f} | Return: {row['return_pct']:7.2f}% | MaxDD: {row['max_drawdown']:5.2f}%")
    else:
        print("   No valid Sharpe ratios calculated")

    print(f"\n{'='*70}\n")

    # Save detailed results
    df_results.to_csv('backtest_results.csv', index=False)
    print("📄 Detailed results saved to 'backtest_results.csv'")

    # Plot the best performing stock
    if plot_best and len(results) > 0:
        best_symbol = df_results.loc[df_results['return_pct'].idxmax(), 'symbol']
        print(f"\n📊 Generating detailed plots for best performer: {best_symbol}\n")
        backtest_symbol(best_symbol, period, initial_cash, strategy_params, plot=True)


if __name__ == "__main__":
    # Run the multi-stock backtest with custom parameters
    run_multi_backtest(
        csv_file=CSV_FILE,
        period=PERIOD,
        initial_cash=INITIAL_CASH,
        strategy_params=STRATEGY_PARAMS,
        plot_best=PLOT_BEST
    )