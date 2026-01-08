# source bt/bin/activate
# cd strategies/SMA_ATR/
# python backtest_multi.py

import matplotlib
matplotlib.use('Agg')
import backtrader as bt
import yfinance as yf
from math import isnan
import matplotlib.pyplot as plt
from sma_atr import SMA_ATR_Exit
import pandas as pd
import csv


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


def backtest_symbol(symbol, period="2y", initial_cash=10_000, plot=False):
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
        cerebro.addstrategy(SMA_ATR_Exit)

        # Broker settings
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=0.0)

        # Analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

        if plot:
            cerebro.addobserver(BuySellArrows, plot=True, subplot=False)
            cerebro.addobserver(bt.observers.Trades, plot=True, subplot=False)

        # Run
        results = cerebro.run()
        strat = results[0]

        # Extract results
        sharpe = strat.analyzers.sharpe.get_analysis()
        dd = strat.analyzers.dd.get_analysis()
        trades = strat.analyzers.trades.get_analysis()

        final_value = cerebro.broker.getvalue()
        total_return = final_value - initial_cash
        return_pct = (final_value / initial_cash - 1) * 100

        result = {
            'symbol': symbol,
            'initial_value': initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'return_pct': return_pct,
            'sharpe': sharpe.get('sharperatio', 0) or 0,
            'max_drawdown': dd['max']['drawdown'] if dd.get('max') else 0,
            'max_drawdown_money': dd['max']['moneydown'] if dd.get('max') else 0,
            'total_trades': trades['total']['total'] if trades.get('total') else 0,
            'wins': trades['won']['total'] if trades.get('won') else 0,
            'losses': trades['lost']['total'] if trades.get('lost') else 0,
            'avg_win': trades['won']['pnl']['average'] if trades.get('won') else 0,
            'avg_loss': trades['lost']['pnl']['average'] if trades.get('lost') else 0,
            'win_rate': (trades['won']['total'] / trades['total']['total'] * 100) if trades.get('total') and trades['total']['total'] > 0 else 0
        }

        # Plot if requested
        if plot:
            plt.style.use('dark_background')
            figs = cerebro.plot(
                style='candlestick',
                iplot=False,
                barup='#597D35',
                bardown='#FF7171',
                volume=False,
            )
            for fig in figs:
                fig[0].savefig(f"{symbol}_backtest.png")

        return result

    except Exception as e:
        print(f"❌ {symbol}: Error - {str(e)}")
        return None


def run_multi_backtest(csv_file='stocks.csv', period="2y", initial_cash=10_000, plot_best=True):
    """Run backtest on multiple stocks from CSV file"""

    # Read symbols from CSV
    symbols = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():  # Skip empty rows
                symbols.append(row[0].strip())

    print(f"\n{'='*70}")
    print(f"RUNNING BACKTEST ON {len(symbols)} STOCKS")
    print(f"{'='*70}\n")

    results = []
    for symbol in symbols:
        print(f"Testing {symbol}...", end=" ")
        result = backtest_symbol(symbol, period, initial_cash, plot=False)
        if result:
            results.append(result)
            print(f"✓ Return: {result['return_pct']:.2f}%")
        else:
            print()

    if not results:
        print("No valid results!")
        return

    # Convert to DataFrame for easy analysis
    df_results = pd.DataFrame(results)

    # Print summary
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")

    total_initial = len(results) * initial_cash
    total_final = df_results['final_value'].sum()
    total_return = total_final - total_initial
    avg_return_pct = df_results['return_pct'].mean()

    print(f"\n💰 Portfolio Performance:")
    print(f"   Stocks Tested:       {len(results)}")
    print(f"   Total Initial Value: ${total_initial:,.2f}")
    print(f"   Total Final Value:   ${total_final:,.2f}")
    print(f"   Total Return:        ${total_return:,.2f} ({(total_return/total_initial)*100:.2f}%)")
    print(f"   Avg Return per Stock: {avg_return_pct:.2f}%")

    print(f"\n📊 Risk Metrics:")
    print(f"   Avg Sharpe Ratio:    {df_results['sharpe'].mean():.3f}")
    print(f"   Avg Max Drawdown:    {df_results['max_drawdown'].mean():.2f}%")

    print(f"\n📈 Trade Statistics:")
    print(f"   Total Trades:        {df_results['total_trades'].sum()}")
    print(f"   Total Wins:          {df_results['wins'].sum()}")
    print(f"   Total Losses:        {df_results['losses'].sum()}")
    print(f"   Overall Win Rate:    {(df_results['wins'].sum() / df_results['total_trades'].sum() * 100):.1f}%")

    print(f"\n🏆 Top 5 Performers:")
    top5 = df_results.nlargest(5, 'return_pct')[['symbol', 'return_pct', 'sharpe', 'total_trades']]
    for idx, row in top5.iterrows():
        print(f"   {row['symbol']:6s}: {row['return_pct']:7.2f}% | Sharpe: {row['sharpe']:5.2f} | Trades: {int(row['total_trades'])}")

    print(f"\n📉 Bottom 5 Performers:")
    bottom5 = df_results.nsmallest(5, 'return_pct')[['symbol', 'return_pct', 'sharpe', 'total_trades']]
    for idx, row in bottom5.iterrows():
        print(f"   {row['symbol']:6s}: {row['return_pct']:7.2f}% | Sharpe: {row['sharpe']:5.2f} | Trades: {int(row['total_trades'])}")

    print(f"\n{'='*70}\n")

    # Save detailed results to CSV
    df_results.to_csv('backtest_results.csv', index=False)
    print("📄 Detailed results saved to 'backtest_results.csv'")

    # Plot the best performing stock
    if plot_best:
        best_symbol = df_results.loc[df_results['return_pct'].idxmax(), 'symbol']
        print(f"\n📊 Generating plot for best performer: {best_symbol}")
        backtest_symbol(best_symbol, period, initial_cash, plot=True)

if __name__ == "__main__":
    # Run the multi-stock backtest
    run_multi_backtest(csv_file='../sp500_2024.csv', period='2y', initial_cash=10_000, plot_best=True)