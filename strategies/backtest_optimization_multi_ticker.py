# source bt/bin/activate
# cd strategies/SMA_ATR/
# python optimize.py

import matplotlib
matplotlib.use('Agg')
import backtrader as bt
import yfinance as yf
from SMA_ATR.sma_atr import SMA_ATR_Exit
import pandas as pd
from itertools import product
from decimal import Decimal
import csv


def backtest_with_params(symbol, params, df_cache, period="2y", initial_cash=10_000):
    """Run a single backtest with given parameters"""

    # Get data from cache or download
    if symbol not in df_cache:
        df = yf.download(symbol, period=period, interval="1d", progress=False)
        if df.empty:
            return None
        df.index = df.index.tz_localize(None)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df_cache[symbol] = df

    df = df_cache[symbol]

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
        # Add verbose=False to suppress strategy logs during optimization
        cerebro.addstrategy(SMA_ATR_Exit, verbose=False, **params)

        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=0.0)

        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

        results_run = cerebro.run()
        strat = results_run[0]

        sharpe = strat.analyzers.sharpe.get_analysis()
        dd = strat.analyzers.dd.get_analysis()
        trades = strat.analyzers.trades.get_analysis()

        final_value = cerebro.broker.getvalue()
        return_pct = (final_value / initial_cash - 1) * 100

        # Safely extract trade statistics
        total_trades = trades.get('total', {}).get('total', 0)

        if total_trades > 0:
            wins = trades.get('won', {}).get('total', 0)
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        else:
            win_rate = 0

        return {
            'return_pct': return_pct,
            'sharpe': sharpe.get('sharperatio', 0) or 0,
            'max_drawdown': dd.get('max', {}).get('drawdown', 0),
            'total_trades': total_trades,
            'win_rate': win_rate,
        }
    except Exception as e:
        print(f"\n❌ Error in backtest for {symbol} with params {params}: {e}")
        import traceback
        traceback.print_exc()
        return None


def optimize_strategy_multi_stock(csv_file, param_grid, period="2y", initial_cash=10_000):
    """
    Optimize strategy parameters across multiple stocks

    Args:
        csv_file: Path to CSV file with stock tickers (one per line)
        param_grid: Dictionary of parameters to test
        period: Time period for backtesting
        initial_cash: Starting capital per stock

    Returns:
        DataFrame with aggregated results for all parameter combinations
    """

    # Read symbols from CSV
    symbols = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                symbols.append(row[0].strip())

    print(f"\n{'='*70}")
    print(f"MULTI-STOCK PARAMETER OPTIMIZATION")
    print(f"{'='*70}\n")
    print(f"Symbols to test: {len(symbols)}")
    print(f"Symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")

    # Download all data first (with caching)
    print(f"\nDownloading data for {len(symbols)} stocks...")
    df_cache = {}
    valid_symbols = []

    for symbol in symbols:
        try:
            df = yf.download(symbol, period=period, interval="1d", progress=False)
            if not df.empty:
                df.index = df.index.tz_localize(None)
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                df_cache[symbol] = df
                valid_symbols.append(symbol)
        except:
            print(f"  ⚠️  Failed to download {symbol}")

    print(f"✓ Successfully downloaded {len(valid_symbols)}/{len(symbols)} stocks\n")

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))

    total_combinations = len(param_combinations)
    total_backtests = total_combinations * len(valid_symbols)

    print(f"Testing {total_combinations} parameter combinations")
    print(f"Total backtests to run: {total_backtests}\n")

    results = []
    backtest_count = 0

    for combo_idx, param_values in enumerate(param_combinations, 1):
        params = dict(zip(param_names, param_values))

        # Test this parameter set across all stocks
        stock_results = []

        for symbol in valid_symbols:
            backtest_count += 1

            if backtest_count % 50 == 0:
                print(f"Progress: {backtest_count}/{total_backtests} ({backtest_count/total_backtests*100:.1f}%)")

            result = backtest_with_params(symbol, params, df_cache, period, initial_cash)
            if result:
                stock_results.append(result)

        # Aggregate results across all stocks
        if stock_results:
            # Calculate aggregate metrics
            avg_return = sum(r['return_pct'] for r in stock_results) / len(stock_results)
            avg_sharpe = sum(r['sharpe'] for r in stock_results) / len(stock_results)
            avg_drawdown = sum(r['max_drawdown'] for r in stock_results) / len(stock_results)
            total_trades = sum(r['total_trades'] for r in stock_results)
            avg_win_rate = sum(r['win_rate'] for r in stock_results) / len(stock_results)

            # Count winning stocks (positive return)
            winning_stocks = sum(1 for r in stock_results if r['return_pct'] > 0)
            stock_win_rate = (winning_stocks / len(stock_results)) * 100

            # Store aggregated result
            aggregated = {
                'fast_len': params.get('fast_len'),
                'slow_len': params.get('slow_len'),
                'atr_len': params.get('atr_len'),
                'atr_mult': params.get('atr_mult'),
                'stop_loss_pct': params.get('stop_loss_pct'),
                'avg_return_pct': avg_return,
                'avg_sharpe': avg_sharpe,
                'avg_max_drawdown': avg_drawdown,
                'total_trades': total_trades,
                'avg_win_rate': avg_win_rate,
                'stock_win_rate': stock_win_rate,
                'stocks_tested': len(stock_results),
                'winning_stocks': winning_stocks
            }

            results.append(aggregated)

    df_results = pd.DataFrame(results)
    return df_results


def print_optimization_results(df_results, metric='avg_return_pct', top_n=5):
    """Print top performing parameter sets"""

    print(f"\n{'='*70}")
    print(f"OPTIMIZATION RESULTS (Ranked by {metric})")
    print(f"{'='*70}\n")

    df_sorted = df_results.sort_values(by=metric, ascending=False)

    print(f"🏆 TOP {top_n} PARAMETER SETS:\n")

    for i, (idx, row) in enumerate(df_sorted.head(top_n).iterrows(), 1):
        print(f"#{i} {'─'*66}")
        print(f"Parameters:")
        print(f"  Fast SMA:      {int(row['fast_len'])}")
        print(f"  Slow SMA:      {int(row['slow_len'])}")
        print(f"  ATR Length:    {int(row['atr_len'])}")
        print(f"  ATR Mult:      {row['atr_mult']:.1f}")
        print(f"  Stop Loss %:   {row['stop_loss_pct']*100:.1f}%")
        print(f"\nAggregate Performance (across {int(row['stocks_tested'])} stocks):")
        print(f"  Avg Return:        {row['avg_return_pct']:7.2f}%")
        print(f"  Avg Sharpe:        {row['avg_sharpe']:7.3f}")
        print(f"  Avg Max Drawdown:  {row['avg_max_drawdown']:7.2f}%")
        print(f"  Total Trades:      {int(row['total_trades'])}")
        print(f"  Avg Trade Win %:   {row['avg_win_rate']:7.1f}%")

        # Only print new metrics if they exist
        if 'avg_rr_ratio' in row:
            print(f"  Avg RR Ratio:      {row['avg_rr_ratio']:7.2f}")
        if 'avg_expectancy' in row:
            print(f"  Avg Expectancy:    ${row['avg_expectancy']:7.2f}")
        if 'avg_profit_factor' in row:
            print(f"  Avg Profit Factor: {row['avg_profit_factor']:7.2f}")
        if 'avg_trade_pnl' in row:
            print(f"  Avg Trade P&L:     ${row['avg_trade_pnl']:7.2f}")

        print(f"  Stock Win Rate:    {row['stock_win_rate']:7.1f}% ({int(row['winning_stocks'])}/{int(row['stocks_tested'])} stocks)")
        print()

    print(f"{'─'*70}")
    print(f"SUMMARY STATISTICS:")
    print(f"  Total Combinations:  {len(df_results)}")
    print(f"  Best {metric}:       {df_sorted.iloc[0][metric]:.2f}")
    print(f"  Worst {metric}:      {df_sorted.iloc[-1][metric]:.2f}")
    print(f"  Average {metric}:    {df_results[metric].mean():.2f}")
    print(f"  Median {metric}:     {df_results[metric].median():.2f}")
    print(f"{'='*70}\n")


def save_results(df_results, filename='optimization_results.csv'):
    """Save all optimization results to CSV"""
    df_results.to_csv(filename, index=False)
    print(f"📄 Full results saved to '{filename}'")


if __name__ == "__main__":
    # Define parameter grid to test
    param_grid = {
        'fast_len': [7, 10, 14, 20],
        'slow_len': [15, 18, 26, 50, 100],
        'atr_len': [7, 10, 14, 20],
        'atr_mult': [Decimal("2.0"), Decimal("3.0"), Decimal("3.5"), Decimal("4.0")],
        'stop_loss_pct': [Decimal("0.05"), Decimal("0.1"), Decimal("0.15")]
    }

    # Run optimization across multiple stocks
    csv_file = "optimization_set.csv"

    results = optimize_strategy_multi_stock(
        csv_file=csv_file,
        param_grid=param_grid,
        period="2y",
        initial_cash=10_000
    )

    if results is not None and not results.empty:
        # Print top 5 by average return
        print_optimization_results(results, metric='avg_return_pct', top_n=5)

        # Also show top 5 by average Sharpe ratio
        print(f"\n{'='*70}")
        print("BONUS: Top 5 by Average Sharpe Ratio")
        print(f"{'='*70}\n")
        print_optimization_results(results, metric='avg_sharpe', top_n=5)

        # Save all results
        save_results(results, filename='multi_stock_optimization_results.csv')