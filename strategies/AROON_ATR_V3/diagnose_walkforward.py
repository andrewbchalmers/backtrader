"""
Diagnostic Script: Walk-Forward vs Regular Backtest Comparison

This script runs the SAME parameters on the SAME data using both methods
to identify what's different.
"""

from decimal import Decimal
import backtrader as bt
import yfinance as yf
from aroon_atr import AroonMultiFilterStrategy
import pandas as pd
from datetime import datetime, timedelta

# Test parameters (from your CSV - Period 1)
TEST_PARAMS = {
    'aroon_len': 18,
    'atr_filter_len': 10,
    'atr_filter_mult': Decimal('2.0'),
    'min_trend_strength': Decimal('30'),
    'max_both_middle': True,
    'stability_bars': 3,
    'use_adx_filter': True,
    'adx_length': 14,
    'adx_threshold': Decimal('25.0'),
    'atr_stop_len': 5,
    'atr_stop_mult': Decimal('3.0'),
    'stop_loss_pct': Decimal('0.10'),
    'take_profit_pct': Decimal('0.13'),
    'enable_peak_exit': True,
    'peak_atr_mult': Decimal('2.0'),
    'peak_atr_period': 14,
    'min_profit_pct_to_activate': Decimal('0.03'),
    'position_size_pct': Decimal('1.0'),
    'verbose': True  # Enable to see trades
}

def run_backtest(symbol, start_date, end_date, params, initial_cash=10000):
    """Run a backtest on specific date range."""

    print(f"\n{'='*70}")
    print(f"Testing {symbol}: {start_date} to {end_date}")
    print(f"{'='*70}")

    # Download data
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)

    if df.empty:
        print(f"❌ No data for {symbol}")
        return None

    df.index = df.index.tz_localize(None)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = ['open', 'high', 'low', 'close', 'volume']

    print(f"\n📊 Data Info:")
    print(f"   Bars: {len(df)}")
    print(f"   Start: {df.index[0].date()}")
    print(f"   End: {df.index[-1].date()}")

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
    cerebro.addstrategy(AroonMultiFilterStrategy, **params)

    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.0005)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    # Run
    results = cerebro.run()
    strat = results[0]

    # Extract results
    sharpe = strat.analyzers.sharpe.get_analysis()
    dd = strat.analyzers.dd.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    final_value = cerebro.broker.getvalue()
    return_pct = (final_value / initial_cash - 1) * 100

    total_trades = trades.get('total', {}).get('total', 0)

    print(f"\n📈 Results:")
    print(f"   Final Value: ${final_value:,.2f}")
    print(f"   Return: {return_pct:.2f}%")
    print(f"   Sharpe: {sharpe.get('sharperatio', 'N/A')}")
    print(f"   Max DD: {dd.get('max', {}).get('drawdown', 0):.2f}%")
    print(f"   Total Trades: {total_trades}")

    if total_trades > 0:
        wins = trades.get('won', {}).get('total', 0)
        losses = trades.get('lost', {}).get('total', 0)
        print(f"   Wins: {wins}, Losses: {losses}")

    return {
        'return_pct': return_pct,
        'sharpe': sharpe.get('sharperatio', None),
        'max_dd': dd.get('max', {}).get('drawdown', 0),
        'total_trades': total_trades,
        'final_value': final_value
    }


def main():
    print("\n" + "="*70)
    print("DIAGNOSTIC: Walk-Forward vs Regular Backtest")
    print("="*70)

    # Test on a single stock from your set
    test_symbol = "AAPL"  # Change to any stock in your optimization set

    # Period 1 from your walk-forward results
    train_start = "2023-01-25"
    train_end = "2024-01-20"
    test_start = "2024-01-21"
    test_end = "2024-04-20"

    print("\n🔬 Testing Training Period")
    train_result = run_backtest(test_symbol, train_start, train_end, TEST_PARAMS)

    print("\n\n🔬 Testing Test Period (Out-of-Sample)")
    test_result = run_backtest(test_symbol, test_start, test_end, TEST_PARAMS)

    # Now test full year for comparison
    print("\n\n🔬 Testing Full Year (For Comparison)")
    full_year_start = "2023-01-01"
    full_year_end = "2023-12-31"
    full_result = run_backtest(test_symbol, full_year_start, full_year_end, TEST_PARAMS)

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    if train_result and test_result and full_result:
        print(f"\n📊 Trade Comparison:")
        print(f"   Train period (12mo):  {train_result['total_trades']} trades")
        print(f"   Test period (3mo):    {test_result['total_trades']} trades")
        print(f"   Full year:            {full_result['total_trades']} trades")

        print(f"\n📊 Return Comparison:")
        print(f"   Train period:  {train_result['return_pct']:7.2f}%")
        print(f"   Test period:   {test_result['return_pct']:7.2f}%")
        print(f"   Full year:     {full_result['return_pct']:7.2f}%")

        print(f"\n📊 Sharpe Comparison:")
        print(f"   Train period:  {train_result['sharpe']}")
        print(f"   Test period:   {test_result['sharpe']}")
        print(f"   Full year:     {full_result['sharpe']}")

        # Check for issues
        print(f"\n🔍 Diagnostic Findings:")

        if train_result['total_trades'] == 0:
            print("   ⚠️  PROBLEM: No trades in training period!")
            print("       Strategy is too restrictive for this data.")

        if test_result['total_trades'] == 0:
            print("   ⚠️  PROBLEM: No trades in test period!")
            print("       This explains 0.0000 composite score.")

        if train_result['sharpe'] is None or train_result['sharpe'] <= 0:
            print("   ⚠️  PROBLEM: Sharpe ratio is None or negative!")
            print("       Composite score formula requires positive Sharpe.")
            print("       This is why all scores are 0.0000.")

        if full_result['total_trades'] > 10:
            print(f"   ✓ Full year has {full_result['total_trades']} trades")
            print("     Strategy DOES work over longer periods.")
            print("     Problem is likely: periods too short (3 months)")

    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("\nThe issue is likely one or more of:")
    print("1. 3-month test periods too short for this strategy")
    print("2. Sharpe ratio calculation fails on short periods")
    print("3. Composite score formula too strict (requires positive Sharpe)")
    print("\nSolutions:")
    print("- Use longer test periods (6 months instead of 3)")
    print("- Change composite score to use abs(return) instead of requiring positive Sharpe")
    print("- Or simply use avg_return as the ranking metric")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()