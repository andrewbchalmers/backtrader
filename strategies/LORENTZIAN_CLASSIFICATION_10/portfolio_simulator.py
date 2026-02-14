"""
Portfolio Simulator - Combines per-stock backtest results into a realistic portfolio.

Simulates a fixed capital pool with max concurrent positions, replaying trade
events chronologically to produce accurate equity curves and metrics.

No backtrader dependency - works with plain dates, floats, and dicts so it can
be reused by optimizer scripts.

Usage:
    from portfolio_simulator import simulate_portfolio, print_portfolio_summary, plot_portfolio

    stock_data = [
        {
            'symbol': 'AAPL',
            'dates': [date(2024,1,2), date(2024,1,3), ...],
            'values': [10000.0, 10050.0, ...],
            'initial_value': 10000.0,
            'trade_log': [
                {'entry_date': date(2024,1,5), 'exit_date': date(2024,1,15)},
                ...
            ]
        },
        ...
    ]

    results = simulate_portfolio(stock_data, initial_capital=100_000, max_positions=10)
    print_portfolio_summary(results)
    plot_portfolio(results, 'portfolio_simulation.png')
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import timedelta


def simulate_portfolio(stock_data, initial_capital=100_000, max_positions=10, spy_df=None,
                       trade_log_file=None):
    """
    Simulate equal-weight portfolio with fixed capital and max concurrent positions.

    Args:
        stock_data: list of dicts per stock (see module docstring for format)
        initial_capital: total portfolio capital
        max_positions: max concurrent positions allowed
        spy_df: optional SPY DataFrame for benchmark (needs 'Close' column, DatetimeIndex)
        trade_log_file: if set, write detailed trade log CSV to this path

    Returns:
        dict with portfolio metrics and equity curve data, or None if no data
    """
    if not stock_data:
        return None

    position_size = initial_capital / max_positions

    # Deduplicate stock_data by symbol (keep first occurrence)
    seen_symbols = set()
    deduped_stock_data = []
    for stock in stock_data:
        if stock['symbol'] not in seen_symbols:
            seen_symbols.add(stock['symbol'])
            deduped_stock_data.append(stock)
    if len(deduped_stock_data) < len(stock_data):
        print(f"  [DEBUG] Deduplicated stock_data: {len(stock_data)} → {len(deduped_stock_data)} "
              f"(removed {len(stock_data) - len(deduped_stock_data)} duplicates)")
    stock_data = deduped_stock_data

    # Build per-stock daily return lookup: {symbol: {date: daily_return}}
    stock_returns = {}
    for stock in stock_data:
        dates = stock['dates']
        values = stock['values']
        if len(dates) < 2 or len(values) < 2:
            continue
        returns = {}
        for i in range(1, len(dates)):
            if values[i - 1] > 0:
                returns[dates[i]] = values[i] / values[i - 1] - 1
            else:
                returns[dates[i]] = 0.0
        stock_returns[stock['symbol']] = returns

    # Build per-stock position day sets from trade_log
    # position_days[symbol] = set of dates where stock has an active position
    position_days = {}
    for stock in stock_data:
        symbol = stock['symbol']
        days = set()
        for trade in stock.get('trade_log', []):
            entry = trade['entry_date']
            exit_d = trade['exit_date']
            for d in stock['dates']:
                if entry <= d <= exit_d:
                    days.add(d)
        position_days[symbol] = days

    # Build entry/exit day lookups
    # entry_days[symbol] = set of dates where a trade starts
    # exit_days[symbol] = set of dates where a trade ends
    entry_days = {}
    exit_days = {}
    for stock in stock_data:
        symbol = stock['symbol']
        entries = set()
        exits = set()
        for trade in stock.get('trade_log', []):
            entries.add(trade['entry_date'])
            exits.add(trade['exit_date'])
        entry_days[symbol] = entries
        exit_days[symbol] = exits

    # Get all unique sorted dates across all stocks
    all_dates = sorted(set(d for stock in stock_data for d in stock['dates']))

    if not all_dates:
        return None

    # Simulate portfolio
    cash = initial_capital
    active_positions = {}  # symbol -> current_value
    equity_curve = []
    positions_held_series = []
    skipped_entries = 0
    total_entries = 0
    max_concurrent = 0

    prev_day_positions = set()  # symbols that were in position yesterday

    # Trade logging
    trade_log_entries = []  # list of dicts for completed trades
    active_trade_info = {}  # symbol -> {entry_date, entry_value, position_alloc}
    skipped_log = []  # list of dicts for skipped entries

    for date in all_dates:
        # Determine which stocks are entering/exiting TODAY
        todays_exits = []
        todays_entries = []

        for stock in stock_data:
            symbol = stock['symbol']
            was_in = symbol in prev_day_positions
            now_in = date in position_days.get(symbol, set())

            # Exit: was in position yesterday, not today (or this is the exit date)
            if was_in and (not now_in or date in exit_days.get(symbol, set())):
                if symbol in active_positions:
                    todays_exits.append(symbol)

            # Entry: not in position yesterday (or just exited), now entering
            if date in entry_days.get(symbol, set()):
                # Don't re-enter if we're already holding this stock
                # (unless it also exited today - handled by exit-first ordering)
                if symbol not in active_positions or symbol in todays_exits:
                    todays_entries.append(symbol)

        # Step 1: Process exits (free capital)
        for symbol in todays_exits:
            if symbol in active_positions:
                # Apply today's return before exiting (exit happens at close)
                daily_ret = stock_returns.get(symbol, {}).get(date, 0.0)
                active_positions[symbol] *= (1 + daily_ret)
                exit_value = active_positions[symbol]
                cash += exit_value
                del active_positions[symbol]

                # Log the completed trade
                if symbol in active_trade_info:
                    info = active_trade_info.pop(symbol)
                    pnl = exit_value - info['entry_alloc']
                    pnl_pct = (exit_value / info['entry_alloc'] - 1) * 100 if info['entry_alloc'] > 0 else 0
                    trade_log_entries.append({
                        'symbol': symbol,
                        'entry_date': info['entry_date'],
                        'exit_date': date,
                        'entry_alloc': info['entry_alloc'],
                        'exit_value': exit_value,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'entry_day_ret': info['entry_day_ret'],
                    })

        # Step 2: Process entries (allocate capital)
        seen_entries = set()
        for symbol in todays_entries:
            if symbol in seen_entries:
                continue  # prevent double-entry for same symbol on same day
            seen_entries.add(symbol)
            total_entries += 1
            if len(active_positions) < max_positions and cash >= position_size:
                active_positions[symbol] = position_size
                cash -= position_size
                # Apply today's return (entry happens at open, price moves by close)
                daily_ret = stock_returns.get(symbol, {}).get(date, 0.0)
                active_positions[symbol] *= (1 + daily_ret)

                # Track entry for logging
                active_trade_info[symbol] = {
                    'entry_date': date,
                    'entry_alloc': position_size,
                    'entry_day_ret': daily_ret,
                }
            else:
                skipped_entries += 1
                skipped_log.append({
                    'symbol': symbol,
                    'date': date,
                    'reason': 'no_slots' if len(active_positions) >= max_positions else 'insufficient_cash',
                    'active_positions': len(active_positions),
                    'cash': cash,
                })

        # Step 3: Mark-to-market positions that are continuing (not entered/exited today)
        for symbol in list(active_positions.keys()):
            if symbol not in todays_exits and symbol not in todays_entries:
                daily_ret = stock_returns.get(symbol, {}).get(date, 0.0)
                active_positions[symbol] *= (1 + daily_ret)

        # Record portfolio value
        total_value = cash + sum(active_positions.values())
        equity_curve.append(total_value)
        positions_held_series.append(len(active_positions))
        max_concurrent = max(max_concurrent, len(active_positions))

        # Update prev_day_positions for next iteration
        prev_day_positions = set()
        for stock in stock_data:
            symbol = stock['symbol']
            if date in position_days.get(symbol, set()):
                prev_day_positions.add(symbol)

    # Log any positions still open at end of simulation
    for symbol, value in active_positions.items():
        if symbol in active_trade_info:
            info = active_trade_info[symbol]
            pnl = value - info['entry_alloc']
            pnl_pct = (value / info['entry_alloc'] - 1) * 100 if info['entry_alloc'] > 0 else 0
            trade_log_entries.append({
                'symbol': symbol,
                'entry_date': info['entry_date'],
                'exit_date': all_dates[-1],
                'entry_alloc': info['entry_alloc'],
                'exit_value': value,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'entry_day_ret': info['entry_day_ret'],
                'still_open': True,
            })

    # Write trade log CSV if requested
    if trade_log_file and trade_log_entries:
        import csv as csv_mod
        with open(trade_log_file, 'w', newline='') as f:
            writer = csv_mod.DictWriter(f, fieldnames=[
                'symbol', 'entry_date', 'exit_date', 'entry_alloc', 'exit_value',
                'pnl', 'pnl_pct', 'entry_day_ret', 'still_open'])
            writer.writeheader()
            for t in sorted(trade_log_entries, key=lambda x: x['entry_date']):
                row = dict(t)
                row.setdefault('still_open', False)
                row['pnl'] = f"{row['pnl']:.2f}"
                row['pnl_pct'] = f"{row['pnl_pct']:.2f}"
                row['entry_alloc'] = f"{row['entry_alloc']:.2f}"
                row['exit_value'] = f"{row['exit_value']:.2f}"
                row['entry_day_ret'] = f"{row['entry_day_ret']:.6f}"
                writer.writerow(row)
        print(f"  [DEBUG] Trade log written to {trade_log_file} ({len(trade_log_entries)} trades, {len(skipped_log)} skipped)")

    equity = np.array(equity_curve)
    dates_array = pd.DatetimeIndex(all_dates)

    # Compute metrics
    daily_returns = np.diff(equity) / equity[:-1]

    # Total return
    final_value = equity[-1]
    total_return_pct = (final_value / initial_capital - 1) * 100

    # Annualized return
    n_days = len(equity)
    years = n_days / 252
    if years > 0 and final_value > 0 and initial_capital > 0:
        annualized_return = ((final_value / initial_capital) ** (1 / years) - 1) * 100
    else:
        annualized_return = 0

    # Sharpe ratio
    if len(daily_returns) > 1:
        avg_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns, ddof=1)
        sharpe = (avg_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
    else:
        sharpe = 0

    # Sortino ratio
    if len(daily_returns) > 1:
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0:
            downside_std = np.std(downside, ddof=1)
            sortino = (np.mean(daily_returns) / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        else:
            sortino = float('inf') if np.mean(daily_returns) > 0 else 0
    else:
        sortino = 0

    # Max drawdown
    peak = equity[0]
    max_dd_pct = 0
    max_dd_money = 0
    drawdown_series = []
    for val in equity:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100 if peak > 0 else 0
        drawdown_series.append(dd)
        if dd > max_dd_pct:
            max_dd_pct = dd
            max_dd_money = peak - val

    # Calmar ratio
    calmar = annualized_return / max_dd_pct if max_dd_pct > 0 else 0

    # Annualized volatility
    if len(daily_returns) > 1:
        volatility = np.std(daily_returns, ddof=1) * np.sqrt(252) * 100
    else:
        volatility = 0

    # Position utilization
    avg_positions = np.mean(positions_held_series) if positions_held_series else 0
    pct_time_fully_invested = (
        sum(1 for p in positions_held_series if p >= max_positions) / len(positions_held_series) * 100
        if positions_held_series else 0
    )
    pct_time_any_position = (
        sum(1 for p in positions_held_series if p > 0) / len(positions_held_series) * 100
        if positions_held_series else 0
    )

    # SPY benchmark
    spy_return_pct = None
    spy_equity = None
    if spy_df is not None and len(spy_df) > 0:
        spy_close = spy_df['Close']
        spy_initial = spy_close.iloc[0]
        spy_final = spy_close.iloc[-1]
        if hasattr(spy_initial, 'item'):
            spy_initial = spy_initial.item()
        if hasattr(spy_final, 'item'):
            spy_final = spy_final.item()
        spy_return_pct = (spy_final / spy_initial - 1) * 100
        spy_shares = initial_capital / spy_initial

        # Build SPY equity curve aligned to portfolio dates
        spy_equity = []
        for date in all_dates:
            try:
                matching = spy_df.loc[spy_df.index.date == date, 'Close']
                if len(matching) > 0:
                    price = matching.iloc[0]
                    if hasattr(price, 'item'):
                        price = price.item()
                else:
                    valid = spy_df.index[spy_df.index.date <= date]
                    if len(valid) > 0:
                        price = spy_df.loc[valid[-1], 'Close']
                        if hasattr(price, 'item'):
                            price = price.item()
                    else:
                        price = spy_initial
                spy_equity.append(spy_shares * price)
            except Exception:
                spy_equity.append(spy_equity[-1] if spy_equity else initial_capital)

    # Stock-level stats
    n_stocks = len(stock_data)
    profitable_stocks = sum(
        1 for s in stock_data
        if len(s['values']) >= 2 and s['values'][-1] > s['initial_value']
    )
    traded_stocks = sum(1 for s in stock_data if len(s.get('trade_log', [])) > 0)

    return {
        'initial_capital': initial_capital,
        'max_positions': max_positions,
        'position_size': position_size,
        'final_value': final_value,
        'total_return_pct': total_return_pct,
        'annualized_return': annualized_return,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'max_drawdown_pct': max_dd_pct,
        'max_drawdown_money': max_dd_money,
        'volatility': volatility,
        'n_stocks': n_stocks,
        'traded_stocks': traded_stocks,
        'profitable_stocks': profitable_stocks,
        'spy_return_pct': spy_return_pct,
        'outperformance': total_return_pct - spy_return_pct if spy_return_pct is not None else None,
        # Position utilization
        'avg_positions_held': avg_positions,
        'max_concurrent_positions': max_concurrent,
        'pct_time_fully_invested': pct_time_fully_invested,
        'pct_time_any_position': pct_time_any_position,
        'total_entries': total_entries,
        'skipped_entries': skipped_entries,
        # Equity curve data for plotting
        'dates': dates_array,
        'equity': equity,
        'drawdown_series': drawdown_series,
        'spy_equity': spy_equity,
        'positions_held_series': positions_held_series,
        # Trade log for debugging
        '_trade_log': trade_log_entries,
        '_skipped_log': skipped_log,
    }


def print_portfolio_summary(results):
    """Print formatted portfolio simulation results."""
    if results is None:
        print("  No portfolio simulation results available.")
        return

    print(f"\n{'='*70}")
    print("SIMULATED PORTFOLIO PERFORMANCE")
    print(f"{'='*70}")

    print(f"\n  Capital Allocation:")
    print(f"    Total Capital:        ${results['initial_capital']:,.0f}")
    print(f"    Max Positions:        {results['max_positions']}")
    print(f"    Position Size:        ${results['position_size']:,.0f}")
    print(f"    Stocks in Universe:   {results['n_stocks']}")
    print(f"    Stocks Traded:        {results['traded_stocks']}")
    print(f"    Profitable Stocks:    {results['profitable_stocks']} "
          f"({results['profitable_stocks']/results['n_stocks']*100:.0f}%)")

    print(f"\n  Portfolio Returns:")
    print(f"    Final Value:          ${results['final_value']:,.2f}")
    print(f"    Total Return:         {results['total_return_pct']:+.2f}%")
    print(f"    Annualized Return:    {results['annualized_return']:+.2f}%")
    if results['spy_return_pct'] is not None:
        print(f"    SPY Return:           {results['spy_return_pct']:+.2f}%")
        print(f"    Outperformance:       {results['outperformance']:+.2f}%")

    print(f"\n  Risk Metrics:")
    print(f"    Sharpe Ratio:         {results['sharpe']:.3f}")
    print(f"    Sortino Ratio:        {results['sortino']:.3f}")
    print(f"    Calmar Ratio:         {results['calmar']:.3f}")
    print(f"    Max Drawdown:         {results['max_drawdown_pct']:.2f}%")
    print(f"    Max Drawdown ($):     ${results['max_drawdown_money']:,.2f}")
    print(f"    Annualized Vol:       {results['volatility']:.2f}%")

    print(f"\n  Position Utilization:")
    print(f"    Avg Positions Held:   {results['avg_positions_held']:.1f} / {results['max_positions']}")
    print(f"    Max Concurrent:       {results['max_concurrent_positions']}")
    print(f"    Time Invested:        {results['pct_time_any_position']:.1f}% of days")
    print(f"    Time Fully Invested:  {results['pct_time_fully_invested']:.1f}% of days")
    print(f"    Trade Entries:        {results['total_entries']}")
    if results['skipped_entries'] > 0:
        print(f"    Skipped (no slots):   {results['skipped_entries']} "
              f"({results['skipped_entries']/results['total_entries']*100:.1f}%)")
    else:
        print(f"    Skipped (no slots):   0")

    print(f"\n{'='*70}")


def print_trade_log_summary(results, max_trades=50):
    """Print a debug summary of trades in the portfolio simulation."""
    if results is None:
        return

    trade_log = results.get('_trade_log', [])
    skipped_log = results.get('_skipped_log', [])

    if not trade_log:
        print("  [DEBUG] No trades in portfolio simulation trade log.")
        return

    trades = sorted(trade_log, key=lambda x: x['entry_date'])
    winners = [t for t in trades if t['pnl'] > 0]
    losers = [t for t in trades if t['pnl'] <= 0]
    total_pnl = sum(t['pnl'] for t in trades)

    print(f"\n  {'='*80}")
    print(f"  PORTFOLIO TRADE LOG DEBUG")
    print(f"  {'='*80}")
    print(f"  Total Trades Executed: {len(trades)}")
    print(f"  Winners: {len(winners)} ({len(winners)/len(trades)*100:.1f}%)")
    print(f"  Losers: {len(losers)} ({len(losers)/len(trades)*100:.1f}%)")
    print(f"  Total P&L: ${total_pnl:,.2f}")
    print(f"  Avg P&L/Trade: ${total_pnl/len(trades):,.2f}")
    if winners:
        print(f"  Avg Win: ${sum(t['pnl'] for t in winners)/len(winners):,.2f} "
              f"({sum(t['pnl_pct'] for t in winners)/len(winners):.2f}%)")
    if losers:
        print(f"  Avg Loss: ${sum(t['pnl'] for t in losers)/len(losers):,.2f} "
              f"({sum(t['pnl_pct'] for t in losers)/len(losers):.2f}%)")
    print(f"  Skipped Entries: {len(skipped_log)}")

    # Show largest winners and losers
    by_pnl = sorted(trades, key=lambda x: x['pnl'])
    print(f"\n  Top 10 Losers:")
    print(f"  {'Symbol':<8} {'Entry':<12} {'Exit':<12} {'Alloc':>10} {'Exit Val':>10} {'P&L':>10} {'P&L%':>8} {'EntryRet':>10}")
    for t in by_pnl[:10]:
        still = ' *' if t.get('still_open') else ''
        print(f"  {t['symbol']:<8} {str(t['entry_date']):<12} {str(t['exit_date']):<12} "
              f"${t['entry_alloc']:>9,.2f} ${t['exit_value']:>9,.2f} ${t['pnl']:>9,.2f} "
              f"{t['pnl_pct']:>7.2f}% {t['entry_day_ret']:>9.4f}{still}")

    print(f"\n  Top 10 Winners:")
    for t in by_pnl[-10:]:
        still = ' *' if t.get('still_open') else ''
        print(f"  {t['symbol']:<8} {str(t['entry_date']):<12} {str(t['exit_date']):<12} "
              f"${t['entry_alloc']:>9,.2f} ${t['exit_value']:>9,.2f} ${t['pnl']:>9,.2f} "
              f"{t['pnl_pct']:>7.2f}% {t['entry_day_ret']:>9.4f}{still}")

    # Show first N trades chronologically
    show = min(max_trades, len(trades))
    print(f"\n  First {show} trades (chronological):")
    print(f"  {'#':<4} {'Symbol':<8} {'Entry':<12} {'Exit':<12} {'Alloc':>10} {'Exit Val':>10} {'P&L':>10} {'P&L%':>8} {'EntryRet':>10}")
    print(f"  {'-'*96}")
    running_pnl = 0
    for i, t in enumerate(trades[:show]):
        running_pnl += t['pnl']
        still = ' *' if t.get('still_open') else ''
        print(f"  {i+1:<4} {t['symbol']:<8} {str(t['entry_date']):<12} {str(t['exit_date']):<12} "
              f"${t['entry_alloc']:>9,.2f} ${t['exit_value']:>9,.2f} ${t['pnl']:>9,.2f} "
              f"{t['pnl_pct']:>7.2f}% {t['entry_day_ret']:>9.4f}{still}")
    print(f"  Running P&L after {show} trades: ${running_pnl:,.2f}")

    # Per-symbol summary
    from collections import defaultdict
    symbol_stats = defaultdict(lambda: {'trades': 0, 'pnl': 0, 'wins': 0})
    for t in trades:
        s = symbol_stats[t['symbol']]
        s['trades'] += 1
        s['pnl'] += t['pnl']
        if t['pnl'] > 0:
            s['wins'] += 1

    print(f"\n  Per-Symbol Summary (top 10 losers by total P&L):")
    print(f"  {'Symbol':<8} {'Trades':>7} {'Wins':>6} {'WR%':>6} {'Total P&L':>12}")
    worst = sorted(symbol_stats.items(), key=lambda x: x[1]['pnl'])
    for sym, s in worst[:10]:
        wr = s['wins'] / s['trades'] * 100 if s['trades'] > 0 else 0
        print(f"  {sym:<8} {s['trades']:>7} {s['wins']:>6} {wr:>5.1f}% ${s['pnl']:>11,.2f}")

    # Show skipped entry distribution
    if skipped_log:
        from collections import Counter
        skip_symbols = Counter(s['symbol'] for s in skipped_log)
        print(f"\n  Most frequently skipped symbols (top 10):")
        for sym, cnt in skip_symbols.most_common(10):
            print(f"    {sym}: {cnt} entries skipped")

    # Reconciliation: compare per-stock backtest returns vs portfolio-realized returns
    print(f"\n  {'='*80}")
    print(f"  RECONCILIATION: Expected vs Realized Returns")
    print(f"  {'='*80}")
    print(f"  Total portfolio P&L from trades: ${total_pnl:,.2f}")
    print(f"  Position size: ${results['position_size']:,.2f}")
    print(f"  Expected portfolio return from trades: {total_pnl / results['initial_capital'] * 100:+.2f}%")
    print(f"  Actual portfolio return: {results['total_return_pct']:+.2f}%")
    print(f"  Difference: {results['total_return_pct'] - total_pnl / results['initial_capital'] * 100:+.2f}%")
    print(f"  {'='*80}\n")


def plot_portfolio(results, filename='portfolio_simulation.png'):
    """Generate portfolio equity curve chart with 3 panels."""
    if results is None:
        return

    dates = results['dates']
    equity = results['equity']

    fig, axes = plt.subplots(3, 1, figsize=(16, 14),
                             gridspec_kw={'height_ratios': [3, 1.5, 1]})
    fig.patch.set_facecolor('#1a1a1a')

    # Panel 1: Equity curves
    ax1 = axes[0]
    ax1.set_facecolor('#2a2a2a')
    ax1.plot(dates, equity, label='Lorentzian Portfolio', linewidth=2, color='#00ff88')
    if results['spy_equity'] is not None:
        ax1.plot(dates, results['spy_equity'], label='SPY Buy & Hold',
                 linewidth=2, color='#ff6b6b', linestyle='--')
    ax1.axhline(y=results['initial_capital'], color='gray', linestyle=':', alpha=0.5)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12, color='white')
    ax1.set_title(
        f'Simulated Portfolio (${results["initial_capital"]:,.0f}, '
        f'max {results["max_positions"]} positions, {results["n_stocks"]} stocks)',
        fontsize=14, fontweight='bold', color='white'
    )
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(colors='white')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Annotate final values
    ax1.annotate(f'${equity[-1]:,.0f} ({results["total_return_pct"]:+.1f}%)',
                 xy=(dates[-1], equity[-1]),
                 xytext=(10, 5), textcoords='offset points',
                 fontsize=10, color='#00ff88', fontweight='bold')
    if results['spy_equity'] is not None:
        ax1.annotate(f'${results["spy_equity"][-1]:,.0f} ({results["spy_return_pct"]:+.1f}%)',
                     xy=(dates[-1], results['spy_equity'][-1]),
                     xytext=(10, -15), textcoords='offset points',
                     fontsize=10, color='#ff6b6b', fontweight='bold')

    # Panel 2: Cumulative returns %
    ax2 = axes[1]
    ax2.set_facecolor('#2a2a2a')
    strategy_rets = [(v / results['initial_capital'] - 1) * 100 for v in equity]
    ax2.plot(dates, strategy_rets, linewidth=2, color='#00ff88', label='Portfolio')
    ax2.fill_between(dates, strategy_rets, 0, alpha=0.2,
                     where=[r >= 0 for r in strategy_rets], color='#00ff88')
    ax2.fill_between(dates, strategy_rets, 0, alpha=0.2,
                     where=[r < 0 for r in strategy_rets], color='#ff6b6b')
    if results['spy_equity'] is not None:
        spy_rets = [(v / results['initial_capital'] - 1) * 100 for v in results['spy_equity']]
        ax2.plot(dates, spy_rets, linewidth=2, color='#ff6b6b', linestyle='--',
                 alpha=0.7, label='SPY')
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_ylabel('Cumulative Return (%)', fontsize=12, color='white')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(colors='white')

    # Panel 3: Drawdown
    ax3 = axes[2]
    ax3.set_facecolor('#2a2a2a')
    dd_series = [-d for d in results['drawdown_series']]
    ax3.fill_between(dates, dd_series, 0, color='#ff6b6b', alpha=0.5)
    ax3.plot(dates, dd_series, color='#ff6b6b', linewidth=1, alpha=0.8)
    ax3.set_ylabel('Drawdown (%)', fontsize=12, color='white')
    ax3.set_xlabel('Date', fontsize=12, color='white')
    ax3.grid(True, alpha=0.2)
    ax3.tick_params(colors='white')

    # Summary stats text
    stats_text = (
        f"Return: {results['total_return_pct']:+.2f}%  |  "
        f"Sharpe: {results['sharpe']:.2f}  |  "
        f"Sortino: {results['sortino']:.2f}  |  "
        f"Max DD: {results['max_drawdown_pct']:.1f}%  |  "
        f"Vol: {results['volatility']:.1f}%  |  "
        f"Avg Pos: {results['avg_positions_held']:.1f}/{results['max_positions']}"
    )
    fig.text(0.5, 0.01, stats_text, ha='center', fontsize=11, color='white',
             style='italic', alpha=0.8)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()
    print(f"  Saved portfolio simulation chart to {filename}")
