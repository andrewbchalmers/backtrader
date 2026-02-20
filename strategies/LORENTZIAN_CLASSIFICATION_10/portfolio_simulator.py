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


def _compute_candidate_score(symbol, ranking_stats):
    """Compute static candidate score from in-sample training stats."""
    stats = ranking_stats.get(symbol)
    if not stats:
        return 0.0
    w = stats['win_rate'] / 100.0
    r = stats['rr_ratio']
    kelly_edge = (w - (1 - w) / r) if r > 0 else 0.0
    # Normalize expectancy to [0,1] range using tanh-like scaling
    exp_norm = stats['expectancy'] / (abs(stats['expectancy']) + 100) if stats['expectancy'] != 0 else 0
    # Profit factor contribution (log scale, capped)
    pf = stats['profit_factor']
    pf_norm = min(pf, 5.0) / 5.0 if pf > 0 else 0.0
    # Weighted composite: kelly edge is most important
    return kelly_edge * 0.5 + exp_norm * 0.3 + pf_norm * 0.2


def _compute_held_score(symbol, ranking_stats, entry_alloc, current_value, entry_date, current_date):
    """Compute dynamic score for a held position (static base + P&L trajectory)."""
    base = _compute_candidate_score(symbol, ranking_stats)
    # Current P&L as percentage
    pnl_pct = (current_value / entry_alloc - 1) if entry_alloc > 0 else 0.0
    days_held = (current_date - entry_date).days

    if pnl_pct > 0:
        # Winners get a bonus — don't replace what's working
        # The longer a winner runs, the more protected it is
        winner_bonus = pnl_pct * 1.0 + min(days_held * 0.001, 0.05)
        return base + winner_bonus
    else:
        # Losers get penalized more as time passes — stale losers are prime replacement targets
        hold_penalty = max(0, (days_held - 15) * 0.003)
        return base + pnl_pct * 0.8 - hold_penalty


def simulate_portfolio(stock_data, initial_capital=100_000, max_positions=10, spy_df=None,
                       debug=False, trade_log_file=None, kelly_fraction=0.0, kelly_stats=None,
                       ranking_stats=None, use_position_replacement=False,
                       replacement_threshold=0.1, replacement_cooldown_days=5,
                       replacement_max_per_day=3,
                       prediction_validation_bars=0, prediction_validation_threshold=-0.5,
                       entry_mode='market', entry_limit_window=3,
                       entry_dip_threshold=-0.5, entry_confirmation_threshold=0.5):
    """
    Simulate equal-weight portfolio with fixed capital and max concurrent positions.
    Optionally uses fractional Kelly criterion for position sizing and position replacement.

    Args:
        stock_data: list of dicts per stock (see module docstring for format)
        initial_capital: total portfolio capital
        max_positions: max concurrent positions allowed
        spy_df: optional SPY DataFrame for benchmark (needs 'Close' column, DatetimeIndex)
        debug: if True, enable trade logging and write CSV / print summary
        trade_log_file: if debug=True and set, write detailed trade log CSV to this path
        kelly_fraction: fraction of Kelly optimal size to use (0.0=equal weight, 0.33=1/3 Kelly)
        kelly_stats: dict of {symbol: {'win_rate': float, 'rr_ratio': float}} from in-sample data
        ranking_stats: dict of {symbol: {'win_rate', 'rr_ratio', 'expectancy', 'profit_factor'}}
        use_position_replacement: if True, replace worst positions when better candidates appear
        replacement_threshold: min score advantage to trigger replacement
        replacement_cooldown_days: min days before a position can be replaced
        replacement_max_per_day: max replacements per calendar day
        prediction_validation_bars: bars to wait before checking if ML prediction was correct (0=off)
        prediction_validation_threshold: force-exit if P&L% is below this after validation bars
        entry_mode: 'market' (immediate), 'dip' (wait for pullback), 'confirmation' (wait for
                    up move), or 'dip_or_confirmation' (fill on either condition)
        entry_limit_window: max bars to wait for limit fill before cancelling (0 or 'market'=immediate)
        entry_dip_threshold: daily return % to qualify as a dip fill (e.g., -0.5)
        entry_confirmation_threshold: daily return % to qualify as confirmation fill (e.g., 0.5)

    Returns:
        dict with portfolio metrics and equity curve data, or None if no data
    """
    if not stock_data:
        return None

    equal_weight_size = initial_capital / max_positions

    # Build per-stock position sizes
    use_kelly = kelly_fraction > 0 and kelly_stats
    stock_position_sizes = {}
    min_kelly_size = equal_weight_size * 0.5  # Floor: skip stocks sized below 50% of equal-weight
    if use_kelly:
        _below_floor = 0
        for stock in stock_data:
            symbol = stock['symbol']
            stats = kelly_stats.get(symbol)
            if stats and stats['rr_ratio'] > 0:
                w = stats['win_rate'] / 100.0  # convert from percentage
                r = stats['rr_ratio']
                kelly_pct = w - (1 - w) / r
                if kelly_pct > 0:
                    size = kelly_fraction * kelly_pct * initial_capital
                    # Cap at equal-weight slot size
                    size = min(size, equal_weight_size)
                    # Floor: skip positions too small to matter
                    if size < min_kelly_size:
                        stock_position_sizes[symbol] = 0.0
                        _below_floor += 1
                    else:
                        stock_position_sizes[symbol] = size
                else:
                    stock_position_sizes[symbol] = 0.0  # negative Kelly = don't trade
            else:
                stock_position_sizes[symbol] = 0.0
        if debug:
            _sized = {s: sz for s, sz in stock_position_sizes.items() if sz > 0}
            _skipped = {s: sz for s, sz in stock_position_sizes.items() if sz <= 0}
            print(f"  [KELLY] Fraction: {kelly_fraction:.0%} of Kelly optimal")
            print(f"  [KELLY] Equal-weight slot: ${equal_weight_size:,.0f}, min floor: ${min_kelly_size:,.0f}")
            print(f"  [KELLY] Sized {len(_sized)} stocks, skipped {len(_skipped)} "
                  f"({_below_floor} below floor, {len(_skipped) - _below_floor} negative/zero)")
            if _sized:
                _avg_sz = np.mean(list(_sized.values()))
                _min_sz = min(_sized.values())
                _max_sz = max(_sized.values())
                print(f"  [KELLY] Position sizes: avg=${_avg_sz:,.0f}, min=${_min_sz:,.0f}, max=${_max_sz:,.0f}")

    position_size = equal_weight_size  # default for non-Kelly stocks

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

    # Always track entry metadata (needed for replacement scoring and debug logging)
    active_trade_info = {}  # symbol -> {entry_date, entry_alloc, entry_day_ret}

    # Trade logging (only when debug=True)
    if debug:
        trade_log_entries = []
        skipped_log = []

    # Position replacement tracking
    do_replacement = use_position_replacement and ranking_stats
    total_replacements = 0
    replacement_score_improvements = []

    # Prediction validation tracking
    do_prediction_validation = prediction_validation_bars > 0
    prediction_validation_exits = 0

    # Limit order tracking
    use_limit_orders = entry_mode != 'market' and entry_limit_window > 0
    pending_orders = {}  # symbol -> {'signal_date': date, 'bars_remaining': N, 'size': sym_size}
    limit_orders_filled = 0
    limit_orders_expired = 0

    for date in all_dates:
        # Step 0: Prediction validation — force-exit positions where ML was wrong
        if do_prediction_validation:
            for symbol in list(active_positions.keys()):
                info = active_trade_info.get(symbol)
                if not info:
                    continue
                days_held = (date - info['entry_date']).days
                if days_held == prediction_validation_bars:
                    pnl_pct = (active_positions[symbol] / info['entry_alloc'] - 1) * 100
                    if pnl_pct < prediction_validation_threshold:
                        # ML prediction was wrong — force exit
                        daily_ret = stock_returns.get(symbol, {}).get(date, 0.0)
                        active_positions[symbol] *= (1 + daily_ret)
                        exit_value = active_positions[symbol]
                        cash += exit_value
                        del active_positions[symbol]
                        evicted_info = active_trade_info.pop(symbol, None)
                        prediction_validation_exits += 1

                        if debug and evicted_info:
                            pnl = exit_value - evicted_info['entry_alloc']
                            pnl_pct_actual = (exit_value / evicted_info['entry_alloc'] - 1) * 100
                            trade_log_entries.append({
                                'symbol': symbol,
                                'entry_date': evicted_info['entry_date'],
                                'exit_date': date,
                                'entry_alloc': evicted_info['entry_alloc'],
                                'exit_value': exit_value,
                                'pnl': pnl,
                                'pnl_pct': pnl_pct_actual,
                                'entry_day_ret': evicted_info['entry_day_ret'],
                                'exit_reason': 'prediction_validation',
                            })

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

                info = active_trade_info.pop(symbol, None)
                if debug and info:
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
        # With limit orders: new signals become pending, filled pending become entries
        if use_limit_orders:
            # Add new signals to pending orders
            for symbol in todays_entries:
                if symbol in pending_orders or symbol in active_positions:
                    continue
                # Determine position size upfront
                if use_kelly:
                    sym_size = stock_position_sizes.get(symbol, 0.0)
                    if sym_size <= 0:
                        total_entries += 1
                        skipped_entries += 1
                        continue
                else:
                    sym_size = position_size
                pending_orders[symbol] = {
                    'signal_date': date,
                    'bars_remaining': entry_limit_window,
                    'size': sym_size,
                }

            # Check pending orders for fill conditions
            filled_symbols = []
            expired_symbols = []
            for sym, order in list(pending_orders.items()):
                # Don't fill on signal day — wait for next bar
                if order['signal_date'] == date:
                    continue
                # Cancel if stock already exited the trade window in the underlying strategy
                if sym not in position_days or date not in position_days.get(sym, set()):
                    expired_symbols.append(sym)
                    continue
                daily_ret_pct = stock_returns.get(sym, {}).get(date, 0.0) * 100
                filled = False
                if entry_mode == 'dip':
                    filled = daily_ret_pct <= entry_dip_threshold
                elif entry_mode == 'confirmation':
                    filled = daily_ret_pct >= entry_confirmation_threshold
                elif entry_mode == 'dip_or_confirmation':
                    filled = (daily_ret_pct <= entry_dip_threshold or
                              daily_ret_pct >= entry_confirmation_threshold)
                if filled:
                    filled_symbols.append(sym)
                else:
                    order['bars_remaining'] -= 1
                    if order['bars_remaining'] <= 0:
                        expired_symbols.append(sym)

            for sym in expired_symbols:
                pending_orders.pop(sym, None)
                limit_orders_expired += 1

            # Replace todays_entries with filled limit orders, preserve sizes
            filled_order_sizes = {}
            todays_entries = filled_symbols
            for sym in filled_symbols:
                filled_order_sizes[sym] = pending_orders.pop(sym)['size']
                limit_orders_filled += 1

        seen_entries = set()
        replacements_today = 0
        for symbol in todays_entries:
            if symbol in seen_entries:
                continue  # prevent double-entry for same symbol on same day
            seen_entries.add(symbol)
            total_entries += 1

            # Determine position size (Kelly or equal-weight)
            if use_limit_orders:
                sym_size = filled_order_sizes.get(symbol, position_size)
            elif use_kelly:
                sym_size = stock_position_sizes.get(symbol, 0.0)
                if sym_size <= 0:
                    skipped_entries += 1
                    if debug:
                        skipped_log.append({
                            'symbol': symbol, 'date': date,
                            'reason': 'negative_kelly',
                            'active_positions': len(active_positions), 'cash': cash,
                        })
                    continue
            else:
                sym_size = position_size

            if len(active_positions) < max_positions and cash >= sym_size:
                # Slot available — enter normally
                active_positions[symbol] = sym_size
                cash -= sym_size
                # For limit order fills, don't apply fill-day return — the fill price
                # already reflects the dip/confirmation move. Applying it again double-counts.
                if use_limit_orders:
                    daily_ret = 0.0
                else:
                    daily_ret = stock_returns.get(symbol, {}).get(date, 0.0)
                    active_positions[symbol] *= (1 + daily_ret)

                active_trade_info[symbol] = {
                    'entry_date': date,
                    'entry_alloc': sym_size,
                    'entry_day_ret': daily_ret,
                }
            elif do_replacement and replacements_today < replacement_max_per_day:
                # Portfolio full — attempt position replacement
                candidate_score = _compute_candidate_score(symbol, ranking_stats)

                # Score all held positions eligible for replacement
                worst_sym = None
                worst_score = float('inf')
                for held_sym, held_value in active_positions.items():
                    held_info = active_trade_info.get(held_sym)
                    if not held_info:
                        continue
                    # Cooldown check
                    days_held = (date - held_info['entry_date']).days
                    if days_held < replacement_cooldown_days:
                        continue
                    held_score = _compute_held_score(
                        held_sym, ranking_stats, held_info['entry_alloc'],
                        held_value, held_info['entry_date'], date
                    )
                    if held_score < worst_score:
                        worst_score = held_score
                        worst_sym = held_sym

                # Replace if candidate is materially better
                if worst_sym and candidate_score > worst_score + replacement_threshold:
                    # Sell worst position (apply today's return first, same as normal exit)
                    daily_ret_exit = stock_returns.get(worst_sym, {}).get(date, 0.0)
                    active_positions[worst_sym] *= (1 + daily_ret_exit)
                    exit_value = active_positions[worst_sym]
                    cash += exit_value
                    del active_positions[worst_sym]
                    evicted_info = active_trade_info.pop(worst_sym, None)

                    if debug and evicted_info:
                        pnl = exit_value - evicted_info['entry_alloc']
                        pnl_pct = (exit_value / evicted_info['entry_alloc'] - 1) * 100 if evicted_info['entry_alloc'] > 0 else 0
                        trade_log_entries.append({
                            'symbol': worst_sym,
                            'entry_date': evicted_info['entry_date'],
                            'exit_date': date,
                            'entry_alloc': evicted_info['entry_alloc'],
                            'exit_value': exit_value,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'entry_day_ret': evicted_info['entry_day_ret'],
                            'replaced_by': symbol,
                        })

                    # Buy candidate
                    if cash >= sym_size:
                        active_positions[symbol] = sym_size
                        cash -= sym_size
                        if use_limit_orders:
                            daily_ret = 0.0
                        else:
                            daily_ret = stock_returns.get(symbol, {}).get(date, 0.0)
                            active_positions[symbol] *= (1 + daily_ret)

                        active_trade_info[symbol] = {
                            'entry_date': date,
                            'entry_alloc': sym_size,
                            'entry_day_ret': daily_ret,
                        }
                        replacements_today += 1
                        total_replacements += 1
                        replacement_score_improvements.append(candidate_score - worst_score)
                    else:
                        # Shouldn't happen normally, but handle gracefully
                        skipped_entries += 1
                else:
                    skipped_entries += 1
                    if debug:
                        skipped_log.append({
                            'symbol': symbol, 'date': date,
                            'reason': 'no_better_replacement' if worst_sym else 'all_in_cooldown',
                            'active_positions': len(active_positions), 'cash': cash,
                        })
            else:
                skipped_entries += 1
                if debug:
                    skipped_log.append({
                        'symbol': symbol, 'date': date,
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
    if debug:
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

    # Trade-level stats from underlying backtest trade logs
    all_trades_pnl = []
    for s in stock_data:
        for t in s.get('trade_log', []):
            pnl = t.get('pnlcomm', t.get('pnl', 0)) or 0
            all_trades_pnl.append(pnl)

    total_trades = len(all_trades_pnl)
    winners = [p for p in all_trades_pnl if p > 0]
    losers = [p for p in all_trades_pnl if p <= 0]
    n_winners = len(winners)
    n_losers = len(losers)
    win_rate = (n_winners / total_trades * 100) if total_trades > 0 else 0
    avg_win = (sum(winners) / n_winners) if n_winners > 0 else 0
    avg_loss = (sum(losers) / n_losers) if n_losers > 0 else 0
    gross_profit = sum(winners)
    gross_loss = abs(sum(losers))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
    expectancy = (sum(all_trades_pnl) / total_trades) if total_trades > 0 else 0
    largest_win = max(winners) if winners else 0
    largest_loss = min(losers) if losers else 0

    result = {
        'initial_capital': initial_capital,
        'max_positions': max_positions,
        'position_size': position_size,
        'kelly_fraction': kelly_fraction,
        'stock_position_sizes': stock_position_sizes if use_kelly else None,
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
        # Position replacement stats
        'total_replacements': total_replacements,
        'avg_replacement_score_improvement': (
            np.mean(replacement_score_improvements) if replacement_score_improvements else 0
        ),
        'use_position_replacement': do_replacement,
        # Prediction validation stats
        'prediction_validation_exits': prediction_validation_exits,
        'prediction_validation_bars': prediction_validation_bars,
        # Limit order stats
        'entry_mode': entry_mode,
        'limit_orders_filled': limit_orders_filled,
        'limit_orders_expired': limit_orders_expired,
        # Trade stats
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'n_winners': n_winners,
        'n_losers': n_losers,
        # Equity curve data for plotting
        'dates': dates_array,
        'equity': equity,
        'drawdown_series': drawdown_series,
        'spy_equity': spy_equity,
        'positions_held_series': positions_held_series,
    }

    if debug:
        result['_trade_log'] = trade_log_entries
        result['_skipped_log'] = skipped_log

    return result


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
    _kelly_frac = results.get('kelly_fraction', 0.0)
    _stock_sizes = results.get('stock_position_sizes')
    if _kelly_frac > 0 and _stock_sizes:
        _sized = [sz for sz in _stock_sizes.values() if sz > 0]
        _skipped_kelly = sum(1 for sz in _stock_sizes.values() if sz <= 0)
        if _sized:
            print(f"    Position Sizing:      {_kelly_frac:.0%} Kelly")
            print(f"    Avg Position Size:    ${np.mean(_sized):,.0f}")
            print(f"    Min Position Size:    ${min(_sized):,.0f}")
            print(f"    Max Position Size:    ${max(_sized):,.0f}")
            print(f"    Equal-Weight Cap:     ${results['position_size']:,.0f}")
            print(f"    Kelly-Skipped Stocks: {_skipped_kelly} (negative edge)")
        else:
            print(f"    Position Sizing:      {_kelly_frac:.0%} Kelly (all stocks skipped!)")
    else:
        print(f"    Position Size:        ${results['position_size']:,.0f} (equal weight)")
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
    if results.get('use_position_replacement'):
        _repl = results['total_replacements']
        _avg_imp = results['avg_replacement_score_improvement']
        print(f"    Replacements:         {_repl}" +
              (f" (avg score improvement: {_avg_imp:.3f})" if _repl > 0 else ""))
    _pv_exits = results.get('prediction_validation_exits', 0)
    _pv_bars = results.get('prediction_validation_bars', 0)
    if _pv_bars > 0:
        print(f"    Prediction Stops:     {_pv_exits} (ML wrong after {_pv_bars} bars)")
    _entry_mode = results.get('entry_mode', 'market')
    if _entry_mode != 'market':
        _lo_filled = results.get('limit_orders_filled', 0)
        _lo_expired = results.get('limit_orders_expired', 0)
        _lo_total = _lo_filled + _lo_expired
        _lo_fill_rate = (_lo_filled / _lo_total * 100) if _lo_total > 0 else 0
        print(f"    Entry Mode:           {_entry_mode}")
        print(f"    Limit Orders Filled:  {_lo_filled}/{_lo_total} ({_lo_fill_rate:.1f}% fill rate)")

    print(f"\n  Trade Statistics:")
    print(f"    Total Trades:         {results['total_trades']}")
    print(f"    Winners:              {results['n_winners']}")
    print(f"    Losers:               {results['n_losers']}")
    print(f"    Win Rate:             {results['win_rate']:.1f}%")
    print(f"    Avg Win:              ${results['avg_win']:,.2f}")
    print(f"    Avg Loss:             ${results['avg_loss']:,.2f}")
    rr = abs(results['avg_win'] / results['avg_loss']) if results['avg_loss'] != 0 else 0
    print(f"    Risk/Reward:          {rr:.2f}")
    pf = results['profit_factor']
    print(f"    Profit Factor:        {pf:.2f}" if pf != float('inf') else f"    Profit Factor:        ∞ (no losses)")
    print(f"    Expectancy:           ${results['expectancy']:,.2f}/trade")
    print(f"    Largest Win:          ${results['largest_win']:,.2f}")
    print(f"    Largest Loss:         ${results['largest_loss']:,.2f}")

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
    _kf = results.get('kelly_fraction', 0.0)
    if _kf > 0:
        print(f"  Position sizing: {_kf:.0%} Kelly (equal-weight cap: ${results['position_size']:,.2f})")
    else:
        print(f"  Position size: ${results['position_size']:,.2f} (equal weight)")
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
        f"Return: {results['total_return_pct']:+.02f}%  |  "
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
