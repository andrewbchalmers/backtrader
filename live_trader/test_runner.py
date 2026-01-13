# test_runner.py
"""
Test mode runner for backtesting strategies
"""

import pandas as pd
from datetime import datetime, timedelta
import time


class TestRunner:
    """Run strategy backtests with detailed performance reporting"""

    def __init__(self, monitor, warmup_days):
        self.monitor = monitor
        self.warmup_days = warmup_days

    def run(self, start_date, end_date, speed=0):
        """
        Run test mode simulation

        Args:
            start_date: Start of test period (YYYY-MM-DD)
            end_date: End of test period (YYYY-MM-DD)
            speed: Days per second (0 = instant)
        """
        print(f"\n🧪 TESTING MODE")
        print(f"Watching {len(self.monitor.watchlist)} symbols")
        print(f"Test Period: {start_date} to {end_date}")
        print(f"Warmup Required: {self.warmup_days} trading days")
        print(f"Speed: {speed} day(s) per second" if speed > 0 else "Speed: Instant")
        print()

        test_start = pd.to_datetime(start_date)
        test_end = pd.to_datetime(end_date)

        # Load historical data
        stock_data = self._load_historical_data(start_date, end_date, test_start)

        if not stock_data:
            print("\n❌ No valid data loaded!")
            return

        # Run simulation
        completed_trades = self._run_simulation(stock_data, test_start, test_end, speed)

        # Print reports
        self._print_performance_report(completed_trades)
        self._print_open_positions()

    def _load_historical_data(self, start_date, end_date, test_start):
        """Load and validate historical data for all symbols"""
        calendar_days = int(self.warmup_days * 1.6)
        data_start = test_start - timedelta(days=calendar_days)

        print(f"Fetching from {data_start.strftime('%Y-%m-%d')} (~{calendar_days} calendar days)")
        print(f"Signals start from {start_date}\n")

        stock_data = {}
        for symbol in self.monitor.watchlist:
            df = self.monitor.get_historical_data(symbol, start_date, end_date)

            if df is not None:
                data_before_test = df[df.index < test_start]
                trading_days_before = len(data_before_test)
                min_required = int(self.warmup_days * 0.8)

                if trading_days_before >= min_required:
                    stock_data[symbol] = df
                    total_days = len(df)
                    test_days = len(df[df.index >= test_start])
                    print(f"✓ {symbol}: {total_days} days ({trading_days_before} warmup + {test_days} test)")
                else:
                    print(f"❌ {symbol}: Insufficient warmup ({trading_days_before} < {min_required})")
            else:
                print(f"❌ {symbol}: No data available")

        return stock_data

    def _run_simulation(self, stock_data, test_start, test_end, speed):
        """Run day-by-day simulation"""
        print(f"\n{'='*60}")
        print(f"STARTING SIMULATION - {len(stock_data)} symbols loaded")
        print(f"{'='*60}\n")

        test_dates = pd.date_range(start=test_start, end=test_end, freq='D')
        completed_trades = []
        total_buys = 0
        total_sells = 0

        for current_date in test_dates:
            print(f"\n📅 {current_date.strftime('%Y-%m-%d')}")

            held_symbols = set(self.monitor.position_manager.list_all().keys())
            if held_symbols:
                print(f"   Holding: {list(held_symbols)}")

            buy_count = 0
            sell_count = 0

            for symbol in stock_data.keys():
                df = stock_data[symbol]
                current_df = df[df.index <= current_date]

                if len(current_df) < self.warmup_days:
                    continue

                if symbol in held_symbols:
                    # Check for exit
                    trade = self._check_exit(symbol, current_df, current_date)
                    if trade:
                        completed_trades.append(trade)
                        sell_count += 1
                        total_sells += 1
                else:
                    # Check for entry
                    if self._check_entry(symbol, current_df, current_date):
                        buy_count += 1
                        total_buys += 1

            if buy_count > 0 or sell_count > 0:
                print(f"   Signals: {buy_count} BUY, {sell_count} SELL")

            if speed > 0:
                time.sleep(1.0 / speed)

        print(f"\n{'='*60}")
        print("SIMULATION COMPLETE")
        print(f"Total Signals: {total_buys} BUY, {total_sells} SELL")
        print(f"{'='*60}\n")

        return completed_trades

    def _check_entry(self, symbol, df, current_date):
        """Check for entry signal"""
        buy_signal = self.monitor.strategy.get_entry_signal(df, self.monitor.params)

        if buy_signal['signal']:
            self.monitor.notifier.send_buy_alert(symbol, buy_signal)
            self.monitor.position_manager.add(
                symbol,
                buy_signal['price'],
                buy_signal['stop_loss'],
                current_date.isoformat()
            )
            return True
        return False

    def _check_exit(self, symbol, df, current_date):
        """Check for exit signal and return trade data if exited"""
        position = self.monitor.position_manager.get(symbol)
        entry_price = position['entry_price']
        current_stop = position['stop_loss']
        entry_date = pd.to_datetime(position['entry_date'])

        sell_signal = self.monitor.strategy.get_exit_signal(
            df, self.monitor.params, entry_price, current_stop
        )

        if sell_signal['signal']:
            exit_price = sell_signal['price']
            pnl_pct = ((exit_price / entry_price) - 1) * 100
            days_held = (current_date - entry_date).days

            trade = {
                'symbol': symbol,
                'entry_date': entry_date.strftime('%Y-%m-%d'),
                'exit_date': current_date.strftime('%Y-%m-%d'),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'days_held': days_held,
                'exit_type': sell_signal['stop_type']
            }

            self.monitor.notifier.send_sell_alert(symbol, sell_signal, entry_price)
            self.monitor.position_manager.remove(symbol)
            return trade
        else:
            self.monitor.position_manager.update_stop(symbol, sell_signal['new_stop'])
            return None

    def _print_performance_report(self, completed_trades):
        """Print detailed performance report"""
        if not completed_trades:
            print("No completed trades during test period\n")
            return

        print(f"\n{'='*80}")
        print("TRADE PERFORMANCE REPORT")
        print(f"{'='*80}\n")

        # Calculate statistics
        winning_trades = [t for t in completed_trades if t['pnl_pct'] > 0]
        losing_trades = [t for t in completed_trades if t['pnl_pct'] <= 0]

        total_trades = len(completed_trades)
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

        avg_win = sum(t['pnl_pct'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl_pct'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        avg_pnl = sum(t['pnl_pct'] for t in completed_trades) / total_trades if total_trades > 0 else 0
        total_pnl = sum(t['pnl_pct'] for t in completed_trades)
        avg_days = sum(t['days_held'] for t in completed_trades) / total_trades if total_trades > 0 else 0

        # Summary statistics
        print(f"Total Trades:        {total_trades}")
        print(f"Winning Trades:      {len(winning_trades)} ({win_rate:.1f}%)")
        print(f"Losing Trades:       {len(losing_trades)} ({100-win_rate:.1f}%)")
        print(f"Average Win:         {avg_win:+.2f}%")
        print(f"Average Loss:        {avg_loss:+.2f}%")
        print(f"Average P&L:         {avg_pnl:+.2f}%")
        print(f"Total P&L:           {total_pnl:+.2f}%")
        print(f"Average Hold Time:   {avg_days:.1f} days")

        if winning_trades and losing_trades:
            profit_factor = abs(sum(t['pnl_pct'] for t in winning_trades) / sum(t['pnl_pct'] for t in losing_trades))
            print(f"Profit Factor:       {profit_factor:.2f}")

        # Individual trades
        print(f"\n{'─'*80}")
        print("Symbol   Entry        Exit         Entry $    Exit $     P&L        Days   Type  ")
        print(f"{'─'*80}")

        for trade in completed_trades:
            pnl_str = f"{trade['pnl_pct']:+.2f}%"
            print(f"{trade['symbol']:<8} {trade['entry_date']:<12} {trade['exit_date']:<12} "
                  f"${trade['entry_price']:<9.2f} ${trade['exit_price']:<9.2f} "
                  f"{pnl_str:<10} {trade['days_held']:<6} {trade['exit_type']:<6}")

        print(f"{'='*80}\n")

    def _print_open_positions(self):
        """Print currently open positions"""
        summary = self.monitor.position_manager.get_summary()
        if summary['count'] > 0:
            print(f"Open Positions: {summary['count']}")
            print(f"Average Risk:   {summary['avg_risk_pct']:.2f}%\n")
            for symbol, pos in self.monitor.position_manager.list_all().items():
                print(f"  {symbol}: Entry ${pos['entry_price']:.2f}, SL ${pos['stop_loss']:.2f}")
        else:
            print("No open positions")
        print()