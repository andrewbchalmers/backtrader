# monitor.py
"""
Live trading monitor - scans for opportunities and manages positions
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
from positions import PositionManager


class LiveTradingMonitor:
    """Monitor stocks and send notifications for trading opportunities"""

    def __init__(self, watchlist, strategy_loader, strategy_params, notifier, warmup_days=300):
        self.watchlist = watchlist
        self.strategy = strategy_loader
        self.params = strategy_params
        self.notifier = notifier
        self.position_manager = PositionManager()
        self.warmup_days = warmup_days
        self.buy_alerts_sent = {}  # Track when buy alerts were sent: {symbol: date}
        self.market_open_notified_date = None  # Track when market open notification was sent

        # Clear any pending_exit flags from previous sessions
        self._clear_pending_exit_flags()

        # Import chart generator
        from chart_generator import ChartGenerator
        self.chart_gen = ChartGenerator(strategy_loader, strategy_params)

        # Start continuous reply listener
        self.notifier.start_listening(self._handle_reply)

    def _clear_pending_exit_flags(self):
        """Clear pending_exit flags on startup (they don't persist across restarts)"""
        positions = self.position_manager.list_all()
        cleared = 0

        for symbol, position in positions.items():
            if position.get('pending_exit'):
                del position['pending_exit']
                if 'exit_alerted_date' in position:
                    del position['exit_alerted_date']
                self.position_manager.positions[symbol] = position
                cleared += 1

        if cleared > 0:
            self.position_manager._save()
            print(f"ℹ️  Cleared {cleared} pending exit flag(s) from previous session")

    def get_live_data(self, symbol, period="1y", interval="1d"):
        """Fetch live data for a symbol

        Args:
            symbol: Stock ticker
            period: Data period (e.g., "1y", "6mo", "60d")
            interval: Bar interval (e.g., "1d", "1h", "15m")
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                return None

            df.index = df.index.tz_localize(None)
            return df

        except Exception as e:
            print(f"❌ Error fetching {symbol} ({interval}): {e}")
            return None

    def get_historical_data(self, symbol, start_date, end_date):
        """Fetch historical data for testing mode with warmup period"""
        try:
            start = pd.to_datetime(start_date)

            # Convert trading days to calendar days
            calendar_days = int(self.warmup_days * 1.6)
            lookback_start = start - timedelta(days=calendar_days)

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=lookback_start.strftime('%Y-%m-%d'),
                                end=end_date,
                                interval="1d")

            if df.empty:
                return None

            df.index = df.index.tz_localize(None)
            return df

        except Exception as e:
            print(f"❌ Error fetching historical data for {symbol}: {e}")
            return None

    def scan_for_opportunities(self):
        """Scan all watchlist stocks for buy opportunities and held positions for exits"""
        print(f"\n{'='*60}")
        print(f"SCANNING AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        # Clean up old buy alerts (remove alerts from previous days)
        today = datetime.now().date()
        self.buy_alerts_sent = {
            symbol: date for symbol, date in self.buy_alerts_sent.items()
            if date == today
        }

        # Get list of held positions
        held_positions = set(self.position_manager.list_all().keys())
        print(f"Currently holding {len(held_positions)} positions: {list(held_positions)}")

        # Scan for BUY opportunities (only symbols we DON'T hold)
        buy_opportunities = 0
        today = datetime.now().date()

        for symbol in self.watchlist:
            try:
                if symbol in held_positions:
                    continue

                # Check if we already sent a BUY alert today
                if symbol in self.buy_alerts_sent:
                    alert_date = self.buy_alerts_sent[symbol]
                    if alert_date == today:
                        # Already sent alert today, skip
                        continue

                df = self.get_live_data(symbol, period="max", interval="1d")
                if df is None or len(df) < 200:
                    continue

                buy_signal = self.strategy.get_entry_signal(df, self.params)

                if buy_signal.get('signal') and buy_signal.get('signal_type', 'BUY') == 'BUY':
                    # Only alert if signal is from today or yesterday (bars_ago <= 1)
                    bars_ago = buy_signal.get('bars_ago', 0)
                    if bars_ago <= 1:
                        self.notifier.send_buy_alert(symbol, buy_signal)
                        self.buy_alerts_sent[symbol] = today  # Mark as alerted today
                        buy_opportunities += 1
                        print(f"🟢 BUY SIGNAL: {symbol} (ML prediction: {buy_signal.get('prediction', 'N/A')})")

            except Exception as e:
                print(f"❌ Error scanning {symbol} for buy: {e}")

        print(f"\nFound {buy_opportunities} buy opportunities")

        # Scan ONLY held positions for SELL opportunities
        if held_positions:
            print(f"\nChecking {len(held_positions)} held positions for exits...")
            for symbol in held_positions:
                try:
                    position = self.position_manager.get(symbol)

                    # Skip if already sent exit alert (waiting for confirmation)
                    if position and position.get('pending_exit'):
                        print(f"⏳ {symbol}: Exit alert already sent, waiting for 'SOLD {symbol}' confirmation")
                        continue

                    self.check_exit(symbol)
                except Exception as e:
                    print(f"❌ Error checking exit for {symbol}: {e}")
        else:
            print("\nNo positions to check for exits")

    def _handle_reply(self, reply):
        """
        Process a single reply from user (called by background listener)

        Args:
            reply: The reply string from user (already uppercased)
        """
        print(f"\n💬 Processing reply: {reply}")

        if reply.startswith("BOUGHT "):
            self.handle_bought_reply(reply)
        elif reply.startswith("SOLD "):
            self.handle_sold_reply(reply)
        elif reply.startswith("LAST "):
            self.handle_last_signal_query(reply)
        elif reply.startswith("BACKTEST "):
            self.handle_backtest_query(reply)
        elif reply == "HOLDING" or reply == "HOLDINGS":
            self.handle_holdings_query()
        else:
            print(f"⚠️  Unknown reply format: {reply}")

    def handle_bought_reply(self, reply):
        """Handle 'BOUGHT SYMBOL' or 'BOUGHT SYMBOL AT PRICE' reply"""
        parts = reply.split()

        if len(parts) < 2:
            print(f"⚠️  Invalid BOUGHT format: {reply}")
            return

        symbol = parts[1].upper()

        if self.position_manager.has_position(symbol):
            print(f"⚠️  Already holding {symbol}")
            return

        # Check if price was provided
        price = None
        if len(parts) >= 4 and parts[2].upper() == "AT":
            try:
                price = float(parts[3].replace('$', ''))
            except ValueError:
                print(f"⚠️  Invalid price in: {reply}")

        # Get current price if not provided
        if price is None:
            df = self.get_live_data(symbol)
            if df is not None:
                price = df['Close'].iloc[-1]
            else:
                print(f"❌ Could not get price for {symbol}")
                return

        # Calculate stop loss using strategy
        df = self.get_live_data(symbol)
        if df is None:
            print(f"❌ Could not get data for {symbol}")
            return

        buy_signal = self.strategy.get_entry_signal(df, self.params)
        stop_loss = buy_signal.get('stop_loss', price * 0.9)

        # Add position
        position = self.position_manager.add(symbol, price, stop_loss)

        if position:
            self.notifier.send_position_confirmation(symbol, price, stop_loss, "added")
            print(f"✅ Auto-added position from reply: {symbol} @ ${price:.2f}")

            # Clear the buy alert tracking since position was confirmed
            if symbol in self.buy_alerts_sent:
                del self.buy_alerts_sent[symbol]

    def handle_sold_reply(self, reply):
        """Handle 'SOLD SYMBOL' reply"""
        parts = reply.split()

        if len(parts) < 2:
            print(f"⚠️  Invalid SOLD format: {reply}")
            return

        symbol = parts[1].upper()
        position = self.position_manager.get(symbol)

        if not position:
            print(f"⚠️  No position found for {symbol}")
            return

        # Get entry price for notification
        entry_price = position['entry_price']

        # Get current price to calculate P&L
        df = self.get_live_data(symbol)
        if df is not None:
            current_price = df['Close'].iloc[-1]
            pnl = ((current_price / entry_price) - 1) * 100
        else:
            current_price = entry_price
            pnl = 0.0

        # Remove the position
        self.position_manager.remove(symbol)

        # Send confirmation with P&L
        self.notifier.send_position_confirmation(symbol, entry_price, current_price, "removed", pnl)
        print(f"✅ Auto-removed position from reply: {symbol} (P&L: {pnl:+.2f}%)")

    def check_exit(self, symbol):
        """Check if held position should be exited"""
        position = self.position_manager.get(symbol)
        if not position:
            return

        entry_price = position['entry_price']
        current_stop = position['stop_loss']

        df = self.get_live_data(symbol)
        if df is None:
            return

        current_price = df['Close'].iloc[-1]

        sell_signal = self.strategy.get_exit_signal(df, self.params, entry_price, current_stop)

        if sell_signal['signal']:
            # Check if stop was hit in the past
            bars_ago = sell_signal.get('bars_ago', 0)
            bar_date = sell_signal.get('bar_date', 'today')

            if bars_ago > 0:
                print(f"⚠️  {symbol} hit stop {bars_ago} bar(s) ago on {bar_date}!")

            # Send SELL alert but DON'T auto-remove position
            # Wait for user to confirm with "SOLD SYMBOL"
            self.notifier.send_sell_alert(symbol, sell_signal, entry_price)
            print(f"🔴 SELL ALERT: {symbol} - Stop hit at ${sell_signal['price']:.2f} on {bar_date}")
            print(f"   ⏳ Waiting for confirmation: Reply 'SOLD {symbol}' to remove position")

            # Mark position as "pending exit" to avoid sending duplicate alerts
            position['pending_exit'] = True
            position['exit_alerted_date'] = bar_date
            self.position_manager.positions[symbol] = position
            self.position_manager._save()

        else:
            # Update trailing stop
            old_stop = current_stop
            new_stop = sell_signal['new_stop']
            self.position_manager.update_stop(symbol, new_stop)

            # Show stop update info (only if it changed significantly)
            if abs(new_stop - old_stop) > 0.01:
                pnl = ((current_price / entry_price) - 1) * 100
                distance_to_stop = ((current_price - new_stop) / current_price) * 100
                print(f"📊 {symbol}: Price ${current_price:.2f} | P&L {pnl:+.2f}% | "
                      f"Stop ${new_stop:.2f} ({distance_to_stop:.1f}% away)")

    def is_market_hours(self):
        """Check if market is open (US Eastern Time)"""
        from datetime import datetime
        import pytz

        # Get current time in US Eastern timezone
        eastern = pytz.timezone('US/Eastern')
        now_et = datetime.now(eastern)

        # Weekend
        if now_et.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

        is_open = market_open <= now_et <= market_close

        if not is_open:
            print(f"Market closed (ET time: {now_et.strftime('%I:%M %p')})")

        return is_open

    def send_market_open_notification(self):
        """Send notification when market opens (once per day)"""
        from datetime import datetime
        import pytz

        eastern = pytz.timezone('US/Eastern')
        now_et = datetime.now(eastern)
        today = now_et.date()

        # Only send once per day
        if self.market_open_notified_date == today:
            return

        # Get position summary for the notification
        summary = self.position_manager.get_summary()
        position_count = summary['count']

        title = "🔔 Market Open"
        message = (
            f"The stock market is now open.\n"
            f"Time: {now_et.strftime('%I:%M %p')} ET\n"
            f"Watching: {len(self.watchlist)} symbols\n"
            f"Active positions: {position_count}"
        )

        self.notifier.send_notification(title, message)
        self.market_open_notified_date = today
        print(f"📢 Market open notification sent")

    def run(self, scan_interval=15):
        """Main loop - scan continuously during market hours"""
        print(f"\n🚀 Live Trading Monitor Started")
        print(f"Watching {len(self.watchlist)} symbols")
        print(f"Active positions: {self.position_manager.get_summary()['count']}")
        print(f"Scan interval: {scan_interval} minutes")
        print(f"Reply listener: ACTIVE (responds instantly to your texts)\n")

        try:
            while True:
                try:
                    if self.is_market_hours():
                        # Send market open notification (once per day)
                        self.send_market_open_notification()

                        self.scan_for_opportunities()
                        print(f"\n💤 Next scan in {scan_interval} minutes...")
                        time.sleep(scan_interval * 60)
                    else:
                        print("Market closed. Sleeping...")
                        time.sleep(300)

                except KeyboardInterrupt:
                    print("\n👋 Monitor stopped")
                    break
                except Exception as e:
                    print(f"❌ Error in main loop: {e}")
                    time.sleep(60)
        finally:
            # Stop the listener when exiting
            self.notifier.stop_listening()

    def handle_last_signal_query(self, reply):
        """
        Handle 'LAST SYMBOL' query - shows last signal for a stock with both 30-day and 3-month charts

        Args:
            reply: The reply string like "LAST NVDA"
        """
        parts = reply.split()

        if len(parts) < 2:
            self.notifier.send_notification(
                "⚠️ Invalid Format",
                "Usage: LAST <SYMBOL>\nExample: LAST NVDA"
            )
            return

        symbol = parts[1].upper()

        # Chart configurations with different intervals
        # 30-day chart: hourly bars for more detail
        # 3-month chart: daily bars for broader view
        chart_configs = [
            {
                'period': '60d',      # Fetch 60 days of hourly data
                'interval': '1h',
                'bars': 30 * 7,       # ~30 days * ~7 trading hours
                'title': '30 Day (Hourly)'
            },
            {
                'period': '2y',       # Fetch 2 years of daily data
                'interval': '1d',
                'bars': 90,           # 90 trading days
                'title': '3 Month (Daily)'
            }
        ]

        print(f"📊 Fetching multi-timeframe data for {symbol}...")

        # Fetch data for each timeframe
        chart_data_list = []
        df_daily = None  # Keep daily data for signal checking

        for config in chart_configs:
            print(f"   Fetching {config['interval']} data (period: {config['period']})...")
            df = self.get_live_data(symbol, period=config['period'], interval=config['interval'])

            if df is not None and len(df) > 0:
                print(f"   Got {len(df)} bars of {config['interval']} data")
                chart_data_list.append({
                    'df': df,
                    'bars': config['bars'],
                    'title': config['title'],
                    'interval': config['interval']
                })

                # Keep daily data for status message calculations
                if config['interval'] == '1d':
                    df_daily = df
            else:
                print(f"   ⚠️ Failed to fetch {config['interval']} data")

        if not chart_data_list:
            self.notifier.send_notification(
                f"❌ {symbol}",
                f"Unable to fetch data for any timeframe"
            )
            return

        # Use daily data for status calculations, fall back to first available
        df = df_daily if df_daily is not None else chart_data_list[0]['df']
        print(f"📊 Using {len(df)} bars for signal detection")

        # Build the status message
        position = self.position_manager.get(symbol)
        if position:
            entry_date = position['entry_date'][:10]
            entry_price = position['entry_price']
            stop_loss = position['stop_loss']

            # Get current price and P&L
            current_price = df['Close'].iloc[-1]
            pnl = ((current_price / entry_price) - 1) * 100

            title = f"📊 {symbol} - Current Position"
            message = (
                f"Signal: BUY\n"
                f"Date: {entry_date}\n"
                f"Entry: ${entry_price:.2f}\n"
                f"Current: ${current_price:.2f}\n"
                f"P&L: {pnl:+.2f}%\n"
                f"Stop: ${stop_loss:.2f}"
            )

            if position.get('pending_exit'):
                message += f"\n\n⚠️ Exit signal active!\nReply: SOLD {symbol}"

        else:
            # Not holding - run ML strategy to detect recent signals
            print(f"📊 Running ML strategy signal detection for {symbol}...")
            signal = self.strategy.get_entry_signal(df, self.params)

            if signal.get('signal'):
                signal_type = signal.get('signal_type', 'BUY')
                signal_date = signal.get('date', df.index[-1])
                if hasattr(signal_date, 'strftime'):
                    signal_date = signal_date.strftime('%Y-%m-%d')
                else:
                    signal_date = str(signal_date)

                signal_price = signal.get('price', df['Close'].iloc[-1])
                current_price = signal.get('current_price', df['Close'].iloc[-1])
                bars_ago = signal.get('bars_ago', 0)
                prediction = signal.get('prediction', 0)

                pnl = ((current_price / signal_price) - 1) * 100

                title = f"📊 {symbol} - Last Signal"
                message = (
                    f"Signal: {signal_type}\n"
                    f"Date: {signal_date} ({bars_ago} bar(s) ago)\n"
                    f"Price then: ${signal_price:.2f}\n"
                    f"Price now: ${current_price:.2f}\n"
                    f"Change: {pnl:+.2f}%\n"
                    f"ML Prediction: {prediction:+d}\n\n"
                    f"⚠️ Not currently holding"
                )
                print(f"✓ Found {signal_type} signal from {signal_date}")
            else:
                title = f"📊 {symbol} - Last Signal"
                message = "No recent BUY/SELL signals in the last 5 bars"
                print(f"✓ No recent signals found for {symbol}")

        # Generate stacked multi-timeframe chart
        print(f"📊 Generating stacked chart for {symbol} (30 Day Hourly + 3 Month Daily)...")
        chart_buffer = self.chart_gen.generate_multi_timeframe_chart(
            symbol, chart_data_list
        )

        if chart_buffer:
            self.notifier.send_notification_with_image(
                title, message, chart_buffer, f"{symbol}_chart.png"
            )
            print(f"✓ Sent stacked chart for {symbol}")
        else:
            # Fallback to text only
            self.notifier.send_notification(title, message)
            print(f"✓ Sent last signal info for {symbol} (text only, chart generation failed)")

    def handle_backtest_query(self, reply):
        """
        Handle 'BACKTEST SYMBOL PERIOD' query - runs a backtest and returns results

        Args:
            reply: The reply string like "BACKTEST NVDA 1Y" or "BACKTEST AAPL 6M"
        """
        parts = reply.split()

        if len(parts) < 2:
            self.notifier.send_notification(
                "⚠️ Invalid Format",
                "Usage: BACKTEST <SYMBOL> [PERIOD]\n"
                "Periods: 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y\n"
                "Example: BACKTEST NVDA 1Y"
            )
            return

        symbol = parts[1].upper()
        period = parts[2].upper() if len(parts) >= 3 else "1Y"

        # Parse period to get start/end dates
        period_map = {
            '1M': 30,
            '3M': 90,
            '6M': 180,
            '1Y': 365,
            '2Y': 730,
            '3Y': 1095,
            '5Y': 1825,
        }

        if period not in period_map:
            self.notifier.send_notification(
                "⚠️ Invalid Period",
                f"Unknown period: {period}\n"
                "Valid periods: 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y"
            )
            return

        days = period_map[period]

        print(f"📊 Running backtest for {symbol} over {period}...")
        self.notifier.send_notification(
            f"⏳ Backtest Started",
            f"Running {period} backtest for {symbol}...\nThis may take a moment."
        )

        try:
            results = self._run_backtest(symbol, days)

            if results is None:
                self.notifier.send_notification(
                    f"❌ Backtest Failed",
                    f"Could not run backtest for {symbol}\nInsufficient data or error occurred."
                )
                return

            # Format results message
            title = f"📊 {symbol} Backtest ({period})"

            # Build detailed message
            message_lines = [
                f"Period: {results['start_date']} to {results['end_date']}",
                f"",
                f"💰 Returns:",
                f"  Total: {results['total_return_pct']:+.2f}%",
                f"  Annualized: {results['annualized_return']:+.2f}%",
                f"  vs SPY: {results['spy_return']:+.2f}%",
                f"  Alpha: {results['total_return_pct'] - results['spy_return']:+.2f}%",
                f"",
                f"📈 Trades:",
                f"  Total: {results['total_trades']}",
                f"  Win Rate: {results['win_rate']:.1f}%",
                f"  Profit Factor: {results['profit_factor']:.2f}",
                f"",
                f"📉 Risk:",
                f"  Max Drawdown: {results['max_drawdown']:.2f}%",
                f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}",
            ]

            if results['total_trades'] > 0:
                message_lines.extend([
                    f"",
                    f"💵 Avg Trade:",
                    f"  Win: ${results['avg_win']:.2f}",
                    f"  Loss: ${results['avg_loss']:.2f}",
                ])

            message = "\n".join(message_lines)

            self.notifier.send_notification(title, message)
            print(f"✓ Sent backtest results for {symbol}")

        except Exception as e:
            print(f"❌ Backtest error: {e}")
            import traceback
            traceback.print_exc()
            self.notifier.send_notification(
                f"❌ Backtest Error",
                f"Error running backtest for {symbol}:\n{str(e)[:100]}"
            )

    def _run_backtest(self, symbol, days):
        """
        Run a backtest for the given symbol and period.

        Args:
            symbol: Stock ticker
            days: Number of calendar days to backtest

        Returns:
            dict: Backtest results or None on failure
        """
        import backtrader as bt
        import numpy as np

        # Calculate dates
        end_date = datetime.now()
        test_start = end_date - timedelta(days=days)

        # Add warmup period (need extra data for ML model)
        warmup_calendar_days = int(self.warmup_days * 1.5)
        data_start = test_start - timedelta(days=warmup_calendar_days)

        print(f"   Fetching data from {data_start.date()} to {end_date.date()}...")

        # Fetch data
        try:
            import yfinance as yf
            df = yf.download(symbol, start=data_start.strftime('%Y-%m-%d'),
                           end=end_date.strftime('%Y-%m-%d'), progress=False)

            if df.empty or len(df) < 100:
                print(f"   ❌ Insufficient data for {symbol}")
                return None

            df.index = df.index.tz_localize(None)

            # Handle multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.columns = [c.lower() for c in df.columns]

            # Find test start index
            test_start_mask = df.index >= pd.Timestamp(test_start)
            if not test_start_mask.any():
                print(f"   ❌ No data after test start date")
                return None

            test_start_idx = test_start_mask.argmax()

            print(f"   Got {len(df)} bars, test period starts at bar {test_start_idx}")

            # Fetch SPY for benchmark
            spy_df = yf.download('SPY', start=test_start.strftime('%Y-%m-%d'),
                                end=end_date.strftime('%Y-%m-%d'), progress=False)
            spy_df.index = spy_df.index.tz_localize(None)

            # Handle multi-level columns from yfinance
            if isinstance(spy_df.columns, pd.MultiIndex):
                spy_df.columns = spy_df.columns.get_level_values(0)

            if len(spy_df) > 0:
                spy_start = float(spy_df['Close'].iloc[0])
                spy_end = float(spy_df['Close'].iloc[-1])
                spy_return = ((spy_end / spy_start) - 1) * 100
            else:
                spy_return = 0

        except Exception as e:
            print(f"   ❌ Data fetch error: {e}")
            return None

        # Run backtrader
        try:
            cerebro = bt.Cerebro(stdstats=False)

            # Add data
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

            # Prepare params
            strategy_params = self.params.copy()
            strategy_params['verbose'] = False
            strategy_params['test_start_idx'] = test_start_idx

            # Add strategy
            cerebro.addstrategy(self.strategy.strategy_class, **strategy_params)

            # Broker settings
            initial_cash = 10000
            cerebro.broker.setcash(initial_cash)
            cerebro.broker.setcommission(commission=0.0)

            # Add analyzers
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")

            # Add portfolio value observer
            class PortfolioValue(bt.Observer):
                lines = ('value',)
                plotinfo = dict(plot=False)
                def next(self):
                    self.lines.value[0] = self._owner.broker.getvalue()
                def prenext(self):
                    self.lines.value[0] = self._owner.broker.getvalue()

            cerebro.addobserver(PortfolioValue)

            print(f"   Running backtest...")
            results = cerebro.run()
            strat = results[0]

            # Extract results
            trades = strat.analyzers.trades.get_analysis()

            # Get portfolio values for test period
            test_values = []
            observer = strat.observers.portfoliovalue
            for i in range(len(observer.lines.value)):
                if i >= test_start_idx:
                    try:
                        val = observer.lines.value.array[i]
                        if not np.isnan(val) and val > 0:
                            test_values.append(val)
                    except:
                        break

            if len(test_values) < 2:
                test_values = [initial_cash, cerebro.broker.getvalue()]

            final_value = test_values[-1]
            total_return_pct = ((final_value / initial_cash) - 1) * 100

            # Calculate metrics
            total_trades = trades.get('total', {}).get('total', 0)

            if total_trades > 0:
                win_count = trades.get('won', {}).get('total', 0)
                win_rate = (win_count / total_trades) * 100

                total_win_pnl = trades.get('won', {}).get('pnl', {}).get('total', 0)
                total_loss_pnl = abs(trades.get('lost', {}).get('pnl', {}).get('total', 0))
                profit_factor = (total_win_pnl / total_loss_pnl) if total_loss_pnl > 0 else 999

                avg_win = trades.get('won', {}).get('pnl', {}).get('average', 0)
                avg_loss = abs(trades.get('lost', {}).get('pnl', {}).get('average', 0))
            else:
                win_rate = 0
                profit_factor = 0
                avg_win = 0
                avg_loss = 0

            # Calculate drawdown from test period
            peak = test_values[0]
            max_dd = 0
            for val in test_values:
                if val > peak:
                    peak = val
                dd = ((peak - val) / peak) * 100
                if dd > max_dd:
                    max_dd = dd

            # Calculate Sharpe ratio
            if len(test_values) > 1:
                daily_returns = []
                for i in range(1, len(test_values)):
                    ret = (test_values[i] / test_values[i-1]) - 1
                    daily_returns.append(ret)

                if len(daily_returns) > 1 and np.std(daily_returns) > 0:
                    sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
                else:
                    sharpe = 0
            else:
                sharpe = 0

            # Annualized return
            test_days = len(test_values)
            years = test_days / 252
            if years > 0:
                annualized = ((final_value / initial_cash) ** (1 / years) - 1) * 100
            else:
                annualized = total_return_pct

            # Get actual date range
            test_df = df.iloc[test_start_idx:]
            start_date_str = test_df.index[0].strftime('%Y-%m-%d')
            end_date_str = test_df.index[-1].strftime('%Y-%m-%d')

            return {
                'start_date': start_date_str,
                'end_date': end_date_str,
                'total_return_pct': total_return_pct,
                'annualized_return': annualized,
                'spy_return': spy_return,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_drawdown': max_dd,
                'sharpe_ratio': sharpe,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'final_value': final_value,
            }

        except Exception as e:
            print(f"   ❌ Backtest execution error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def handle_holdings_query(self):
        """Handle 'HOLDING' or 'HOLDINGS' query - shows all current positions with P&L"""
        positions = self.position_manager.list_all()

        if not positions:
            self.notifier.send_notification(
                "📭 No Holdings",
                "You are not currently holding any positions"
            )
            print("✓ Sent holdings query response (empty)")
            return

        # Build holdings message
        title = f"📊 Holdings ({len(positions)})"
        message_lines = []

        total_pnl = 0
        count = 0

        for symbol, pos in positions.items():
            entry_price = pos['entry_price']
            entry_date = pos['entry_date'][:10]

            # Get current price
            df = self.get_live_data(symbol)
            if df is not None:
                current_price = df['Close'].iloc[-1]
                pnl = ((current_price / entry_price) - 1) * 100
                total_pnl += pnl
                count += 1

                pending_exit = " ⚠️ SELL" if pos.get('pending_exit') else ""
                message_lines.append(
                    f"{symbol}: {pnl:+.2f}%{pending_exit}\n"
                    f"  Entry: ${entry_price:.2f} ({entry_date})\n"
                    f"  Now: ${current_price:.2f}"
                )
            else:
                message_lines.append(
                    f"{symbol}: N/A\n"
                    f"  Entry: ${entry_price:.2f} ({entry_date})\n"
                    f"  (Unable to fetch price)"
                )

        message = "\n\n".join(message_lines)

        # Add summary
        if count > 0:
            avg_pnl = total_pnl / count
            message += f"\n\n{'─'*20}\nAvg P&L: {avg_pnl:+.2f}%"

        self.notifier.send_notification(title, message)
        print(f"✓ Sent holdings query response ({len(positions)} positions)")