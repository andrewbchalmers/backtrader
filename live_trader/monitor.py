# monitor.py
"""
Live trading monitor - scans for opportunities and manages positions
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import gc
import threading
from positions import PositionManager

# Suppress yfinance and urllib verbose logging/errors
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)


class LiveTradingMonitor:
    """Monitor stocks and send notifications for trading opportunities"""

    # Valid timeframes and their configurations
    TIMEFRAMES = {
        '1M': {'interval': '1m', 'period': '7d', 'description': '1 Minute'},
        '5M': {'interval': '5m', 'period': '60d', 'description': '5 Minutes'},
        '15M': {'interval': '15m', 'period': '60d', 'description': '15 Minutes'},
        '30M': {'interval': '30m', 'period': '60d', 'description': '30 Minutes'},
        '1H': {'interval': '1h', 'period': '730d', 'description': '1 Hour'},
        '4H': {'interval': '4h', 'period': '730d', 'description': '4 Hours'},
        '1D': {'interval': '1d', 'period': 'max', 'description': '1 Day'},
    }

    def __init__(self, watchlist, strategy_loader, strategy_params, notifier, warmup_days=300):
        self.watchlist = watchlist
        self.strategy = strategy_loader
        self.params = strategy_params
        self.notifier = notifier
        self.position_manager = PositionManager()
        self.warmup_days = warmup_days
        self.buy_alerts_sent = {}  # Track when buy alerts were sent: {symbol: date}
        self.market_open_notified_date = None  # Track when market open notification was sent
        self.current_timeframe = '1D'  # Default to daily bars
        self.ml_lock = threading.Lock()  # Prevent concurrent ML operations
        self.pending_command = None  # Command to process between scan symbols

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

    def get_live_data(self, symbol, period=None, interval=None, use_current_timeframe=True):
        """Fetch live data for a symbol

        Args:
            symbol: Stock ticker
            period: Data period (e.g., "1y", "6mo", "60d"). If None, uses current timeframe setting.
            interval: Bar interval (e.g., "1d", "1h", "15m"). If None, uses current timeframe setting.
            use_current_timeframe: If True and period/interval not specified, use current timeframe
        """
        try:
            # Use current timeframe settings if not explicitly specified
            if use_current_timeframe and (period is None or interval is None):
                tf_config = self.TIMEFRAMES[self.current_timeframe]
                if period is None:
                    period = tf_config['period']
                if interval is None:
                    interval = tf_config['interval']

            # Use yf.download instead of Ticker.history - handles connections better
            df = yf.download(symbol, period=period, interval=interval, progress=False,
                           auto_adjust=True, prepost=False, threads=False)

            if df.empty:
                return None

            # Handle multi-level columns from yf.download
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten to just the price column names (Open, High, Low, Close, Volume)
                df.columns = [col[0] for col in df.columns]

            # Remove any duplicate columns (keep first)
            df = df.loc[:, ~df.columns.duplicated()]

            df.index = df.index.tz_localize(None)
            return df

        except Exception as e:
            print(f"\n❌ Error fetching {symbol}: {e}")
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
        tf_config = self.TIMEFRAMES[self.current_timeframe]
        print(f"\n{'='*60}")
        print(f"SCANNING AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Timeframe: {self.current_timeframe} ({tf_config['description']})")
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

        # Filter watchlist to only symbols we need to scan
        symbols_to_scan = [s for s in self.watchlist
                          if s not in held_positions
                          and self.buy_alerts_sent.get(s) != today]
        total_to_scan = len(symbols_to_scan)
        scan_start = time.time()
        print(f"Scanning {total_to_scan} symbols for buy opportunities...")

        for idx, symbol in enumerate(symbols_to_scan):
            try:
                # Skip if position was added during scan (via BOUGHT reply)
                if self.position_manager.has_position(symbol):
                    print(f"   [{idx+1}/{total_to_scan}] {symbol}: skipped (now holding)")
                    continue

                # Progress indicator - each on new line for debugging
                print(f"   [{idx+1}/{total_to_scan}] {symbol}: fetching...", end='', flush=True)

                # Small delay to avoid rate limiting
                time.sleep(0.1)

                df = self.get_live_data(symbol)
                if df is None or len(df) < 200:
                    print(" skipped (no data)")
                    continue

                # Limit to 1000 bars - balance between ML accuracy and scan speed
                if len(df) > 1000:
                    df = df.iloc[-1000:]

                print(f" {len(df)} bars...", end='', flush=True)

                # Run ML signal detection (with lock to prevent concurrent operations)
                with self.ml_lock:
                    buy_signal = self.strategy.get_entry_signal(df, self.params)

                # Clean up dataframe after use
                del df

                if buy_signal.get('signal') and buy_signal.get('signal_type', 'BUY') == 'BUY':
                    bars_ago = buy_signal.get('bars_ago', 0)
                    print(f"SIGNAL! (bars_ago={bars_ago})")
                    # Only alert if signal is recent (within last 3 bars)
                    if bars_ago <= 3:
                        self.notifier.send_buy_alert(symbol, buy_signal)
                        self.buy_alerts_sent[symbol] = today  # Mark as alerted today
                        buy_opportunities += 1
                        print(f"🟢 BUY SIGNAL: {symbol} (ML prediction: {buy_signal.get('prediction', 'N/A')})")
                    else:
                        print(f"   ⏭️  Skipped {symbol} - signal too old ({bars_ago} bars ago)")
                else:
                    print("no signal")

            except Exception as e:
                print(f" ERROR: {e}")

            # Check for pending commands between symbols
            if self.pending_command:
                cmd = self.pending_command
                self.pending_command = None
                print(f"\n⏸️  Pausing scan to process: {cmd}")
                if cmd.startswith("LAST "):
                    self.handle_last_signal_query(cmd)
                elif cmd.startswith("BACKTEST "):
                    self.handle_backtest_query(cmd)
                elif cmd.startswith("ANALYZE "):
                    self.handle_analyze_query(cmd)
                elif cmd.startswith("COMPARE "):
                    self.handle_compare_query(cmd)
                print(f"▶️  Resuming scan...\n")

            # Periodic garbage collection every 25 symbols
            if idx > 0 and idx % 25 == 0:
                gc.collect()

        # Final cleanup
        gc.collect()

        scan_duration = time.time() - scan_start
        print(f"\r   Scanning: {total_to_scan}/{total_to_scan} - Done!          ")
        print(f"✅ Scan complete in {scan_duration:.1f}s - Found {buy_opportunities} buy opportunities")

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

        # Quick commands - process immediately
        if reply.startswith("BOUGHT "):
            self.handle_bought_reply(reply)
        elif reply.startswith("SOLD "):
            self.handle_sold_reply(reply)
        elif reply.startswith("TIMEFRAME "):
            self.handle_timeframe_command(reply)
        elif reply == "HOLDING" or reply == "HOLDINGS":
            self.handle_holdings_query()
        elif reply == "TIMEFRAME":
            self.handle_timeframe_query()
        # Heavy commands - defer during active scan, process immediately when market closed
        elif reply.startswith("LAST ") or reply.startswith("BACKTEST ") or reply.startswith("ANALYZE ") or reply.startswith("COMPARE "):
            if self.is_market_hours():
                # During market hours, defer to avoid interrupting scans
                if self.pending_command is None:
                    self.pending_command = reply
                    print(f"⏳ Will process '{reply}' momentarily...")
                else:
                    print(f"⚠️ Already have a pending command, ignoring: {reply}")
            else:
                # Market closed - no scan running, process immediately
                print(f"⏳ Processing '{reply}' now (market closed)...")
                if reply.startswith("LAST "):
                    self.handle_last_signal_query(reply)
                elif reply.startswith("BACKTEST "):
                    self.handle_backtest_query(reply)
                elif reply.startswith("ANALYZE "):
                    self.handle_analyze_query(reply)
                elif reply.startswith("COMPARE "):
                    self.handle_compare_query(reply)
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

        # Calculate stop loss using simple percentage (avoid slow ML call)
        stop_loss_pct = self.params.get('stop_loss_pct', 0.05)
        if hasattr(stop_loss_pct, '__float__'):
            stop_loss_pct = float(stop_loss_pct)
        stop_loss = price * (1 - stop_loss_pct)

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
        tf_config = self.TIMEFRAMES[self.current_timeframe]
        print(f"\n🚀 Live Trading Monitor Started")
        print(f"Watching {len(self.watchlist)} symbols")
        print(f"Active positions: {self.position_manager.get_summary()['count']}")
        print(f"Timeframe: {self.current_timeframe} ({tf_config['description']})")
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

        # Send immediate acknowledgment
        print(f"⏳ Fetching signal data for {symbol}... this may take a moment")

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
            with self.ml_lock:
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

        # Generate stacked multi-timeframe chart (with lock to prevent concurrent ML)
        print(f"📊 Generating stacked chart for {symbol} (30 Day Hourly + 3 Month Daily)...")
        with self.ml_lock:
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

        # Send immediate acknowledgment
        print(f"⏳ Running backtest for {symbol} ({period})... this may take a minute")

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

        tf_config = self.TIMEFRAMES[self.current_timeframe]
        print(f"📊 Running backtest for {symbol} over {period} ({self.current_timeframe} bars)...")
        self.notifier.send_notification(
            f"⏳ Backtest Started",
            f"Running {period} backtest for {symbol}...\n"
            f"Timeframe: {self.current_timeframe} ({tf_config['description']})\n"
            f"This may take a moment."
        )

        try:
            # Run backtest with lock to prevent concurrent ML operations
            with self.ml_lock:
                results = self._run_backtest(symbol, days)

            if results is None:
                self.notifier.send_notification(
                    f"❌ Backtest Failed",
                    f"Could not run backtest for {symbol}\nInsufficient data or error occurred."
                )
                return

            # Format results message
            title = f"📊 {symbol} Backtest ({period} / {self.current_timeframe})"

            # Build detailed message
            message_lines = [
                f"Period: {results['start_date']} to {results['end_date']}",
                f"Timeframe: {self.current_timeframe} ({tf_config['description']})",
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

        # Get current timeframe settings
        tf_config = self.TIMEFRAMES[self.current_timeframe]
        interval = tf_config['interval']
        tf_description = tf_config['description']

        # Calculate dates
        end_date = datetime.now()
        test_start = end_date - timedelta(days=days)

        # Add warmup period (need extra data for ML model)
        # For intraday, we need more calendar days to get enough bars
        if interval in ['1m', '5m', '15m', '30m', '1h', '4h']:
            # Intraday: ~7 trading hours per day, need more calendar days
            bars_per_day = {'1m': 390, '5m': 78, '15m': 26, '30m': 13, '1h': 7, '4h': 2}
            bpd = bars_per_day.get(interval, 7)
            warmup_calendar_days = int((self.warmup_days / bpd) * 1.5) + 60
        else:
            warmup_calendar_days = int(self.warmup_days * 1.5)

        data_start = test_start - timedelta(days=warmup_calendar_days)

        print(f"   Timeframe: {self.current_timeframe} ({tf_description})")
        print(f"   Fetching data from {data_start.date()} to {end_date.date()}...")

        # Fetch data
        try:
            import yfinance as yf

            # For intraday data, yfinance has limitations on how far back we can go
            # Use period parameter for intraday to get maximum available data
            if interval in ['1m', '5m', '15m', '30m', '1h', '4h']:
                # Map interval to yfinance max period
                max_periods = {
                    '1m': '7d',
                    '5m': '60d',
                    '15m': '60d',
                    '30m': '60d',
                    '1h': '730d',
                    '4h': '730d',
                }
                period = max_periods.get(interval, '60d')
                df = yf.download(symbol, period=period, interval=interval, progress=False)
            else:
                df = yf.download(symbol, start=data_start.strftime('%Y-%m-%d'),
                               end=end_date.strftime('%Y-%m-%d'), interval=interval, progress=False)

            if df.empty or len(df) < 100:
                print(f"   ❌ Insufficient data for {symbol}")
                return None

            df.index = df.index.tz_localize(None)

            # Handle multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.columns = [c.lower() for c in df.columns]

            # Find test start index based on the requested test period
            # For intraday, calculate based on number of bars we want to test
            total_bars = len(df)

            if interval in ['1m', '5m', '15m', '30m', '1h', '4h']:
                # For intraday: calculate how many bars correspond to the test period
                bars_per_day = {'1m': 390, '5m': 78, '15m': 26, '30m': 13, '1h': 7, '4h': 2}
                bpd = bars_per_day.get(interval, 7)
                test_bars = int(days * bpd * 5 / 7)  # Approximate trading days

                # Ensure we have enough warmup
                min_warmup = min(self.warmup_days, total_bars // 2)
                test_start_idx = max(min_warmup, total_bars - test_bars)
            else:
                # Daily: use date-based calculation
                test_start_mask = df.index >= pd.Timestamp(test_start)
                if not test_start_mask.any():
                    print(f"   ❌ No data after test start date")
                    return None
                test_start_idx = test_start_mask.argmax()

            # Ensure test_start_idx is valid
            if test_start_idx >= total_bars - 10:
                test_start_idx = max(0, total_bars - 100)

            print(f"   Got {len(df)} bars ({interval}), test period starts at bar {test_start_idx}")

            # Fetch SPY for benchmark (use same interval for fair comparison)
            if interval in ['1m', '5m', '15m', '30m', '1h', '4h']:
                spy_df = yf.download('SPY', period=period, interval=interval, progress=False)
            else:
                spy_df = yf.download('SPY', start=test_start.strftime('%Y-%m-%d'),
                                    end=end_date.strftime('%Y-%m-%d'), progress=False)
            spy_df.index = spy_df.index.tz_localize(None)

            # Handle multi-level columns from yfinance
            if isinstance(spy_df.columns, pd.MultiIndex):
                spy_df.columns = spy_df.columns.get_level_values(0)

            # Calculate SPY return over equivalent test period
            if len(spy_df) > test_start_idx:
                spy_test_df = spy_df.iloc[test_start_idx:]
                if len(spy_test_df) > 0:
                    spy_start = float(spy_test_df['Close'].iloc[0])
                    spy_end = float(spy_test_df['Close'].iloc[-1])
                    spy_return = ((spy_end / spy_start) - 1) * 100
                else:
                    spy_return = 0
            elif len(spy_df) > 0:
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

            # Calculate Sharpe ratio (with proper annualization for timeframe)
            if len(test_values) > 1:
                bar_returns = []
                for i in range(1, len(test_values)):
                    ret = (test_values[i] / test_values[i-1]) - 1
                    bar_returns.append(ret)

                if len(bar_returns) > 1 and np.std(bar_returns) > 0:
                    # Annualization factor depends on bar frequency
                    if interval in ['1m', '5m', '15m', '30m', '1h', '4h']:
                        bars_per_day = {'1m': 390, '5m': 78, '15m': 26, '30m': 13, '1h': 7, '4h': 2}
                        bpd = bars_per_day.get(interval, 7)
                        annualization_factor = np.sqrt(252 * bpd)
                    else:
                        annualization_factor = np.sqrt(252)

                    sharpe = (np.mean(bar_returns) / np.std(bar_returns)) * annualization_factor
                else:
                    sharpe = 0
            else:
                sharpe = 0

            # Annualized return (account for bar frequency)
            test_bars = len(test_values)
            if interval in ['1m', '5m', '15m', '30m', '1h', '4h']:
                # Convert bars to approximate trading days
                bars_per_day = {'1m': 390, '5m': 78, '15m': 26, '30m': 13, '1h': 7, '4h': 2}
                bpd = bars_per_day.get(interval, 7)
                test_days = test_bars / bpd
            else:
                test_days = test_bars

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
                'timeframe': self.current_timeframe,
                'interval': interval,
                'total_bars': total_bars,
                'test_bars': len(test_values),
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

            # Get current market price (not timeframe dependent)
            try:
                ticker = yf.Ticker(symbol)
                current_price = ticker.fast_info.get('lastPrice') or ticker.info.get('regularMarketPrice')
                if current_price is None:
                    raise ValueError("No price available")
                current_price = float(current_price)
                pnl = float(((current_price / entry_price) - 1) * 100)
                total_pnl += pnl
                count += 1

                pending_exit = " ⚠️ SELL" if pos.get('pending_exit') else ""
                message_lines.append(
                    f"{symbol}: {pnl:+.2f}%{pending_exit}\n"
                    f"  Entry: ${entry_price:.2f} ({entry_date})\n"
                    f"  Now: ${current_price:.2f}"
                )
            except Exception:
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

    def handle_timeframe_command(self, reply):
        """
        Handle 'TIMEFRAME SET <TF>' command - changes the operating timeframe

        Args:
            reply: The reply string like "TIMEFRAME SET 15M" or "TIMEFRAME SET 1H"
        """
        parts = reply.split()

        if len(parts) < 3 or parts[1] != "SET":
            self.notifier.send_notification(
                "⚠️ Invalid Format",
                "Usage: TIMEFRAME SET <TF>\n\n"
                "Valid timeframes:\n"
                "  1M  - 1 Minute\n"
                "  5M  - 5 Minutes\n"
                "  15M - 15 Minutes\n"
                "  30M - 30 Minutes\n"
                "  1H  - 1 Hour\n"
                "  4H  - 4 Hours\n"
                "  1D  - 1 Day\n\n"
                "Example: TIMEFRAME SET 15M"
            )
            return

        new_tf = parts[2].upper()

        if new_tf not in self.TIMEFRAMES:
            valid_tfs = ", ".join(self.TIMEFRAMES.keys())
            self.notifier.send_notification(
                "⚠️ Invalid Timeframe",
                f"Unknown timeframe: {new_tf}\n\n"
                f"Valid options: {valid_tfs}"
            )
            return

        old_tf = self.current_timeframe
        self.current_timeframe = new_tf
        tf_config = self.TIMEFRAMES[new_tf]

        # Warn about limitations of shorter timeframes
        warnings = []
        if new_tf in ['1M', '5M']:
            warnings.append("⚠️ Very short timeframes may have insufficient data for ML model warmup")
        if new_tf in ['1M', '5M', '15M', '30M']:
            warnings.append("⚠️ Intraday data limited to ~60 days history")

        warning_text = "\n".join(warnings) if warnings else ""

        message = (
            f"Timeframe changed:\n"
            f"  {old_tf} → {new_tf}\n\n"
            f"Now using: {tf_config['description']} bars\n"
            f"Data period: {tf_config['period']}"
        )

        if warning_text:
            message += f"\n\n{warning_text}"

        self.notifier.send_notification(f"⏱️ Timeframe: {new_tf}", message)
        print(f"✓ Timeframe changed from {old_tf} to {new_tf}")

    def handle_timeframe_query(self):
        """Handle 'TIMEFRAME' query - shows current timeframe and available options"""
        tf_config = self.TIMEFRAMES[self.current_timeframe]

        # Build list of available timeframes
        tf_list = []
        for tf, config in self.TIMEFRAMES.items():
            marker = "→ " if tf == self.current_timeframe else "  "
            tf_list.append(f"{marker}{tf}: {config['description']}")

        message = (
            f"Current: {self.current_timeframe} ({tf_config['description']})\n"
            f"Data period: {tf_config['period']}\n\n"
            f"Available timeframes:\n" +
            "\n".join(tf_list) +
            f"\n\nTo change: TIMEFRAME SET <TF>"
        )

        self.notifier.send_notification("⏱️ Timeframe Settings", message)
        print(f"✓ Sent timeframe query response (current: {self.current_timeframe})")

    def handle_analyze_query(self, reply):
        """
        Handle 'ANALYZE SYMBOL' query - performs deep fundamental analysis of a stock

        Args:
            reply: The reply string like "ANALYZE NVDA"
        """
        parts = reply.split()

        if len(parts) < 2:
            self.notifier.send_notification(
                "⚠️ Invalid Format",
                "Usage: ANALYZE <SYMBOL>\nExample: ANALYZE NVDA"
            )
            return

        symbol = parts[1].upper()

        print(f"📊 Running deep analysis for {symbol}...")

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info or info.get('regularMarketPrice') is None:
                self.notifier.send_notification(
                    f"❌ {symbol}",
                    "Unable to fetch data for this symbol"
                )
                return

            # === BASIC INFO ===
            company_name = info.get('shortName', symbol)
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            current_price = info.get('regularMarketPrice', 0)
            currency = info.get('currency', 'USD')

            # === VALUATION METRICS ===
            pe_trailing = info.get('trailingPE')
            pe_forward = info.get('forwardPE')
            ps_ratio = info.get('priceToSalesTrailing12Months')
            pb_ratio = info.get('priceToBook')
            ev_ebitda = info.get('enterpriseToEbitda')

            # === PROFITABILITY ===
            profit_margin = info.get('profitMargins')
            operating_margin = info.get('operatingMargins')
            roe = info.get('returnOnEquity')
            roa = info.get('returnOnAssets')

            # === GROWTH ===
            revenue_growth = info.get('revenueGrowth')
            earnings_growth = info.get('earningsGrowth')
            earnings_quarterly_growth = info.get('earningsQuarterlyGrowth')

            # === PEG RATIO (calculate if not available) ===
            peg_ratio = info.get('pegRatio')
            if not peg_ratio and pe_trailing and earnings_growth and earnings_growth > 0:
                peg_ratio = pe_trailing / (earnings_growth * 100)

            # === FINANCIAL HEALTH ===
            current_ratio = info.get('currentRatio')
            debt_to_equity = info.get('debtToEquity')
            free_cash_flow = info.get('freeCashflow')
            operating_cash_flow = info.get('operatingCashflow')

            # === DIVIDENDS ===
            dividend_yield = info.get('dividendYield')
            dividend_rate = info.get('dividendRate')  # Annual dividend per share
            payout_ratio = info.get('payoutRatio')

            # === PRICE LEVELS ===
            fifty_two_week_high = info.get('fiftyTwoWeekHigh', 0)
            fifty_two_week_low = info.get('fiftyTwoWeekLow', 0)
            fifty_day_avg = info.get('fiftyDayAverage', 0)
            two_hundred_day_avg = info.get('twoHundredDayAverage', 0)

            # === ANALYST DATA ===
            target_high = info.get('targetHighPrice')
            target_low = info.get('targetLowPrice')
            target_mean = info.get('targetMeanPrice')
            recommendation = info.get('recommendationKey', 'N/A')
            num_analysts = info.get('numberOfAnalystOpinions', 0)

            # === CALCULATE FAIR VALUE ESTIMATES ===
            fair_values = []
            fair_value_details = []

            # Method 1: Graham Number (for value stocks)
            eps = info.get('trailingEps')
            book_value = info.get('bookValue')
            if eps and eps > 0 and book_value and book_value > 0:
                import math
                graham = math.sqrt(22.5 * eps * book_value)
                fair_values.append(graham)
                fair_value_details.append(f"Graham: ${graham:.2f}")

            # Method 2: DCF-lite using FCF yield
            market_cap = info.get('marketCap')
            if free_cash_flow and market_cap and free_cash_flow > 0:
                fcf_yield = free_cash_flow / market_cap
                # Assume 10% required return, calculate implied value
                if fcf_yield > 0.02:  # At least 2% FCF yield
                    dcf_value = current_price * (fcf_yield / 0.08)  # 8% target yield
                    fair_values.append(dcf_value)
                    fair_value_details.append(f"FCF-based: ${dcf_value:.2f}")

            # Method 3: PE-based fair value
            if pe_trailing and pe_trailing > 0 and eps and eps > 0:
                # Use sector average PE or 15 as baseline
                sector_pe = 18 if sector in ['Technology', 'Healthcare'] else 15
                pe_fair_value = eps * sector_pe
                fair_values.append(pe_fair_value)
                fair_value_details.append(f"PE-based: ${pe_fair_value:.2f}")

            # Method 4: Analyst target
            if target_mean:
                fair_values.append(target_mean)
                fair_value_details.append(f"Analyst avg: ${target_mean:.2f}")

            # Calculate average fair value
            if fair_values:
                avg_fair_value = sum(fair_values) / len(fair_values)
                upside = ((avg_fair_value / current_price) - 1) * 100
            else:
                avg_fair_value = None
                upside = None

            # === CALCULATE FAIR PE ===
            fair_pe = None
            if peg_ratio and peg_ratio > 0 and earnings_growth:
                # Fair PE = PEG of 1 * Growth Rate
                growth_rate = abs(earnings_growth) * 100
                fair_pe = growth_rate * 1.0  # PEG = 1

            # === BUILD MESSAGE ===
            lines = [
                f"{company_name}",
                f"{sector} › {industry}",
                f"",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"💰 PRICE",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"   Current        ${current_price:.2f}",
                f"   Year High      ${fifty_two_week_high:.2f}",
                f"   Year Low       ${fifty_two_week_low:.2f}",
            ]

            # Position in 52-week range
            if fifty_two_week_high > fifty_two_week_low:
                range_position = ((current_price - fifty_two_week_low) /
                                  (fifty_two_week_high - fifty_two_week_low)) * 100
                if range_position > 80:
                    range_desc = "Near high"
                elif range_position > 60:
                    range_desc = "Upper"
                elif range_position > 40:
                    range_desc = "Middle"
                elif range_position > 20:
                    range_desc = "Lower"
                else:
                    range_desc = "Near low"
                lines.append(f"   In Range       {range_desc} ({range_position:.0f}%)")

            # Trend info
            if fifty_day_avg and two_hundred_day_avg:
                if current_price > fifty_day_avg > two_hundred_day_avg:
                    trend = "📈 Uptrend"
                elif current_price < fifty_day_avg < two_hundred_day_avg:
                    trend = "📉 Downtrend"
                else:
                    trend = "➡️ Sideways"
                lines.append(f"   Trend          {trend}")

            lines.extend([
                f"",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"📈 VALUATION",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            ])

            if pe_trailing:
                if pe_trailing < 15:
                    pe_desc = "Cheap"
                elif pe_trailing < 25:
                    pe_desc = "Fair"
                elif pe_trailing < 40:
                    pe_desc = "Pricey"
                else:
                    pe_desc = "Expensive"
                lines.append(f"   P/E Ratio      {pe_trailing:.1f}x  ({pe_desc})")
            if pe_forward:
                lines.append(f"   Forward P/E    {pe_forward:.1f}x")
            if fair_pe:
                lines.append(f"   Fair P/E       ~{fair_pe:.0f}x")
            if peg_ratio:
                if peg_ratio < 1:
                    peg_desc = "Undervalued"
                elif peg_ratio < 2:
                    peg_desc = "Fair"
                else:
                    peg_desc = "Overvalued"
                lines.append(f"   PEG Ratio      {peg_ratio:.2f}  ({peg_desc})")
            if pb_ratio:
                if pb_ratio < 1:
                    pb_desc = "Below book"
                elif pb_ratio < 3:
                    pb_desc = "Reasonable"
                else:
                    pb_desc = "Premium"
                lines.append(f"   Price/Book     {pb_ratio:.1f}x  ({pb_desc})")

            lines.extend([
                f"",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"💵 PROFITABILITY",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            ])
            if profit_margin:
                if profit_margin > 0.20:
                    margin_desc = "Excellent"
                elif profit_margin > 0.10:
                    margin_desc = "Good"
                elif profit_margin > 0:
                    margin_desc = "Low"
                else:
                    margin_desc = "Negative"
                lines.append(f"   Profit Margin  {profit_margin*100:.1f}%  ({margin_desc})")
            if roe:
                if roe > 0.20:
                    roe_desc = "Excellent"
                elif roe > 0.10:
                    roe_desc = "Good"
                else:
                    roe_desc = "Poor"
                lines.append(f"   Return/Equity  {roe*100:.1f}%  ({roe_desc})")

            lines.extend([
                f"",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"🚀 GROWTH",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            ])
            if revenue_growth:
                if revenue_growth > 0.25:
                    rev_desc = "Fast"
                elif revenue_growth > 0.10:
                    rev_desc = "Solid"
                elif revenue_growth > 0:
                    rev_desc = "Slow"
                else:
                    rev_desc = "Shrinking"
                lines.append(f"   Revenue        {revenue_growth*100:+.1f}%  ({rev_desc})")
            if earnings_growth:
                if earnings_growth > 0.25:
                    earn_desc = "Strong"
                elif earnings_growth > 0:
                    earn_desc = "Growing"
                else:
                    earn_desc = "Declining"
                lines.append(f"   Earnings       {earnings_growth*100:+.1f}%  ({earn_desc})")

            lines.extend([
                f"",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"🏦 FINANCIAL HEALTH",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            ])
            if current_ratio:
                if current_ratio > 2:
                    cr_desc = "Strong"
                elif current_ratio > 1:
                    cr_desc = "OK"
                else:
                    cr_desc = "Weak"
                lines.append(f"   Liquidity      {current_ratio:.1f}x  ({cr_desc})")
            if debt_to_equity:
                if debt_to_equity < 50:
                    de_desc = "Low"
                elif debt_to_equity < 100:
                    de_desc = "Moderate"
                else:
                    de_desc = "High"
                lines.append(f"   Debt/Equity    {debt_to_equity:.0f}%  ({de_desc})")
            if free_cash_flow:
                fcf_b = free_cash_flow / 1e9
                if free_cash_flow > 0:
                    lines.append(f"   Free Cash      ${fcf_b:.1f}B/yr 👍")
                else:
                    lines.append(f"   Cash Burn      ${abs(fcf_b):.1f}B/yr 👎")

            if dividend_yield:
                lines.extend([
                    f"",
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                    f"💸 DIVIDENDS",
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                    f"   Yield          {dividend_yield*100:.2f}%",
                ])
                if dividend_rate:
                    lines.append(f"   Per Share      ${dividend_rate:.2f}/year")
                    # Calculate quarterly payment
                    quarterly = dividend_rate / 4
                    lines.append(f"                  (${quarterly:.2f}/quarter)")
                if payout_ratio:
                    if payout_ratio < 0.5:
                        payout_desc = "Safe"
                    elif payout_ratio < 0.8:
                        payout_desc = "Moderate"
                    else:
                        payout_desc = "Risky"
                    lines.append(f"   Payout Ratio   {payout_ratio*100:.0f}%  ({payout_desc})")

            lines.extend([
                f"",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"🎯 FAIR VALUE ESTIMATE",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            ])
            if fair_value_details:
                for detail in fair_value_details:
                    # Parse and reformat
                    lines.append(f"   {detail}")
            if avg_fair_value:
                lines.append(f"   ─────────────────────")
                lines.append(f"   Average        ${avg_fair_value:.2f}")
                lines.append(f"   Current        ${current_price:.2f}")
                if upside:
                    if upside > 20:
                        verdict = f"UNDERVALUED {upside:+.0f}% 🟢"
                    elif upside > 5:
                        verdict = f"Slightly under {upside:+.0f}% 🟢"
                    elif upside > -5:
                        verdict = f"Fairly priced 🟡"
                    elif upside > -20:
                        verdict = f"Slightly over {upside:+.0f}% 🟡"
                    else:
                        verdict = f"OVERVALUED {upside:+.0f}% 🔴"
                    lines.append(f"   Verdict        {verdict}")

            lines.extend([
                f"",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"👔 ANALYST OPINIONS ({num_analysts})",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            ])
            if recommendation:
                rec_friendly = {
                    'strongBuy': 'Strong Buy 🟢🟢',
                    'strong_buy': 'Strong Buy 🟢🟢',
                    'buy': 'Buy 🟢',
                    'hold': 'Hold 🟡',
                    'sell': 'Sell 🔴',
                    'strongSell': 'Strong Sell 🔴🔴',
                    'strong_sell': 'Strong Sell 🔴🔴',
                }.get(recommendation, recommendation.replace('_', ' ').title())
                lines.append(f"   Rating         {rec_friendly}")
            if target_mean:
                target_upside = ((target_mean / current_price) - 1) * 100
                lines.append(f"   Target         ${target_mean:.2f}  ({target_upside:+.0f}%)")
            if target_low and target_high:
                lines.append(f"   Range          ${target_low:.2f} - ${target_high:.2f}")

            # === OVERALL ASSESSMENT ===
            lines.extend([
                f"",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"📋 BOTTOM LINE",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            ])

            # Score the stock with user-friendly reasons
            score = 0
            positives = []
            negatives = []

            # Valuation
            if pe_trailing:
                if pe_trailing < 20:
                    score += 1
                    positives.append("Reasonably priced")
                elif pe_trailing > 40:
                    negatives.append("Expensive valuation")

            if peg_ratio:
                if peg_ratio < 1.5:
                    score += 1
                    positives.append("Good value for growth")
                elif peg_ratio > 2.5:
                    negatives.append("Overpriced for growth")

            # Growth
            if revenue_growth and revenue_growth > 0.1:
                score += 1
                positives.append("Growing sales")
            elif revenue_growth and revenue_growth < 0:
                negatives.append("Shrinking revenue")

            if earnings_growth and earnings_growth > 0.1:
                score += 1
                positives.append("Growing profits")
            elif earnings_growth and earnings_growth < 0:
                negatives.append("Declining earnings")

            # Profitability
            if roe and roe > 0.15:
                score += 1
                positives.append("Profitable business")
            if profit_margin and profit_margin > 0.1:
                score += 1
                positives.append("Good margins")
            elif profit_margin and profit_margin < 0:
                negatives.append("Losing money")

            # Financial health
            if current_ratio and current_ratio > 1.5:
                score += 1
                positives.append("Financially stable")
            elif current_ratio and current_ratio < 1:
                negatives.append("Cash flow concerns")

            if debt_to_equity and debt_to_equity < 100:
                score += 1
                positives.append("Low debt")
            elif debt_to_equity and debt_to_equity > 200:
                negatives.append("High debt load")

            # Upside
            if upside and upside > 15:
                score += 1
                positives.append("Looks undervalued")
            elif upside and upside < -20:
                negatives.append("Looks overvalued")

            # Generate assessment
            if score >= 7:
                assessment = "Strong Buy 🟢🟢"
            elif score >= 5:
                assessment = "Buy 🟢"
            elif score >= 3:
                assessment = "Hold 🟡"
            elif score >= 1:
                assessment = "Caution 🔴"
            else:
                assessment = "Avoid 🔴🔴"

            lines.append(f"")
            lines.append(f"   Rating         {assessment}")
            lines.append(f"   Score          {score}/9")

            if positives:
                lines.append(f"")
                lines.append(f"   ✅ Strengths")
                for p in positives[:4]:
                    lines.append(f"      • {p}")

            if negatives:
                lines.append(f"")
                lines.append(f"   ⚠️ Concerns")
                for n in negatives[:3]:
                    lines.append(f"      • {n}")

            message = "\n".join(lines)

            self.notifier.send_notification(f"📊 {symbol} Analysis", message)
            print(f"✓ Sent analysis for {symbol}")

        except Exception as e:
            print(f"❌ Analysis error for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            self.notifier.send_notification(
                f"❌ Analysis Error",
                f"Error analyzing {symbol}:\n{str(e)[:100]}"
            )

    def handle_compare_query(self, reply):
        """
        Handle 'COMPARE SYMBOL1 SYMBOL2 ...' query - compares multiple stocks

        Args:
            reply: The reply string like "COMPARE AAPL MSFT GOOGL"
        """
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        from io import BytesIO

        parts = reply.split()

        if len(parts) < 3:
            self.notifier.send_notification(
                "⚠️ Invalid Format",
                "Usage: COMPARE <SYMBOL1> <SYMBOL2> [SYMBOL3] ...\n"
                "Example: COMPARE AAPL MSFT GOOGL"
            )
            return

        symbols = [s.upper() for s in parts[1:]]

        if len(symbols) > 8:
            self.notifier.send_notification(
                "⚠️ Too Many Stocks",
                "Please compare 8 or fewer stocks at a time."
            )
            return

        print(f"📊 Comparing {len(symbols)} stocks: {', '.join(symbols)}...")

        # Fetch data for all stocks
        stock_data = {}
        failed_symbols = []

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info

                if not info or info.get('regularMarketPrice') is None:
                    failed_symbols.append(symbol)
                    continue

                # Extract key metrics for comparison
                pe_ratio = info.get('trailingPE')
                earnings_growth = info.get('earningsGrowth')

                # Get PEG ratio - calculate if not available
                peg_ratio = info.get('pegRatio')
                if not peg_ratio and pe_ratio and earnings_growth and earnings_growth > 0:
                    # PEG = P/E / (Earnings Growth * 100)
                    peg_ratio = pe_ratio / (earnings_growth * 100)

                # Fetch 1 year of historical data for sparkline chart
                hist = ticker.history(period='1y')
                hist_prices = hist['Close'].values if len(hist) > 0 else None

                stock_data[symbol] = {
                    'name': info.get('shortName', symbol),
                    'price': info.get('regularMarketPrice', 0),
                    'pe_ratio': pe_ratio,
                    'forward_pe': info.get('forwardPE'),
                    'peg_ratio': peg_ratio,
                    'pb_ratio': info.get('priceToBook'),
                    'profit_margin': info.get('profitMargins'),
                    'roe': info.get('returnOnEquity'),
                    'revenue_growth': info.get('revenueGrowth'),
                    'earnings_growth': earnings_growth,
                    'current_ratio': info.get('currentRatio'),
                    'debt_to_equity': info.get('debtToEquity'),
                    'dividend_yield': info.get('dividendYield'),
                    'target_mean': info.get('targetMeanPrice'),
                    'recommendation': info.get('recommendationKey'),
                    'market_cap': info.get('marketCap'),
                    'history': hist_prices,
                }

                print(f"   ✓ Fetched {symbol}")

            except Exception as e:
                print(f"   ❌ Failed to fetch {symbol}: {e}")
                failed_symbols.append(symbol)

        if len(stock_data) < 2:
            self.notifier.send_notification(
                "❌ Comparison Failed",
                f"Need at least 2 valid stocks to compare.\n"
                f"Failed: {', '.join(failed_symbols)}"
            )
            return

        # Calculate upside for each stock
        upside_values = {}
        for s, d in stock_data.items():
            if d['target_mean'] and d['price']:
                upside_values[s] = ((d['target_mean'] / d['price']) - 1) * 100

        # Calculate fair value for each stock
        fair_values = {}
        value_gaps = {}
        for symbol, data in stock_data.items():
            estimates = []
            price = data['price']

            # Method 1: PE-based (using sector-appropriate PE of 18 for tech, 15 otherwise)
            if data['pe_ratio'] and data['pe_ratio'] > 0:
                eps = price / data['pe_ratio']
                fair_pe = 18  # Could vary by sector but keeping it simple
                pe_fair = eps * fair_pe
                estimates.append(pe_fair)

            # Method 2: Analyst target
            if data['target_mean']:
                estimates.append(data['target_mean'])

            # Method 3: PEG-based (fair value at PEG=1)
            if data['peg_ratio'] and data['peg_ratio'] > 0 and data['earnings_growth'] and data['earnings_growth'] > 0:
                # If PEG > 1, stock is overvalued; calculate what price would give PEG=1
                fair_peg_price = price / data['peg_ratio']
                estimates.append(fair_peg_price)

            if estimates and price:
                avg_fair = sum(estimates) / len(estimates)
                fair_values[symbol] = avg_fair
                value_gaps[symbol] = ((avg_fair / price) - 1) * 100  # Positive = undervalued
            else:
                fair_values[symbol] = None
                value_gaps[symbol] = None

        # Find best values FIRST (used for both scoring and highlighting)
        pe_values = {s: d['pe_ratio'] for s, d in stock_data.items() if d['pe_ratio'] and d['pe_ratio'] > 0}
        peg_values = {s: d['peg_ratio'] for s, d in stock_data.items() if d['peg_ratio'] and d['peg_ratio'] > 0}
        margin_values = {s: d['profit_margin'] for s, d in stock_data.items() if d['profit_margin']}
        roe_values = {s: d['roe'] for s, d in stock_data.items() if d['roe']}
        rev_values = {s: d['revenue_growth'] for s, d in stock_data.items() if d['revenue_growth']}
        earn_values = {s: d['earnings_growth'] for s, d in stock_data.items() if d['earnings_growth']}
        de_values = {s: d['debt_to_equity'] for s, d in stock_data.items() if d['debt_to_equity'] is not None}
        div_values = {s: d['dividend_yield'] for s, d in stock_data.items() if d['dividend_yield']}
        gap_values = {s: v for s, v in value_gaps.items() if v is not None}

        best_pe = min(pe_values.values()) if pe_values else None
        best_peg = min(peg_values.values()) if peg_values else None
        best_margin = max(margin_values.values()) if margin_values else None
        best_roe = max(roe_values.values()) if roe_values else None
        best_rev = max(rev_values.values()) if rev_values else None
        best_earn = max(earn_values.values()) if earn_values else None
        best_de = min(de_values.values()) if de_values else None
        best_div = max(div_values.values()) if div_values else None
        best_upside = max(upside_values.values()) if upside_values else None
        best_gap = max(gap_values.values()) if gap_values else None  # Most undervalued

        # Score each stock - now includes bonus for being BEST in each category
        # Base scoring uses thresholds, but being BEST in category adds bonus points
        scores = {}
        for symbol, data in stock_data.items():
            score = 0

            # P/E Ratio (lower is better)
            if data['pe_ratio'] and data['pe_ratio'] > 0:
                if data['pe_ratio'] < 15:
                    score += 2
                elif data['pe_ratio'] < 25:
                    score += 1
                # BONUS: Best P/E among compared stocks
                if best_pe and data['pe_ratio'] == best_pe:
                    score += 1

            # PEG Ratio (lower is better)
            if data['peg_ratio'] and data['peg_ratio'] > 0:
                if data['peg_ratio'] < 1:
                    score += 2
                elif data['peg_ratio'] < 1.5:
                    score += 1
                # BONUS: Best PEG among compared stocks
                if best_peg and data['peg_ratio'] == best_peg:
                    score += 1

            # Profit Margin (higher is better)
            if data['profit_margin'] and data['profit_margin'] > 0.15:
                score += 2
            elif data['profit_margin'] and data['profit_margin'] > 0.08:
                score += 1
            # BONUS: Best margin among compared stocks
            if data['profit_margin'] and best_margin and data['profit_margin'] == best_margin:
                score += 1

            # ROE (higher is better)
            if data['roe'] and data['roe'] > 0.18:
                score += 2
            elif data['roe'] and data['roe'] > 0.10:
                score += 1
            # BONUS: Best ROE among compared stocks
            if data['roe'] and best_roe and data['roe'] == best_roe:
                score += 1

            # Revenue Growth (higher is better)
            if data['revenue_growth'] and data['revenue_growth'] > 0.15:
                score += 2
            elif data['revenue_growth'] and data['revenue_growth'] > 0.05:
                score += 1
            # BONUS: Best revenue growth among compared stocks
            if data['revenue_growth'] and best_rev and data['revenue_growth'] == best_rev:
                score += 1

            # Earnings Growth (higher is better)
            if data['earnings_growth'] and data['earnings_growth'] > 0.15:
                score += 2
            elif data['earnings_growth'] and data['earnings_growth'] > 0.05:
                score += 1
            # BONUS: Best earnings growth among compared stocks
            if data['earnings_growth'] and best_earn and data['earnings_growth'] == best_earn:
                score += 1

            if data['current_ratio'] and data['current_ratio'] > 1.5:
                score += 1

            # Debt/Equity (lower is better)
            if data['debt_to_equity'] and data['debt_to_equity'] < 80:
                score += 1
            # BONUS: Best (lowest) debt among compared stocks
            if data['debt_to_equity'] is not None and best_de is not None and data['debt_to_equity'] == best_de:
                score += 1

            if data['recommendation'] in ['strongBuy', 'strong_buy', 'buy']:
                score += 1

            # Analyst Upside - BONUS for best upside
            if symbol in upside_values and best_upside and upside_values[symbol] == best_upside:
                score += 1

            # Value gap scoring (undervalued = good)
            if value_gaps.get(symbol) is not None:
                gap = value_gaps[symbol]
                if gap > 15:
                    score += 2  # Significantly undervalued
                elif gap > 5:
                    score += 1  # Slightly undervalued
                # BONUS: Best (most undervalued) value gap among compared stocks
                if best_gap and gap == best_gap:
                    score += 1

            scores[symbol] = score

        # Find winner
        winner = max(scores, key=scores.get)
        winner_score = scores[winner]

        # Calculate max possible score (dynamic based on categories)
        max_score = 16 + 10  # Base 16 + up to 10 bonus points for being best

        # Build table data
        symbols_list = list(stock_data.keys())
        metrics = [
            'Price',
            'Fair Value',
            'Value Gap',
            'P/E Ratio',
            'PEG Ratio',
            'Profit Margin',
            'Return on Equity',
            'Revenue Growth',
            'Earnings Growth',
            'Debt/Equity',
            'Dividend Yield',
            'Analyst Upside',
            'Analyst Rating',
            'SCORE',
        ]

        # Create cell data and colors
        cell_data = []
        cell_colors = []

        # Colors
        header_color = '#2C3E50'
        best_color = '#27AE60'
        winner_color = '#F39C12'
        normal_color = '#FFFFFF'
        alt_color = '#F8F9FA'

        for i, metric in enumerate(metrics):
            row_data = []
            row_colors = []
            bg = alt_color if i % 2 == 0 else normal_color

            for symbol in symbols_list:
                data = stock_data[symbol]
                is_best = False
                val = "N/A"

                if metric == 'Price':
                    val = f"${data['price']:.2f}" if data['price'] else "N/A"
                elif metric == 'Fair Value':
                    if fair_values.get(symbol):
                        val = f"${fair_values[symbol]:.2f}"
                    else:
                        val = "N/A"
                elif metric == 'Value Gap':
                    if value_gaps.get(symbol) is not None:
                        gap = value_gaps[symbol]
                        if gap > 0:
                            val = f"+{gap:.0f}%"  # Undervalued
                        else:
                            val = f"{gap:.0f}%"  # Overvalued
                        is_best = best_gap and value_gaps[symbol] == best_gap
                elif metric == 'P/E Ratio':
                    if data['pe_ratio']:
                        val = f"{data['pe_ratio']:.1f}"
                        is_best = best_pe and data['pe_ratio'] == best_pe
                elif metric == 'PEG Ratio':
                    if data['peg_ratio'] and data['peg_ratio'] > 0:
                        val = f"{data['peg_ratio']:.2f}"
                        is_best = best_peg and data['peg_ratio'] == best_peg
                elif metric == 'Profit Margin':
                    if data['profit_margin']:
                        val = f"{data['profit_margin']*100:.1f}%"
                        is_best = best_margin and data['profit_margin'] == best_margin
                elif metric == 'Return on Equity':
                    if data['roe']:
                        val = f"{data['roe']*100:.1f}%"
                        is_best = best_roe and data['roe'] == best_roe
                elif metric == 'Revenue Growth':
                    if data['revenue_growth']:
                        val = f"{data['revenue_growth']*100:+.1f}%"
                        is_best = best_rev and data['revenue_growth'] == best_rev
                elif metric == 'Earnings Growth':
                    if data['earnings_growth']:
                        val = f"{data['earnings_growth']*100:+.1f}%"
                        is_best = best_earn and data['earnings_growth'] == best_earn
                elif metric == 'Debt/Equity':
                    if data['debt_to_equity'] is not None:
                        val = f"{data['debt_to_equity']:.0f}%"
                        is_best = best_de is not None and data['debt_to_equity'] == best_de
                elif metric == 'Dividend Yield':
                    if data['dividend_yield']:
                        val = f"{data['dividend_yield']*100:.2f}%"
                        is_best = best_div and data['dividend_yield'] == best_div
                    else:
                        val = "—"
                elif metric == 'Analyst Upside':
                    if symbol in upside_values:
                        val = f"{upside_values[symbol]:+.1f}%"
                        is_best = best_upside and upside_values[symbol] == best_upside
                elif metric == 'Analyst Rating':
                    rec = data['recommendation']
                    if rec:
                        val = {
                            'strongBuy': 'Strong Buy',
                            'strong_buy': 'Strong Buy',
                            'buy': 'Buy',
                            'hold': 'Hold',
                            'sell': 'Sell',
                            'strongSell': 'Strong Sell',
                            'strong_sell': 'Strong Sell',
                        }.get(rec, rec.replace('_', ' ').title())
                elif metric == 'SCORE':
                    val = f"{scores[symbol]}"
                    is_best = symbol == winner

                row_data.append(val)
                if metric == 'SCORE' and symbol == winner:
                    row_colors.append(winner_color)
                elif is_best:
                    row_colors.append(best_color)
                else:
                    row_colors.append(bg)

            cell_data.append(row_data)
            cell_colors.append(row_colors)

        # Create figure with GridSpec for table + sparkline charts
        from matplotlib.gridspec import GridSpec
        import numpy as np

        fig_width = 3 + len(symbols_list) * 1.8
        fig_height = 1 + len(metrics) * 0.4 + 3.5  # Extra space for charts and winner

        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = GridSpec(3, 1, height_ratios=[6, 1.5, 1], hspace=0.15)

        # Top section: Table
        ax_table = fig.add_subplot(gs[0])
        ax_table.axis('off')

        # Create table
        table = ax_table.table(
            cellText=cell_data,
            rowLabels=metrics,
            colLabels=symbols_list,
            cellColours=cell_colors,
            rowColours=['#ECF0F1'] * len(metrics),
            colColours=[header_color] * len(symbols_list),
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)

        # Style header cells
        for j in range(len(symbols_list)):
            cell = table[(0, j)]
            cell.set_text_props(weight='bold', color='white')
            cell.set_fontsize(12)

        # Style row labels
        for i in range(len(metrics)):
            cell = table[(i + 1, -1)]
            cell.set_text_props(weight='bold')
            if metrics[i] == 'SCORE':
                cell.set_facecolor('#34495E')
                cell.set_text_props(weight='bold', color='white')

        # Style SCORE row
        for j in range(len(symbols_list)):
            cell = table[(len(metrics), j)]
            cell.set_text_props(weight='bold')

        # Add title
        plt.suptitle('Stock Comparison', fontsize=16, fontweight='bold', y=0.98)

        # Middle section: Sparkline charts (1 year price history)
        ax_charts = fig.add_subplot(gs[1])
        ax_charts.axis('off')

        # Create mini subplots for each stock's sparkline
        num_stocks = len(symbols_list)
        chart_axes = []

        # Calculate positions for mini charts (evenly spaced across the width)
        chart_width = 0.8 / num_stocks
        chart_left_margin = 0.15  # Account for row labels

        for idx, symbol in enumerate(symbols_list):
            # Position each mini chart
            left = chart_left_margin + idx * (0.85 / num_stocks)
            bottom = 0.35  # Position within the middle section
            width = 0.8 / num_stocks - 0.02
            height = 0.55

            # Create inset axes for each sparkline
            ax_spark = fig.add_axes([left, bottom * 0.28 + 0.22, width, height * 0.12])
            chart_axes.append(ax_spark)

            data = stock_data[symbol]
            hist = data.get('history')

            if hist is not None and len(hist) > 10:
                # Normalize to percentage change from start
                prices = np.array(hist)
                pct_change = ((prices / prices[0]) - 1) * 100

                # Determine color based on overall trend
                if prices[-1] > prices[0]:
                    line_color = '#27AE60'  # Green for up
                    fill_color = '#27AE6030'
                else:
                    line_color = '#E74C3C'  # Red for down
                    fill_color = '#E74C3C30'

                # Plot sparkline
                ax_spark.plot(pct_change, color=line_color, linewidth=1.5)
                ax_spark.fill_between(range(len(pct_change)), pct_change, 0,
                                      color=fill_color, alpha=0.3)
                ax_spark.axhline(y=0, color='#888888', linewidth=0.5, linestyle='-')

                # Add YTD return label
                ytd_return = pct_change[-1]
                ax_spark.text(0.5, -0.25, f"{ytd_return:+.0f}% (1Y)",
                             transform=ax_spark.transAxes,
                             fontsize=8, ha='center', va='top',
                             color=line_color, fontweight='bold')
            else:
                ax_spark.text(0.5, 0.5, 'No data', transform=ax_spark.transAxes,
                             fontsize=8, ha='center', va='center', color='#888888')

            # Clean up axes
            ax_spark.set_xlim(0, len(hist) if hist is not None and len(hist) > 0 else 1)
            ax_spark.set_xticks([])
            ax_spark.set_yticks([])
            for spine in ax_spark.spines.values():
                spine.set_visible(False)

        # Add "1Y Chart" label on the left
        fig.text(0.08, 0.30, '1Y Chart', fontsize=10, fontweight='bold',
                 ha='center', va='center', rotation=0,
                 bbox=dict(boxstyle='round', facecolor='#ECF0F1', edgecolor='none'))

        # Bottom section: Winner announcement
        ax_winner = fig.add_subplot(gs[2])
        ax_winner.axis('off')

        winner_data = stock_data[winner]
        winner_text = f"🏆 WINNER: {winner} ({winner_data['name']})\nScore: {winner_score} points"

        # Why it won
        reasons = []
        if winner_data['pe_ratio'] and winner_data['pe_ratio'] < 20:
            reasons.append("Reasonable valuation")
        if winner_data['profit_margin'] and winner_data['profit_margin'] > 0.15:
            reasons.append("High margins")
        if winner_data['roe'] and winner_data['roe'] > 0.15:
            reasons.append("Strong returns")
        if winner_data['revenue_growth'] and winner_data['revenue_growth'] > 0.10:
            reasons.append("Growing revenue")
        if winner_data['debt_to_equity'] and winner_data['debt_to_equity'] < 80:
            reasons.append("Low debt")
        if winner in upside_values and upside_values[winner] > 15:
            reasons.append("High upside")
        if value_gaps.get(winner) and value_gaps[winner] > 10:
            reasons.append("Undervalued")

        if reasons:
            winner_text += "\nStrengths: " + ", ".join(reasons[:3])

        ax_winner.text(0.5, 0.5, winner_text, transform=ax_winner.transAxes,
                       fontsize=12, ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor=winner_color, alpha=0.3))

        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        plt.close(fig)

        # Build text message (for notification body)
        message = f"Comparing: {', '.join(symbols_list)}\n\n"
        message += f"🏆 Winner: {winner} ({winner_data['name']})\n"
        message += f"Score: {winner_score} points"
        if reasons:
            message += f"\nStrengths: {', '.join(reasons[:3])}"

        if failed_symbols:
            message += f"\n\n⚠️ Failed to fetch: {', '.join(failed_symbols)}"

        # Send with image
        self.notifier.send_notification_with_image(
            f"📊 {' vs '.join(symbols_list)}",
            message,
            buf,
            "comparison.png"
        )
        print(f"✓ Sent comparison for {len(stock_data)} stocks. Winner: {winner}")