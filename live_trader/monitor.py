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

        # Import chart generator
        from chart_generator import ChartGenerator
        self.chart_gen = ChartGenerator(strategy_loader, strategy_params)

        # Start continuous reply listener
        self.notifier.start_listening(self._handle_reply)

    def get_live_data(self, symbol, period="1y"):
        """Fetch live data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval="1d")

            if df.empty:
                return None

            df.index = df.index.tz_localize(None)
            return df

        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
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

                df = self.get_live_data(symbol)
                if df is None or len(df) < 200:
                    continue

                buy_signal = self.strategy.get_entry_signal(df, self.params)

                if buy_signal['signal']:
                    self.notifier.send_buy_alert(symbol, buy_signal)
                    self.buy_alerts_sent[symbol] = today  # Mark as alerted today
                    buy_opportunities += 1

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
        Handle 'LAST SYMBOL' query - shows last signal for a stock with chart

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

        # Chart generator has its own warmup calculation
        chart_days = 30
        warmup_needed = self.chart_gen.warmup_days
        total_days_needed = warmup_needed + chart_days

        # Convert trading days to calendar days with buffer
        calendar_days = int(total_days_needed * 1.6)

        # Get data (yfinance period must be enough)
        if calendar_days <= 90:
            period = "3mo"
        elif calendar_days <= 180:
            period = "6mo"
        else:
            period = "1y"

        df = self.get_live_data(symbol, period=period)

        if df is None or len(df) < warmup_needed:
            self.notifier.send_notification(
                f"❌ {symbol}",
                f"Unable to fetch sufficient data (need {warmup_needed} days for indicators)"
            )
            return

        # Generate chart
        print(f"📊 Generating chart for {symbol} (warmup: {warmup_needed} days, display: {chart_days} days)...")
        chart_buffer = self.chart_gen.generate_chart(symbol, df, days=chart_days)

        # Check if we're currently holding it
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
            # Not holding - check for recent signals
            buy_signal_found = False
            for i in range(min(5, len(df))):
                idx = -(i + 1)
                current_df = df.iloc[:len(df) + idx + 1]

                if len(current_df) < warmup_needed:
                    continue

                signal = self.strategy.get_entry_signal(current_df, self.params)

                if signal['signal']:
                    signal_date = df.index[idx].strftime('%Y-%m-%d')
                    signal_price = signal['price']
                    current_price = df['Close'].iloc[-1]
                    pnl = ((current_price / signal_price) - 1) * 100
                    days_ago = i

                    title = f"📊 {symbol} - Last Signal"
                    message = (
                        f"Signal: BUY\n"
                        f"Date: {signal_date} ({days_ago} day(s) ago)\n"
                        f"Price then: ${signal_price:.2f}\n"
                        f"Price now: ${current_price:.2f}\n"
                        f"Change: {pnl:+.2f}%\n\n"
                        f"⚠️ Not currently holding"
                    )
                    buy_signal_found = True
                    break

            if not buy_signal_found:
                title = f"📊 {symbol} - Last Signal"
                message = "No recent signals in the last 5 days"

        # Send with chart
        if chart_buffer:
            self.notifier.send_notification_with_image(
                title, message, chart_buffer, f"{symbol}_chart.png"
            )
            print(f"✓ Sent last signal info with chart for {symbol}")
        else:
            # Fallback to text only
            self.notifier.send_notification(title, message)
            print(f"✓ Sent last signal info for {symbol} (chart generation failed)")

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