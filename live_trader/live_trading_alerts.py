# live_trading_alerts.py
"""
Live Trading Alert System - REFACTORED
Scans stocks in real-time and sends notifications for buy/sell opportunities
Uses Pushbullet for notifications and reply-based position management
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import importlib.util
from pushbullet import Pushbullet
from positions import PositionManager

# ============================================================================
# CONFIGURATION
# ============================================================================
# Pushbullet API key (get from https://www.pushbullet.com/#settings/account)
PUSHBULLET_API_KEY = "o.ptYJ8W8YpFEnDVZ1CL4vO9N7suOvJURG"

# Testing mode
TESTING_MODE = False  # Set to True for simulated backtesting
TEST_START_DATE = "2024-06-01"
TEST_END_DATE = "2024-12-31"
TEST_SPEED = 10  # Days to simulate per second (higher = faster, 0 = instant)

# Strategy configuration
STRATEGY_MODULE = "../strategies/SMA_ATR_POS_MOM/sma_atr"
STRATEGY_CLASS = "Strategy"
STRATEGY_PARAMS = {
    'fast_len': 7,
    'slow_len': 50,
    'trend_len': 200,
    'atr_len': 10,
    'atr_mult': 3.0,
    'stop_loss_pct': 0.1,
    'verbose': False
}

# Watchlist
WATCHLIST_FILE = "../strategies/nasdaq_2025.csv"

# Data lookback period (default minimum)
DATA_LOOKBACK_DAYS = 300

# Scan interval (minutes) - only used in live mode
SCAN_INTERVAL = 15


# ============================================================================
# AUTO-CALCULATE WARMUP PERIOD FROM STRATEGY
# ============================================================================
def calculate_warmup_days(strategy_params):
    """
    Automatically calculate required warmup days from strategy parameters
    Looks for common parameter names and adds safety buffer

    Returns: Recommended warmup days
    """
    max_period = 0

    # Common parameter names for lookback periods
    lookback_params = [
        'trend_len', 'slow_len', 'fast_len', 'atr_len',
        'ma_period', 'sma_period', 'ema_period', 'rsi_period',
        'bb_period', 'macd_slow', 'lookback', 'period', 'length'
    ]

    for param_name, param_value in strategy_params.items():
        if any(key in param_name.lower() for key in lookback_params):
            if isinstance(param_value, (int, float)):
                max_period = max(max_period, int(param_value))

    if max_period > 0:
        # Add 50% buffer for indicator stabilization
        recommended = int(max_period * 1.5)
        return max(recommended, DATA_LOOKBACK_DAYS)  # Use at least the default
    else:
        return DATA_LOOKBACK_DAYS

# Calculate actual warmup to use (done at module load time)
ACTUAL_WARMUP_DAYS = calculate_warmup_days(STRATEGY_PARAMS)
print(f"ℹ️  Warmup period: {ACTUAL_WARMUP_DAYS} days (auto-calculated from strategy parameters)")
# ============================================================================


class PushbulletNotifier:
    """Send notifications via Pushbullet and handle user replies"""

    def __init__(self, api_key):
        self.pb = Pushbullet(api_key)
        self.last_check_time = time.time()
        print(f"✓ Pushbullet initialized")

    def send_notification(self, title, message):
        """Send push notification"""
        try:
            push = self.pb.push_note(title, message)
            print(f"✓ Notification sent: {title}")
            return True
        except Exception as e:
            print(f"❌ Notification failed: {e}")
            return False

    def check_for_replies(self):
        """
        Check for new pushes (replies from user)

        Returns:
            list: List of reply strings from user
        """
        try:
            pushes = self.pb.get_pushes(modified_after=self.last_check_time)
            self.last_check_time = time.time()

            replies = []
            for push in pushes:
                # Only look at pushes we received (not ones we sent)
                if push.get('direction') == 'self' and push.get('body'):
                    body = push.get('body', '').strip().upper()
                    replies.append(body)

            return replies

        except Exception as e:
            print(f"❌ Error checking replies: {e}")
            return []


class StrategyLoader:
    """Dynamically load any backtrader strategy"""

    def __init__(self, module_name, class_name):
        self.module_name = module_name
        self.class_name = class_name
        self.strategy_class = self._load_strategy()

    def _load_strategy(self):
        """Load strategy class from module"""
        try:
            spec = importlib.util.spec_from_file_location(
                self.module_name,
                f"{self.module_name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            strategy_class = getattr(module, self.class_name)
            print(f"✓ Loaded strategy: {self.class_name} from {self.module_name}.py")
            return strategy_class

        except Exception as e:
            print(f"❌ Failed to load strategy: {e}")
            raise

    def get_entry_signal(self, df, params):
        """
        Extract entry logic from strategy
        Returns: {'signal': bool, 'price': float, 'stop_loss': float, 'atr': float}
        """
        indicators = self._calculate_indicators(df, params)

        # Check for crossover (common pattern)
        if 'fast_sma' in indicators and 'slow_sma' in indicators:
            fast = indicators['fast_sma']
            slow = indicators['slow_sma']

            # Crossover detected
            if fast.iloc[-1] > slow.iloc[-1] and fast.iloc[-2] <= slow.iloc[-2]:
                # Check trend filter if present
                if 'trend_ma' in indicators:
                    trend = indicators['trend_ma']
                    if trend.iloc[-1] <= trend.iloc[-2]:  # Declining trend
                        return {'signal': False}

                # Calculate stop loss
                close = df['Close'].iloc[-1]
                atr = indicators.get('atr', pd.Series([0])).iloc[-1]

                atr_mult = params.get('atr_mult', 3.0)
                stop_pct = params.get('stop_loss_pct', 0.1)

                atr_stop = close - atr_mult * atr if atr > 0 else close * (1 - stop_pct)
                pct_stop = close * (1 - stop_pct)
                stop_loss = max(atr_stop, pct_stop)

                return {
                    'signal': True,
                    'price': close,
                    'stop_loss': stop_loss,
                    'atr': atr
                }

        return {'signal': False}

    def get_exit_signal(self, df, params, entry_price, current_stop):
        """
        Extract exit logic from strategy
        Returns: {'signal': bool, 'price': float, 'stop_type': str, 'new_stop': float}
        """
        indicators = self._calculate_indicators(df, params)

        close = df['Close'].iloc[-1]
        atr = indicators.get('atr', pd.Series([0])).iloc[-1]

        # Update trailing stop
        atr_mult = params.get('atr_mult', 3.0)
        stop_pct = params.get('stop_loss_pct', 0.1)

        atr_stop = close - atr_mult * atr if atr > 0 else entry_price * (1 - stop_pct)
        pct_stop = entry_price * (1 - stop_pct)
        new_stop = max(atr_stop, pct_stop, current_stop)

        # Check if stop hit
        if close <= new_stop:
            return {
                'signal': True,
                'price': close,
                'stop_type': 'STOP',
                'new_stop': new_stop
            }

        return {'signal': False, 'new_stop': new_stop}

    def _calculate_indicators(self, df, params):
        """Calculate common indicators"""
        indicators = {}

        # SMAs
        if 'fast_len' in params:
            indicators['fast_sma'] = df['Close'].rolling(window=params['fast_len']).mean()
        if 'slow_len' in params:
            indicators['slow_sma'] = df['Close'].rolling(window=params['slow_len']).mean()
        if 'trend_len' in params:
            indicators['trend_ma'] = df['Close'].rolling(window=params['trend_len']).mean()

        # ATR
        if 'atr_len' in params:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            indicators['atr'] = true_range.rolling(window=params['atr_len']).mean()

        return indicators


class LiveTradingMonitor:
    """Monitor stocks and send notifications for trading opportunities"""

    def __init__(self, watchlist, strategy_loader, strategy_params, notifier):
        self.watchlist = watchlist
        self.strategy = strategy_loader
        self.params = strategy_params
        self.notifier = notifier
        self.position_manager = PositionManager()

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

    def get_historical_data(self, symbol, start_date, end_date, warmup_days=None):
        """
        Fetch historical data for testing mode with warmup period

        Args:
            symbol: Stock ticker
            start_date: Start of test period
            end_date: End of test period
            warmup_days: TRADING days of data to fetch before start_date for indicator warmup
                        If None, uses ACTUAL_WARMUP_DAYS

        Returns:
            DataFrame with data from (start_date - warmup_days) to end_date
        """
        try:
            if warmup_days is None:
                warmup_days = ACTUAL_WARMUP_DAYS

            start = pd.to_datetime(start_date)

            # Convert trading days to calendar days (roughly 1.4x due to weekends/holidays)
            # Add extra buffer to ensure we get enough trading days
            calendar_days = int(warmup_days * 1.6)  # ~60% extra for safety
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

        # First, check for user replies
        self.handle_user_replies()

        # Get list of held positions
        held_positions = set(self.position_manager.list_all().keys())
        print(f"Currently holding {len(held_positions)} positions: {list(held_positions)}")

        # Scan for BUY opportunities (only symbols we DON'T hold)
        buy_opportunities = 0
        for symbol in self.watchlist:
            try:
                if symbol in held_positions:
                    # Skip - we'll check for exits separately
                    continue

                # Get live data
                df = self.get_live_data(symbol)
                if df is None:
                    print(f"⚠️  No data for {symbol}")
                    continue

                if len(df) < 200:  # Need enough data for indicators
                    print(f"⚠️  Insufficient data for {symbol}: {len(df)} days")
                    continue

                # Check for buy signal
                buy_signal = self.strategy.get_entry_signal(df, self.params)

                if buy_signal['signal']:
                    self.send_buy_alert(symbol, buy_signal)
                    buy_opportunities += 1

            except Exception as e:
                print(f"❌ Error scanning {symbol} for buy: {e}")
                import traceback
                traceback.print_exc()

        print(f"\nFound {buy_opportunities} buy opportunities")

        # Scan ONLY held positions for SELL opportunities
        if held_positions:
            print(f"\nChecking {len(held_positions)} held positions for exits...")
            for symbol in held_positions:
                try:
                    self.check_exit(symbol)
                except Exception as e:
                    print(f"❌ Error checking exit for {symbol}: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print("\nNo positions to check for exits")

    def handle_user_replies(self):
        """Process user replies from Pushbullet"""
        replies = self.notifier.check_for_replies()

        if replies:
            print(f"\n💬 Processing {len(replies)} reply(ies)...")

        for reply in replies:
            # Parse "BOUGHT SYMBOL" or "BOUGHT SYMBOL AT PRICE"
            if reply.startswith("BOUGHT "):
                self.handle_bought_reply(reply)

            # Parse "SOLD SYMBOL"
            elif reply.startswith("SOLD "):
                self.handle_sold_reply(reply)

            else:
                print(f"⚠️  Unknown reply format: {reply}")

    def handle_bought_reply(self, reply):
        """
        Handle 'BOUGHT SYMBOL' or 'BOUGHT SYMBOL AT PRICE' reply

        Args:
            reply: The reply string from user
        """
        parts = reply.split()

        if len(parts) < 2:
            print(f"⚠️  Invalid BOUGHT format: {reply}")
            return

        symbol = parts[1].upper()

        # Check if already holding
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
            # Send confirmation notification
            risk_pct = ((price - stop_loss) / price) * 100
            title = f"✅ Position Added: {symbol}"
            message = (
                f"Entry: ${price:.2f}\n"
                f"Stop Loss: ${stop_loss:.2f}\n"
                f"Risk: {risk_pct:.2f}%\n"
                f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            self.notifier.send_notification(title, message)
            print(f"✅ Auto-added position from reply: {symbol} @ ${price:.2f}")

    def handle_sold_reply(self, reply):
        """
        Handle 'SOLD SYMBOL' reply

        Args:
            reply: The reply string from user
        """
        parts = reply.split()

        if len(parts) < 2:
            print(f"⚠️  Invalid SOLD format: {reply}")
            return

        symbol = parts[1].upper()

        position = self.position_manager.remove(symbol)

        if position:
            # Send confirmation notification
            title = f"🗑️ Position Removed: {symbol}"
            message = f"Entry was: ${position['entry_price']:.2f}"
            self.notifier.send_notification(title, message)
            print(f"✅ Auto-removed position from reply: {symbol}")

    def check_exit(self, symbol):
        """Check if held position should be exited"""
        position = self.position_manager.get(symbol)
        if not position:
            return

        entry_price = position['entry_price']
        current_stop = position['stop_loss']

        # Get live data
        df = self.get_live_data(symbol)
        if df is None:
            return

        # Check for sell signal using strategy
        sell_signal = self.strategy.get_exit_signal(df, self.params, entry_price, current_stop)

        if sell_signal['signal']:
            self.send_sell_alert(symbol, sell_signal, entry_price)
            self.position_manager.remove(symbol)
        else:
            # Update trailing stop
            self.position_manager.update_stop(symbol, sell_signal['new_stop'])

    def send_buy_alert(self, symbol, signal):
        """Send buy opportunity notification"""
        risk_pct = ((signal['price'] - signal['stop_loss']) / signal['price']) * 100
        title = f"🟢 BUY {symbol}"
        message = (
            f"Price: ${signal['price']:.2f}\n"
            f"Stop Loss: ${signal['stop_loss']:.2f}\n"
            f"Risk: {risk_pct:.2f}%\n\n"
            f"Reply: BOUGHT {symbol}\n"
            f"Or: BOUGHT {symbol} AT <price>"
        )
        self.notifier.send_notification(title, message)
        print(f"🟢 BUY ALERT: {symbol} @ ${signal['price']:.2f}, SL @ ${signal['stop_loss']:.2f}")

    def send_sell_alert(self, symbol, signal, entry_price):
        """Send sell alert notification"""
        pnl = ((signal['price'] / entry_price) - 1) * 100
        title = f"🔴 SELL {symbol}"
        message = (
            f"Price: ${signal['price']:.2f}\n"
            f"Entry: ${entry_price:.2f}\n"
            f"P&L: {pnl:+.2f}%\n"
            f"Type: {signal['stop_type']}\n\n"
            f"Reply: SOLD {symbol}"
        )
        self.notifier.send_notification(title, message)
        print(f"🔴 SELL ALERT: {symbol} @ ${signal['price']:.2f}, P&L: {pnl:+.2f}%")

    def is_market_hours(self):
        """Check if market is open (US Eastern Time)"""
        now = datetime.now()

        # Weekend
        if now.weekday() >= 5:
            return False

        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = now.replace(hour=9, minute=30, second=0)
        market_close = now.replace(hour=16, minute=0, second=0)

        return market_open <= now <= market_close

    def run(self):
        """Main loop - scan continuously during market hours"""
        print(f"\n🚀 Live Trading Monitor Started")
        print(f"Strategy: {STRATEGY_CLASS} from {STRATEGY_MODULE}.py")
        print(f"Watching {len(self.watchlist)} symbols")
        print(f"Active positions: {self.position_manager.get_summary()['count']}")
        print(f"Scan interval: {SCAN_INTERVAL} minutes\n")

        while True:
            try:
                if self.is_market_hours():
                    self.scan_for_opportunities()
                    print(f"\n💤 Next scan in {SCAN_INTERVAL} minutes...")
                    time.sleep(SCAN_INTERVAL * 60)
                else:
                    print("Market closed. Sleeping...")
                    time.sleep(300)

            except KeyboardInterrupt:
                print("\n👋 Monitor stopped")
                break
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                time.sleep(60)

    def run_test_mode(self, start_date, end_date, speed=1):
        """
        Run in testing mode - simulates live trading from historical data

        The key improvement: Fetches warmup data BEFORE the test period so indicators
        are ready from day 1 of testing. This works for any strategy with any lookback period.

        Args:
            start_date: Date to start simulation (YYYY-MM-DD) - signals generated from this date
            end_date: Date to end simulation (YYYY-MM-DD)
            speed: Days to simulate per second (1 = 1 day/sec, 0 = instant)
        """
        print(f"\n🧪 TESTING MODE")
        print(f"Strategy: {STRATEGY_CLASS} from {STRATEGY_MODULE}.py")
        print(f"Watching {len(self.watchlist)} symbols")
        print(f"Test Period: {start_date} to {end_date}")
        print(f"Warmup Required: {ACTUAL_WARMUP_DAYS} trading days")
        print(f"Speed: {speed} day(s) per second" if speed > 0 else "Speed: Instant (no delay)")
        print()

        test_start = pd.to_datetime(start_date)
        test_end = pd.to_datetime(end_date)

        # Calculate actual data fetch start (with extra buffer for weekends/holidays)
        calendar_days = int(ACTUAL_WARMUP_DAYS * 1.6)
        data_start = test_start - timedelta(days=calendar_days)
        print(f"Fetching from {data_start.strftime('%Y-%m-%d')} (~{calendar_days} calendar days for {ACTUAL_WARMUP_DAYS} trading days)")
        print(f"Signals will be generated from {start_date} onwards\n")

        # Fetch all historical data WITH warmup period
        stock_data = {}
        for symbol in self.watchlist:
            df = self.get_historical_data(symbol, start_date, end_date, ACTUAL_WARMUP_DAYS)

            if df is not None:
                # Verify we have enough TRADING data for warmup
                data_before_test = df[df.index < test_start]
                trading_days_before = len(data_before_test)

                # We need at least 80% of the requested warmup days (in trading days)
                # Since ACTUAL_WARMUP_DAYS is specified as indicator periods, it's already trading days
                min_required = int(ACTUAL_WARMUP_DAYS * 0.8)

                if trading_days_before >= min_required:
                    stock_data[symbol] = df
                    total_days = len(df)
                    test_days = len(df[df.index >= test_start])
                    print(f"✓ {symbol}: {total_days} days total ({trading_days_before} warmup + {test_days} test)")
                else:
                    print(f"❌ {symbol}: Insufficient warmup ({trading_days_before} trading days < {min_required} required)")
            else:
                print(f"❌ {symbol}: No data available")

        if not stock_data:
            print("\n❌ No valid data loaded!")
            return

        print(f"\n{'='*60}")
        print(f"STARTING SIMULATION - {len(stock_data)} symbols loaded")
        print(f"Indicators are pre-warmed, signals start immediately!")
        print(f"{'='*60}\n")

        # Simulate ONLY the test period (warmup data is already loaded)
        test_dates = pd.date_range(start=test_start, end=test_end, freq='D')

        total_buys = 0
        total_sells = 0

        # Track completed trades for report
        completed_trades = []

        for current_date in test_dates:
            print(f"\n📅 {current_date.strftime('%Y-%m-%d')}")

            held_symbols = set(self.position_manager.list_all().keys())
            if held_symbols:
                print(f"   Holding: {list(held_symbols)}")

            buy_count = 0
            sell_count = 0

            # Check each symbol
            for symbol in stock_data.keys():
                df = stock_data[symbol]

                # Get data up to current date
                # This includes ALL warmup data, so indicators are always valid
                current_df = df[df.index <= current_date]

                # Indicators should work from day 1 since we have warmup data
                # But let's verify we have minimum data just in case
                if len(current_df) < ACTUAL_WARMUP_DAYS:
                    # This should rarely happen now
                    continue

                # Check if we have a position
                if symbol in held_symbols:
                    # Check for exit
                    position = self.position_manager.get(symbol)
                    entry_price = position['entry_price']
                    current_stop = position['stop_loss']
                    entry_date = pd.to_datetime(position['entry_date'])

                    sell_signal = self.strategy.get_exit_signal(
                        current_df, self.params, entry_price, current_stop
                    )

                    if sell_signal['signal']:
                        # Calculate trade metrics
                        exit_price = sell_signal['price']
                        pnl_pct = ((exit_price / entry_price) - 1) * 100
                        days_held = (current_date - entry_date).days

                        # Record completed trade
                        completed_trades.append({
                            'symbol': symbol,
                            'entry_date': entry_date.strftime('%Y-%m-%d'),
                            'exit_date': current_date.strftime('%Y-%m-%d'),
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl_pct': pnl_pct,
                            'days_held': days_held,
                            'exit_type': sell_signal['stop_type']
                        })

                        self.send_sell_alert(symbol, sell_signal, entry_price)
                        self.position_manager.remove(symbol)
                        sell_count += 1
                        total_sells += 1
                    else:
                        # Update trailing stop
                        self.position_manager.update_stop(symbol, sell_signal['new_stop'])
                else:
                    # Check for entry
                    buy_signal = self.strategy.get_entry_signal(current_df, self.params)

                    if buy_signal['signal']:
                        self.send_buy_alert(symbol, buy_signal)
                        # Auto-add position in test mode
                        self.position_manager.add(
                            symbol,
                            buy_signal['price'],
                            buy_signal['stop_loss'],
                            current_date.isoformat()
                        )
                        buy_count += 1
                        total_buys += 1

            if buy_count > 0 or sell_count > 0:
                print(f"   Signals: {buy_count} BUY, {sell_count} SELL")

            # Sleep based on speed setting
            if speed > 0:
                time.sleep(1.0 / speed)  # speed = days per second, so sleep = seconds per day

        print(f"\n{'='*60}")
        print("SIMULATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total Signals: {total_buys} BUY, {total_sells} SELL\n")

        # Show trade performance report
        if completed_trades:
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
            print(f"Symbol   Entry        Exit         Entry $    Exit $     P&L        Days   Type  ")
            print(f"{'─'*80}")

            for trade in completed_trades:
                pnl_str = f"{trade['pnl_pct']:+.2f}%"
                print(f"{trade['symbol']:<8} {trade['entry_date']:<12} {trade['exit_date']:<12} "
                      f"${trade['entry_price']:<9.2f} ${trade['exit_price']:<9.2f} "
                      f"{pnl_str:<10} {trade['days_held']:<6} {trade['exit_type']:<6}")

            print(f"{'='*80}\n")
        else:
            print("No completed trades during test period\n")

        # Show open positions
        summary = self.position_manager.get_summary()
        if summary['count'] > 0:
            print(f"Open Positions: {summary['count']}")
            print(f"Average Risk:   {summary['avg_risk_pct']:.2f}%\n")
            for symbol, pos in self.position_manager.list_all().items():
                print(f"  {symbol}: Entry ${pos['entry_price']:.2f}, SL ${pos['stop_loss']:.2f}")
        else:
            print("No open positions")
        print()


def load_watchlist(filename):
    """Load watchlist from CSV file"""
    symbols = []
    with open(filename, 'r') as f:
        for line in f:
            symbol = line.strip()
            if symbol and not symbol.startswith('#'):
                symbols.append(symbol)
    return symbols


if __name__ == "__main__":
    # Setup Pushbullet notifier
    notifier = PushbulletNotifier(PUSHBULLET_API_KEY)

    # Load strategy
    strategy_loader = StrategyLoader(STRATEGY_MODULE, STRATEGY_CLASS)

    # Load watchlist
    watchlist = load_watchlist(WATCHLIST_FILE)

    # Create monitor
    monitor = LiveTradingMonitor(watchlist, strategy_loader, STRATEGY_PARAMS, notifier)

    # Choose mode
    if TESTING_MODE:
        monitor.run_test_mode(TEST_START_DATE, TEST_END_DATE, TEST_SPEED)
    else:
        monitor.run()