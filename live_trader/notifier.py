# notifier.py
"""
Pushbullet notification handler for live trading alerts
"""

import time
import threading
from pushbullet import Pushbullet


class PushbulletNotifier:
    """Send notifications via Pushbullet and handle user replies"""

    def __init__(self, api_key):
        self.pb = Pushbullet(api_key)
        self.last_check_time = time.time()
        self.reply_callback = None
        self.listener_thread = None
        self.listening = False
        self.processed_push_ids = set()  # Track processed pushes to avoid duplicates

        # Clean up old pushes on startup to free storage
        self._cleanup_old_pushes()

        print(f"✓ Pushbullet initialized")

    def _cleanup_old_pushes(self, keep_recent=100):
        """Delete old pushes to free up storage space (for free tier)"""
        try:
            pushes = self.pb.get_pushes()

            if len(pushes) > keep_recent:
                # Keep only the most recent ones
                old_pushes = pushes[keep_recent:]
                deleted = 0

                for push in old_pushes:
                    try:
                        self.pb.delete_push(push.get('iden'))
                        deleted += 1
                    except:
                        pass

                if deleted > 0:
                    print(f"ℹ️  Cleaned up {deleted} old push(es) to free storage")
        except Exception as e:
            print(f"⚠️  Could not cleanup old pushes: {e}")

    def send_notification(self, title, message):
        """Send push notification"""
        try:
            push = self.pb.push_note(title, message)
            print(f"✓ Notification sent: {title}")
            return True
        except Exception as e:
            print(f"❌ Notification failed: {e}")
            return False

    def send_notification_with_image(self, title, message, image_buffer, filename="chart.png"):
        """
        Send push notification with an attached image
        Auto-cleans old pushes to stay within free tier limits

        Args:
            title: Notification title
            message: Notification message
            image_buffer: BytesIO buffer containing PNG image
            filename: Filename for the attachment

        Returns:
            bool: Success status
        """
        try:
            # Clean up old pushes before uploading new image
            self._cleanup_old_pushes(keep_recent=100)

            # Upload the file
            file_data = self.pb.upload_file(image_buffer, filename)

            # Send the file with message
            push = self.pb.push_file(
                **file_data,
                title=title,
                body=message
            )

            print(f"✓ Notification with image sent: {title}")
            return True

        except Exception as e:
            # If file upload fails (Pro required or limit hit), send text only
            if "pushbullet_pro_required" in str(e) or "rate_limit" in str(e):
                print(f"⚠️  Image upload failed (Pro required or limit hit), sending text only")
                return self.send_notification(title, message)
            else:
                print(f"❌ Notification with image failed: {e}")
                import traceback
                traceback.print_exc()
                return False

    def start_listening(self, callback):
        """
        Start continuous listener for user replies in background thread

        Args:
            callback: Function to call when reply received, signature: callback(reply_text)
        """
        if self.listening:
            print("⚠️  Listener already running")
            return

        self.reply_callback = callback
        self.listening = True
        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener_thread.start()
        print("✓ Reply listener started (running in background)")

    def stop_listening(self):
        """Stop the reply listener"""
        self.listening = False
        if self.listener_thread:
            self.listener_thread.join(timeout=5)
        print("✓ Reply listener stopped")

    def _listen_loop(self):
        """Background loop that continuously checks for replies"""
        print("🎧 Listening for replies...")

        while self.listening:
            try:
                replies = self._check_for_new_replies()

                for reply in replies:
                    if self.reply_callback:
                        # Call the callback function with the reply
                        self.reply_callback(reply)

                # Check every 5 seconds
                time.sleep(5)

            except Exception as e:
                print(f"❌ Error in listener loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(10)  # Wait longer on error

    def _check_for_new_replies(self):
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
                # Skip if we've already processed this push
                push_id = push.get('iden')
                if push_id in self.processed_push_ids:
                    continue

                # Look for pushes with text body
                if push.get('body'):
                    body = push.get('body', '').strip()
                    direction = push.get('direction', 'unknown')
                    push_type = push.get('type', 'unknown')

                    # Only process if it's a NOTE type and NOT sent by us
                    if push_type != 'note':
                        continue

                    # Skip if it's outgoing (we sent it)
                    if direction == 'outgoing':
                        self.processed_push_ids.add(push_id)
                        continue

                    # Convert to uppercase for processing
                    body_upper = body.upper()

                    # Check if it starts with a known response pattern (our notifications)
                    # These start with emojis or specific formats we send
                    if body_upper.startswith(('ENTRY:', '✅', '🟢', '🔴', '📊', '⚠️', '❌', '📭')):
                        # This is one of our notification responses, ignore it
                        self.processed_push_ids.add(push_id)
                        continue

                    # Accept all valid command formats
                    valid_commands = ['BOUGHT ', 'SOLD ', 'LAST ', 'BACKTEST ', 'HOLDING', 'HOLDINGS']
                    if any(body_upper.startswith(cmd) or body_upper == cmd for cmd in valid_commands):
                        replies.append(body_upper)
                        self.processed_push_ids.add(push_id)
                        print(f"📩 Received reply: {body_upper}")
                    else:
                        # Not a command we recognize, mark as processed
                        self.processed_push_ids.add(push_id)

            # Cleanup old IDs (keep only last 100)
            if len(self.processed_push_ids) > 100:
                self.processed_push_ids = set(list(self.processed_push_ids)[-100:])

            return replies

        except Exception as e:
            print(f"❌ Error checking replies: {e}")
            return []

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
        self.send_notification(title, message)
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
        self.send_notification(title, message)
        print(f"🔴 SELL ALERT: {symbol} @ ${signal['price']:.2f}, P&L: {pnl:+.2f}%")

    def send_position_confirmation(self, symbol, entry_price, exit_price_or_stop, action="added", pnl=None):
        """Send position add/remove confirmation"""
        from datetime import datetime

        if action == "added":
            stop_loss = exit_price_or_stop
            risk_pct = ((entry_price - stop_loss) / entry_price) * 100
            title = f"✅ {symbol} REGISTERED AS BOUGHT"
            message = (
                f"Entry: ${entry_price:.2f}\n"
                f"Stop Loss: ${stop_loss:.2f}\n"
                f"Risk: {risk_pct:.2f}%\n"
                f"Time: {datetime.now().strftime('%H:%M:%S')}"
            )
        else:  # removed
            title = f"✅ {symbol} REGISTERED AS SOLD"
            message = f"Entry was: ${entry_price:.2f}\n"

            if pnl is not None:
                message += f"P&L: {pnl:+.2f}%\n"

            if exit_price_or_stop is not None:
                message += f"Exit price: ${exit_price_or_stop:.2f}\n"

            message += f"Time: {datetime.now().strftime('%H:%M:%S')}"

        self.send_notification(title, message)