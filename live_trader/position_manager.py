# position_manager.py
"""
CLI tool to manually manage positions
Now uses the shared PositionManager class for consistency

Usage:
  python position_manager.py add NVDA 184.86
  python position_manager.py list
  python position_manager.py remove NVDA
  python position_manager.py clear
  python position_manager.py summary
"""

import sys
from datetime import datetime
from positions import PositionManager

try:
    from live_trading_alerts import (
        StrategyLoader,
        PushbulletNotifier,
        STRATEGY_PARAMS,
        STRATEGY_MODULE,
        STRATEGY_CLASS,
        PUSHBULLET_API_KEY,
        LiveTradingMonitor,
        WATCHLIST_FILE,
        load_watchlist
    )
    IMPORTS_AVAILABLE = True
except ImportError:
    print("⚠️  Could not import from live_trading_alerts.py")
    print("Some features (like stop loss calculation) may be limited")
    IMPORTS_AVAILABLE = False


def add_position(symbol, price):
    """Add a new position with automatic stop loss calculation"""
    price = float(price)

    if IMPORTS_AVAILABLE:
        # Use strategy to calculate proper stop loss
        try:
            watchlist = load_watchlist(WATCHLIST_FILE)
            notifier = PushbulletNotifier(PUSHBULLET_API_KEY)
            strategy_loader = StrategyLoader(STRATEGY_MODULE, STRATEGY_CLASS)
            monitor = LiveTradingMonitor(watchlist, strategy_loader, STRATEGY_PARAMS, notifier)

            # Get live data and calculate stop
            df = monitor.get_live_data(symbol)
            if df is None:
                print(f"❌ Could not fetch data for {symbol}")
                return

            buy_signal = monitor.strategy.get_entry_signal(df, STRATEGY_PARAMS)
            stop_loss = buy_signal.get('stop_loss', price * 0.9)

        except Exception as e:
            print(f"⚠️  Error calculating stop loss: {e}")
            stop_loss = price * 0.9  # Fallback to 10%
    else:
        # Simple fallback
        stop_loss = price * 0.9

    # Add position
    pm = PositionManager()
    position = pm.add(symbol, price, stop_loss)

    if position:
        risk_pct = ((price - stop_loss) / price) * 100

        # Send notification if available
        if IMPORTS_AVAILABLE:
            try:
                notifier = PushbulletNotifier(PUSHBULLET_API_KEY)
                title = f"✅ Position Added: {symbol}"
                message = (
                    f"Entry: ${price:.2f}\n"
                    f"Stop Loss: ${stop_loss:.2f}\n"
                    f"Risk: {risk_pct:.2f}%\n"
                    f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                )
                notifier.send_notification(title, message)
            except:
                pass

        print(f"\n✅ Position added successfully!")
        print(f"   Symbol:     {symbol}")
        print(f"   Entry:      ${price:.2f}")
        print(f"   Stop Loss:  ${stop_loss:.2f}")
        print(f"   Risk:       {risk_pct:.2f}%\n")


def list_positions():
    """List all active positions"""
    pm = PositionManager()
    positions = pm.list_all()

    if not positions:
        print("\n📭 No active positions\n")
        return

    print(f"\n{'='*70}")
    print("ACTIVE POSITIONS")
    print(f"{'='*70}\n")

    for symbol, data in positions.items():
        entry_price = data['entry_price']
        stop_loss = data['stop_loss']
        entry_date = datetime.fromisoformat(data['entry_date']).strftime('%Y-%m-%d %H:%M')
        risk_pct = ((entry_price - stop_loss) / entry_price) * 100
        risk_dollars = entry_price - stop_loss

        print(f"📊 {symbol}")
        print(f"   Entry:      ${entry_price:.2f}")
        print(f"   Stop Loss:  ${stop_loss:.2f}")
        print(f"   Date:       {entry_date}")
        print(f"   Risk:       {risk_pct:.2f}% (${risk_dollars:.2f})")
        print()

    # Show summary
    summary = pm.get_summary()
    print(f"{'─'*70}")
    print(f"Total Positions:  {summary['count']}")
    print(f"Average Risk:     {summary['avg_risk_pct']:.2f}%")
    print(f"{'='*70}\n")


def remove_position(symbol):
    """Remove a position"""
    pm = PositionManager()
    position = pm.remove(symbol)

    if position:
        # Send notification if available
        if IMPORTS_AVAILABLE:
            try:
                notifier = PushbulletNotifier(PUSHBULLET_API_KEY)
                title = f"🗑️ Position Removed: {symbol}"
                message = f"Entry was: ${position['entry_price']:.2f}"
                notifier.send_notification(title, message)
            except:
                pass

        print(f"\n✅ Removed position: {symbol}")
        print(f"   Entry was: ${position['entry_price']:.2f}\n")


def clear_all_positions():
    """Clear all positions"""
    pm = PositionManager()
    summary = pm.get_summary()

    if summary['count'] == 0:
        print("\n📭 No positions to clear\n")
        return

    print(f"\n⚠️  You are about to remove {summary['count']} position(s):")
    for symbol in pm.list_all().keys():
        print(f"   - {symbol}")

    confirm = input("\nAre you sure? (yes/no): ")

    if confirm.lower() in ['yes', 'y']:
        count = pm.clear_all()

        # Send notification if available
        if IMPORTS_AVAILABLE:
            try:
                notifier = PushbulletNotifier(PUSHBULLET_API_KEY)
                notifier.send_notification("🗑️ All Positions Cleared", f"Removed {count} positions")
            except:
                pass

        print(f"\n✅ Cleared {count} positions\n")
    else:
        print("\n❌ Cancelled\n")


def show_summary():
    """Show position summary statistics"""
    pm = PositionManager()
    summary = pm.get_summary()

    print(f"\n{'='*70}")
    print("POSITION SUMMARY")
    print(f"{'='*70}\n")

    if summary['count'] == 0:
        print("📭 No active positions\n")
        return

    print(f"Total Positions:    {summary['count']}")
    print(f"Average Risk:       {summary['avg_risk_pct']:.2f}%")
    print(f"Total Risk:         {summary['total_risk_pct']:.2f}%")

    # Show position list
    print(f"\nPositions:")
    for symbol in pm.list_all().keys():
        print(f"  • {symbol}")

    print(f"\n{'='*70}\n")


def show_help():
    """Show help message"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                        POSITION MANAGER                              ║
║                  Manual Position Management Tool                     ║
╚══════════════════════════════════════════════════════════════════════╝

USAGE:
  python position_manager.py <command> [arguments]

COMMANDS:
  add SYMBOL PRICE       Add a position with automatic stop loss
  list                   List all active positions with details
  remove SYMBOL          Remove a specific position
  clear                  Remove all positions (with confirmation)
  summary                Show position statistics summary
  help                   Show this help message

EXAMPLES:
  # Add position after buying NVDA at $184.86
  python position_manager.py add NVDA 184.86
  
  # List all active positions
  python position_manager.py list
  
  # Show summary statistics
  python position_manager.py summary
  
  # Remove position after selling
  python position_manager.py remove NVDA
  
  # Clear all positions
  python position_manager.py clear

NOTES:
  • Stop loss is calculated automatically using your strategy settings
  • Pushbullet notifications are sent if configured
  • Positions are shared with live_trading_alerts.py
  • Use this for manual position management or as backup to Pushbullet replies

╔══════════════════════════════════════════════════════════════════════╗
║  TIP: For live trading, reply to notifications with "BOUGHT SYMBOL"  ║
║       This tool is for manual/backup position management             ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


def main():
    if len(sys.argv) < 2:
        show_help()
        return

    command = sys.argv[1].lower()

    if command == "add":
        if len(sys.argv) != 4:
            print("❌ Usage: python position_manager.py add SYMBOL PRICE")
            print("   Example: python position_manager.py add NVDA 184.86")
            return
        symbol = sys.argv[2].upper()
        price = sys.argv[3]
        try:
            add_position(symbol, price)
        except ValueError:
            print(f"❌ Invalid price: {price}")

    elif command == "list":
        list_positions()

    elif command == "remove":
        if len(sys.argv) != 3:
            print("❌ Usage: python position_manager.py remove SYMBOL")
            print("   Example: python position_manager.py remove NVDA")
            return
        symbol = sys.argv[2].upper()
        remove_position(symbol)

    elif command == "clear":
        clear_all_positions()

    elif command == "summary":
        show_summary()

    elif command == "help":
        show_help()

    else:
        print(f"❌ Unknown command: {command}\n")
        show_help()


if __name__ == "__main__":
    main()