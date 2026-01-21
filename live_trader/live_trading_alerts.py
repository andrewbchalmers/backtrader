# live_trading_alerts.py
"""
Live Trading Alert System - REFACTORED
Main entry point - clean and simple!
"""

from notifier import PushbulletNotifier
from strategy_loader import StrategyLoader, calculate_warmup_days
from monitor import LiveTradingMonitor
from test_runner import TestRunner

# ============================================================================
# CONFIGURATION
# ============================================================================
# Pushbullet API key
PUSHBULLET_API_KEY = "o.ptYJ8W8YpFEnDVZ1CL4vO9N7suOvJURG"

# Testing mode
TESTING_MODE = False  # Set to True for backtesting, False for live trading
TEST_START_DATE = "2024-06-01"
TEST_END_DATE = "2024-12-31"
TEST_SPEED = 0  # Days per second (0 = instant, 1 = 1 day/sec, 10 = 10 days/sec)

# Strategy configuration
STRATEGY_MODULE = "../strategies/AROON_FILTER_ATR/aroon_atr"
STRATEGY_CLASS = "AroonATRStrategy"
STRATEGY_PARAMS = {
    'aroon_len': 20,
    'atr_entry_sma_period': 20,
    'atr_filter_mult': 1.2,
    'stop_loss_pct': 0.05,
    'verbose': False,
    'atr_exit_len': 14,
    'atr_exit_mult': 3.0,
    'use_atr_filter': True,
    'take_profit_pct': 0.13,
    'use_take_profit': True,
    'position_size_pct': '0.95'
}

# Watchlist
WATCHLIST_FILE = "../strategies/sp500_2025.csv"

# Live mode scan interval (minutes)
SCAN_INTERVAL = 5

# Auto-calculate warmup period from strategy
WARMUP_DAYS = calculate_warmup_days(STRATEGY_PARAMS, default_days=300)
print(f"ℹ️  Warmup period: {WARMUP_DAYS} trading days (auto-calculated)")
# ============================================================================


def load_watchlist(filename):
    """Load watchlist from CSV file"""
    symbols = []
    with open(filename, 'r') as f:
        for line in f:
            symbol = line.strip()
            if symbol and not symbol.startswith('#'):
                symbols.append(symbol)
    return symbols


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("LIVE TRADING ALERT SYSTEM")
    print("="*60 + "\n")

    # Initialize components
    notifier = PushbulletNotifier(PUSHBULLET_API_KEY)
    strategy_loader = StrategyLoader(STRATEGY_MODULE, STRATEGY_CLASS)
    watchlist = load_watchlist(WATCHLIST_FILE)

    # Create monitor
    monitor = LiveTradingMonitor(
        watchlist=watchlist,
        strategy_loader=strategy_loader,
        strategy_params=STRATEGY_PARAMS,
        notifier=notifier,
        warmup_days=WARMUP_DAYS
    )

    # Run in selected mode
    if TESTING_MODE:
        print(f"Mode: TEST/BACKTEST")
        print(f"Strategy: {STRATEGY_CLASS}")
        test_runner = TestRunner(monitor, WARMUP_DAYS)
        test_runner.run(TEST_START_DATE, TEST_END_DATE, TEST_SPEED)
    else:
        print(f"Mode: LIVE TRADING")
        print(f"Strategy: {STRATEGY_CLASS}")
        monitor.run(SCAN_INTERVAL)


if __name__ == "__main__":
    main()