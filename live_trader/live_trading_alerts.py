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
STRATEGY_MODULE = "../strategies/LORENTZIAN_CLASSIFICATION_3/lorentzian_classification"
STRATEGY_CLASS = "LorentzianClassificationStrategy"
STRATEGY_PARAMS = {
    # === General Settings ===
    'neighbors_count': 10,
    'max_bars_back': 2000,
    'feature_count': 5,

    # Label Mode: False=mean-reversion (expect reversals), True=trend-following (expect continuation)
    'trend_following_labels': False,

    # Re-entry Mode: True=enter anytime signal is favorable, False=only enter on signal flip
    'allow_reentry': True,

    # Minimum prediction strength to trade (0=any signal, higher=stronger consensus required)
    'min_prediction_strength': 4,

    # === Feature 1: RSI (Momentum) ===
    'f1_type': 'RSI',
    'f1_param_a': 14,
    'f1_param_b': 1,

    # === Feature 2: ADX (Trend Strength) ===
    'f2_type': 'ADX',
    'f2_param_a': 14,
    'f2_param_b': 1,

    # === Feature 3: ATR Ratio (Normalized Volatility) ===
    'f3_type': 'ATRR',
    'f3_param_a': 14,
    'f3_param_b': 1,

    # === Feature 4: Price Position (Range Location) ===
    'f4_type': 'PP',
    'f4_param_a': 20,
    'f4_param_b': 1,

    # === Feature 5: Efficiency Ratio (Trend Quality) ===
    'f5_type': 'ER',
    'f5_param_a': 10,
    'f5_param_b': 1,

    # === Filters ===
    'use_volatility_filter': True,
    'use_regime_filter': True,
    'regime_threshold': -0.1,
    'use_adx_filter': False,
    'adx_threshold': 20,
    'use_ema_filter': False,
    'ema_period': 50,
    'use_sma_filter': False,
    'sma_period': 200,

    # === Kernel Settings ===
    'use_kernel_filter': False,
    'use_kernel_smoothing': False,
    'kernel_lookback': 8,
    'kernel_rel_weight': 8.0,
    'kernel_start_bar': 25,
    'kernel_lag': 2,

    # === Exit Settings ===
    'use_dynamic_exits': True,
    'bars_to_hold': 10000,

    # === RSI Exit Settings ===
    'use_rsi_exit': False,
    'rsi_exit_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,

    # === Kernel Exit Settings ===
    'use_kernel_exit': True,

    # === Risk Management ===
    'position_size_pct': 0.95,
    'stop_loss_pct': 0.05,
    'use_stop_loss': True,
    'long_only': True,

    # === Display ===
    'verbose': False,
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