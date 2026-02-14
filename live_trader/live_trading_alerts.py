# live_trading_alerts.py
"""
Live Trading Alert System - REFACTORED
Main entry point - clean and simple!
"""

from decimal import Decimal
from notifier import PushbulletNotifier
from strategy_loader import StrategyLoader, calculate_lookback
from monitor import LiveTradingMonitor
from test_runner import TestRunner

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load peer universe from classification CSV (same as backtest code)
_peer_universe = 'SPY,QQQ,IWM,TLT,GLD,XLE,EFA'
try:
    with open('../strategies/classification_set.csv') as _f:
        _symbols = [line.strip() for line in _f if line.strip()]
        if _symbols:
            _peer_universe = ','.join(_symbols)
except FileNotFoundError:
    pass

# Pushbullet API key
PUSHBULLET_API_KEY = "o.ptYJ8W8YpFEnDVZ1CL4vO9N7suOvJURG"

# Testing mode
TESTING_MODE = False  # Set to True for backtesting, False for live trading
TEST_START_DATE = "2024-06-01"
TEST_END_DATE = "2024-12-31"
TEST_SPEED = 0  # Days per second (0 = instant, 1 = 1 day/sec, 10 = 10 days/sec)

# Strategy configuration
STRATEGY_MODULE = "../strategies/LORENTZIAN_CLASSIFICATION_8/lorentzian_classification"
STRATEGY_CLASS = "LorentzianClassificationStrategy"
STRATEGY_PARAMS = {
    # === General Settings ===
    'neighbors_count': 5,
    'max_bars_back': 5000,
    'feature_count': 9,
    'trend_following_labels': False,
    'allow_reentry': True,
    'min_prediction_strength': 20,

    # === Label Settings ===
    'label_lookahead': 3,
    'label_dead_zone': 0.15,
    'use_magnitude_labels': True,

    # === Feature 1: Relative Strength Momentum ===
    'f1_type': 'RSM',
    'f1_param_a': 10,
    'f1_param_b': 126,

    # === Feature 2: Volume Anomaly ===
    'f2_type': 'VA',
    'f2_param_a': 20,
    'f2_param_b': 1,

    # === Feature 3: Multi-Timeframe Divergence ===
    'f3_type': 'MTD',
    'f3_param_a': 5,
    'f3_param_b': 60,

    # === Feature 4: Mean Reversion Z-Score ===
    'f4_type': 'ZS',
    'f4_param_a': 50,
    'f4_param_b': 1,

    # === Feature 5: Efficiency Ratio ===
    'f5_type': 'ER',
    'f5_param_a': 10,
    'f5_param_b': 1,

    # === Feature 6: Volume-Price Divergence ===
    'f6_type': 'VPD',
    'f6_param_a': 14,
    'f6_param_b': 1,

    # === Feature 7: Momentum Acceleration ===
    'f7_type': 'MACC',
    'f7_param_a': 5,
    'f7_param_b': 5,

    # === Feature 8: OBV Trend ===
    'f8_type': 'OBVT',
    'f8_param_a': 20,
    'f8_param_b': 3,

    # === Feature 9: Candle Structure ===
    'f9_type': 'CS',
    'f9_param_a': 5,
    'f9_param_b': 2,

    # === Filters ===
    'use_volatility_filter': True,
    'use_regime_filter': True,
    'regime_threshold': 1,
    'regime_period': 'weekly',
    'use_regime_direction': True,
    'regime_stability_min': 0.0,
    'regime_stability_window': 60,
    'regime_max_flips': 8,
    'use_adx_filter': True,
    'adx_threshold': 14,
    'use_ema_filter': False,
    'ema_period': 50,
    'use_sma_filter': False,
    'sma_period': 200,

    # === Kernel Settings ===
    'use_kernel_filter': False,
    'use_kernel_smoothing': False,
    'kernel_lookback': 20,
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
    'use_kernel_exit': False,

    # === ATR Trailing Stop Exit Settings ===
    'use_trailing_atr_exit': True,
    'trailing_atr_mult': 2.5,
    'trailing_atr_warmup': 3,

    # === Loss Penalty (ML bearish feedback after losing trades) ===
    'use_loss_penalty': True,
    'loss_penalty_amount': 0,
    'loss_penalty_decay': 0.90,

    # === Risk Management ===
    'position_size_pct': Decimal('0.95'),
    'stop_loss_pct': Decimal('0.05'),
    'use_stop_loss': True,
    'long_only': True,

    # === Fundamental / Earnings Settings ===
    'use_fundamental_filter': True,
    'fundamental_quality_weight': 0.2,
    'fundamental_momentum_weight': 0.3,
    'earnings_blackout_before': 5,
    'earnings_blackout_after': 2,
    'close_before_earnings': True,
    'min_trending_probability': 40,
    'full_position_threshold': 70,
    'reduced_position_pct': Decimal('0.75'),
    'min_quality_score': 30,
    'min_momentum_score': 30,
    'fundamental_symbol': '',

    # === Cross-Symbol Training ===
    'use_cross_symbol_training': True,
    'cross_symbol_etfs': _peer_universe,
    'cross_symbol_lookback_years': 5,
    'use_regime_balancing': True,
    'cross_symbol_auto_peers': True,
    'cross_symbol_target_symbol': '',  # Set per-symbol by strategy_loader
    'cross_symbol_max_peers': 7,

    # === Display ===
    'verbose': False,
}

# Watchlist
WATCHLIST_FILE = "../strategies/sp500_2025.csv"

# Live mode scan interval (minutes)
SCAN_INTERVAL = 5

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

    # Calculate lookback from strategy class + params (matches backtest.py)
    WARMUP_DAYS = calculate_lookback(strategy_loader.strategy_class, STRATEGY_PARAMS)
    print(f"ℹ️  Warmup period: {WARMUP_DAYS} bars (calculated from strategy)")

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