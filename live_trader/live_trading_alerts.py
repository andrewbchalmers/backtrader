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

# Finnhub API key (free at finnhub.io) — leave blank to disable news monitoring
FINNHUB_API_KEY = "d77bmrhr01qp6afl1npgd77bmrhr01qp6afl1nq0"

# Testing mode
TESTING_MODE = False  # Set to True for backtesting, False for live trading
TEST_START_DATE = "2024-06-01"
TEST_END_DATE = "2024-12-31"
TEST_SPEED = 0  # Days per second (0 = instant, 1 = 1 day/sec, 10 = 10 days/sec)

# Strategy configuration
STRATEGY_MODULE = "../strategies/LORENTZIAN_CLASSIFICATION_10/lorentzian_classification"
STRATEGY_CLASS = "LorentzianClassificationStrategy"
STRATEGY_PARAMS = {
    # === General Settings ===
    'neighbors_count': 11,
    'max_bars_back': 7000,
    'feature_count': 8,
    'trend_following_labels': True,
    'allow_reentry': True,
    'min_prediction_strength': 20,       # Percentile rank threshold (0–100 normalised scale)
    'min_raw_prediction': 0.0,           # Raw ATR-unit threshold (0 = disabled)
    'prediction_norm_window': 200,       # Rolling window used to compute the percentile rank
    'min_bullish_accuracy': 50,          # Minimum ML bullish accuracy % to act on a BUY signal
    'min_bullish_accuracy_samples': 20,  # Min historical bullish predictions before filter activates

    # === Label Settings ===
    'label_lookahead': 4,
    'label_dead_zone': 0.18,
    'use_forward_labels': True,
    'use_magnitude_labels': True,

    # === Features (8 active, f9-f14 defined but below feature_count cutoff) ===
    # Feature 1: Relative Strength Momentum — where is this stock's momentum ranked historically?
    'f1_type': 'RSM',
    'f1_param_a': 40,    # Momentum period: two months
    'f1_param_b': 252,   # Lookback: full year percentile ranking

    # Feature 2: Efficiency Ratio — is price moving directionally or chopping?
    'f2_type': 'ER',
    'f2_param_a': 25,    # Trend quality over ~1 month
    'f2_param_b': 13,

    # Feature 3: Momentum Acceleration — short ROC vs medium ROC, ATR-normalised
    'f3_type': 'MACC',
    'f3_param_a': 5,     # Short momentum window
    'f3_param_b': 20,    # Medium momentum window

    # Feature 4: Streak Pattern — consecutive directional moves = trend persistence
    'f4_type': 'STRK',
    'f4_param_a': 30,    # Max streak length
    'f4_param_b': 3,     # ATR multiplier for magnitude

    # Feature 5: Volatility Compression — coiled spring before breakout
    'f5_type': 'VCOMP',
    'f5_param_a': 4,     # Recent volatility window
    'f5_param_b': 16,    # Lookback volatility window

    # Feature 6: Momentum Persistence — pullback within trend vs reversal
    'f6_type': 'MPER',
    'f6_param_a': 4,     # Short momentum period
    'f6_param_b': 20,    # Medium momentum period

    # Feature 7: Volume-Momentum Coupling — volume engaged during consolidation
    'f7_type': 'VMC',
    'f7_param_a': 3,     # Recent volume/momentum window
    'f7_param_b': 40,    # Baseline volume average period

    # Feature 8: Candle Structure — trend-quality candles (strong bodies, small wicks)
    'f8_type': 'CS',
    'f8_param_a': 5,
    'f8_param_b': 2,

    # Features 9-14 (inactive — below feature_count cutoff)
    'f9_type': 'CS',
    'f9_param_a': 5,
    'f9_param_b': 2,
    'f10_type': 'CS',
    'f10_param_a': 5,
    'f10_param_b': 2,
    'f11_type': 'CS',
    'f11_param_a': 5,
    'f11_param_b': 2,
    'f12_type': 'VCOMP',
    'f12_param_a': 4,
    'f12_param_b': 16,
    'f13_type': 'MPER',
    'f13_param_a': 4,
    'f13_param_b': 20,
    'f14_type': 'VMC',
    'f14_param_a': 5,
    'f14_param_b': 40,

    # === Filters ===
    'use_volatility_filter': False,
    'use_regime_filter': True,
    'regime_threshold': Decimal('0'),   # 0=block bearish only
    'regime_period': 'weekly',
    'use_regime_direction': True,
    'regime_stability_min': 0.0,
    'regime_stability_window': 10,
    'regime_max_flips': 2,
    'use_adx_filter': False,
    'adx_threshold': 14,
    'use_ema_filter': False,
    'ema_period': 400,
    'ema_slope_lookback': 20,
    'use_sma_filter': False,
    'sma_period': 400,
    'sma_slope_lookback': 20,

    # === SPY Market Regime Filter ===
    'use_spy_filter': True,
    'spy_regime_threshold': Decimal('0'),  # 0=block bearish SPY
    'spy_regime_period': 'weekly',

    # === Kernel Settings ===
    'use_kernel_filter': False,
    'use_kernel_smoothing': False,
    'kernel_lookback': 8,
    'kernel_rel_weight': 8.0,
    'kernel_start_bar': 25,
    'kernel_lag': 2,

    # === Exit Settings ===
    'use_dynamic_exits': False,
    'bars_to_hold': 100000,

    # === RSI Exit Settings ===
    'use_rsi_exit': False,
    'rsi_exit_period': 14,
    'rsi_overbought': 80,   # Widened — avoids cutting winning trends short
    'rsi_oversold': 20,

    # === Kernel Exit Settings ===
    'use_kernel_exit': False,

    # === Chandelier ATR Trailing Stop ===
    # Wider settings give trends room to breathe through normal pullbacks
    'use_trailing_atr_exit': True,
    'trailing_atr_warmup': 5,
    'chandelier_start_mult': 3.5,          # Wider initial stop than mean-reversion (3.0)
    'chandelier_end_mult': 1,              # Final stop
    'chandelier_tighten_bars': 40,         # Slower tightening — trends need more time
    'chandelier_mult_mode': 'blend',
    'chandelier_profit_atr_threshold': 1.5,
    'chandelier_breakeven_atr': 0.0,       # Requires more profit before locking break-even

    # === Prediction Validation Exit ===
    'use_prediction_exit': True,
    'prediction_exit_threshold': -20.0,    # Exit only on strongly bearish ML signal

    # === Loss Penalty ===
    'use_loss_penalty': True,
    'loss_penalty_amount': 0,
    'loss_penalty_decay': 0.90,

    # === Risk Management ===
    'position_size_pct': Decimal('0.95'),
    'stop_loss_pct': Decimal('0.05'),
    'use_stop_loss': False,
    'long_only': True,

    # === Fundamental / Earnings Settings ===
    'use_fundamental_filter': True,
    'fundamental_quality_weight': 0.2,
    'fundamental_momentum_weight': 0.3,
    'earnings_blackout_before': 5,
    'earnings_blackout_after': 2,
    'close_before_earnings': True,
    'min_trending_probability': 20,
    'full_position_threshold': 50,
    'reduced_position_pct': Decimal('0.75'),
    'min_quality_score': 20,
    'min_momentum_score': 20,
    'use_earnings_reaction': True,
    'earnings_reaction_days': 5,
    'min_earnings_reaction_pct': -5.0,
    'max_days_since_earnings': 100,
    'staleness_decay_rate': 0.005,
    'fundamental_symbol': '',           # Set per-symbol by strategy_loader

    # === Cross-Symbol Training ===
    'use_cross_symbol_training': True,
    'cross_symbol_etfs': _peer_universe,
    'cross_symbol_lookback_years': 5,
    'use_regime_balancing': False,
    'cross_symbol_auto_peers': True,    # Auto-select sector peers per symbol
    'cross_symbol_target_symbol': '',   # Set per-symbol by strategy_loader
    'cross_symbol_max_peers': 7,

    # === Backtest control ===
    'test_start_idx': 0,

    # === Display ===
    'verbose': False,
}

# Watchlist
WATCHLIST_FILE = "../strategies/sp500_2025.csv"

# Live mode scan interval (minutes)
SCAN_INTERVAL = 5

# Portfolio management
PORTFOLIO_CAPITAL = 100_000   # Starting/current cash for dynamic position sizing
MAX_POSITIONS     = 25        # Hard cap — no buy alerts sent when portfolio is full

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
        warmup_days=WARMUP_DAYS,
        portfolio_capital=PORTFOLIO_CAPITAL,
        max_positions=MAX_POSITIONS,
        finnhub_api_key=FINNHUB_API_KEY,
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