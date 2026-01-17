"""
Aroon Crossover Multi-Filter Trend Strategy (Enhanced)

A sophisticated trend-following strategy that combines:
- Aroon indicator crossovers with state-based bullish signals
- Multi-layer sideways market detection and filtering
- ADX trend strength confirmation
- ATR-based volatility filtering
- Peak drawdown exits (trailing from highest high)
- Multiple exit methods (ATR stop, fixed SL/TP, bearish reversal)

Entry Logic:
    - Aroon Up crosses above Aroon Down and maintains bullish state
    - Aroon Up > 70 and Aroon Down < 30 (strong trend)
    - Trend strength (|AroonUp - AroonDown|) > threshold
    - Both Aroons NOT in middle range (30-70) if filter enabled
    - Aroon stability check over multiple bars
    - ADX > threshold (strong trending market)
    - ATR below baseline (low volatility, calm market)

Exit Logic:
    - ATR trailing stop (dynamic)
    - Fixed percentage stop-loss (10% default)
    - Fixed take-profit target (13% default)
    - Bearish Aroon signal (AroonDown crosses above AroonUp)
    - Peak drawdown exit (from highest high, optional)
"""

from decimal import Decimal
import backtrader as bt


class AroonMultiFilterStrategy(bt.Strategy):
    """
    Enhanced Aroon crossover strategy with comprehensive sideways market filtering,
    ADX confirmation, and peak drawdown exit management.
    """

    plotinfo = dict(
        plot=True,
        subplot=False,
        plotlinelabels=True
    )

    params = (
        # Aroon indicator
        ('aroon_len', 24),

        # ATR volatility filter (for entry - prefer LOW volatility)
        ('atr_filter_len', 14),
        ('atr_filter_baseline_len', 40),  # Baseline period for volatility comparison
        ('atr_filter_mult', Decimal('2.5')),  # ATR must be < SMA * multiplier

        # ATR trailing stop (for exit)
        ('atr_stop_len', 5),
        ('atr_stop_mult', Decimal('4.0')),

        # Risk management
        ('stop_loss_pct', Decimal('0.05')),    # 10% stop loss
        ('take_profit_pct', Decimal('0.13')),  # 13% take profit

        # Dynamic peak drawdown exit (ATR-based)
        ('enable_peak_exit', False),
        ('peak_atr_mult', Decimal('2.0')),           # Peak drawdown = Peak - (ATR * multiplier)
        ('peak_atr_period', 14),                      # ATR period for peak calculation
        ('min_profit_pct_to_activate', Decimal('0.03')),  # 3% min profit to activate

        # Sideways market detection
        ('min_trend_strength', Decimal('25.0')),     # Min |AroonUp - AroonDown|
        ('max_both_middle', True),                    # Reject when both in 30-70 range
        ('stability_bars', 3),                        # Aroon consistency check

        # ADX trend confirmation
        ('use_adx_filter', False),
        ('adx_length', 14),
        ('adx_threshold', Decimal('25.0')),

        # Position sizing
        ('position_size_pct', Decimal('0.95')),  # 100% of equity

        # Logging
        ('verbose', True),
    )

    def __init__(self):
        """Initialize indicators and state variables."""
        self._init_indicators()
        self._init_state()

    def _init_indicators(self):
        """Initialize all technical indicators."""
        # Aroon indicator
        self.aroon = bt.indicators.AroonUpDown(
            self.data,
            period=self.p.aroon_len
        )

        # ATR indicators
        self.atr_filter = bt.indicators.ATR(
            self.data,
            period=self.p.atr_filter_len
        )
        self.atr_stop = bt.indicators.ATR(
            self.data,
            period=self.p.atr_stop_len
        )

        # ATR for dynamic peak drawdown calculation
        self.atr_peak = bt.indicators.ATR(
            self.data,
            period=self.p.peak_atr_period
        )

        # ATR baseline for volatility filter (longer-term average)
        self.atr_baseline = bt.indicators.SMA(
            self.atr_filter,
            period=self.p.atr_filter_baseline_len
        )

        # ADX calculation (manual implementation)
        # Note: backtrader has DirectionalMovementIndex but we'll build it manually
        # for exact Pine Script parity
        self._init_adx_components()

    def _init_adx_components(self):
        """Initialize ADX calculation components."""
        # True Range
        self.tr = bt.indicators.Max(
            self.data.high - self.data.low,
            bt.indicators.Max(
                abs(self.data.high - self.data.close(-1)),
                abs(self.data.low - self.data.close(-1))
            )
        )

        # Smoothed True Range
        self.tr_smooth = bt.indicators.SmoothedMovingAverage(
            self.tr,
            period=self.p.adx_length
        )

        # Plus/Minus Directional Movement (need custom calculation)
        # We'll calculate these in next() for accuracy

        # For now, use backtrader's built-in DMI
        self.dmi = bt.indicators.DirectionalMovementIndex(
            self.data,
            period=self.p.adx_length
        )

    def _init_state(self):
        """Initialize strategy state variables."""
        self.order = None
        self.atr_stop_price = None
        self.exiting = False

        # Peak drawdown tracking
        self.position_peak = None
        self.peak_drawdown_stop = None

        # Bullish state tracking (state-based signal)
        self.bullish_state = False

        # Aroon stability tracking
        self.aroon_up_history = []
        self.aroon_down_history = []

    def next(self):
        """Main strategy logic called on each bar."""
        # Update Aroon history for stability check
        self._update_aroon_history()

        # Force exit on final bar
        if self._is_final_bar():
            self._force_close_position()
            return

        # Skip if order is pending
        if self.order:
            return

        # Update bullish state based on crossovers
        self._update_bullish_state()

        # Try to enter new position
        if not self.position:
            self._check_entry_signal()

        # Manage existing position
        else:
            self._manage_position()

    def _update_aroon_history(self):
        """Update Aroon history for stability checks."""
        # Keep last N bars for stability analysis
        max_history = self.p.stability_bars + 1

        self.aroon_up_history.append(self.aroon.aroonup[0])
        self.aroon_down_history.append(self.aroon.aroondown[0])

        if len(self.aroon_up_history) > max_history:
            self.aroon_up_history.pop(0)
        if len(self.aroon_down_history) > max_history:
            self.aroon_down_history.pop(0)

    def _is_final_bar(self):
        """Check if current bar is the last available bar."""
        return len(self.data) == self.data.buflen()

    def _force_close_position(self):
        """Close position on final bar to avoid skewed results."""
        if self.position:
            if self.p.verbose:
                print("FINAL BAR - Closing position")
            self.order = self.sell(size=self.position.size)
            self.exiting = True

    def _update_bullish_state(self):
        """
        Update bullish state based on Aroon crossovers.
        State persists until bearish crossover occurs.
        """
        # Detect crossovers
        cross_up = (
                self.aroon.aroonup[0] > self.aroon.aroondown[0] and
                self.aroon.aroonup[-1] <= self.aroon.aroondown[-1]
        )
        cross_down = (
                self.aroon.aroondown[0] > self.aroon.aroonup[0] and
                self.aroon.aroondown[-1] <= self.aroon.aroonup[-1]
        )

        if cross_up:
            self.bullish_state = True
            if self.p.verbose:
                print(f"Aroon Crossover UP - Bullish state activated")

        if cross_down:
            self.bullish_state = False
            if self.p.verbose:
                print(f"Aroon Crossover DOWN - Bullish state deactivated")

    def _check_entry_signal(self):
        """Check for entry conditions and execute buy order if met."""
        # Base bullish signal: state + strong trend positioning
        base_bullish = self._check_base_bullish_signal()

        if not base_bullish:
            return

        # Apply sideways market filters
        passes_sideways_filters = self._check_sideways_filters()

        if not passes_sideways_filters:
            return

        # Check volatility filter (prefer LOW volatility)
        low_volatility = self._check_low_volatility()

        if not low_volatility:
            if self.p.verbose:
                print(f"Entry blocked: High volatility detected")
            return

        # All conditions met - execute entry
        self._execute_entry()

    def _check_base_bullish_signal(self):
        """
        Check base bullish signal conditions.

        Returns:
            bool: True if bullish state active with strong positioning
        """
        base_bullish = (
                self.bullish_state and
                self.aroon.aroonup[0] > 70 and
                self.aroon.aroondown[0] < 30
        )

        return base_bullish

    def _check_sideways_filters(self):
        """
        Apply all sideways market detection filters.

        Returns:
            bool: True if NOT in sideways market
        """
        # 1. Trend strength filter
        trend_strength = abs(self.aroon.aroonup[0] - self.aroon.aroondown[0])
        if trend_strength < float(self.p.min_trend_strength):
            if self.p.verbose:
                print(f"Entry blocked: Weak trend strength ({trend_strength:.1f} < {float(self.p.min_trend_strength):.1f})")
            return False

        # 2. Both in middle range filter
        if self.p.max_both_middle:
            both_in_middle = (
                    30 <= self.aroon.aroonup[0] <= 70 and
                    30 <= self.aroon.aroondown[0] <= 70
            )
            if both_in_middle:
                if self.p.verbose:
                    print(f"Entry blocked: Both Aroons in middle range (sideways)")
                return False

        # 3. Aroon stability check
        aroon_stable = self._check_aroon_stability()
        if not aroon_stable:
            if self.p.verbose:
                print(f"Entry blocked: Aroon not stable over {self.p.stability_bars} bars")
            return False

        # 4. ADX trend confirmation
        if self.p.use_adx_filter:
            strong_trend = self.dmi.adx[0] > float(self.p.adx_threshold)
            if not strong_trend:
                if self.p.verbose:
                    print(f"Entry blocked: ADX too low ({self.dmi.adx[0]:.1f} < {float(self.p.adx_threshold):.1f})")
                return False

        return True

    def _check_aroon_stability(self):
        """
        Check if Aroon has been stable over recent bars.

        Returns:
            bool: True if Aroon Up consistently > 50 and Down consistently < 50
        """
        if len(self.aroon_up_history) < self.p.stability_bars + 1:
            return True  # Not enough history yet

        # Check last N bars (excluding current)
        for i in range(1, self.p.stability_bars + 1):
            idx = -(i + 1)  # Go back in history
            if idx >= -len(self.aroon_up_history):
                if self.aroon_up_history[idx] < 50:
                    return False
                if self.aroon_down_history[idx] > 50:
                    return False

        return True

    def _check_low_volatility(self):
        """
        Check if current ATR indicates low volatility (calm market).

        Returns:
            bool: True if ATR < baseline * multiplier
        """
        threshold = self.atr_baseline[0] * float(self.p.atr_filter_mult)
        is_low = self.atr_filter[0] < threshold

        return is_low

    def _execute_entry(self):
        """Execute buy order with position sizing."""
        size = self._calculate_position_size()

        if size > 0:
            self.order = self.buy(size=size)
            self._reset_exit_tracking()
            self._log_entry()
        elif self.p.verbose:
            print("SIZE IS 0 OR NEGATIVE, NOT BUYING")

    def _calculate_position_size(self):
        """
        Calculate position size based on equity.

        Returns:
            int: Number of shares to buy
        """
        equity = self.broker.getvalue()
        position_value = equity * float(self.p.position_size_pct)
        size = int(position_value / self.data.close[0])
        return size

    def _reset_exit_tracking(self):
        """Reset all exit tracking variables."""
        self.exiting = False
        self.atr_stop_price = None
        self.position_peak = None
        self.peak_drawdown_stop = None

    def _log_entry(self):
        """Log entry signal details."""
        if not self.p.verbose:
            return

        trend_strength = abs(self.aroon.aroonup[0] - self.aroon.aroondown[0])
        adx_str = f", ADX: {self.dmi.adx[0]:.1f}" if self.p.use_adx_filter else ""

        print(
            f"BUY SIGNAL: Aroon({self.aroon.aroonup[0]:.1f}/{self.aroon.aroondown[0]:.1f}), "
            f"Trend Strength: {trend_strength:.1f}{adx_str}, "
            f"ATR: {self.atr_filter[0]:.2f}"
        )

    def _manage_position(self):
        """Manage stops and exits for existing position."""
        if self.exiting:
            return

        # Update all tracking levels
        self._update_atr_trailing_stop()
        self._update_peak_tracking()

        # Check for exit conditions
        exit_reason = self._check_exit_conditions()

        if exit_reason:
            self._execute_exit(exit_reason)

    def _update_atr_trailing_stop(self):
        """Update ATR-based trailing stop (trails upward only)."""
        atr_stop = (
                self.data.close[0] -
                float(self.p.atr_stop_mult) * self.atr_stop[0]
        )

        if self.atr_stop_price is None:
            self.atr_stop_price = atr_stop
        else:
            self.atr_stop_price = max(self.atr_stop_price, atr_stop)

    def _update_peak_tracking(self):
        """
        Update peak price and dynamic ATR-based drawdown stop.

        Instead of using a fixed percentage, the drawdown threshold is calculated
        as: Peak - (ATR * multiplier), which automatically adapts to each stock's
        volatility characteristics.
        """
        if not self.p.enable_peak_exit:
            return

        # Track highest high since entry
        if self.position_peak is None:
            self.position_peak = self.data.high[0]
        else:
            self.position_peak = max(self.position_peak, self.data.high[0])

        # Calculate current profit percentage
        entry_price = self.position.price
        current_profit_pct = (self.data.close[0] - entry_price) / entry_price

        # Activate peak drawdown stop only after minimum profit reached
        if current_profit_pct >= float(self.p.min_profit_pct_to_activate):
            # Dynamic ATR-based drawdown: Peak - (ATR * multiplier)
            # This automatically adjusts to the stock's volatility
            atr_drawdown = self.atr_peak[0] * float(self.p.peak_atr_mult)
            self.peak_drawdown_stop = self.position_peak - atr_drawdown

            if self.p.verbose and self.peak_drawdown_stop != getattr(self, '_last_logged_peak_stop', None):
                peak_dd_pct = (atr_drawdown / self.position_peak) * 100
                print(
                    f"   Peak tracking: Peak=${self.position_peak:.2f}, "
                    f"Stop=${self.peak_drawdown_stop:.2f} "
                    f"(ATR-based: {atr_drawdown:.2f} / {peak_dd_pct:.2f}%)"
                )
                self._last_logged_peak_stop = self.peak_drawdown_stop
        else:
            self.peak_drawdown_stop = None

    def _check_exit_conditions(self):
        """
        Check all exit conditions.

        Returns:
            str or None: Exit reason if condition met, None otherwise
        """
        current_price = self.data.close[0]
        entry_price = self.position.price

        # Calculate stop/target levels
        stop_loss_price = entry_price * (1 - float(self.p.stop_loss_pct))
        take_profit_price = entry_price * (1 + float(self.p.take_profit_pct))

        # Check exit conditions in priority order

        # 1. Peak drawdown (highest priority if enabled)
        if self.peak_drawdown_stop is not None:
            if current_price <= self.peak_drawdown_stop:
                return "PEAK DRAWDOWN"

        # 2. Take profit
        if current_price >= take_profit_price:
            return "TAKE PROFIT"

        # 3. Stop loss
        if current_price <= stop_loss_price:
            return "STOP LOSS"

        # 4. ATR trailing stop
        if self.atr_stop_price and current_price < self.atr_stop_price:
            return "ATR TRAILING STOP"

        # 5. Bearish Aroon signal
        bearish_signal = self._detect_bearish_signal()
        if bearish_signal:
            return "BEARISH AROON"

        return None

    def _detect_bearish_signal(self):
        """
        Detect bearish Aroon signal (crossover down).

        Returns:
            bool: True if AroonDown crosses above AroonUp with Down > 50
        """
        cross_down = (
                self.aroon.aroondown[0] > self.aroon.aroonup[0] and
                self.aroon.aroondown[-1] <= self.aroon.aroonup[-1]
        )

        return cross_down and self.aroon.aroondown[0] > 50

    def _execute_exit(self, exit_reason):
        """Execute sell order and log exit details."""
        if self.p.verbose:
            entry_price = self.position.price
            current_price = self.data.close[0]
            pnl = (current_price - entry_price) * self.position.size
            pnl_pct = (current_price - entry_price) / entry_price * 100

            peak_str = ""
            if self.position_peak:
                peak_str = f", Peak: ${self.position_peak:.2f}"

            print(
                f"{exit_reason}! | "
                f"Entry: ${entry_price:.2f} | "
                f"Exit: ${current_price:.2f}{peak_str} | "
                f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)"
            )

        self.order = self.sell(size=self.position.size)
        self.exiting = True

    def notify_order(self, order):
        """Handle order status notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if self.p.verbose:
            if order.status == order.Completed:
                self._log_order_completion(order)
            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                print(
                    f"ORDER FAILED: {self.data.datetime.date(0)}, "
                    f"status={order.getstatusname()}"
                )

        self.order = None

    def _log_order_completion(self, order):
        """Log completed order details."""
        order_type = "BUY" if order.isbuy() else "SELL"
        print(
            f"{order_type} Executed: {self.data.datetime.date(0)}, "
            f"size={order.executed.size}, "
            f"price={order.executed.price:.2f}"
        )

        if order.issell():
            self.exiting = False
            self._reset_exit_tracking()