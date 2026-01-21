"""
AROON Filter with ATR Strategy

A trend-following strategy that uses:
- Aroon indicator crossovers to identify trend changes
- ATR filter to avoid entering during volatile/noisy markets
- ATR-based trailing stop, fixed stop-loss, and optional take-profit exits

Entry Logic:
    - Aroon Up crosses above Aroon Down (bullish signal)
    - ATR is below its moving average (calm market filter)

Exit Logic:
    - ATR trailing stop (dynamic, follows price)
    - Fixed percentage stop-loss (2% default)
    - Optional take-profit target (13% default)
"""

from decimal import Decimal
import backtrader as bt


class AroonATRStrategy(bt.Strategy):
    """
    Aroon crossover strategy with ATR-based entry filter and multiple exit methods.
    """

    plotinfo = dict(
        plot=True,
        subplot=False,
        plotlinelabels=True
    )

    params = (
        # Aroon indicator
        ('aroon_len', 20),

        # ATR entry filter (identifies calm vs volatile markets)
        ('atr_entry_len', 10),
        ('atr_entry_sma_period', 20),
        ('atr_filter_mult', Decimal('1.2')),  # 1.0 = exact SMA, >1.0 = looser filter
        ('use_atr_filter', True),

        # ATR trailing stop
        ('atr_exit_len', 14),
        ('atr_exit_mult', Decimal('3.0')),

        # Risk management
        ('stop_loss_pct', Decimal('0.05')),    # 2% stop loss
        ('take_profit_pct', Decimal('0.13')),  # 13% take profit
        ('use_take_profit', True),
        ('position_size_pct', Decimal('0.95')),

        # Logging
        ('verbose', True),
    )

    def __init__(self):
        """Initialize indicators and state variables."""
        self._init_indicators()
        self._init_state()

    def _init_indicators(self):
        """Initialize all technical indicators."""
        # Trend identification
        self.aroon = bt.indicators.AroonUpDown(
            self.data,
            period=self.p.aroon_len
        )

        # Volatility measures
        self.atr_entry = bt.indicators.ATR(
            self.data,
            period=self.p.atr_entry_len
        )
        self.atr_exit = bt.indicators.ATR(
            self.data,
            period=self.p.atr_exit_len
        )

        # ATR filter threshold
        self.atr_entry_sma = bt.indicators.SMA(
            self.atr_entry,
            period=self.p.atr_entry_sma_period
        )

    def _init_state(self):
        """Initialize strategy state variables."""
        self.order = None
        self.stop_price = None   # ATR trailing stop level
        self.sl_price = None     # Fixed percentage stop level
        self.tp_price = None     # Take profit target level
        self.exiting = False

    def next(self):
        """Main strategy logic called on each bar."""
        # Force exit on final bar
        if self._is_final_bar():
            self._force_close_position()
            return

        # Skip if order is pending
        if self.order:
            return

        # Try to enter new position
        if not self.position:
            self._check_entry_signal()

        # Manage existing position
        else:
            self._manage_position()

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

    def _check_entry_signal(self):
        """Check for entry conditions and execute buy order if met."""
        # Detect Aroon crossover
        aroon_cross = self._detect_aroon_crossover()

        # Check ATR filter
        atr_is_low, atr_threshold = self._check_atr_filter()

        # Execute entry if conditions met
        if aroon_cross and atr_is_low:
            self._execute_entry(atr_threshold)
        elif aroon_cross and not atr_is_low:
            self._log_blocked_entry(atr_threshold)

    def _detect_aroon_crossover(self):
        """
        Detect bullish Aroon crossover.

        Returns:
            bool: True if Aroon Up crosses above Aroon Down
        """
        return (
                self.aroon.aroonup[0] > self.aroon.aroondown[0] and
                self.aroon.aroonup[-1] <= self.aroon.aroondown[-1]
        )

    def _check_atr_filter(self):
        """
        Check if ATR is below threshold (calm market filter).

        Returns:
            tuple: (is_below_threshold, threshold_value)
        """
        if not self.p.use_atr_filter:
            return True, 0

        atr_threshold = self.atr_entry_sma[0] * float(self.p.atr_filter_mult)
        atr_is_low = self.atr_entry[0] < atr_threshold

        return atr_is_low, atr_threshold

    def _execute_entry(self, atr_threshold):
        """Execute buy order with position sizing."""
        size = self._calculate_position_size()

        if size > 0:
            self.order = self.buy(size=size)
            self._reset_exit_prices()
            self._log_entry(atr_threshold)
        elif self.p.verbose:
            print("SIZE IS 0 OR NEGATIVE, NOT BUYING")

    def _calculate_position_size(self):
        """
        Calculate position size based on available cash.

        Returns:
            int: Number of shares to buy
        """
        cash = self.broker.getcash()
        position_value = cash * float(self.p.position_size_pct)
        size = int(position_value / self.data.close[0])
        return size

    def _reset_exit_prices(self):
        """Reset all exit price tracking variables."""
        self.exiting = False
        self.stop_price = None
        self.sl_price = None
        self.tp_price = None

    def _log_entry(self, atr_threshold):
        """Log entry signal details."""
        if not self.p.verbose:
            return

        if self.p.use_atr_filter:
            pct_below = (
                    (atr_threshold - self.atr_entry[0]) / atr_threshold * 100
            )
            mult_str = (
                f" (threshold: SMA × {float(self.p.atr_filter_mult):.1f})"
                if float(self.p.atr_filter_mult) != 1.0
                else ""
            )
            print(
                f"BUY SIGNAL: AROON Cross + Low ATR | "
                f"ATR: {self.atr_entry[0]:.2f} < {atr_threshold:.2f}{mult_str} "
                f"({pct_below:.1f}% below)"
            )
        else:
            print("BUY SIGNAL: AROON Cross (ATR filter disabled)")

    def _log_blocked_entry(self, atr_threshold):
        """Log when entry is blocked by ATR filter."""
        if not self.p.verbose:
            return

        pct_above = (
            (self.atr_entry[0] - atr_threshold) / atr_threshold * 100
            if atr_threshold > 0
            else 0
        )
        mult_str = (
            f" (×{float(self.p.atr_filter_mult):.1f})"
            if float(self.p.atr_filter_mult) != 1.0
            else ""
        )
        print(
            f"AROON Cross blocked: ATR too high (noisy) | "
            f"ATR: {self.atr_entry[0]:.2f} >= {atr_threshold:.2f}{mult_str} "
            f"(+{pct_above:.1f}% above)"
        )

    def _manage_position(self):
        """Manage stops and exits for existing position."""
        if self.exiting:
            return

        # Update all stop levels
        self._update_atr_trailing_stop()
        self._update_fixed_stop_loss()
        self._update_take_profit()

        # Check for exit conditions
        exit_reason = self._check_exit_conditions()

        if exit_reason:
            self._execute_exit(exit_reason)

    def _update_atr_trailing_stop(self):
        """Update ATR-based trailing stop (trails upward only)."""
        atr_stop = (
                self.data.close[0] -
                float(self.p.atr_exit_mult) * self.atr_exit[0]
        )

        if self.stop_price is None:
            self.stop_price = atr_stop
        else:
            self.stop_price = max(self.stop_price, atr_stop)

    def _update_fixed_stop_loss(self):
        """Update fixed percentage stop-loss."""
        sl_stop = self.position.price * (1 - float(self.p.stop_loss_pct))

        if self.sl_price is None:
            self.sl_price = sl_stop
        else:
            self.sl_price = max(self.sl_price, sl_stop)

    def _update_take_profit(self):
        """Update take profit target."""
        if not self.p.use_take_profit:
            return

        if self.tp_price is None:
            self.tp_price = (
                    self.position.price * (1 + float(self.p.take_profit_pct))
            )

    def _check_exit_conditions(self):
        """
        Check all exit conditions.

        Returns:
            str or None: Exit reason if condition met, None otherwise
        """
        current_price = self.data.close[0]

        # Check in priority order
        if current_price <= self.sl_price:
            return "STOP LOSS HIT"

        if current_price <= self.stop_price:
            return "ATR TRAILING STOP HIT"

        if self.p.use_take_profit and self.tp_price:
            if current_price >= self.tp_price:
                return "TAKE PROFIT HIT"

        return None

    def _execute_exit(self, exit_reason):
        """Execute sell order and log exit details."""
        if self.p.verbose:
            pnl = (
                    (self.data.close[0] - self.position.price) *
                    self.position.size
            )
            print(
                f"{exit_reason}! | "
                f"Entry: ${self.position.price:.2f} | "
                f"Exit: ${self.data.close[0]:.2f} | "
                f"P&L: ${pnl:.2f}"
            )

        self.order = self.sell(size=self.position.size)
        self._reset_exit_prices()

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