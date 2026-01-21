"""
RSI Donchian Half-Life Strategy

A mean-reversion strategy that uses:
- RSI threshold for oversold/overbought conditions
- Donchian channel filter to confirm price is above upper band (breakout confirmation)
- Half-Life exit timer for mean reversion timing
- Fixed stop-loss and take-profit exits

Entry Logic:
    - RSI(14) crosses below threshold (e.g., 30 for oversold)
    - Price is above Donchian(20) upper band (confirming strength)

Exit Logic:
    - Half-Life exit timer crosses threshold (mean reversion expected)
    - Fixed percentage stop-loss (5% default)
    - Fixed take-profit target (8% default)

Based on generic_strategy_generator results:
    Total Return: 4.41%
    Sharpe Ratio: 0.660
    Max Drawdown: 10.81%
    Profit Factor: 3.69
    Win Rate: 71.43%
"""

from decimal import Decimal
import backtrader as bt
import math


class HalfLifeExit(bt.Indicator):
    """
    Half-Life Exit Timer - signals based on Ornstein-Uhlenbeck mean reversion timing.

    Uses the OU half-life to create an exit timing signal.
    The idea: if you enter a mean reversion trade, you should expect
    the price to revert within approximately 1-2 half-lives.

    Returns a value from 0 to 100:
    - 0: Just entered (or half-life is very long)
    - 50: One half-life has passed since significant deviation
    - 100: Two half-lives have passed (strong exit signal)
    """
    lines = ('exit_signal',)
    params = (
        ('period', 50),
        ('deviation_threshold', 1.5),
    )

    def __init__(self):
        self.addminperiod(self.p.period)
        self.bars_since_entry = 0
        self.current_halflife = 50  # Default estimate
        self.in_trade = False

    def next(self):
        period = self.p.period

        # Calculate current z-score to detect entry/exit conditions
        prices = [self.data[-i] for i in range(min(period, len(self)))]
        if len(prices) < 10:
            self.lines.exit_signal[0] = 0
            return

        prices = prices[::-1]
        mean_price = sum(prices) / len(prices)
        variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
        std_price = math.sqrt(variance) if variance > 0 else 0.001

        current_zscore = (self.data[0] - mean_price) / std_price

        # Estimate half-life using AR(1)
        if len(prices) >= 20:
            y = prices[1:]
            x = prices[:-1]
            n = len(x)
            mean_x = sum(x) / n
            mean_y = sum(y) / n
            numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
            denominator = sum((xi - mean_x) ** 2 for xi in x)

            if denominator > 0.0001:
                b = numerator / denominator
                if 0 < b < 1:
                    self.current_halflife = -math.log(2) / math.log(b)
                    self.current_halflife = max(1, min(self.current_halflife, 200))

        # Track trade timing
        # If price is significantly deviated, reset the counter (new entry)
        if abs(current_zscore) > self.p.deviation_threshold:
            self.bars_since_entry = 0
            self.in_trade = True
        elif self.in_trade:
            self.bars_since_entry += 1

        # Calculate exit signal (0-100 scale)
        if self.current_halflife > 0 and self.in_trade:
            # How many half-lives have passed?
            halflives_passed = self.bars_since_entry / self.current_halflife
            # Scale to 0-100, capping at 2 half-lives = 100
            exit_signal = min(100, halflives_passed * 50)
            self.lines.exit_signal[0] = exit_signal
        else:
            self.lines.exit_signal[0] = 0

        # Reset if price returned to mean
        if abs(current_zscore) < 0.5:
            self.in_trade = False
            self.bars_since_entry = 0


class RSIDonchianHalfLifeStrategy(bt.Strategy):
    """
    RSI threshold entry with Donchian filter and Half-Life exit strategy.
    """

    plotinfo = dict(
        plot=True,
        subplot=False,
        plotlinelabels=True
    )

    params = (
        # RSI parameters
        ('rsi_period', 14),
        ('rsi_threshold', 30),  # Buy when RSI < threshold (oversold)

        # Donchian filter
        ('donchian_period', 20),
        ('use_donchian_filter', True),

        # Half-Life exit
        ('halflife_period', 50),
        ('halflife_exit_threshold', 50),  # Exit when halflife signal > threshold

        # Risk management
        ('stop_loss_pct', Decimal('0.05')),    # 5% stop loss
        ('take_profit_pct', Decimal('0.08')),  # 8% take profit
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
        # RSI for entry signal
        self.rsi = bt.indicators.RSI(
            self.data.close,
            period=self.p.rsi_period
        )

        # Donchian channel for filter
        self.donchian_high = bt.indicators.Highest(
            self.data.high,
            period=self.p.donchian_period
        )
        self.donchian_low = bt.indicators.Lowest(
            self.data.low,
            period=self.p.donchian_period
        )

        # Half-Life exit timer
        self.halflife_exit = HalfLifeExit(
            self.data.close,
            period=self.p.halflife_period
        )

    def _init_state(self):
        """Initialize strategy state variables."""
        self.order = None
        self.sl_price = None     # Fixed percentage stop level
        self.tp_price = None     # Take profit target level
        self.exiting = False
        self.prev_halflife = 0   # For crossover detection

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

        # Update previous halflife for crossover detection
        self.prev_halflife = self.halflife_exit.exit_signal[0]

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
        # RSI threshold condition (oversold)
        rsi_signal = self.rsi[0] < self.p.rsi_threshold

        # Donchian filter - price above upper band (strength confirmation)
        if self.p.use_donchian_filter:
            donchian_filter = self.data.close[0] > self.donchian_high[-1]
        else:
            donchian_filter = True

        # Execute entry if conditions met
        if rsi_signal and donchian_filter:
            self._execute_entry()
        elif rsi_signal and not donchian_filter and self.p.verbose:
            print(
                f"RSI signal blocked: Price ${self.data.close[0]:.2f} "
                f"not above Donchian high ${self.donchian_high[-1]:.2f}"
            )

    def _execute_entry(self):
        """Execute buy order with position sizing."""
        size = self._calculate_position_size()

        if size > 0:
            self.order = self.buy(size=size)
            self._reset_exit_prices()
            self._log_entry()
        elif self.p.verbose:
            print("SIZE IS 0 OR NEGATIVE, NOT BUYING")

    def _calculate_position_size(self):
        """Calculate position size based on available cash."""
        cash = self.broker.getcash()
        position_value = cash * float(self.p.position_size_pct)
        size = int(position_value / self.data.close[0])
        return size

    def _reset_exit_prices(self):
        """Reset all exit price tracking variables."""
        self.exiting = False
        self.sl_price = None
        self.tp_price = None

    def _log_entry(self):
        """Log entry signal details."""
        if not self.p.verbose:
            return

        print(
            f"BUY SIGNAL: RSI({self.p.rsi_period})={self.rsi[0]:.1f} < {self.p.rsi_threshold} | "
            f"Price ${self.data.close[0]:.2f} > Donchian({self.p.donchian_period}) ${self.donchian_high[-1]:.2f}"
        )

    def _manage_position(self):
        """Manage stops and exits for existing position."""
        if self.exiting:
            return

        # Update stop levels
        self._update_fixed_stop_loss()
        self._update_take_profit()

        # Check for exit conditions
        exit_reason = self._check_exit_conditions()

        if exit_reason:
            self._execute_exit(exit_reason)

    def _update_fixed_stop_loss(self):
        """Update fixed percentage stop-loss."""
        if self.sl_price is None:
            self.sl_price = self.position.price * (1 - float(self.p.stop_loss_pct))

    def _update_take_profit(self):
        """Update take profit target."""
        if not self.p.use_take_profit:
            return

        if self.tp_price is None:
            self.tp_price = self.position.price * (1 + float(self.p.take_profit_pct))

    def _check_exit_conditions(self):
        """Check all exit conditions."""
        current_price = self.data.close[0]

        # Check stop loss first (highest priority)
        if current_price <= self.sl_price:
            return "STOP LOSS HIT"

        # Check take profit
        if self.p.use_take_profit and self.tp_price:
            if current_price >= self.tp_price:
                return "TAKE PROFIT HIT"

        # Check half-life exit (crossover above threshold)
        if (self.halflife_exit.exit_signal[0] > self.p.halflife_exit_threshold and
                self.prev_halflife <= self.p.halflife_exit_threshold):
            return f"HALF-LIFE EXIT (signal={self.halflife_exit.exit_signal[0]:.1f})"

        return None

    def _execute_exit(self, exit_reason):
        """Execute sell order and log exit details."""
        if self.p.verbose:
            pnl = (self.data.close[0] - self.position.price) * self.position.size
            pnl_pct = (self.data.close[0] / self.position.price - 1) * 100
            print(
                f"{exit_reason}! | "
                f"Entry: ${self.position.price:.2f} | "
                f"Exit: ${self.data.close[0]:.2f} | "
                f"P&L: ${pnl:.2f} ({pnl_pct:+.1f}%)"
            )

        self.order = self.sell(size=self.position.size)
        self._reset_exit_prices()
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


# Alias for import compatibility
Strategy = RSIDonchianHalfLifeStrategy
