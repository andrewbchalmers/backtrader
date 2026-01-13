from decimal import Decimal
import backtrader as bt

class Strategy(bt.Strategy):
    # Enable plotting for the strategy
    plotinfo = dict(
        plot=True,
        subplot=True,
        plotlinelabels=True
    )

    params = dict(
        # SMA lengths
        fast_len=14,
        slow_len=26,
        # ATR exit
        atr_len=10,
        atr_mult=Decimal("3.0"),
        trend_len=200,
        stop_loss_pct=Decimal("0.1"),
        # Trend filter
        verbose=True,
    )

    def __init__(self):
        # --------------------------
        # Indicators
        # --------------------------
        self.fast = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.fast_len)
        self.slow = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.slow_len)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_len)
        self.crossover = bt.indicators.CrossOver(self.fast, self.slow)

        # 200 MA trend filter
        self.trend_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.trend_len)
        # Calculate slope of 200 MA (positive = rising, negative = falling)
        self.trend_slope = self.trend_ma - self.trend_ma(-1)

        # Track open orders & stops
        self.order = None
        self.stop_price = None  # ATR trailing stop
        self.sl_price = None    # Fixed percentage stop
        self.exiting = False

        # Track when strategy is ready (all indicators have enough data)
        self.indicators_ready = False

    def prenext(self):
        """Called when not all indicators have enough data yet"""
        # Check if all indicators are ready
        # The longest indicator (trend_ma at 200 periods) needs to be ready
        if len(self) >= self.p.trend_len:
            self.indicators_ready = True
            if self.p.verbose:
                print(f"✓ All indicators ready at bar {len(self)} ({self.data.datetime.date(0)})")
        # Don't trade during prenext - indicators aren't ready yet

    # --------------------------
    # Helper: cancel existing order
    # --------------------------
    def cancel_order(self):
        if self.order:
            self.cancel(self.order)
            self.order = None

    # --------------------------
    # Entry signal
    # --------------------------
    def next(self):
        # Safety check - ensure all indicators are ready
        if not self.indicators_ready:
            return

        # Do nothing if an order is pending
        if self.order:
            return

        # --------------------------
        # Entry: fast SMA crosses above slow SMA
        # --------------------------
        if self.crossover > 0:
            # Check 200 MA trend filter - only buy if 200 MA is rising (not negative slope)
            if self.trend_slope[0] < 0:
                if self.p.verbose:
                    print(f"  ENTRY BLOCKED: 200 MA is declining (slope={self.trend_slope[0]:.4f})")
                return

            cash = self.broker.getcash()
            size = int((cash * 0.98) / self.data.close[0])

            if size > 0:
                self.order = self.buy(size=size)
                self.exiting = False
                self.stop_price = None
                self.sl_price = None
                if self.p.verbose:
                    print(f"  200 MA Filter: PASSED (slope={self.trend_slope[0]:.4f})")
            else:
                if self.p.verbose:
                    print(f"  SIZE IS 0 OR NEGATIVE, NOT BUYING")

        # --------------------------
        # Manage existing position exits
        # --------------------------
        if self.position:
            if self.exiting:
                return

            # ATR trailing stop
            atr_stop = self.data.close[0] - float(self.p.atr_mult) * self.atr[0]
            if self.stop_price is None:
                self.stop_price = atr_stop
            else:
                # Trailing only upward for long
                self.stop_price = max(self.stop_price, atr_stop)

            # Fixed % stop-loss
            sl_stop = self.position.price * (1 - float(self.p.stop_loss_pct))
            if self.sl_price is None:
                self.sl_price = sl_stop
            else:
                self.sl_price = max(self.sl_price, sl_stop)

            # Check if either stop is hit
            if self.data.close[0] <= self.stop_price or self.data.close[0] <= self.sl_price:
                if self.data.close[0] <= self.sl_price:
                    if self.p.verbose:
                        print(f"STOP LOSS HIT!")
                self.order = self.sell(size=self.position.size)
                self.stop_price = None
                self.sl_price = None
                return

    # --------------------------
    # Order notification
    # --------------------------
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted - no action required
            return

        if self.p.verbose:
            if order.status in [order.Completed]:
                if order.isbuy():
                    print(f"BUY Executed: {self.data.datetime.date(0)}, size={order.executed.size}, price={order.executed.price}")
                elif order.issell():
                    print(f"SELL Executed: {self.data.datetime.date(0)}, size={order.executed.size}, price={order.executed.price}")
                    self.exiting = False
            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                print(f"ORDER FAILED: {self.data.datetime.date(0)}, status={order.getstatusname()}")

        self.order = None
