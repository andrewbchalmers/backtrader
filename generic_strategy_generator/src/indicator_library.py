"""
Indicator Library - Factory for Creating Backtrader Indicators
"""

import backtrader as bt


class IndicatorFactory:
    """Factory class for creating backtrader indicators dynamically"""

    @staticmethod
    def create_indicator(data, indicator_name, *params):
        """
        Create a backtrader indicator based on name and parameters

        Args:
            data: Backtrader data feed
            indicator_name: Name of the indicator
            *params: Variable parameters for the indicator

        Returns:
            Backtrader indicator instance
        """
        indicator_name = indicator_name.lower().strip()

        # Moving Averages
        if indicator_name == 'sma':
            return IndicatorFactory._create_sma(data, *params)
        elif indicator_name == 'ema':
            return IndicatorFactory._create_ema(data, *params)
        elif indicator_name == 'wma':
            return IndicatorFactory._create_wma(data, *params)
        elif indicator_name == 'dema':
            return IndicatorFactory._create_dema(data, *params)
        elif indicator_name == 'tema':
            return IndicatorFactory._create_tema(data, *params)
        elif indicator_name == 'kama':
            return IndicatorFactory._create_kama(data, *params)
        elif indicator_name == 'zlema':
            return IndicatorFactory._create_zlema(data, *params)
        elif indicator_name == 'hma':
            return IndicatorFactory._create_hma(data, *params)

        # Volatility Indicators
        elif indicator_name == 'bb':
            return IndicatorFactory._create_bb(data, *params)
        elif indicator_name == 'atr':
            return IndicatorFactory._create_atr(data, *params)
        elif indicator_name == 'natr':
            return IndicatorFactory._create_natr(data, *params)
        elif indicator_name == 'keltner':
            return IndicatorFactory._create_keltner(data, *params)
        elif indicator_name == 'donchian':
            return IndicatorFactory._create_donchian(data, *params)

        # Momentum Oscillators
        elif indicator_name == 'rsi':
            return IndicatorFactory._create_rsi(data, *params)
        elif indicator_name == 'stoch':
            return IndicatorFactory._create_stochastic(data, *params)
        elif indicator_name == 'stochrsi':
            return IndicatorFactory._create_stoch_rsi(data, *params)
        elif indicator_name == 'cci':
            return IndicatorFactory._create_cci(data, *params)
        elif indicator_name == 'williams':
            return IndicatorFactory._create_williams_r(data, *params)
        elif indicator_name == 'roc':
            return IndicatorFactory._create_roc(data, *params)
        elif indicator_name == 'momentum':
            return IndicatorFactory._create_momentum(data, *params)
        elif indicator_name == 'zscore':
            return IndicatorFactory._create_zscore(data, *params)
        elif indicator_name == 'hurst':
            return IndicatorFactory._create_hurst(data, *params)
        elif indicator_name == 'varratio':
            return IndicatorFactory._create_variance_ratio(data, *params)
        elif indicator_name == 'ouhalflife':
            return IndicatorFactory._create_ou_halflife(data, *params)
        elif indicator_name == 'rsiret':
            return IndicatorFactory._create_rsi_returns(data, *params)
        elif indicator_name == 'meantouch':
            return IndicatorFactory._create_mean_touch(data, *params)
        elif indicator_name == 'halflifeexit':
            return IndicatorFactory._create_halflife_exit(data, *params)
        elif indicator_name == 'tsi':
            return IndicatorFactory._create_tsi(data, *params)
        elif indicator_name == 'ultimate':
            return IndicatorFactory._create_ultimate_oscillator(data, *params)

        # Trend Indicators
        elif indicator_name == 'macd':
            return IndicatorFactory._create_macd(data, *params)
        elif indicator_name == 'adx':
            return IndicatorFactory._create_adx(data, *params)
        elif indicator_name == 'dmi':
            return IndicatorFactory._create_dmi(data, *params)
        elif indicator_name == 'aroon':
            return IndicatorFactory._create_aroon(data, *params)
        elif indicator_name == 'psar':
            return IndicatorFactory._create_parabolic_sar(data, *params)
        elif indicator_name == 'supertrend':
            return IndicatorFactory._create_supertrend(data, *params)

        # Volume Indicators
        elif indicator_name == 'obv':
            return IndicatorFactory._create_obv(data, *params)
        elif indicator_name == 'vwap':
            return IndicatorFactory._create_vwap(data, *params)
        elif indicator_name == 'mfi':
            return IndicatorFactory._create_mfi(data, *params)
        elif indicator_name == 'adl':
            return IndicatorFactory._create_adl(data, *params)
        elif indicator_name == 'cmf':
            return IndicatorFactory._create_cmf(data, *params)

        # Support/Resistance
        elif indicator_name == 'pivot':
            return IndicatorFactory._create_pivot_point(data, *params)
        elif indicator_name == 'zigzag':
            return IndicatorFactory._create_zigzag(data, *params)

        # Price Action
        elif indicator_name == 'high':
            return IndicatorFactory._create_highest(data, *params)
        elif indicator_name == 'low':
            return IndicatorFactory._create_lowest(data, *params)
        elif indicator_name == 'avgprice':
            return IndicatorFactory._create_avg_price(data, *params)

        else:
            raise ValueError(f"Unknown indicator: {indicator_name}")

    @staticmethod
    def _create_sma(data, period=20):
        """Simple Moving Average"""
        return bt.indicators.SimpleMovingAverage(data.close, period=int(period))

    @staticmethod
    def _create_ema(data, period=20):
        """Exponential Moving Average"""
        return bt.indicators.ExponentialMovingAverage(data.close, period=int(period))

    @staticmethod
    def _create_wma(data, period=20):
        """Weighted Moving Average"""
        return bt.indicators.WeightedMovingAverage(data.close, period=int(period))

    @staticmethod
    def _create_dema(data, period=20):
        """Double Exponential Moving Average"""
        return bt.indicators.DoubleExponentialMovingAverage(data.close, period=int(period))

    @staticmethod
    def _create_tema(data, period=20):
        """Triple Exponential Moving Average"""
        return bt.indicators.TripleExponentialMovingAverage(data.close, period=int(period))

    @staticmethod
    def _create_kama(data, period=30):
        """Kaufman Adaptive Moving Average"""
        return bt.indicators.AdaptiveMovingAverage(data.close, period=int(period))

    @staticmethod
    def _create_zlema(data, period=20):
        """Zero Lag Exponential Moving Average"""
        return bt.indicators.ZeroLagExponentialMovingAverage(data.close, period=int(period))

    @staticmethod
    def _create_hma(data, period=20):
        """Hull Moving Average"""
        return bt.indicators.HullMovingAverage(data.close, period=int(period))

    @staticmethod
    def _create_bb(data, period=20, devfactor=2.0):
        """Bollinger Bands"""
        return bt.indicators.BollingerBands(data.close, period=int(period), devfactor=float(devfactor))

    @staticmethod
    def _create_atr(data, period=14, multiplier=1.0):
        """Average True Range (with optional multiplier for stops)"""
        atr = bt.indicators.AverageTrueRange(data, period=int(period))
        if float(multiplier) != 1.0:
            return atr * float(multiplier)
        return atr

    @staticmethod
    def _create_natr(data, period=14):
        """Normalized Average True Range"""
        atr = bt.indicators.AverageTrueRange(data, period=int(period))
        return (atr / data.close) * 100

    @staticmethod
    def _create_keltner(data, period=20, atrdist=2.0):
        """Keltner Channel - manually implemented

        Keltner Channels use EMA as the middle line and ATR for the bands:
        - Middle line = EMA of typical price
        - Upper band = EMA + (ATR * multiplier)
        - Lower band = EMA - (ATR * multiplier)
        """
        # Calculate typical price (High + Low + Close) / 3
        typical_price = (data.high + data.low + data.close) / 3.0

        # Middle line is EMA of typical price
        middle = bt.indicators.ExponentialMovingAverage(typical_price, period=int(period))

        # Calculate ATR
        atr = bt.indicators.AverageTrueRange(data, period=int(period))

        # Upper and lower bands
        upper = middle + (atr * float(atrdist))
        lower = middle - (atr * float(atrdist))

        # Return middle as primary indicator with top/bot/mid attributes
        middle.top = upper
        middle.bot = lower
        middle.mid = middle

        return middle

    @staticmethod
    def _create_donchian(data, period=20):
        """Donchian Channel - manually implemented"""
        # Donchian Channel shows the highest high and lowest low over a period
        highest = bt.indicators.Highest(data.high, period=int(period))
        lowest = bt.indicators.Lowest(data.low, period=int(period))
        middle = (highest + lowest) / 2.0

        # Return the middle line as the primary indicator
        # Store upper and lower as attributes for breakout strategies
        middle.top = highest
        middle.bot = lowest
        return middle

    @staticmethod
    def _create_rsi(data, period=14):
        """Relative Strength Index"""
        return bt.indicators.RSI(data.close, period=int(period))

    @staticmethod
    def _create_stochastic(data, period=14, period_dfast=3, period_dslow=3):
        """Stochastic Oscillator"""
        return bt.indicators.Stochastic(
            data,
            period=int(period),
            period_dfast=int(period_dfast),
            period_dslow=int(period_dslow)
        )

    @staticmethod
    def _create_stoch_rsi(data, period=14, pfast=3, pslow=3):
        """Stochastic RSI - manually implemented

        StochRSI applies the Stochastic formula to RSI values:
        StochRSI = (RSI - Lowest(RSI, period)) / (Highest(RSI, period) - Lowest(RSI, period)) * 100
        """
        # Calculate RSI
        rsi = bt.indicators.RSI(data.close, period=int(period))

        # Get highest and lowest RSI over the period
        highest_rsi = bt.indicators.Highest(rsi, period=int(period))
        lowest_rsi = bt.indicators.Lowest(rsi, period=int(period))

        # Calculate StochRSI
        stoch_rsi = 100.0 * (rsi - lowest_rsi) / (highest_rsi - lowest_rsi + 0.000001)  # Avoid division by zero

        # Apply smoothing (fast K and slow D)
        if int(pfast) > 1:
            stoch_rsi = bt.indicators.SimpleMovingAverage(stoch_rsi, period=int(pfast))

        if int(pslow) > 1:
            stoch_rsi_slow = bt.indicators.SimpleMovingAverage(stoch_rsi, period=int(pslow))
            # Store slow line as attribute for crossover strategies
            stoch_rsi.slow = stoch_rsi_slow

        return stoch_rsi

    @staticmethod
    def _create_cci(data, period=20):
        """Commodity Channel Index"""
        return bt.indicators.CCI(data, period=int(period))

    @staticmethod
    def _create_williams_r(data, period=14):
        """Williams %R"""
        return bt.indicators.WilliamsR(data, period=int(period))

    @staticmethod
    def _create_roc(data, period=12):
        """Rate of Change"""
        return bt.indicators.RateOfChange(data.close, period=int(period))

    @staticmethod
    def _create_momentum(data, period=12):
        """Momentum"""
        return bt.indicators.Momentum(data.close, period=int(period))

    @staticmethod
    def _create_zscore(data, period=20):
        """Z-Score - measures how many standard deviations price is from its mean

        Z-Score = (Price - SMA) / StdDev

        Useful for mean reversion strategies:
        - Z-Score > 2: Price is 2+ std devs above mean (potentially overbought)
        - Z-Score < -2: Price is 2+ std devs below mean (potentially oversold)
        """
        sma = bt.indicators.SimpleMovingAverage(data.close, period=int(period))
        stddev = bt.indicators.StandardDeviation(data.close, period=int(period))
        zscore = (data.close - sma) / (stddev + 0.000001)  # Avoid division by zero
        return zscore

    @staticmethod
    def _create_hurst(data, period=100):
        """Hurst Exponent - measures persistence/anti-persistence of a time series

        Uses R/S (Rescaled Range) analysis:
        - H > 0.5: Trending/persistent series (momentum strategies work)
        - H = 0.5: Random walk (no predictable pattern)
        - H < 0.5: Mean-reverting/anti-persistent (mean reversion strategies work)

        Typical usage:
        - H > 0.6: Strong trend, use trend-following
        - H < 0.4: Strong mean reversion
        """
        import math

        class HurstExponent(bt.Indicator):
            lines = ('hurst',)
            params = (('period', 100),)

            def __init__(self):
                self.addminperiod(self.p.period)

            def next(self):
                # Get the price series for the period
                prices = [self.data[i] for i in range(-self.p.period + 1, 1)]

                # Calculate returns
                returns = [prices[i] - prices[i-1] for i in range(1, len(prices))]

                if len(returns) < 2:
                    self.lines.hurst[0] = 0.5
                    return

                # Calculate mean of returns
                mean_ret = sum(returns) / len(returns)

                # Calculate cumulative deviation from mean
                cumdev = []
                running_sum = 0
                for r in returns:
                    running_sum += (r - mean_ret)
                    cumdev.append(running_sum)

                # Range of cumulative deviations
                R = max(cumdev) - min(cumdev)

                # Standard deviation of returns
                variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
                S = math.sqrt(variance) if variance > 0 else 0.000001

                # R/S ratio
                if S > 0.000001 and R > 0:
                    RS = R / S
                    # Hurst exponent: H = log(R/S) / log(n)
                    n = len(returns)
                    if RS > 0 and n > 1:
                        self.lines.hurst[0] = math.log(RS) / math.log(n)
                    else:
                        self.lines.hurst[0] = 0.5
                else:
                    self.lines.hurst[0] = 0.5

        return HurstExponent(data.close, period=int(period))

    @staticmethod
    def _create_variance_ratio(data, period=20, k=2):
        """Lo-MacKinlay Variance Ratio - tests for random walk in returns

        Compares variance of k-period returns to variance of 1-period returns:
        VR(k) = Var(k-period returns) / (k * Var(1-period returns))

        Under random walk hypothesis, VR = 1
        - VR > 1: Positive autocorrelation (trending/momentum)
        - VR < 1: Negative autocorrelation (mean-reverting)
        - VR = 1: Random walk (no predictable pattern)

        Typical usage:
        - VR > 1.1: Trending behavior, use momentum strategies
        - VR < 0.9: Mean-reverting behavior, use mean reversion strategies
        """
        import math

        class VarianceRatio(bt.Indicator):
            lines = ('varratio',)
            params = (('period', 20), ('k', 2),)

            def __init__(self):
                # Need enough data for k-period returns over the lookback period
                self.addminperiod(self.p.period * self.p.k)

            def next(self):
                k = self.p.k
                period = self.p.period

                # Get prices for calculation
                # We need period*k + 1 prices to calculate period k-returns
                total_needed = period * k + 1
                prices = [self.data[-i] for i in range(total_needed)]
                prices = prices[::-1]  # Reverse to chronological order

                if len(prices) < total_needed:
                    self.lines.varratio[0] = 1.0
                    return

                # Calculate 1-period returns (log returns for better statistical properties)
                returns_1 = []
                for i in range(1, len(prices)):
                    if prices[i-1] > 0 and prices[i] > 0:
                        returns_1.append(math.log(prices[i] / prices[i-1]))
                    else:
                        returns_1.append(0)

                # Calculate k-period returns
                returns_k = []
                for i in range(k, len(prices)):
                    if prices[i-k] > 0 and prices[i] > 0:
                        returns_k.append(math.log(prices[i] / prices[i-k]))
                    else:
                        returns_k.append(0)

                if len(returns_1) < 2 or len(returns_k) < 2:
                    self.lines.varratio[0] = 1.0
                    return

                # Calculate variances
                mean_1 = sum(returns_1) / len(returns_1)
                var_1 = sum((r - mean_1) ** 2 for r in returns_1) / (len(returns_1) - 1)

                mean_k = sum(returns_k) / len(returns_k)
                var_k = sum((r - mean_k) ** 2 for r in returns_k) / (len(returns_k) - 1)

                # Variance Ratio = Var(k-period) / (k * Var(1-period))
                if var_1 > 0.000001:
                    vr = var_k / (k * var_1)
                    self.lines.varratio[0] = vr
                else:
                    self.lines.varratio[0] = 1.0

        return VarianceRatio(data.close, period=int(period), k=int(k))

    @staticmethod
    def _create_ou_halflife(data, period=50):
        """Ornstein-Uhlenbeck Half-Life - measures mean reversion speed

        The OU process models mean-reverting behavior:
        dP = theta * (mu - P) * dt + sigma * dW

        Half-life = ln(2) / theta = time for price to revert halfway to mean

        Estimated via AR(1) regression: P(t) = a + b * P(t-1) + error
        Then: half-life = -ln(2) / ln(b)

        Interpretation:
        - Short half-life (< 10 days): Fast mean reversion, good for mean reversion strategies
        - Medium half-life (10-50 days): Moderate mean reversion
        - Long half-life (> 50 days): Slow/no mean reversion, closer to random walk

        Lower values indicate stronger mean reversion opportunities.
        """
        import math

        class OUHalfLife(bt.Indicator):
            lines = ('halflife',)
            params = (('period', 50),)

            def __init__(self):
                self.addminperiod(self.p.period)

            def next(self):
                period = self.p.period

                # Get price series
                prices = [self.data[-i] for i in range(period)]
                prices = prices[::-1]  # Reverse to chronological order

                if len(prices) < period:
                    self.lines.halflife[0] = 100.0  # Default to long half-life
                    return

                # AR(1) regression: P(t) = a + b * P(t-1)
                # We need to estimate b using least squares
                # y = P(t), x = P(t-1)
                y = prices[1:]  # P(t)
                x = prices[:-1]  # P(t-1)

                n = len(x)
                if n < 2:
                    self.lines.halflife[0] = 100.0
                    return

                # Calculate means
                mean_x = sum(x) / n
                mean_y = sum(y) / n

                # Calculate slope (b) using least squares
                # b = sum((x - mean_x) * (y - mean_y)) / sum((x - mean_x)^2)
                numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
                denominator = sum((x[i] - mean_x) ** 2 for i in range(n))

                if denominator < 0.000001:
                    self.lines.halflife[0] = 100.0
                    return

                b = numerator / denominator

                # Half-life = -ln(2) / ln(b)
                # b should be between 0 and 1 for mean reversion
                # b close to 1 means slow reversion (random walk)
                # b close to 0 means fast reversion

                if b <= 0 or b >= 1:
                    # No valid mean reversion (trending or invalid)
                    self.lines.halflife[0] = 100.0
                else:
                    halflife = -math.log(2) / math.log(b)
                    # Cap at reasonable values
                    halflife = max(0.1, min(halflife, 500.0))
                    self.lines.halflife[0] = halflife

        return OUHalfLife(data.close, period=int(period))

    @staticmethod
    def _create_rsi_returns(data, period=14):
        """RSI of Returns - applies RSI to price returns instead of price

        Standard RSI measures overbought/oversold based on price movement.
        RSI of Returns measures momentum in the returns themselves.

        Calculation:
        1. Calculate daily returns: ret = (close - close[-1]) / close[-1]
        2. Apply RSI formula to the returns series

        Interpretation (similar to standard RSI):
        - RSI_ret > 70: Returns have been consistently positive (overbought momentum)
        - RSI_ret < 30: Returns have been consistently negative (oversold momentum)
        - RSI_ret = 50: Neutral momentum

        Can be more responsive than price-based RSI for momentum detection.
        """
        # Calculate percentage returns
        returns = (data.close - data.close(-1)) / data.close(-1) * 100

        # Apply RSI to returns
        # RSI needs positive values, so we shift returns by adding 100
        # This transforms returns from roughly (-10, +10) to (90, 110)
        shifted_returns = returns + 100

        return bt.indicators.RSI(shifted_returns, period=int(period))

    @staticmethod
    def _create_mean_touch(data, period=20):
        """Mean Touch Exit - measures distance from mean for mean reversion exits

        Returns a normalized value indicating how close price is to the mean:
        - Value near 0: Price is at or near the mean (signal to exit MR trade)
        - Value > 0: Price is above the mean
        - Value < 0: Price is below the mean

        For mean reversion exits:
        - If you bought on oversold (below mean), exit when value approaches 0 or goes positive
        - The indicator returns (close - SMA) / SMA * 100 (percentage deviation)

        Typical exit signal: When value crosses above 0 (price touches mean from below)
        """
        sma = bt.indicators.SimpleMovingAverage(data.close, period=int(period))
        # Percentage deviation from mean
        mean_touch = (data.close - sma) / sma * 100
        return mean_touch

    @staticmethod
    def _create_halflife_exit(data, period=50):
        """Half-Life Exit Timer - signals based on OU mean reversion timing

        Uses the Ornstein-Uhlenbeck half-life to create an exit timing signal.
        The idea: if you enter a mean reversion trade, you should expect
        the price to revert within approximately 1-2 half-lives.

        Returns a value from 0 to 100:
        - 0: Just entered (or half-life is very long)
        - 50: One half-life has passed since significant deviation
        - 100: Two half-lives have passed (strong exit signal)

        The indicator tracks bars since price was significantly deviated from mean
        and compares to the estimated half-life.

        Typical exit: When value > 50-70 (one half-life elapsed)
        """
        import math

        class HalfLifeExit(bt.Indicator):
            lines = ('exit_signal',)
            params = (('period', 50), ('deviation_threshold', 1.5),)

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

        return HalfLifeExit(data.close, period=int(period))

    @staticmethod
    def _create_tsi(data, period1=25, period2=13):
        """True Strength Index"""
        return bt.indicators.TrueStrengthIndicator(data.close, period1=int(period1), period2=int(period2))

    @staticmethod
    def _create_ultimate_oscillator(data, p1=7, p2=14, p3=28):
        """Ultimate Oscillator"""
        return bt.indicators.UltimateOscillator(data, p1=int(p1), p2=int(p2), p3=int(p3))

    @staticmethod
    def _create_macd(data, period_me1=12, period_me2=26, period_signal=9):
        """MACD"""
        return bt.indicators.MACD(
            data.close,
            period_me1=int(period_me1),
            period_me2=int(period_me2),
            period_signal=int(period_signal)
        )

    @staticmethod
    def _create_adx(data, period=14):
        """Average Directional Index"""
        return bt.indicators.AverageDirectionalMovementIndex(data, period=int(period))

    @staticmethod
    def _create_dmi(data, period=14):
        """Directional Movement Index"""
        return bt.indicators.DirectionalMovementIndex(data, period=int(period))

    @staticmethod
    def _create_aroon(data, period=25):
        """Aroon Indicator"""
        return bt.indicators.AroonIndicator(data, period=int(period))

    @staticmethod
    def _create_parabolic_sar(data, af=0.02, afmax=0.20):
        """Parabolic SAR"""
        return bt.indicators.ParabolicSAR(data, af=float(af), afmax=float(afmax))

    @staticmethod
    def _create_supertrend(data, period=10, multiplier=3.0):
        """SuperTrend Indicator"""
        atr = bt.indicators.AverageTrueRange(data, period=int(period))
        hl_avg = (data.high + data.low) / 2
        upper = hl_avg + (float(multiplier) * atr)
        lower = hl_avg - (float(multiplier) * atr)
        return upper, lower  # Returns tuple

    @staticmethod
    def _create_obv(data, period=1):
        """On Balance Volume - manually implemented

        OBV adds/subtracts volume based on price direction:
        - If close > previous close: OBV += volume
        - If close < previous close: OBV -= volume
        - If close = previous close: OBV unchanged
        """
        # Determine if price went up or down
        price_change = data.close - data.close(-1)

        # Volume is positive when price goes up, negative when down
        signed_volume = bt.If(price_change > 0, data.volume,
                              bt.If(price_change < 0, -data.volume, 0.0))

        # OBV is the cumulative sum of signed volume
        # Use a custom indicator to maintain running sum
        class OBV(bt.Indicator):
            lines = ('obv',)

            def __init__(self):
                self.addminperiod(1)

            def next(self):
                if len(self) == 1:
                    self.lines.obv[0] = self.data[0]
                else:
                    self.lines.obv[0] = self.lines.obv[-1] + self.data[0]

        return OBV(signed_volume)

    @staticmethod
    def _create_vwap(data, period=1):
        """Volume Weighted Average Price"""
        # Simple VWAP calculation
        typical_price = (data.high + data.low + data.close) / 3
        return bt.indicators.SumN(typical_price * data.volume, period=int(period)) / bt.indicators.SumN(data.volume, period=int(period))

    @staticmethod
    def _create_mfi(data, period=14):
        """Money Flow Index - manually implemented

        MFI is like RSI but uses volume:
        1. Typical Price = (High + Low + Close) / 3
        2. Money Flow = Typical Price * Volume
        3. Positive Flow = sum of money flow when price increases
        4. Negative Flow = sum of money flow when price decreases
        5. MFI = 100 - (100 / (1 + Money Flow Ratio))
        """
        # Calculate typical price
        typical_price = (data.high + data.low + data.close) / 3.0

        # Calculate money flow (typical price * volume)
        money_flow = typical_price * data.volume

        # Create a line that's 1 when price goes up, 0 when down
        # Compare current typical price to previous
        up_down = typical_price - typical_price(-1)

        # Positive money flow (when price increases)
        positive_flow = bt.If(up_down > 0, money_flow, 0.0)

        # Negative money flow (when price decreases)
        negative_flow = bt.If(up_down < 0, money_flow, 0.0)

        # Sum over period
        positive_mf_sum = bt.indicators.SumN(positive_flow, period=int(period))
        negative_mf_sum = bt.indicators.SumN(negative_flow, period=int(period))

        # Money Flow Ratio
        mf_ratio = positive_mf_sum / (negative_mf_sum + 0.000001)  # Avoid division by zero

        # MFI
        mfi = 100.0 - (100.0 / (1.0 + mf_ratio))

        return mfi

    @staticmethod
    def _create_adl(data, period=1):
        """Accumulation/Distribution Line - manually implemented

        ADL measures the cumulative flow of money in/out:
        1. Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        2. Money Flow Volume = Money Flow Multiplier * Volume
        3. ADL = cumulative sum of Money Flow Volume
        """
        # Money Flow Multiplier
        mf_multiplier = ((data.close - data.low) - (data.high - data.close)) / (data.high - data.low + 0.000001)

        # Money Flow Volume
        mf_volume = mf_multiplier * data.volume

        # ADL is cumulative sum
        class ADL(bt.Indicator):
            lines = ('adl',)

            def __init__(self):
                self.addminperiod(1)

            def next(self):
                if len(self) == 1:
                    self.lines.adl[0] = self.data[0]
                else:
                    self.lines.adl[0] = self.lines.adl[-1] + self.data[0]

        return ADL(mf_volume)

    @staticmethod
    def _create_cmf(data, period=20):
        """Chaikin Money Flow"""
        mfv = ((data.close - data.low) - (data.high - data.close)) / (data.high - data.low) * data.volume
        return bt.indicators.SumN(mfv, period=int(period)) / bt.indicators.SumN(data.volume, period=int(period))

    @staticmethod
    def _create_pivot_point(data, period=1):
        """Pivot Point"""
        return bt.indicators.PivotPoint(data)

    @staticmethod
    def _create_zigzag(data, deviation=5.0):
        """ZigZag Indicator"""
        # Simple ZigZag implementation
        return bt.indicators.Highest(data.high, period=int(deviation))

    @staticmethod
    def _create_highest(data, period=20):
        """Highest value over period"""
        return bt.indicators.Highest(data.high, period=int(period))

    @staticmethod
    def _create_lowest(data, period=20):
        """Lowest value over period"""
        return bt.indicators.Lowest(data.low, period=int(period))

    @staticmethod
    def _create_avg_price(data, period=1):
        """Average Price"""
        return (data.open + data.high + data.low + data.close) / 4


class IndicatorSignal:
    """Helper class to define indicator signals and conditions"""

    @staticmethod
    def crossover(ind1, ind2):
        """Returns True when ind1 crosses above ind2"""
        return bt.indicators.CrossOver(ind1, ind2)

    @staticmethod
    def crossunder(ind1, ind2):
        """Returns True when ind1 crosses below ind2"""
        return bt.indicators.CrossDown(ind1, ind2)

    @staticmethod
    def above(ind1, ind2):
        """Returns True when ind1 is above ind2"""
        return ind1 > ind2

    @staticmethod
    def below(ind1, ind2):
        """Returns True when ind1 is below ind2"""
        return ind1 < ind2

    @staticmethod
    def above_threshold(indicator, threshold):
        """Returns True when indicator is above threshold"""
        return indicator > threshold

    @staticmethod
    def below_threshold(indicator, threshold):
        """Returns True when indicator is below threshold"""
        return indicator < threshold


def parse_indicator_csv(csv_path):
    """
    Parse indicator CSV file and return list of indicator definitions.
    Automatically detects single vs dual indicators.

    Single format: indicator_name, param1, param2, ...
    Dual format: indicator1, param1, indicator2, param2, ...

    Args:
        csv_path: Path to CSV file

    Returns:
        Tuple of (single_indicators, dual_indicators)
        - single_indicators: List of (indicator_name, [params]) tuples
        - dual_indicators: List of ((ind1_name, [params1]), (ind2_name, [params2])) tuples
    """
    import csv

    single_indicators = []
    dual_indicators = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for line_num, row in enumerate(reader, 1):
            if not row or row[0].startswith('#'):
                continue

            # Strip inline comments from all cells (e.g., "3  # comment" -> "3")
            cleaned_row = []
            for cell in row:
                if '#' in cell:
                    cell = cell.split('#')[0]
                cleaned_row.append(cell.strip())
            row = cleaned_row

            # Skip if row is now empty after comment stripping
            if not row or not row[0]:
                continue

            # Strategy: Collect all params after first value until we hit another non-numeric
            indicator_name = row[0].strip()
            params = []
            second_indicator_idx = None

            for i in range(1, len(row)):
                cell = row[i].strip()
                if not cell:
                    continue

                try:
                    # Try to parse as number
                    float(cell.replace(',', '.'))
                    params.append(cell)
                except ValueError:
                    # Found non-numeric - this is a second indicator (dual mode)
                    second_indicator_idx = i
                    break

            # Determine if this is single or dual indicator
            if second_indicator_idx is None:
                # Single indicator: only numeric params found
                single_indicators.append((indicator_name, params))
            else:
                # Dual indicator: found second indicator name
                ind1_name = indicator_name
                ind1_params = params
                ind2_name = row[second_indicator_idx].strip()
                ind2_params = [p.strip() for p in row[second_indicator_idx+1:] if p.strip()]

                dual_indicators.append(((ind1_name, ind1_params), (ind2_name, ind2_params)))

    return single_indicators, dual_indicators


def parse_dual_indicator_csv(csv_path):
    """
    Parse dual indicator CSV file for crossover strategies
    Format: indicator1,param1,[param2,...],indicator2,param1,[param2,...]

    Args:
        csv_path: Path to CSV file

    Returns:
        List of tuples ((indicator1_name, [params1]), (indicator2_name, [params2]))
    """
    import csv
    import os

    if not os.path.exists(csv_path):
        return []

    dual_indicators = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for line_num, row in enumerate(reader, 1):
            if not row or row[0].strip().startswith('#'):
                continue

            # Strip inline comments from all cells (e.g., "3  # comment" -> "3")
            cleaned_row = []
            for cell in row:
                if '#' in cell:
                    cell = cell.split('#')[0]
                cleaned_row.append(cell.strip())
            row = cleaned_row

            # Skip if row is now empty after comment stripping
            if not row or not row[0]:
                continue

            # Expected format: ind1, param1, [param2, ...], ind2, param1, [param2, ...]
            # Strategy: Collect numeric params after ind1, then find the next non-numeric (ind2)

            if len(row) < 3:
                print(f"Warning line {line_num}: Dual indicator needs at least 3 values (ind1, param, ind2). Skipping: {row}")
                continue

            ind1_name = row[0].strip()
            ind1_params = []
            ind2_start = None

            # Collect parameters for first indicator (all numeric values after ind1_name)
            for i in range(1, len(row)):
                cell = row[i].strip()
                if not cell:
                    continue

                try:
                    # Try to parse as number
                    float(cell.replace(',', '.'))
                    ind1_params.append(cell)
                except ValueError:
                    # Found non-numeric, this must be the second indicator name
                    ind2_start = i
                    break

            if ind2_start is None or ind2_start >= len(row):
                print(f"Warning line {line_num}: Could not find second indicator in row: {row}")
                continue

            # Parse second indicator
            ind2_name = row[ind2_start].strip()
            ind2_params = []
            for i in range(ind2_start + 1, len(row)):
                cell = row[i].strip()
                if cell:
                    ind2_params.append(cell)

            if not ind1_params:
                print(f"Warning line {line_num}: First indicator has no parameters: {row}")
                continue

            dual_indicators.append(((ind1_name, ind1_params), (ind2_name, ind2_params)))

    return dual_indicators


def parse_filter_csv(csv_path):
    """
    Parse filter indicator CSV file
    Format: indicator,param1,[param2,...],filter_type

    Filter types: above, below, rising, high

    Args:
        csv_path: Path to CSV file

    Returns:
        List of tuples (indicator_name, [params], filter_type)
    """
    import csv
    import os

    if not os.path.exists(csv_path):
        print(f"DEBUG: Filter file not found: {csv_path}")
        return []

    print(f"DEBUG: Parsing filter file: {csv_path}")
    filters = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for line_num, row in enumerate(reader, 1):
            if not row or row[0].strip().startswith('#'):
                continue

            if len(row) < 2:
                print(f"Warning line {line_num}: Filter needs at least indicator and filter_type. Skipping: {row}")
                continue

            # Format: indicator, param1, [param2, ...], filter_type
            # Last column is always the filter type
            indicator_name = row[0].strip()

            # Strip inline comments from filter_type (e.g., "above  # comment" -> "above")
            filter_type_raw = row[-1].strip()
            if '#' in filter_type_raw:
                filter_type = filter_type_raw.split('#')[0].strip().lower()
            else:
                filter_type = filter_type_raw.lower()

            # Everything in between is parameters
            params = [p.strip() for p in row[1:-1] if p.strip()]

            print(f"DEBUG: Line {line_num}: indicator={indicator_name}, params={params}, filter_type={filter_type}")

            # Validate filter type
            valid_types = ['above', 'below', 'rising', 'high']
            if filter_type not in valid_types:
                print(f"Warning line {line_num}: Invalid filter type '{filter_type}'. Must be one of {valid_types}. Skipping.")
                continue

            filters.append((indicator_name, params, filter_type))
            print(f"DEBUG: Added filter: {filters[-1]}")

    print(f"DEBUG: Total filters parsed: {len(filters)}")
    return filters


def get_indicator_description(indicator_name, params):
    """
    Get a human-readable description of an indicator

    Args:
        indicator_name: Name of the indicator
        params: List of parameters

    Returns:
        String description
    """
    params_str = ','.join(str(p) for p in params)
    return f"{indicator_name.upper()}({params_str})"