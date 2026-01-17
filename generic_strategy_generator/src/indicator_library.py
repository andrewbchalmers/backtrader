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
            filter_type = row[-1].strip().lower()

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