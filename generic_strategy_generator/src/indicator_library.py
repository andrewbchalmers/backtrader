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
            indicator_name: Name of the indicator (sma, ema, bb, atr, rsi, macd)
            *params: Variable parameters for the indicator
            
        Returns:
            Backtrader indicator instance
        """
        indicator_name = indicator_name.lower().strip()
        
        if indicator_name == 'sma':
            return IndicatorFactory._create_sma(data, *params)
        elif indicator_name == 'ema':
            return IndicatorFactory._create_ema(data, *params)
        elif indicator_name == 'bb':
            return IndicatorFactory._create_bb(data, *params)
        elif indicator_name == 'atr':
            return IndicatorFactory._create_atr(data, *params)
        elif indicator_name == 'rsi':
            return IndicatorFactory._create_rsi(data, *params)
        elif indicator_name == 'macd':
            return IndicatorFactory._create_macd(data, *params)
        elif indicator_name == 'stoch':
            return IndicatorFactory._create_stochastic(data, *params)
        elif indicator_name == 'cci':
            return IndicatorFactory._create_cci(data, *params)
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
    def _create_rsi(data, period=14):
        """Relative Strength Index"""
        return bt.indicators.RSI(data.close, period=int(period))
    
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
    def _create_stochastic(data, period=14, period_dfast=3, period_dslow=3):
        """Stochastic Oscillator"""
        return bt.indicators.Stochastic(
            data,
            period=int(period),
            period_dfast=int(period_dfast),
            period_dslow=int(period_dslow)
        )
    
    @staticmethod
    def _create_cci(data, period=20):
        """Commodity Channel Index"""
        return bt.indicators.CCI(data, period=int(period))


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
    Parse indicator CSV file and return list of indicator definitions
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        List of tuples (indicator_name, [params])
    """
    import csv
    
    indicators = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            indicator_name = row[0].strip()
            params = [p.strip() for p in row[1:] if p.strip()]
            indicators.append((indicator_name, params))
    
    return indicators


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
