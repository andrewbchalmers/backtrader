"""
Stock Classifier - Categorize stocks by behavior before strategy testing

Classification Dimensions:
1. TREND BEHAVIOR: Trending vs Mean-Reverting vs Mixed
   - Uses: ADX (trend strength)

2. VOLATILITY: High vs Medium vs Low
   - Uses: ATR as percentage of price

3. BREAKOUT PROPENSITY: Breakout-prone vs Range-bound vs Mixed
   - Uses: Donchian channel break frequency

4. PRICE TIER: Low (<$20) vs Mid ($20-100) vs High (>$100)
   - Uses: Average closing price

5. MOVEMENT INTENSITY: High-Beta vs Medium-Beta vs Low-Beta
   - Uses: Average daily return magnitude

6. LIQUIDITY: High vs Medium vs Low
   - Uses: Average daily dollar volume
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import json
import os


class TrendBehavior(Enum):
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    MIXED = "mixed"


class VolatilityLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class BreakoutPropensity(Enum):
    BREAKOUT_PRONE = "breakout_prone"
    RANGE_BOUND = "range_bound"
    MIXED = "mixed"


class PriceTier(Enum):
    LOW = "low"           # < $20
    MID = "mid"           # $20 - $100
    HIGH = "high"         # > $100


class MovementIntensity(Enum):
    HIGH_BETA = "high_beta"
    MEDIUM_BETA = "medium_beta"
    LOW_BETA = "low_beta"


class LiquidityLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class StockClassification:
    """Classification result for a single stock"""
    symbol: str
    trend_behavior: TrendBehavior
    volatility_level: VolatilityLevel
    breakout_propensity: BreakoutPropensity
    price_tier: PriceTier
    movement_intensity: MovementIntensity
    liquidity_level: LiquidityLevel

    # Raw metrics for debugging/analysis
    avg_adx: float
    avg_atr_pct: float
    breakout_frequency: float
    avg_price: float
    avg_daily_return_magnitude: float
    avg_dollar_volume: float

    # Composite classification
    strategy_affinity: List[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'symbol': self.symbol,
            'trend_behavior': self.trend_behavior.value,
            'volatility_level': self.volatility_level.value,
            'breakout_propensity': self.breakout_propensity.value,
            'price_tier': self.price_tier.value,
            'movement_intensity': self.movement_intensity.value,
            'liquidity_level': self.liquidity_level.value,
            'avg_adx': self.avg_adx,
            'avg_atr_pct': self.avg_atr_pct,
            'breakout_frequency': self.breakout_frequency,
            'avg_price': self.avg_price,
            'avg_daily_return_magnitude': self.avg_daily_return_magnitude,
            'avg_dollar_volume': self.avg_dollar_volume,
            'strategy_affinity': self.strategy_affinity
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'StockClassification':
        """Create from dictionary (for loading from cache)"""
        return cls(
            symbol=data['symbol'],
            trend_behavior=TrendBehavior(data['trend_behavior']),
            volatility_level=VolatilityLevel(data['volatility_level']),
            breakout_propensity=BreakoutPropensity(data['breakout_propensity']),
            price_tier=PriceTier(data['price_tier']),
            movement_intensity=MovementIntensity(data['movement_intensity']),
            liquidity_level=LiquidityLevel(data['liquidity_level']),
            avg_adx=data['avg_adx'],
            avg_atr_pct=data['avg_atr_pct'],
            breakout_frequency=data['breakout_frequency'],
            avg_price=data['avg_price'],
            avg_daily_return_magnitude=data['avg_daily_return_magnitude'],
            avg_dollar_volume=data['avg_dollar_volume'],
            strategy_affinity=data['strategy_affinity']
        )


class StockClassifier:
    """
    Classifies stocks by behavioral characteristics using technical metrics.

    Uses ADX, ATR, and price action analysis to categorize stocks into
    groups suitable for different strategy types.
    """

    def __init__(self, config: dict):
        """
        Initialize with classification thresholds from config.

        Args:
            config: Configuration dictionary containing thresholds
        """
        self.config = config
        classification_config = config.get('classification', {})
        self.thresholds = classification_config.get('thresholds', self._default_thresholds())
        self.lookback_period = classification_config.get('lookback_days', 252)
        self.data_dir = config.get('data', {}).get('data_dir', 'data/historical')

    def _default_thresholds(self) -> dict:
        """Default classification thresholds"""
        return {
            # Trend behavior (ADX-based)
            'adx_trending': 25,           # ADX > 25 = trending
            'adx_mean_reverting': 20,     # ADX < 20 = mean reverting

            # Volatility (ATR as % of price)
            'atr_pct_high': 3.0,          # ATR% > 3% = high volatility
            'atr_pct_low': 1.5,           # ATR% < 1.5% = low volatility

            # Breakout propensity (% of days with breakouts)
            'breakout_freq_high': 0.15,   # > 15% = breakout prone
            'breakout_freq_low': 0.05,    # < 5% = range bound

            # Price tiers
            'price_low': 20,              # < $20 = low price
            'price_high': 100,            # > $100 = high price

            # Movement intensity (avg daily return magnitude %)
            'movement_high': 2.0,         # > 2% = high beta-like
            'movement_low': 0.8,          # < 0.8% = low beta-like

            # Liquidity (avg daily dollar volume in millions)
            'liquidity_high': 100,        # > $100M = high liquidity
            'liquidity_low': 10,          # < $10M = low liquidity
        }

    def classify_stock(self, symbol: str, data: pd.DataFrame) -> Optional[StockClassification]:
        """
        Classify a single stock based on historical data.

        Args:
            symbol: Stock ticker symbol
            data: DataFrame with OHLCV data (columns: open, high, low, close, volume)

        Returns:
            StockClassification object or None if insufficient data
        """
        if data is None or len(data) < 60:
            return None

        # Calculate classification metrics
        metrics = self._calculate_metrics(data)

        if metrics is None:
            return None

        # Classify each dimension
        trend = self._classify_trend(metrics)
        volatility = self._classify_volatility(metrics)
        breakout = self._classify_breakout(metrics)
        price = self._classify_price(metrics)
        movement = self._classify_movement(metrics)
        liquidity = self._classify_liquidity(metrics)

        # Determine strategy affinity based on classification
        affinity = self._determine_strategy_affinity(
            trend, volatility, breakout, price, movement
        )

        return StockClassification(
            symbol=symbol,
            trend_behavior=trend,
            volatility_level=volatility,
            breakout_propensity=breakout,
            price_tier=price,
            movement_intensity=movement,
            liquidity_level=liquidity,
            avg_adx=metrics['avg_adx'],
            avg_atr_pct=metrics['avg_atr_pct'],
            breakout_frequency=metrics['breakout_freq'],
            avg_price=metrics['avg_price'],
            avg_daily_return_magnitude=metrics['avg_return_mag'],
            avg_dollar_volume=metrics['avg_dollar_volume'],
            strategy_affinity=affinity
        )

    def _calculate_metrics(self, data: pd.DataFrame) -> Optional[dict]:
        """Calculate all classification metrics from price data"""
        try:
            # Normalize column names
            df = data.copy()
            df.columns = [col.lower() for col in df.columns]

            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values

            # Remove any NaN values
            mask = ~(np.isnan(close) | np.isnan(high) | np.isnan(low) | np.isnan(volume))
            close = close[mask]
            high = high[mask]
            low = low[mask]
            volume = volume[mask]

            if len(close) < 60:
                return None

            # ADX calculation (14-period)
            avg_adx = self._calculate_avg_adx(high, low, close, period=14)

            # ATR as percentage of price (14-period)
            avg_atr_pct = self._calculate_avg_atr_pct(high, low, close, period=14)

            # Breakout frequency (20-period Donchian)
            breakout_freq = self._calculate_breakout_frequency(high, low, close, period=20)

            # Average price
            avg_price = np.mean(close)

            # Average daily return magnitude
            returns = np.abs(np.diff(close) / close[:-1]) * 100
            avg_return_mag = np.mean(returns) if len(returns) > 0 else 0

            # Average dollar volume (in millions)
            avg_dollar_volume = np.mean(volume * close) / 1_000_000

            return {
                'avg_adx': avg_adx,
                'avg_atr_pct': avg_atr_pct,
                'breakout_freq': breakout_freq,
                'avg_price': avg_price,
                'avg_return_mag': avg_return_mag,
                'avg_dollar_volume': avg_dollar_volume,
            }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return None

    def _calculate_avg_adx(self, high, low, close, period=14) -> float:
        """Calculate average ADX over the data period"""
        if len(close) < period * 3:
            return 0.0

        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Wilder's smoothing (exponential moving average with alpha = 1/period)
        def wilder_smooth(arr, period):
            """Wilder's smoothing: EMA with alpha = 1/period"""
            alpha = 1.0 / period
            result = np.zeros(len(arr))
            # First value is simple average
            result[period-1] = np.mean(arr[:period])
            # Subsequent values use EMA formula
            for i in range(period, len(arr)):
                result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
            return result[period-1:]

        if len(tr) < period:
            return 0.0

        # Smooth TR, +DM, -DM
        atr = wilder_smooth(tr, period)
        smoothed_plus_dm = wilder_smooth(plus_dm, period)
        smoothed_minus_dm = wilder_smooth(minus_dm, period)

        # Avoid division by zero
        atr_safe = np.where(atr == 0, 1e-10, atr)

        # Calculate +DI and -DI (should be 0-100)
        plus_di = 100 * smoothed_plus_dm / atr_safe
        minus_di = 100 * smoothed_minus_dm / atr_safe

        # Calculate DX
        di_sum = plus_di + minus_di
        di_sum = np.where(di_sum == 0, 1e-10, di_sum)
        di_diff = np.abs(plus_di - minus_di)
        dx = 100 * di_diff / di_sum

        if len(dx) < period:
            return np.mean(dx) if len(dx) > 0 else 0.0

        # ADX is smoothed DX
        adx = wilder_smooth(dx, period)

        # Return average ADX (should be 0-100)
        return np.mean(adx) if len(adx) > 0 else 0.0

    def _calculate_avg_atr_pct(self, high, low, close, period=14) -> float:
        """Calculate ATR as percentage of price"""
        if len(close) < period + 1:
            return 0.0

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        if len(tr) < period:
            return 0.0

        # Simple moving average of TR
        atr = np.convolve(tr, np.ones(period)/period, mode='valid')
        close_aligned = close[period:]

        if len(close_aligned) == 0 or len(atr) == 0:
            return 0.0

        # Align lengths
        min_len = min(len(atr), len(close_aligned))
        atr = atr[:min_len]
        close_aligned = close_aligned[:min_len]

        # Avoid division by zero
        close_aligned = np.where(close_aligned == 0, 1e-10, close_aligned)

        atr_pct = (atr / close_aligned) * 100
        return np.mean(atr_pct)

    def _calculate_breakout_frequency(self, high, low, close, period=20) -> float:
        """Calculate frequency of Donchian channel breakouts"""
        if len(close) <= period:
            return 0.0

        breakout_count = 0
        total_days = len(close) - period

        for i in range(period, len(close)):
            # Previous period's high/low (excluding current bar)
            highest = np.max(high[i-period:i])
            lowest = np.min(low[i-period:i])

            # Check for breakout
            if close[i] > highest or close[i] < lowest:
                breakout_count += 1

        return breakout_count / total_days if total_days > 0 else 0.0

    def _classify_trend(self, metrics: dict) -> TrendBehavior:
        """Classify trend behavior based on ADX"""
        adx = metrics['avg_adx']
        if adx >= self.thresholds['adx_trending']:
            return TrendBehavior.TRENDING
        elif adx <= self.thresholds['adx_mean_reverting']:
            return TrendBehavior.MEAN_REVERTING
        return TrendBehavior.MIXED

    def _classify_volatility(self, metrics: dict) -> VolatilityLevel:
        """Classify volatility level based on ATR%"""
        atr_pct = metrics['avg_atr_pct']
        if atr_pct >= self.thresholds['atr_pct_high']:
            return VolatilityLevel.HIGH
        elif atr_pct <= self.thresholds['atr_pct_low']:
            return VolatilityLevel.LOW
        return VolatilityLevel.MEDIUM

    def _classify_breakout(self, metrics: dict) -> BreakoutPropensity:
        """Classify breakout propensity"""
        freq = metrics['breakout_freq']
        if freq >= self.thresholds['breakout_freq_high']:
            return BreakoutPropensity.BREAKOUT_PRONE
        elif freq <= self.thresholds['breakout_freq_low']:
            return BreakoutPropensity.RANGE_BOUND
        return BreakoutPropensity.MIXED

    def _classify_price(self, metrics: dict) -> PriceTier:
        """Classify price tier"""
        price = metrics['avg_price']
        if price < self.thresholds['price_low']:
            return PriceTier.LOW
        elif price > self.thresholds['price_high']:
            return PriceTier.HIGH
        return PriceTier.MID

    def _classify_movement(self, metrics: dict) -> MovementIntensity:
        """Classify movement intensity (beta-like behavior)"""
        movement = metrics['avg_return_mag']
        if movement >= self.thresholds['movement_high']:
            return MovementIntensity.HIGH_BETA
        elif movement <= self.thresholds['movement_low']:
            return MovementIntensity.LOW_BETA
        return MovementIntensity.MEDIUM_BETA

    def _classify_liquidity(self, metrics: dict) -> LiquidityLevel:
        """Classify liquidity level"""
        dollar_vol = metrics['avg_dollar_volume']
        if dollar_vol >= self.thresholds['liquidity_high']:
            return LiquidityLevel.HIGH
        elif dollar_vol <= self.thresholds['liquidity_low']:
            return LiquidityLevel.LOW
        return LiquidityLevel.MEDIUM

    def _determine_strategy_affinity(
        self,
        trend: TrendBehavior,
        volatility: VolatilityLevel,
        breakout: BreakoutPropensity,
        price: PriceTier,
        movement: MovementIntensity
    ) -> List[str]:
        """
        Determine which strategy types are best suited for this stock.

        Returns list of strategy affinities:
        - 'trend_following': MA crossovers, Aroon, ADX-based strategies
        - 'mean_reversion': RSI oversold/overbought, BB mean reversion
        - 'breakout': Donchian breakouts, BB momentum breakouts
        - 'momentum': MACD, ROC, momentum-based entries
        - 'volatility_based': ATR stops, Keltner channels
        """
        affinities = []

        # Trend-following affinity
        if trend == TrendBehavior.TRENDING:
            affinities.append('trend_following')
            if movement in [MovementIntensity.HIGH_BETA, MovementIntensity.MEDIUM_BETA]:
                affinities.append('momentum')

        # Mean-reversion affinity
        if trend == TrendBehavior.MEAN_REVERTING:
            affinities.append('mean_reversion')

        # Breakout affinity
        if breakout == BreakoutPropensity.BREAKOUT_PRONE:
            affinities.append('breakout')
            if volatility == VolatilityLevel.HIGH:
                affinities.append('momentum')

        # Volatility-based strategies work well with high/medium volatility
        if volatility in [VolatilityLevel.HIGH, VolatilityLevel.MEDIUM]:
            affinities.append('volatility_based')

        # Mixed behavior gets all affinities
        if trend == TrendBehavior.MIXED and not affinities:
            affinities = ['trend_following', 'mean_reversion', 'breakout', 'momentum', 'volatility_based']

        # Default to mixed if no clear affinity
        if not affinities:
            affinities = ['mixed']

        return list(set(affinities))  # Remove duplicates

    def classify_all(
        self,
        symbols: List[str],
        data_loader=None,
        progress_callback=None
    ) -> Dict[str, StockClassification]:
        """
        Classify all stocks in the list.

        Args:
            symbols: List of stock symbols
            data_loader: Object with load_stock_data_raw method, or None to load from CSV
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping symbol to classification
        """
        classifications = {}

        for i, symbol in enumerate(symbols):
            try:
                # Load data
                if data_loader and hasattr(data_loader, 'load_stock_data_raw'):
                    data = data_loader.load_stock_data_raw(symbol)
                else:
                    data = self._load_from_csv(symbol)

                if data is not None and len(data) >= 60:
                    classification = self.classify_stock(symbol, data)
                    if classification:
                        classifications[symbol] = classification

                if progress_callback:
                    progress_callback(i + 1, len(symbols), symbol)

            except Exception as e:
                print(f"Warning: Could not classify {symbol}: {e}")

        return classifications

    def _load_from_csv(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load stock data from CSV file"""
        csv_path = os.path.join(self.data_dir, f"{symbol}.csv")

        if not os.path.exists(csv_path):
            return None

        try:
            # Try loading with MultiIndex (yfinance format)
            df = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)
            df.columns = df.columns.get_level_values(0)
        except:
            # Fall back to simple format
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

        return df

    def get_stocks_by_affinity(
        self,
        classifications: Dict[str, StockClassification],
        affinity: str
    ) -> List[str]:
        """Get list of stocks matching a strategy affinity"""
        return [
            symbol for symbol, clf in classifications.items()
            if affinity in clf.strategy_affinity
        ]

    def get_stocks_by_criteria(
        self,
        classifications: Dict[str, StockClassification],
        trend: Optional[TrendBehavior] = None,
        volatility: Optional[VolatilityLevel] = None,
        breakout: Optional[BreakoutPropensity] = None,
        price: Optional[PriceTier] = None,
        movement: Optional[MovementIntensity] = None,
        liquidity: Optional[LiquidityLevel] = None,
    ) -> List[str]:
        """Get stocks matching specific classification criteria"""
        matching = []

        for symbol, clf in classifications.items():
            if trend and clf.trend_behavior != trend:
                continue
            if volatility and clf.volatility_level != volatility:
                continue
            if breakout and clf.breakout_propensity != breakout:
                continue
            if price and clf.price_tier != price:
                continue
            if movement and clf.movement_intensity != movement:
                continue
            if liquidity and clf.liquidity_level != liquidity:
                continue
            matching.append(symbol)

        return matching

    def save_to_cache(self, classifications: Dict[str, StockClassification], cache_path: str):
        """Save classifications to JSON cache file"""
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        data = {
            symbol: clf.to_dict()
            for symbol, clf in classifications.items()
        }

        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_cache(self, cache_path: str) -> Dict[str, StockClassification]:
        """Load classifications from JSON cache file"""
        if not os.path.exists(cache_path):
            return {}

        with open(cache_path, 'r') as f:
            data = json.load(f)

        return {
            symbol: StockClassification.from_dict(clf_data)
            for symbol, clf_data in data.items()
        }


class StrategyMatcher:
    """
    Matches strategies to appropriate stock pools based on classification.
    """

    # Map entry indicators to strategy affinities
    INDICATOR_AFFINITY_MAP = {
        # Trend-following indicators
        'sma': ['trend_following'],
        'ema': ['trend_following'],
        'wma': ['trend_following'],
        'dema': ['trend_following'],
        'tema': ['trend_following'],
        'kama': ['trend_following'],
        'zlema': ['trend_following'],
        'hma': ['trend_following'],
        'aroon': ['trend_following'],
        'adx': ['trend_following'],
        'dmi': ['trend_following'],
        'psar': ['trend_following'],
        'supertrend': ['trend_following'],

        # Mean-reversion indicators
        'rsi': ['mean_reversion'],
        'stoch': ['mean_reversion'],
        'stochrsi': ['mean_reversion'],
        'cci': ['mean_reversion'],
        'williams': ['mean_reversion'],

        # Breakout indicators
        'donchian': ['breakout'],
        'bb': ['breakout', 'mean_reversion'],  # BB can be both
        'keltner': ['breakout', 'volatility_based'],
        'high': ['breakout'],
        'low': ['breakout'],

        # Momentum indicators
        'macd': ['momentum', 'trend_following'],
        'roc': ['momentum'],
        'momentum': ['momentum'],
        'tsi': ['momentum'],
        'ultimate': ['momentum'],

        # Volatility-based
        'atr': ['volatility_based'],
        'natr': ['volatility_based'],

        # Volume indicators (work with most strategies)
        'obv': ['mixed'],
        'vwap': ['mixed'],
        'mfi': ['mean_reversion'],
        'adl': ['mixed'],
        'cmf': ['mixed'],
    }

    def __init__(self, config: dict):
        self.config = config
        classification_config = config.get('classification', {})
        self.strict_matching = classification_config.get('strict_matching', False)

    def get_matching_stocks(
        self,
        strategy_config: dict,
        classifications: Dict[str, StockClassification]
    ) -> List[str]:
        """
        Get list of stocks that match a strategy's characteristics.

        Args:
            strategy_config: Strategy configuration dictionary
            classifications: Stock classifications

        Returns:
            List of matching stock symbols
        """
        entry_indicator = strategy_config.get('entry_indicator', '').lower()

        # Get strategy affinities based on entry indicator
        affinities = self.INDICATOR_AFFINITY_MAP.get(entry_indicator, ['mixed'])

        # Find stocks that match any of the affinities
        matching_stocks = set()
        for affinity in affinities:
            if affinity == 'mixed':
                # Mixed matches all stocks
                matching_stocks = set(classifications.keys())
                break
            for symbol, clf in classifications.items():
                if affinity in clf.strategy_affinity:
                    matching_stocks.add(symbol)

        # Apply additional filters based on entry type
        entry_type = strategy_config.get('entry_type', 'crossover')

        if entry_type == 'breakout' and matching_stocks:
            # For breakout strategies, prefer breakout-prone stocks
            breakout_stocks = {
                s for s in matching_stocks
                if classifications[s].breakout_propensity != BreakoutPropensity.RANGE_BOUND
            }
            if breakout_stocks:
                matching_stocks = breakout_stocks

        # If strict matching is disabled or no matches found, return all stocks
        if not self.strict_matching and not matching_stocks:
            return list(classifications.keys())

        if not matching_stocks:
            return list(classifications.keys())

        return list(matching_stocks)

    def get_strategy_affinity(self, strategy_config: dict) -> List[str]:
        """Get the strategy affinity categories for a given strategy"""
        entry_indicator = strategy_config.get('entry_indicator', '').lower()
        return self.INDICATOR_AFFINITY_MAP.get(entry_indicator, ['mixed'])


def print_classification_summary(classifications: Dict[str, StockClassification]):
    """Print a summary of stock classifications"""
    from collections import Counter

    if not classifications:
        print("No stocks classified.")
        return

    print()
    print("=" * 60)
    print("STOCK CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total stocks classified: {len(classifications)}")
    print()

    # Count by trend behavior
    trends = Counter(clf.trend_behavior.value for clf in classifications.values())
    print("Trend Behavior:")
    for trend, count in sorted(trends.items()):
        pct = count / len(classifications) * 100
        print(f"  {trend:20s}: {count:4d} ({pct:5.1f}%)")
    print()

    # Count by volatility
    vols = Counter(clf.volatility_level.value for clf in classifications.values())
    print("Volatility Level:")
    for vol, count in sorted(vols.items()):
        pct = count / len(classifications) * 100
        print(f"  {vol:20s}: {count:4d} ({pct:5.1f}%)")
    print()

    # Count by breakout propensity
    breakouts = Counter(clf.breakout_propensity.value for clf in classifications.values())
    print("Breakout Propensity:")
    for bo, count in sorted(breakouts.items()):
        pct = count / len(classifications) * 100
        print(f"  {bo:20s}: {count:4d} ({pct:5.1f}%)")
    print()

    # Count by price tier
    prices = Counter(clf.price_tier.value for clf in classifications.values())
    print("Price Tier:")
    for price, count in sorted(prices.items()):
        pct = count / len(classifications) * 100
        print(f"  {price:20s}: {count:4d} ({pct:5.1f}%)")
    print()

    # Count by movement intensity
    movements = Counter(clf.movement_intensity.value for clf in classifications.values())
    print("Movement Intensity:")
    for mov, count in sorted(movements.items()):
        pct = count / len(classifications) * 100
        print(f"  {mov:20s}: {count:4d} ({pct:5.1f}%)")
    print()

    # Count by liquidity
    liquidities = Counter(clf.liquidity_level.value for clf in classifications.values())
    print("Liquidity Level:")
    for liq, count in sorted(liquidities.items()):
        pct = count / len(classifications) * 100
        print(f"  {liq:20s}: {count:4d} ({pct:5.1f}%)")
    print()

    # Count by strategy affinity
    affinities = Counter()
    for clf in classifications.values():
        for aff in clf.strategy_affinity:
            affinities[aff] += 1
    print("Strategy Affinities:")
    for aff, count in sorted(affinities.items()):
        pct = count / len(classifications) * 100
        print(f"  {aff:20s}: {count:4d} ({pct:5.1f}%)")

    print()
    print("=" * 60)


def filter_stocks_by_behavior(
    classifications: Dict[str, StockClassification],
    behavior: str
) -> List[str]:
    """
    Filter stocks by a specific behavior characteristic.

    Args:
        classifications: Dictionary of stock classifications
        behavior: Behavior to filter by. Options:
            - 'trending', 'mean_reverting' (trend behavior)
            - 'breakout_prone', 'range_bound' (breakout propensity)
            - 'high_volatility', 'low_volatility' (volatility level)
            - 'low_price', 'high_price' (price tier)
            - 'high_beta', 'low_beta' (movement intensity)

    Returns:
        List of matching stock symbols
    """
    matching = []

    for symbol, clf in classifications.items():
        # Trend behavior
        if behavior == 'trending' and clf.trend_behavior == TrendBehavior.TRENDING:
            matching.append(symbol)
        elif behavior == 'mean_reverting' and clf.trend_behavior == TrendBehavior.MEAN_REVERTING:
            matching.append(symbol)

        # Breakout propensity
        elif behavior == 'breakout_prone' and clf.breakout_propensity == BreakoutPropensity.BREAKOUT_PRONE:
            matching.append(symbol)
        elif behavior == 'range_bound' and clf.breakout_propensity == BreakoutPropensity.RANGE_BOUND:
            matching.append(symbol)

        # Volatility
        elif behavior == 'high_volatility' and clf.volatility_level == VolatilityLevel.HIGH:
            matching.append(symbol)
        elif behavior == 'low_volatility' and clf.volatility_level == VolatilityLevel.LOW:
            matching.append(symbol)

        # Price tier
        elif behavior == 'low_price' and clf.price_tier == PriceTier.LOW:
            matching.append(symbol)
        elif behavior == 'high_price' and clf.price_tier == PriceTier.HIGH:
            matching.append(symbol)

        # Movement intensity
        elif behavior == 'high_beta' and clf.movement_intensity == MovementIntensity.HIGH_BETA:
            matching.append(symbol)
        elif behavior == 'low_beta' and clf.movement_intensity == MovementIntensity.LOW_BETA:
            matching.append(symbol)

    return matching


def filter_strategies_by_type(
    strategies: List[dict],
    strategy_type: str
) -> List[dict]:
    """
    Filter strategies to only those matching a specific type.

    Args:
        strategies: List of strategy configuration dictionaries
        strategy_type: Type to filter by: 'trend_following', 'mean_reversion',
                      'breakout', 'momentum', 'volatility_based'

    Returns:
        Filtered list of strategies
    """
    # Map strategy types to their associated indicators
    TYPE_TO_INDICATORS = {
        'trend_following': ['sma', 'ema', 'wma', 'dema', 'tema', 'kama', 'zlema', 'hma',
                           'aroon', 'adx', 'dmi', 'psar', 'supertrend', 'macd'],
        'mean_reversion': ['rsi', 'stoch', 'stochrsi', 'cci', 'williams', 'bb', 'mfi'],
        'breakout': ['donchian', 'bb', 'keltner', 'high', 'low'],
        'momentum': ['macd', 'roc', 'momentum', 'tsi', 'ultimate'],
        'volatility_based': ['atr', 'natr', 'keltner', 'bb'],
    }

    allowed_indicators = TYPE_TO_INDICATORS.get(strategy_type, [])

    if not allowed_indicators:
        return strategies  # Unknown type, return all

    filtered = []
    for strategy in strategies:
        entry_indicator = strategy.get('entry_indicator', '').lower()
        if entry_indicator in allowed_indicators:
            filtered.append(strategy)

    return filtered
