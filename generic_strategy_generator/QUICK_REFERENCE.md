# Quick Indicator Reference Card

## Format: indicator_name,param1,param2,...

## MOVING AVERAGES (Fast to Slow Response)
sma,20          # Simple Moving Average
ema,20          # Exponential Moving Average (faster)
wma,20          # Weighted Moving Average
dema,20         # Double EMA (very fast)
tema,20         # Triple EMA (very fast)
hma,20          # Hull MA (smooth & fast)
zlema,20        # Zero Lag EMA (fastest)
kama,30         # Kaufman Adaptive (smart)

## VOLATILITY & BANDS
bb,20,2         # Bollinger Bands (period, std dev)
keltner,20,2    # Keltner Channel (period, ATR mult)
donchian,20     # Donchian Channel (breakout)
atr,14,2        # Average True Range (period, mult)
natr,14         # Normalized ATR (%)

## OSCILLATORS (0-100 scale)
rsi,14          # RSI (30=oversold, 70=overbought)
stoch,14,3,3    # Stochastic (period, dfast, dslow)
stochrsi,14,3,3 # Stochastic RSI
cci,20          # CCI (-100 to +100)
williams,14     # Williams %R (-80 to -20)
mfi,14          # Money Flow Index (with volume)

## MOMENTUM (Zero-line crossovers)
roc,12          # Rate of Change (%)
momentum,12     # Momentum (price change)
tsi,25,13       # True Strength Index
ultimate,7,14,28 # Ultimate Oscillator

## TREND INDICATORS
macd,12,26,9    # MACD (fast, slow, signal)
adx,14          # ADX (>25 = strong trend)
dmi,14          # Directional Movement (+DI/-DI)
aroon,25        # Aroon (time since high/low)
psar,0.02,0.20  # Parabolic SAR (af, afmax)
supertrend,10,3 # SuperTrend (period, mult)

## VOLUME INDICATORS
obv,1           # On Balance Volume
vwap,1          # Volume Weighted Avg Price
adl,1           # Accumulation/Distribution
cmf,20          # Chaikin Money Flow

## PRICE ACTION
high,20         # Highest high over period
low,20          # Lowest low over period
avgprice,1      # Average of OHLC
pivot,1         # Pivot points

## POPULAR STRATEGY COMBINATIONS

### Trend Following
Entry:  ema,20 | adx,14 | macd,12,26,9
Exit:   atr,14,2 | psar,0.02,0.20

### Mean Reversion
Entry:  rsi,14 | bb,20,2 | stoch,14,3,3
Exit:   ema,20 | rsi,30

### Breakout
Entry:  donchian,20 | high,20 | adx,14
Exit:   atr,14,3 | low,20

### Momentum
Entry:  macd,12,26,9 | rsi,14 | roc,12
Exit:   macd,12,26,9 | cci,20

### Volume Confirmation
Entry:  ema,20 | obv,1 | mfi,14
Exit:   vwap,1 | cmf,20

## COMMON PARAMETER VALUES

RSI:        7, 14, 21, 30
SMA/EMA:    10, 20, 50, 100, 200
BB:         20,2 (standard) | 20,3 (wider)
ATR:        10, 14, 20 (mult: 2-3 for stops)
MACD:       12,26,9 (standard)
Stoch:      5,3,3 (fast) | 14,3,3 (standard)
ADX:        14, 20
Donchian:   20 (turtle), 55 (long-term)
Aroon:      25 (standard)

## INDICATOR SELECTION TIPS

1. Mix Types: Use trend + momentum + volatility
2. Avoid Redundancy: Don't use SMA(20) and EMA(20) together
3. Timeframe: Shorter periods = more signals (but more noise)
4. Volume: Always good for confirmation
5. Start Simple: 2-3 indicators is often enough
6. Test Different Periods: Standard values aren't always optimal

## ENTRY/EXIT TYPE GUIDE

CROSSOVER:    Price crosses indicator (MA, MACD, PSAR)
THRESHOLD:    Indicator crosses level (RSI>30, CCI>-100)
BREAKOUT:     Price breaks bands (BB, Donchian, Keltner)
STOP_LOSS:    Fixed % or ATR-based stop
TRAILING_STOP: ATR-based trailing stop
TAKE_PROFIT:  Fixed % profit target

## FILES TO EDIT

entry_indicators.csv         # Your entry signals
exit_indicators.csv          # Your exit signals
entry_indicators_extended.csv # 50+ pre-configured entries
exit_indicators_extended.csv  # 30+ pre-configured exits

## TESTING WORKFLOW

1. Start with entry_indicators_extended.csv (50+ indicators)
2. Run: python main.py --download-data
3. Review results/top_strategies.csv
4. Pick best indicators from top strategies
5. Create focused entry/exit CSV with just those
6. Re-test with focused set
7. Iterate!

## NEED MORE INFO?

See INDICATOR_REFERENCE.md for detailed documentation
on every indicator, parameters, and use cases.
