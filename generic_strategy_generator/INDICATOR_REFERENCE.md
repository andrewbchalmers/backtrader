# Indicator Reference Guide

This document lists all supported indicators, their parameters, and typical use cases.

## Table of Contents
1. [Moving Averages](#moving-averages)
2. [Volatility Indicators](#volatility-indicators)
3. [Momentum Oscillators](#momentum-oscillators)
4. [Trend Indicators](#trend-indicators)
5. [Volume Indicators](#volume-indicators)
6. [Support/Resistance](#supportresistance)
7. [Price Action](#price-action)

---

## Moving Averages

### SMA - Simple Moving Average
**Format:** `sma,period`
**Example:** `sma,20`
**Parameters:**
- `period`: Number of periods (default: 20)

**Description:** Simple arithmetic average of closing prices over N periods.
**Use Cases:** 
- Entry: Buy when price crosses above SMA
- Exit: Sell when price crosses below SMA
- Popular periods: 10, 20, 50, 200

---

### EMA - Exponential Moving Average
**Format:** `ema,period`
**Example:** `ema,12`
**Parameters:**
- `period`: Number of periods (default: 20)

**Description:** Weighted average giving more importance to recent prices.
**Use Cases:**
- Faster response than SMA
- Entry: Cross above EMA
- Exit: Cross below EMA
- Popular periods: 5, 12, 20, 50

---

### WMA - Weighted Moving Average
**Format:** `wma,period`
**Example:** `wma,10`
**Parameters:**
- `period`: Number of periods (default: 20)

**Description:** Linear weighted average, more weight on recent data.
**Use Cases:** Similar to EMA but with linear weighting

---

### DEMA - Double Exponential Moving Average
**Format:** `dema,period`
**Example:** `dema,20`
**Parameters:**
- `period`: Number of periods (default: 20)

**Description:** Applies EMA twice to reduce lag.
**Use Cases:** Faster trend identification with less lag

---

### TEMA - Triple Exponential Moving Average
**Format:** `tema,period`
**Example:** `tema,20`
**Parameters:**
- `period`: Number of periods (default: 20)

**Description:** Applies EMA three times for even less lag.
**Use Cases:** Very responsive to price changes

---

### KAMA - Kaufman Adaptive Moving Average
**Format:** `kama,period`
**Example:** `kama,30`
**Parameters:**
- `period`: Number of periods (default: 30)

**Description:** Adapts to market volatility and trends automatically.
**Use Cases:** Works well in both trending and ranging markets

---

### HMA - Hull Moving Average
**Format:** `hma,period`
**Example:** `hma,20`
**Parameters:**
- `period`: Number of periods (default: 20)

**Description:** Weighted MA that reduces lag and increases smoothness.
**Use Cases:** Fast trend identification with smooth curves

---

### ZLEMA - Zero Lag Exponential Moving Average
**Format:** `zlema,period`
**Example:** `zlema,20`
**Parameters:**
- `period`: Number of periods (default: 20)

**Description:** Attempts to eliminate lag from EMA.
**Use Cases:** Very responsive trend following

---

## Volatility Indicators

### BB - Bollinger Bands
**Format:** `bb,period,deviation`
**Example:** `bb,20,2`
**Parameters:**
- `period`: SMA period (default: 20)
- `deviation`: Number of standard deviations (default: 2.0)

**Description:** Three bands - middle (SMA), upper and lower (±2 std dev).
**Use Cases:**
- Entry: Buy when price touches lower band
- Exit: Sell when price touches upper band
- Breakout: Buy when price breaks above upper band

---

### ATR - Average True Range
**Format:** `atr,period,multiplier`
**Example:** `atr,14,2`
**Parameters:**
- `period`: Number of periods (default: 14)
- `multiplier`: Multiplier for stop loss (default: 1.0)

**Description:** Measures market volatility.
**Use Cases:**
- Stop loss placement: Entry price ± (ATR × multiplier)
- Volatility breakout entries
- Position sizing based on volatility

---

### NATR - Normalized Average True Range
**Format:** `natr,period`
**Example:** `natr,14`
**Parameters:**
- `period`: Number of periods (default: 14)

**Description:** ATR as percentage of price.
**Use Cases:** Compare volatility across different price levels

---

### Keltner Channel
**Format:** `keltner,period,atrdist`
**Example:** `keltner,20,2`
**Parameters:**
- `period`: EMA period (default: 20)
- `atrdist`: ATR multiplier for bands (default: 2.0)

**Description:** EMA with bands based on ATR.
**Use Cases:** Similar to Bollinger Bands but uses ATR

---

### Donchian Channel
**Format:** `donchian,period`
**Example:** `donchian,20`
**Parameters:**
- `period`: Lookback period (default: 20)

**Description:** Upper band = highest high, lower band = lowest low.
**Use Cases:**
- Breakout entries
- Turtle trading strategy
- Popular period: 20

---

## Momentum Oscillators

### RSI - Relative Strength Index
**Format:** `rsi,period`
**Example:** `rsi,14`
**Parameters:**
- `period`: Number of periods (default: 14)

**Description:** Oscillator between 0-100 measuring speed and magnitude of price changes.
**Use Cases:**
- Entry: Buy when RSI crosses above 30 (oversold)
- Exit: Sell when RSI crosses below 70 (overbought)
- Divergence: Price makes new high but RSI doesn't
- Popular periods: 7, 14, 21

---

### Stochastic Oscillator
**Format:** `stoch,period,period_dfast,period_dslow`
**Example:** `stoch,14,3,3`
**Parameters:**
- `period`: %K period (default: 14)
- `period_dfast`: Fast %D period (default: 3)
- `period_dslow`: Slow %D period (default: 3)

**Description:** Compares closing price to price range over period.
**Use Cases:**
- Entry: %K crosses above %D below 20
- Exit: %K crosses below %D above 80
- Overbought: Above 80
- Oversold: Below 20

---

### StochRSI - Stochastic RSI
**Format:** `stochrsi,period,pfast,pslow`
**Example:** `stochrsi,14,3,3`
**Parameters:**
- `period`: RSI period (default: 14)
- `pfast`: Fast smoothing (default: 3)
- `pslow`: Slow smoothing (default: 3)

**Description:** Stochastic oscillator applied to RSI.
**Use Cases:** More sensitive than regular Stochastic

---

### CCI - Commodity Channel Index
**Format:** `cci,period`
**Example:** `cci,20`
**Parameters:**
- `period`: Number of periods (default: 20)

**Description:** Measures variation from statistical mean.
**Use Cases:**
- Entry: Cross above -100
- Exit: Cross below +100
- Overbought: Above +100
- Oversold: Below -100

---

### Williams %R
**Format:** `williams,period`
**Example:** `williams,14`
**Parameters:**
- `period`: Lookback period (default: 14)

**Description:** Momentum indicator similar to Stochastic (inverted scale).
**Use Cases:**
- Entry: Cross above -80
- Exit: Cross below -20
- Overbought: Above -20
- Oversold: Below -80

---

### ROC - Rate of Change
**Format:** `roc,period`
**Example:** `roc,12`
**Parameters:**
- `period`: Lookback period (default: 12)

**Description:** Percentage change in price over N periods.
**Use Cases:**
- Entry: ROC crosses above 0
- Exit: ROC crosses below 0
- Divergence signals

---

### Momentum
**Format:** `momentum,period`
**Example:** `momentum,12`
**Parameters:**
- `period`: Lookback period (default: 12)

**Description:** Current price minus price N periods ago.
**Use Cases:**
- Entry: Momentum crosses above 0
- Exit: Momentum crosses below 0

---

### TSI - True Strength Index
**Format:** `tsi,period1,period2`
**Example:** `tsi,25,13`
**Parameters:**
- `period1`: Long period (default: 25)
- `period2`: Short period (default: 13)

**Description:** Double smoothed momentum indicator.
**Use Cases:**
- Entry: TSI crosses above signal line
- Overbought/oversold levels
- Divergence detection

---

### Ultimate Oscillator
**Format:** `ultimate,p1,p2,p3`
**Example:** `ultimate,7,14,28`
**Parameters:**
- `p1`: Short period (default: 7)
- `p2`: Medium period (default: 14)
- `p3`: Long period (default: 28)

**Description:** Uses three timeframes to reduce false signals.
**Use Cases:**
- Entry: Cross above 30
- Exit: Cross below 70
- Divergence signals

---

## Trend Indicators

### MACD - Moving Average Convergence Divergence
**Format:** `macd,fast,slow,signal`
**Example:** `macd,12,26,9`
**Parameters:**
- `fast`: Fast EMA period (default: 12)
- `slow`: Slow EMA period (default: 26)
- `signal`: Signal line period (default: 9)

**Description:** Two EMAs and their difference (histogram).
**Use Cases:**
- Entry: MACD crosses above signal line
- Exit: MACD crosses below signal line
- Histogram: Momentum strength
- Zero line: Trend direction

---

### ADX - Average Directional Index
**Format:** `adx,period`
**Example:** `adx,14`
**Parameters:**
- `period`: Number of periods (default: 14)

**Description:** Measures trend strength (0-100).
**Use Cases:**
- ADX > 25: Strong trend
- ADX < 20: Weak/no trend
- Use with +DI/-DI for direction
- Entry: ADX rising above 25

---

### DMI - Directional Movement Index
**Format:** `dmi,period`
**Example:** `dmi,14`
**Parameters:**
- `period`: Number of periods (default: 14)

**Description:** Shows trend direction with +DI and -DI lines.
**Use Cases:**
- Entry: +DI crosses above -DI
- Exit: -DI crosses above +DI
- Use with ADX for confirmation

---

### Aroon Indicator
**Format:** `aroon,period`
**Example:** `aroon,25`
**Parameters:**
- `period`: Lookback period (default: 25)

**Description:** Measures time since highest high and lowest low.
**Use Cases:**
- Entry: Aroon Up crosses above Aroon Down
- Aroon Up > 70: Strong uptrend
- Aroon Down > 70: Strong downtrend

---

### PSAR - Parabolic SAR
**Format:** `psar,af,afmax`
**Example:** `psar,0.02,0.20`
**Parameters:**
- `af`: Acceleration factor (default: 0.02)
- `afmax`: Maximum acceleration (default: 0.20)

**Description:** Stop and reverse indicator (dots above/below price).
**Use Cases:**
- Entry: PSAR switches from above to below price
- Exit: PSAR switches from below to above price
- Trailing stop

---

### SuperTrend
**Format:** `supertrend,period,multiplier`
**Example:** `supertrend,10,3`
**Parameters:**
- `period`: ATR period (default: 10)
- `multiplier`: ATR multiplier (default: 3.0)

**Description:** Trend-following indicator using ATR.
**Use Cases:**
- Entry: Price crosses above SuperTrend
- Exit: Price crosses below SuperTrend
- Trend identification

---

## Volume Indicators

### OBV - On Balance Volume
**Format:** `obv,1`
**Example:** `obv,1`
**Parameters:** None (use 1 as placeholder)

**Description:** Cumulative volume based on price direction.
**Use Cases:**
- Confirmation: OBV should follow price trend
- Divergence: Price up but OBV down (bearish)
- Entry: OBV breaks out before price

---

### VWAP - Volume Weighted Average Price
**Format:** `vwap,1`
**Example:** `vwap,1`
**Parameters:** None (use 1 as placeholder)

**Description:** Average price weighted by volume.
**Use Cases:**
- Institutional reference point
- Entry: Buy when price crosses above VWAP
- Exit: Sell when price crosses below VWAP

---

### MFI - Money Flow Index
**Format:** `mfi,period`
**Example:** `mfi,14`
**Parameters:**
- `period`: Number of periods (default: 14)

**Description:** RSI using volume-weighted typical price.
**Use Cases:**
- Entry: MFI crosses above 20 (oversold)
- Exit: MFI crosses below 80 (overbought)
- Divergence signals

---

### ADL - Accumulation/Distribution Line
**Format:** `adl,1`
**Example:** `adl,1`
**Parameters:** None (use 1 as placeholder)

**Description:** Cumulative indicator using volume and close location.
**Use Cases:**
- Trend confirmation
- Divergence detection
- Money flow analysis

---

### CMF - Chaikin Money Flow
**Format:** `cmf,period`
**Example:** `cmf,20`
**Parameters:**
- `period`: Number of periods (default: 20)

**Description:** Volume-weighted average of accumulation/distribution.
**Use Cases:**
- Entry: CMF crosses above 0
- Exit: CMF crosses below 0
- Above 0.25: Strong buying pressure
- Below -0.25: Strong selling pressure

---

## Support/Resistance

### Pivot Point
**Format:** `pivot,1`
**Example:** `pivot,1`
**Parameters:** None (use 1 as placeholder)

**Description:** Calculates pivot levels (support/resistance).
**Use Cases:**
- Entry: Buy at support levels
- Exit: Sell at resistance levels
- Intraday trading reference

---

## Price Action

### HIGH - Highest High
**Format:** `high,period`
**Example:** `high,20`
**Parameters:**
- `period`: Lookback period (default: 20)

**Description:** Highest high over N periods.
**Use Cases:**
- Resistance levels
- Breakout entries: Price > HIGH
- Popular periods: 20, 50

---

### LOW - Lowest Low
**Format:** `low,period`
**Example:** `low,20`
**Parameters:**
- `period`: Lookback period (default: 20)

**Description:** Lowest low over N periods.
**Use Cases:**
- Support levels
- Breakout entries: Price < LOW
- Popular periods: 20, 50

---

### AVGPRICE - Average Price
**Format:** `avgprice,1`
**Example:** `avgprice,1`
**Parameters:** None (use 1 as placeholder)

**Description:** Average of OHLC (Open, High, Low, Close).
**Use Cases:**
- Alternative to close price
- Smoothed price reference

---

## Strategy Combinations Examples

### Trend Following
```csv
# Entry
ema,20
ema,50
adx,14

# Exit
atr,14,2
psar,0.02,0.20
```

### Mean Reversion
```csv
# Entry
rsi,14
bb,20,2
stoch,14,3,3

# Exit
rsi,30
bb,20,2
```

### Momentum
```csv
# Entry
macd,12,26,9
rsi,14
roc,12

# Exit
macd,12,26,9
cci,20
```

### Breakout
```csv
# Entry
donchian,20
high,20
adx,14

# Exit
atr,14,3
low,20
```

### Volume-Based
```csv
# Entry
obv,1
mfi,14
cmf,20

# Exit
mfi,14
vwap,1
```

---

## Tips for Indicator Selection

1. **Avoid Redundancy**: Don't use multiple similar indicators (e.g., SMA and EMA of same period)
2. **Mix Types**: Combine trend + momentum + volatility indicators
3. **Test Parameters**: Common periods may not be optimal for your strategy
4. **Consider Timeframe**: Short periods for day trading, longer for swing trading
5. **Volume Confirmation**: Add volume indicators to price-based signals
6. **Start Simple**: Begin with 1-2 indicators, add complexity as needed

---

## Common Parameter Ranges

| Indicator | Parameter | Common Values | Notes |
|-----------|-----------|---------------|-------|
| SMA/EMA | Period | 10, 20, 50, 200 | Shorter = faster signals |
| RSI | Period | 7, 14, 21 | 14 is standard |
| BB | Period | 20 | Industry standard |
| BB | StdDev | 2, 2.5, 3 | 2 is standard |
| ATR | Period | 10, 14, 20 | 14 is standard |
| MACD | Fast | 12 | Standard |
| MACD | Slow | 26 | Standard |
| MACD | Signal | 9 | Standard |
| Stochastic | Period | 5, 14, 21 | 14 is standard |
| ADX | Period | 14, 20 | 14 is standard |

---

## Backtrader Indicator Documentation

For more details on backtrader indicators:
https://www.backtrader.com/docu/indautoref/
