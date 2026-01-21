#!/usr/bin/env python3
"""
Find stocks exhibiting mean reversion characteristics.
Metrics used:
- Hurst Exponent (< 0.5 indicates mean reversion)
- ADF Test (stationarity test - lower p-value = more mean reverting)
- Half-life of mean reversion
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# Candidate stocks to analyze (large universe - ETFs, sectors known for mean reversion)
CANDIDATES = [
    # Utilities (often mean reverting)
    'AES', 'OGE', 'NRG', 'ALE', 'AVA', 'BKH', 'IDA', 'OGS', 'POR', 'SR',
    'CNP', 'OTTR', 'HE', 'AEP', 'EIX', 'PCG', 'FE', 'DTE', 'WE', 'LNT',

    # REITs (income focused, often range-bound)
    'IRM', 'CUBE', 'LSI', 'MAA', 'EQR', 'AVB', 'ESS', 'UDR', 'CPT', 'AIRC',
    'ARE', 'BXP', 'SLG', 'VNO', 'KIM', 'REG', 'FRT', 'BRX', 'SITC', 'ROIC',
    'HR', 'OHI', 'SBRA', 'MPW', 'DOC', 'PEAK', 'CTRE', 'LTC', 'NHI', 'GMRE',

    # Consumer Staples (defensive, range-bound)
    'EL', 'COTY', 'MNST', 'TAP', 'STZ', 'BF-B', 'SAM', 'FIZZ', 'CELH', 'COKE',
    'FLO', 'THS', 'BGS', 'SMPL', 'FRPT', 'HAIN', 'SFM', 'SPTN', 'CHEF', 'UNFI',
    'KR', 'ACI', 'GO', 'IMKTA', 'WMK', 'VLGEA', 'NGVC', 'ANDE',

    # Healthcare (stable revenue, often range-bound)
    'CAH', 'MCK', 'COR', 'HSIC', 'PDCO', 'OMI', 'BDX', 'BAX', 'ZBH', 'HOLX',
    'DXCM', 'PODD', 'ALGN', 'XRAY', 'NVST', 'TFX', 'GMED', 'MMSI', 'LIVN',
    'MOH', 'CNC', 'HUM', 'OSCR', 'ALHC', 'CLOV', 'CLVR',

    # Financials (value/dividend, often range-bound)
    'TFC', 'FITB', 'RF', 'HBAN', 'CFG', 'KEY', 'MTB', 'ZION', 'CMA', 'FHN',
    'PNC', 'USB', 'WFC', 'BAC', 'C', 'MS', 'SCHW', 'IBKR', 'LPLA', 'RJF',
    'AXP', 'DFS', 'SYF', 'ALLY', 'COF', 'CACC', 'OMF', 'SLM', 'NAVI',
    'PRU', 'MET', 'AFL', 'UNM', 'GL', 'PFG', 'VOYA', 'LNC', 'EQH', 'AIG',

    # Telecom (mature, stable)
    'LUMN', 'FYBR', 'USM', 'SHEN', 'LILA', 'LILAK', 'ATUS', 'CABO',

    # Materials (cyclical but often range-bound)
    'IP', 'PKG', 'WRK', 'SEE', 'SON', 'BLL', 'CCK', 'OI', 'ARD', 'GPK',
    'AVY', 'BERY', 'SLVM', 'AMCR', 'ATR', 'AOS', 'SWK', 'ALLE', 'MAS',
    'FBHS', 'BLDR', 'BLD', 'JELD', 'DOOR', 'AWI', 'TILE', 'FND', 'FLOR',

    # Energy MLPs/Midstream (income focused)
    'KMI', 'WMB', 'OKE', 'ET', 'EPD', 'MPLX', 'PAA', 'TRGP', 'AM', 'HESM',
    'ENB', 'TRP', 'KNTK', 'DTM', 'NS', 'CEQP', 'GEL', 'ENLC', 'WES', 'DKL',

    # Industrial (mature, often range-bound)
    'SNA', 'GWW', 'FAST', 'WSO', 'AIT', 'WESCO', 'DSGR', 'HDS', 'CNH', 'AGCO',
    'PCAR', 'OSK', 'TTC', 'ALV', 'LEA', 'BWA', 'APTV', 'VC', 'DAN', 'AXL',

    # Bond-like ETFs (naturally mean-reverting)
    'VCSH', 'SPAB', 'BIV', 'BSV', 'GOVT', 'VGSH', 'SCHO', 'NEAR', 'MINT', 'FLOT',
    'IGIB', 'IGSB', 'USIG', 'SUSC', 'QLTA', 'ANGL', 'SJNK', 'HYLB', 'USHY', 'JNK',

    # Low volatility/Dividend ETFs
    'DVY', 'VIG', 'SDY', 'DGRO', 'DGRW', 'NOBL', 'SPHD', 'HDV', 'FVD', 'CDC',
    'VYM', 'SPYD', 'PFF', 'PGX', 'PFFD', 'DIV', 'KBWD', 'QYLD', 'RYLD', 'XYLD',

    # International stable (often range-bound)
    'EWZ', 'EWW', 'EWY', 'EWT', 'EWH', 'EWS', 'EWA', 'EWC', 'EWG', 'EWU',
    'EWQ', 'EWI', 'EWP', 'EWN', 'EWO', 'EWK', 'EWD', 'NORW', 'EDEN', 'EFNL',
]

def calculate_hurst_exponent(series, max_lag=100):
    """Calculate Hurst exponent using R/S analysis."""
    series = np.array(series)
    n = len(series)
    if n < max_lag:
        max_lag = n // 2

    lags = range(2, max_lag)
    tau = []

    for lag in lags:
        # Calculate standard deviation of lagged differences
        std = np.std(np.subtract(series[lag:], series[:-lag]))
        tau.append(std)

    # Calculate Hurst using log-log regression
    try:
        log_lags = np.log(list(lags))
        log_tau = np.log(tau)

        # Remove any infinities
        mask = np.isfinite(log_lags) & np.isfinite(log_tau)
        if np.sum(mask) < 10:
            return 0.5

        poly = np.polyfit(log_lags[mask], log_tau[mask], 1)
        hurst = poly[0]
        return hurst
    except:
        return 0.5

def calculate_half_life(series):
    """Calculate half-life of mean reversion using OLS."""
    series = np.array(series)
    lag = series[:-1]
    ret = series[1:] - series[:-1]

    # Remove NaN/Inf
    mask = np.isfinite(lag) & np.isfinite(ret)
    lag = lag[mask]
    ret = ret[mask]

    if len(lag) < 30:
        return np.inf

    # OLS regression: ret = alpha + beta * lag
    lag_mean = np.mean(lag)
    ret_mean = np.mean(ret)

    beta = np.sum((lag - lag_mean) * (ret - ret_mean)) / np.sum((lag - lag_mean) ** 2)

    if beta >= 0:
        return np.inf  # No mean reversion

    half_life = -np.log(2) / beta
    return half_life

def analyze_stock(symbol, start_date, end_date):
    """Analyze a single stock for mean reversion characteristics."""
    try:
        # Download data
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)

        if len(data) < 100:
            return None

        # Use adjusted close
        if 'Adj Close' in data.columns:
            prices = data['Adj Close'].values
        elif ('Adj Close', symbol) in data.columns:
            prices = data[('Adj Close', symbol)].values
        else:
            prices = data['Close'].values if 'Close' in data.columns else data[('Close', symbol)].values

        prices = prices[~np.isnan(prices)]

        if len(prices) < 100:
            return None

        # Calculate log prices for analysis
        log_prices = np.log(prices)

        # 1. Hurst Exponent
        hurst = calculate_hurst_exponent(log_prices)

        # 2. ADF Test (stationarity of returns and price deviations from MA)
        returns = np.diff(log_prices)

        # Price deviation from 20-day MA
        ma = pd.Series(prices).rolling(20).mean().values
        deviation = (prices[19:] - ma[19:]) / ma[19:]
        deviation = deviation[~np.isnan(deviation)]

        try:
            adf_result = adfuller(deviation, maxlag=10)
            adf_pvalue = adf_result[1]
        except:
            adf_pvalue = 1.0

        # 3. Half-life
        half_life = calculate_half_life(deviation)

        # 4. Additional: autocorrelation of returns
        if len(returns) > 20:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        else:
            autocorr = 0

        return {
            'symbol': symbol,
            'hurst': hurst,
            'adf_pvalue': adf_pvalue,
            'half_life': half_life,
            'autocorr': autocorr,
            'data_points': len(prices)
        }
    except Exception as e:
        return None

def main():
    # Load existing stocks
    existing = pd.read_csv('inputs/stocks.csv')['symbol'].tolist()
    existing = [s.strip() for s in existing if pd.notna(s) and s.strip()]

    print(f"Existing stocks: {len(existing)}")
    print(f"Candidates to analyze: {len(CANDIDATES)}")

    # Filter out already existing
    to_analyze = [s for s in CANDIDATES if s not in existing]
    print(f"New candidates to analyze: {len(to_analyze)}")

    # Test period
    start_date = "2024-01-01"
    end_date = "2026-01-01"

    results = []

    print("\nAnalyzing stocks for mean reversion characteristics...")
    print("-" * 70)

    for i, symbol in enumerate(to_analyze):
        print(f"Analyzing {symbol} ({i+1}/{len(to_analyze)})...", end='\r')
        result = analyze_stock(symbol, start_date, end_date)
        if result:
            results.append(result)

    print("\n" + "-" * 70)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    if len(df) == 0:
        print("No valid results found!")
        return

    # Score mean reversion:
    # - Hurst < 0.5 (lower is better)
    # - ADF p-value < 0.05 (lower is better)
    # - Half-life between 5 and 60 days (tradeable range)
    # - Negative autocorrelation (more negative is better)

    # Filter for mean-reverting characteristics
    df_mr = df[
        (df['hurst'] < 0.5) &  # Mean reverting Hurst
        (df['adf_pvalue'] < 0.1) &  # Somewhat stationary
        (df['half_life'] > 3) &  # Not too fast
        (df['half_life'] < 100) &  # Not too slow
        (df['autocorr'] < 0.1)  # Not trending
    ].copy()

    # Calculate composite score (lower is more mean reverting)
    df_mr['mr_score'] = (
        df_mr['hurst'] * 2 +  # Weight Hurst heavily
        df_mr['adf_pvalue'] +  # Lower p-value is better
        df_mr['half_life'] / 100 +  # Normalize half-life
        df_mr['autocorr']  # Negative is better
    )

    df_mr = df_mr.sort_values('mr_score')

    print(f"\nFound {len(df_mr)} mean-reverting stocks:")
    print("=" * 80)
    print(f"{'Symbol':<10} {'Hurst':>8} {'ADF p-val':>10} {'Half-life':>10} {'Autocorr':>10} {'Score':>8}")
    print("-" * 80)

    for _, row in df_mr.head(50).iterrows():
        print(f"{row['symbol']:<10} {row['hurst']:>8.3f} {row['adf_pvalue']:>10.4f} {row['half_life']:>10.1f} {row['autocorr']:>10.3f} {row['mr_score']:>8.3f}")

    # Save top mean reverting stocks
    output_file = 'inputs/mean_reverting_stocks.csv'
    df_mr[['symbol']].to_csv(output_file, index=False)
    print(f"\nSaved {len(df_mr)} mean-reverting stocks to {output_file}")

    # Also show all results for reference
    print("\n\nAll analyzed stocks (sorted by Hurst):")
    print("-" * 80)
    df_sorted = df.sort_values('hurst')
    for _, row in df_sorted.head(30).iterrows():
        hl = f"{row['half_life']:.1f}" if row['half_life'] < 1000 else "inf"
        print(f"{row['symbol']:<10} Hurst: {row['hurst']:.3f}, ADF: {row['adf_pvalue']:.4f}, HL: {hl}, AC: {row['autocorr']:.3f}")

    # Save full results
    df.to_csv('inputs/mean_reversion_analysis.csv', index=False)
    print(f"\nFull analysis saved to inputs/mean_reversion_analysis.csv")

if __name__ == "__main__":
    main()
