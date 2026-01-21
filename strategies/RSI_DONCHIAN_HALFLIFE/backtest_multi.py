# source bt/bin/activate
# cd strategies/RSI_DONCHIAN_HALFLIFE/
# python backtest_multi.py

import matplotlib
matplotlib.use('Agg')
import backtrader as bt
import yfinance as yf
from math import isnan
import matplotlib.pyplot as plt
from rsi_donchian_halflife import Strategy
import pandas as pd
import numpy as np
import csv
import os
import json
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


# ============================================================================
# STOCK CLASSIFICATION (from generic_strategy_generator)
# ============================================================================

class TrendBehavior(Enum):
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    MIXED = "mixed"


class VolatilityLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class StockClassification:
    """Classification result for a single stock"""
    symbol: str
    trend_behavior: TrendBehavior
    volatility_level: VolatilityLevel
    avg_adx: float
    avg_atr_pct: float

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'trend_behavior': self.trend_behavior.value,
            'volatility_level': self.volatility_level.value,
            'avg_adx': self.avg_adx,
            'avg_atr_pct': self.avg_atr_pct,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'StockClassification':
        return cls(
            symbol=data['symbol'],
            trend_behavior=TrendBehavior(data['trend_behavior']),
            volatility_level=VolatilityLevel(data['volatility_level']),
            avg_adx=data['avg_adx'],
            avg_atr_pct=data['avg_atr_pct'],
        )


class StockClassifier:
    """Classifies stocks by behavioral characteristics using ADX and ATR."""

    def __init__(self, adx_mean_reverting=20, atr_pct_high=3.0, atr_pct_low=1.5):
        self.adx_mean_reverting = adx_mean_reverting
        self.adx_trending = 25
        self.atr_pct_high = atr_pct_high
        self.atr_pct_low = atr_pct_low

    def classify_stock(self, symbol: str, data: pd.DataFrame) -> Optional[StockClassification]:
        """Classify a single stock based on historical data."""
        if data is None or len(data) < 60:
            return None

        try:
            df = data.copy()
            df.columns = [col.lower() for col in df.columns]

            close = df['close'].values
            high = df['high'].values
            low = df['low'].values

            mask = ~(np.isnan(close) | np.isnan(high) | np.isnan(low))
            close = close[mask]
            high = high[mask]
            low = low[mask]

            if len(close) < 60:
                return None

            # Calculate ADX
            avg_adx = self._calculate_avg_adx(high, low, close, period=14)

            # Calculate ATR%
            avg_atr_pct = self._calculate_avg_atr_pct(high, low, close, period=14)

            # Classify trend behavior
            if avg_adx >= self.adx_trending:
                trend = TrendBehavior.TRENDING
            elif avg_adx <= self.adx_mean_reverting:
                trend = TrendBehavior.MEAN_REVERTING
            else:
                trend = TrendBehavior.MIXED

            # Classify volatility
            if avg_atr_pct >= self.atr_pct_high:
                volatility = VolatilityLevel.HIGH
            elif avg_atr_pct <= self.atr_pct_low:
                volatility = VolatilityLevel.LOW
            else:
                volatility = VolatilityLevel.MEDIUM

            return StockClassification(
                symbol=symbol,
                trend_behavior=trend,
                volatility_level=volatility,
                avg_adx=avg_adx,
                avg_atr_pct=avg_atr_pct,
            )
        except Exception as e:
            print(f"Error classifying {symbol}: {e}")
            return None

    def _calculate_avg_adx(self, high, low, close, period=14) -> float:
        """Calculate average ADX."""
        if len(close) < period * 3:
            return 0.0

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        def wilder_smooth(arr, period):
            alpha = 1.0 / period
            result = np.zeros(len(arr))
            result[period-1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
            return result[period-1:]

        if len(tr) < period:
            return 0.0

        atr = wilder_smooth(tr, period)
        smoothed_plus_dm = wilder_smooth(plus_dm, period)
        smoothed_minus_dm = wilder_smooth(minus_dm, period)

        atr_safe = np.where(atr == 0, 1e-10, atr)
        plus_di = 100 * smoothed_plus_dm / atr_safe
        minus_di = 100 * smoothed_minus_dm / atr_safe

        di_sum = plus_di + minus_di
        di_sum = np.where(di_sum == 0, 1e-10, di_sum)
        di_diff = np.abs(plus_di - minus_di)
        dx = 100 * di_diff / di_sum

        if len(dx) < period:
            return np.mean(dx) if len(dx) > 0 else 0.0

        adx = wilder_smooth(dx, period)
        return np.mean(adx) if len(adx) > 0 else 0.0

    def _calculate_avg_atr_pct(self, high, low, close, period=14) -> float:
        """Calculate ATR as percentage of price."""
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

        atr = np.convolve(tr, np.ones(period)/period, mode='valid')
        close_aligned = close[period:]

        if len(close_aligned) == 0 or len(atr) == 0:
            return 0.0

        min_len = min(len(atr), len(close_aligned))
        atr = atr[:min_len]
        close_aligned = close_aligned[:min_len]
        close_aligned = np.where(close_aligned == 0, 1e-10, close_aligned)

        atr_pct = (atr / close_aligned) * 100
        return np.mean(atr_pct)


# ============================================================================
# CONFIGURATION
# ============================================================================
STRATEGY_PARAMS = {
    'rsi_period': 14,
    'rsi_threshold': 30,
    'donchian_period': 20,
    'use_donchian_filter': True,
    'halflife_period': 50,
    'halflife_exit_threshold': 50,
    'stop_loss_pct': Decimal("0.05"),
    'take_profit_pct': Decimal("0.08"),
    'use_take_profit': True,
    'verbose': False
}

# Stock list - use path relative to this file or absolute path
CSV_FILE = '../../generic_strategy_generator/inputs/stocks.csv'
CLASSIFICATION_CACHE = 'classification_cache.json'
PERIOD = "2y"
INITIAL_CASH = 10_000
PLOT_BEST = True

# Classification filter
USE_CLASSIFICATION = True
FILTER_MEAN_REVERTING = True  # Only test mean-reverting stocks


# ============================================================================
# OBSERVERS
# ============================================================================

class BuySellArrows(bt.observers.BuySell):
    plotlines = dict(
        buy=dict(marker='^', markersize=8, color='lime', fillstyle='full', ls=''),
        sell=dict(marker='v', markersize=8, color='red', fillstyle='full', ls='')
    )

    def next(self):
        super(BuySellArrows, self).next()
        if self.lines.buy[0] and not isnan(self.lines.buy[0]):
            self.lines.buy[0] = self.data.low[0] * 0.97
        if self.lines.sell[0] and not isnan(self.lines.sell[0]):
            self.lines.sell[0] = self.data.high[0] * 1.03


class PortfolioValue(bt.Observer):
    lines = ('value',)
    plotinfo = dict(plot=False, subplot=False)

    def next(self):
        self.lines.value[0] = self._owner.broker.getvalue()

    def prenext(self):
        self.lines.value[0] = self._owner.broker.getvalue()


# ============================================================================
# BACKTEST FUNCTIONS
# ============================================================================

def backtest_symbol(symbol, period="2y", initial_cash=10_000, strategy_params=None, plot=False):
    """Run backtest for a single symbol and return results."""
    try:
        df = yf.download(symbol, period=period, interval="1d", progress=False)

        if df.empty:
            return None

        df.index = df.index.tz_localize(None)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']

        # Download SPY for benchmark
        start_date = df.index[0]
        end_date = df.index[-1]
        spy_df = yf.download('SPY', start=start_date, end=end_date, progress=False)
        spy_df.index = spy_df.index.tz_localize(None)

        spy_initial_price = spy_df['Close'].iloc[0]
        spy_final_price = spy_df['Close'].iloc[-1]

        if hasattr(spy_initial_price, 'item'):
            spy_initial_price = spy_initial_price.item()
        if hasattr(spy_final_price, 'item'):
            spy_final_price = spy_final_price.item()

        spy_shares = initial_cash / spy_initial_price
        spy_final_value = spy_shares * spy_final_price
        spy_return = (spy_final_value / initial_cash - 1) * 100

        # Setup cerebro
        cerebro = bt.Cerebro(stdstats=False)
        data = bt.feeds.PandasData(
            dataname=df,
            datetime=None,
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume'
        )
        cerebro.adddata(data)

        if strategy_params:
            cerebro.addstrategy(Strategy, **strategy_params)
        else:
            cerebro.addstrategy(Strategy)

        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=0.0)

        # Analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")

        if plot:
            cerebro.addobserver(BuySellArrows, plot=True, subplot=False)
            cerebro.addobserver(bt.observers.Trades, plot=True, subplot=False)
            cerebro.addobserver(PortfolioValue)

        results = cerebro.run()
        strat = results[0]

        # Extract results
        sharpe = strat.analyzers.sharpe.get_analysis()
        dd = strat.analyzers.dd.get_analysis()
        trades = strat.analyzers.trades.get_analysis()
        sqn = strat.analyzers.sqn.get_analysis()

        final_value = cerebro.broker.getvalue()
        total_return = final_value - initial_cash
        return_pct = (final_value / initial_cash - 1) * 100

        total_trades = trades.get('total', {}).get('total', 0)

        if total_trades > 0:
            wins = trades.get('won', {}).get('total', 0)
            losses = trades.get('lost', {}).get('total', 0)
            win_rate = (wins / total_trades * 100)

            avg_win = trades.get('won', {}).get('pnl', {}).get('average', 0)
            avg_loss = abs(trades.get('lost', {}).get('pnl', {}).get('average', 0))
            rr_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0

            total_win_pnl = trades.get('won', {}).get('pnl', {}).get('total', 0)
            total_loss_pnl = abs(trades.get('lost', {}).get('pnl', {}).get('total', 0))
            profit_factor = (total_win_pnl / total_loss_pnl) if total_loss_pnl > 0 else 0

            expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
            avg_trade_len = trades.get('len', {}).get('average', 0)
        else:
            win_rate = rr_ratio = profit_factor = expectancy = 0
            avg_win = avg_loss = avg_trade_len = 0
            wins = losses = 0

        max_dd_pct = dd.get('max', {}).get('drawdown', 0)
        sqn_score = sqn.get('sqn', 0)

        days_traded = len(df)
        years = days_traded / 252
        annualized_return = ((final_value / initial_cash) ** (1 / years) - 1) * 100 if years > 0 else 0

        result = {
            'symbol': symbol,
            'final_value': final_value,
            'return_pct': return_pct,
            'annualized_return': annualized_return,
            'spy_return': spy_return,
            'outperformance': return_pct - spy_return,
            'sharpe': sharpe.get('sharperatio', 0) or 0,
            'sqn': sqn_score,
            'max_drawdown': max_dd_pct,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'rr_ratio': rr_ratio,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_trade_len': avg_trade_len,
        }

        if plot:
            plt.style.use('dark_background')
            figs = cerebro.plot(
                style='candlestick',
                iplot=False,
                barup='#597D35',
                bardown='#FF7171',
                volume=False,
            )
            for fig in figs:
                fig[0].savefig(f"{symbol}_backtest.png", dpi=150, bbox_inches='tight')

        return result

    except Exception as e:
        print(f"❌ {symbol}: Error - {str(e)}")
        return None


def classify_stocks(symbols: List[str], cache_path: str = None) -> Dict[str, StockClassification]:
    """Classify all stocks, using cache if available."""
    classifications = {}

    # Try to load from cache
    if cache_path and os.path.exists(cache_path):
        print(f"Loading classifications from cache: {cache_path}")
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)

        # Check if all symbols are in cache
        cached_symbols = set(cache_data.keys())
        current_symbols = set(symbols)
        missing = current_symbols - cached_symbols

        if not missing:
            print(f"✓ Cache is current ({len(cache_data)} stocks)")
            for symbol, data in cache_data.items():
                if symbol in current_symbols:
                    classifications[symbol] = StockClassification.from_dict(data)
            return classifications
        else:
            print(f"Cache is stale: {len(missing)} stocks missing. Regenerating...")

    # Classify stocks
    print("Classifying stocks...")
    classifier = StockClassifier()

    for i, symbol in enumerate(symbols):
        print(f"  [{i+1}/{len(symbols)}] Classifying {symbol}...", end='\r')
        try:
            df = yf.download(symbol, period="2y", interval="1d", progress=False)
            if not df.empty:
                df.index = df.index.tz_localize(None)
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df.columns = ['open', 'high', 'low', 'close', 'volume']

                classification = classifier.classify_stock(symbol, df)
                if classification:
                    classifications[symbol] = classification
        except Exception as e:
            pass

    print()

    # Save to cache
    if cache_path and classifications:
        cache_data = {s: c.to_dict() for s, c in classifications.items()}
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"✓ Saved classifications to {cache_path}")

    return classifications


def run_multi_backtest(csv_file='stocks.csv', period="2y", initial_cash=10_000,
                       strategy_params=None, plot_best=True,
                       use_classification=True, filter_mean_reverting=True):
    """Run backtest on multiple stocks from CSV file."""

    # Read symbols from CSV
    symbols = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip() and row[0].strip() != 'symbol':
                symbols.append(row[0].strip())

    print(f"\n{'='*70}")
    print(f"RSI/DONCHIAN/HALFLIFE STRATEGY - MULTI-STOCK BACKTEST")
    print(f"{'='*70}\n")
    print(f"Total stocks in list: {len(symbols)}")

    # Classify and filter stocks
    if use_classification:
        classifications = classify_stocks(symbols, CLASSIFICATION_CACHE)

        # Print classification summary
        trend_counts = {}
        for clf in classifications.values():
            trend = clf.trend_behavior.value
            trend_counts[trend] = trend_counts.get(trend, 0) + 1

        print(f"\nClassification Summary:")
        for trend, count in sorted(trend_counts.items()):
            pct = count / len(classifications) * 100
            print(f"  {trend:15s}: {count:4d} ({pct:5.1f}%)")

        if filter_mean_reverting:
            # Filter to only mean-reverting stocks
            filtered_symbols = [
                s for s in symbols
                if s in classifications and
                classifications[s].trend_behavior == TrendBehavior.MEAN_REVERTING
            ]
            print(f"\nFiltered to mean-reverting stocks: {len(filtered_symbols)}")
            symbols = filtered_symbols

    if not symbols:
        print("No stocks to test!")
        return

    print(f"\nStrategy Configuration:")
    print(f"  Entry: RSI({strategy_params.get('rsi_period', 14)}) < {strategy_params.get('rsi_threshold', 30)}")
    print(f"         when price > DONCHIAN({strategy_params.get('donchian_period', 20)})")
    print(f"  Exit: HALFLIFE({strategy_params.get('halflife_period', 50)}) > {strategy_params.get('halflife_exit_threshold', 50)}")
    print(f"        + {float(strategy_params.get('stop_loss_pct', 0.05))*100}% SL")
    print(f"        + {float(strategy_params.get('take_profit_pct', 0.08))*100}% TP")
    print()

    results = []
    for symbol in symbols:
        print(f"Testing {symbol}...", end=" ")
        result = backtest_symbol(symbol, period, initial_cash, strategy_params, plot=False)
        if result:
            results.append(result)
            print(f"✓ Return: {result['return_pct']:7.2f}% | vs SPY: {result['outperformance']:+7.2f}% | Trades: {result['total_trades']}")
        else:
            print()

    if not results:
        print("No valid results!")
        return

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Print summary
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")

    total_initial = len(results) * initial_cash
    total_final = df_results['final_value'].sum()
    total_return = total_final - total_initial
    avg_return_pct = df_results['return_pct'].mean()
    avg_spy_return = df_results['spy_return'].mean()

    print(f"\n💰 Portfolio Performance:")
    print(f"   Stocks Tested:         {len(results)}")
    print(f"   Total Initial Value:   ${total_initial:,.2f}")
    print(f"   Total Final Value:     ${total_final:,.2f}")
    print(f"   Total Return:          ${total_return:,.2f} ({(total_return/total_initial)*100:.2f}%)")
    print(f"   Avg Return per Stock:  {avg_return_pct:.2f}%")
    print(f"   Avg SPY Return:        {avg_spy_return:.2f}%")
    print(f"   Avg Outperformance:    {df_results['outperformance'].mean():.2f}%")
    print(f"   Stocks Beat SPY:       {len(df_results[df_results['outperformance'] > 0])} ({len(df_results[df_results['outperformance'] > 0])/len(results)*100:.1f}%)")

    print(f"\n📉 Risk Metrics:")
    avg_sharpe = df_results['sharpe'].mean()
    print(f"   Avg Sharpe Ratio:      {avg_sharpe:.3f}" if not pd.isna(avg_sharpe) else "   Avg Sharpe Ratio:      N/A")
    print(f"   Avg SQN:               {df_results['sqn'].mean():.2f}")
    print(f"   Avg Max Drawdown:      {df_results['max_drawdown'].mean():.2f}%")

    print(f"\n📈 Trade Statistics:")
    total_trades = int(df_results['total_trades'].sum())
    total_wins = int(df_results['wins'].sum())
    total_losses = int(df_results['losses'].sum())
    print(f"   Total Trades:          {total_trades}")
    print(f"   Total Wins:            {total_wins}")
    print(f"   Total Losses:          {total_losses}")
    if total_trades > 0:
        print(f"   Overall Win Rate:      {(total_wins / total_trades * 100):.1f}%")
    print(f"   Avg Profit Factor:     {df_results['profit_factor'].mean():.2f}")
    print(f"   Avg Expectancy:        ${df_results['expectancy'].mean():.2f}")

    print(f"\n🏆 Top 5 Performers (by Return %):")
    top5 = df_results.nlargest(5, 'return_pct')[['symbol', 'return_pct', 'outperformance', 'total_trades', 'win_rate']]
    for idx, row in top5.iterrows():
        print(f"   {row['symbol']:6s}: {row['return_pct']:7.2f}% | vs SPY: {row['outperformance']:+7.2f}% | Trades: {int(row['total_trades'])} | WR: {row['win_rate']:.0f}%")

    print(f"\n📉 Bottom 5 Performers:")
    bottom5 = df_results.nsmallest(5, 'return_pct')[['symbol', 'return_pct', 'outperformance', 'total_trades']]
    for idx, row in bottom5.iterrows():
        print(f"   {row['symbol']:6s}: {row['return_pct']:7.2f}% | vs SPY: {row['outperformance']:+7.2f}% | Trades: {int(row['total_trades'])}")

    print(f"\n{'='*70}\n")

    # Save results
    df_results.to_csv('backtest_results.csv', index=False)
    print("📄 Detailed results saved to 'backtest_results.csv'")

    # Plot best performer
    if plot_best and len(results) > 0:
        best_symbol = df_results.loc[df_results['return_pct'].idxmax(), 'symbol']
        print(f"\n📊 Generating plot for best performer: {best_symbol}\n")
        backtest_symbol(best_symbol, period, initial_cash, strategy_params, plot=True)


if __name__ == "__main__":
    run_multi_backtest(
        csv_file=CSV_FILE,
        period=PERIOD,
        initial_cash=INITIAL_CASH,
        strategy_params=STRATEGY_PARAMS,
        plot_best=PLOT_BEST,
        use_classification=USE_CLASSIFICATION,
        filter_mean_reverting=FILTER_MEAN_REVERTING,
    )
