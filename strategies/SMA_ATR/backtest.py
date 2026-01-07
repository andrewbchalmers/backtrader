# source bt/bin/activate
# cd strategies/SMA_ATR/
# python backtest.py

import matplotlib
matplotlib.use('Agg')  # <-- put this at the very top, before backtrader import
import backtrader as bt
import yfinance as yf
from math import isnan
import matplotlib.pyplot as plt
from sma_atr import SMA_ATR_Exit


class BuySellArrows(bt.observers.BuySell):
    plotlines = dict(
        buy=dict(marker='^', markersize=8, color='lime', fillstyle='full', ls=''),
        sell=dict(marker='v', markersize=8, color='red', fillstyle='full', ls='')
    )

    def next(self):
        # Call parent to get the actual buy/sell signals
        super(BuySellArrows, self).next()

        # Only offset if there's actually a buy signal (non-zero/non-NaN)
        if self.lines.buy[0] and not isnan(self.lines.buy[0]):
            self.lines.buy[0] = self.data.low[0] * 0.97

        # Only offset if there's actually a sell signal
        if self.lines.sell[0] and not isnan(self.lines.sell[0]):
            self.lines.sell[0] = self.data.high[0] * 1.03


# 1️⃣ Download data
symbol = "SOFI"
df = yf.download(symbol, period="2y", interval="1d")
df.index = df.index.tz_localize(None)  # ensure naive timestamps
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.columns = ['open', 'high', 'low', 'close', 'volume']

# 2️⃣ Backtrader setup
cerebro = bt.Cerebro(stdstats=False)

# Use PandasData properly
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
cerebro.addstrategy(SMA_ATR_Exit)

# Broker
cerebro.broker.setcash(10_000)
cerebro.broker.setcommission(commission=0.0)
cerebro.broker.set_shortcash(False)

# Analyzers
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

# 2️⃣a Add observers for buy/sell markers and trades
cerebro.addobserver(BuySellArrows, plot=True, subplot=False)
cerebro.addobserver(bt.observers.Trades, plot=True, subplot=False)

# 3️⃣ Run
results = cerebro.run()
strat = results[0]
# Get analyzer results
sharpe = strat.analyzers.sharpe.get_analysis()
dd = strat.analyzers.dd.get_analysis()
trades = strat.analyzers.trades.get_analysis()

# Print formatted results
print("\n" + "="*50)
print("BACKTEST RESULTS")
print("="*50)

print(f"\n💰 Portfolio Performance:")
print(f"   Starting Value: ${10_000:,.2f}")
print(f"   Final Value:    ${cerebro.broker.getvalue():,.2f}")
print(f"   Total Return:   ${cerebro.broker.getvalue() - 10_000:,.2f} ({(cerebro.broker.getvalue()/10_000 - 1)*100:.2f}%)")

print(f"\n📊 Risk Metrics:")
print(f"   Sharpe Ratio:     {sharpe.get('sharperatio', 0):.3f}")
print(f"   Max Drawdown:     {dd['max']['drawdown']:.2f}%")
print(f"   Max Drawdown ($): ${dd['max']['moneydown']:,.2f}")

print(f"\n📈 Trade Statistics:")
print(f"   Total Trades:     {trades['total']['total']}")
print(f"   Wins:             {trades['won']['total']} ({trades['won']['total']/trades['total']['total']*100:.1f}%)")
print(f"   Losses:           {trades['lost']['total']} ({trades['lost']['total']/trades['total']['total']*100:.1f}%)")
print(f"   Avg Win:          ${trades['won']['pnl']['average']:,.2f}")
print(f"   Avg Loss:         ${trades['lost']['pnl']['average']:,.2f}")
print(f"   Avg Trade P&L:    ${trades['pnl']['net']['average']:,.2f}")
print(f"   Win Rate:         {trades['won']['total']/trades['total']['total']*100:.1f}%")

print("="*50 + "\n")

plt.style.use('dark_background')

# 4️⃣ Plot
figs = cerebro.plot(
    style='candlestick',
    iplot=False,
    barup='#597D35',
    bardown='#FF7171',
    volume=False,
)

# Save all figures (typically 1) to PNG
for fig in figs:
    fig[0].savefig("amd_backtest.png")

exit(0)
