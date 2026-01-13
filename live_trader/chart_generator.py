# chart_generator.py
"""
Generate technical charts with indicators and signals
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import io


class ChartGenerator:
    """Generate technical analysis charts"""

    def __init__(self, strategy_loader, strategy_params):
        self.strategy = strategy_loader
        self.params = strategy_params

    def generate_chart(self, symbol, df, days=30):
        """
        Generate a technical chart for a symbol

        Args:
            symbol: Stock ticker
            df: DataFrame with OHLC data
            days: Number of days to show

        Returns:
            BytesIO: PNG image buffer
        """
        # Get last N days
        df_chart = df.tail(days).copy()

        if len(df_chart) == 0:
            return None

        # Calculate indicators for the chart period (need more data for calculation)
        lookback = max(self.params.get('trend_len', 200),
                       self.params.get('slow_len', 50))
        df_calc = df.tail(days + lookback).copy()
        indicators = self.strategy._calculate_indicators(df_calc, self.params)

        # Get the indicators for just the chart period
        fast_sma = indicators.get('fast_sma', pd.Series()).tail(days)
        slow_sma = indicators.get('slow_sma', pd.Series()).tail(days)
        trend_ma = indicators.get('trend_ma', pd.Series()).tail(days)
        atr = indicators.get('atr', pd.Series()).tail(days)

        # Find buy/sell signals in this period
        buy_signals = []
        sell_signals = []

        for i in range(1, len(df_chart)):
            current_df = df.iloc[:-(len(df_chart) - i - 1)] if i < len(df_chart) - 1 else df

            # Check for buy signal
            signal = self.strategy.get_entry_signal(current_df, self.params)
            if signal['signal']:
                buy_signals.append((df_chart.index[i], df_chart['Close'].iloc[i]))

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                       gridspec_kw={'height_ratios': [3, 1]},
                                       facecolor='#1e1e1e')

        # Style
        for ax in [ax1, ax2]:
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white', which='both')
            ax.spines['bottom'].set_color('#404040')
            ax.spines['top'].set_color('#404040')
            ax.spines['left'].set_color('#404040')
            ax.spines['right'].set_color('#404040')

        # --- Main chart: Price and SMAs ---
        ax1.plot(df_chart.index, df_chart['Close'],
                 label='Close', color='#00d4ff', linewidth=2, zorder=5)

        if not fast_sma.empty:
            ax1.plot(df_chart.index, fast_sma,
                     label=f'Fast SMA ({self.params.get("fast_len", 7)})',
                     color='#00ff88', linewidth=1.5, alpha=0.8)

        if not slow_sma.empty:
            ax1.plot(df_chart.index, slow_sma,
                     label=f'Slow SMA ({self.params.get("slow_len", 50)})',
                     color='#ff6b6b', linewidth=1.5, alpha=0.8)

        if not trend_ma.empty:
            ax1.plot(df_chart.index, trend_ma,
                     label=f'Trend MA ({self.params.get("trend_len", 200)})',
                     color='#ffd93d', linewidth=1.5, alpha=0.7, linestyle='--')

        # Plot buy signals
        if buy_signals:
            dates, prices = zip(*buy_signals)
            ax1.scatter(dates, prices, color='#00ff00', marker='^',
                        s=200, label='BUY Signal', zorder=10, edgecolors='white', linewidths=2)

        # Plot sell signals (if any)
        if sell_signals:
            dates, prices = zip(*sell_signals)
            ax1.scatter(dates, prices, color='#ff0000', marker='v',
                        s=200, label='SELL Signal', zorder=10, edgecolors='white', linewidths=2)

        ax1.set_title(f'{symbol} - Last {days} Days',
                      color='white', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Price ($)', color='white', fontsize=12)
        ax1.legend(loc='upper left', framealpha=0.9, facecolor='#2a2a2a',
                   edgecolor='#404040', labelcolor='white')
        ax1.grid(True, alpha=0.2, color='#404040')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

        # --- ATR subplot ---
        if not atr.empty:
            ax2.plot(df_chart.index, atr, color='#ff6b6b', linewidth=2)
            ax2.fill_between(df_chart.index, 0, atr, alpha=0.3, color='#ff6b6b')
            ax2.set_ylabel('ATR', color='white', fontsize=12)
            ax2.set_xlabel('Date', color='white', fontsize=12)
            ax2.grid(True, alpha=0.2, color='#404040')
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

        # Format x-axis
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, facecolor='#1e1e1e',
                    edgecolor='none', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        return buf

    def generate_holdings_chart(self, positions, monitor):
        """
        Generate a chart showing all holdings with P&L

        Args:
            positions: Dictionary of positions
            monitor: LiveTradingMonitor instance for fetching data

        Returns:
            BytesIO: PNG image buffer
        """
        if not positions:
            return None

        # Prepare data
        symbols = []
        pnls = []
        colors = []

        for symbol, pos in positions.items():
            entry_price = pos['entry_price']
            df = monitor.get_live_data(symbol)

            if df is not None:
                current_price = df['Close'].iloc[-1]
                pnl = ((current_price / entry_price) - 1) * 100
                symbols.append(symbol)
                pnls.append(pnl)
                colors.append('#00ff88' if pnl > 0 else '#ff6b6b')

        if not symbols:
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=(12, max(6, len(symbols) * 0.5)),
                               facecolor='#1e1e1e')
        ax.set_facecolor('#1e1e1e')

        # Horizontal bar chart
        y_pos = np.arange(len(symbols))
        bars = ax.barh(y_pos, pnls, color=colors, edgecolor='white', linewidth=1.5)

        # Add P&L labels on bars
        for i, (bar, pnl) in enumerate(zip(bars, pnls)):
            width = bar.get_width()
            label_x = width + (0.5 if width > 0 else -0.5)
            ax.text(label_x, bar.get_y() + bar.get_height()/2,
                    f'{pnl:+.2f}%',
                    ha='left' if width > 0 else 'right',
                    va='center', color='white', fontweight='bold', fontsize=10)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(symbols, color='white', fontsize=11)
        ax.set_xlabel('P&L (%)', color='white', fontsize=12)
        ax.set_title(f'Holdings P&L ({len(symbols)} positions)',
                     color='white', fontsize=16, fontweight='bold', pad=20)

        # Add zero line
        ax.axvline(x=0, color='white', linewidth=1, linestyle='--', alpha=0.5)

        # Style
        ax.tick_params(colors='white', which='both')
        ax.spines['bottom'].set_color('#404040')
        ax.spines['top'].set_color('#404040')
        ax.spines['left'].set_color('#404040')
        ax.spines['right'].set_color('#404040')
        ax.grid(True, alpha=0.2, color='#404040', axis='x')

        plt.tight_layout()

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, facecolor='#1e1e1e',
                    edgecolor='none', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        return buf