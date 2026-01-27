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
        self.warmup_days = self._calculate_warmup()

    def _calculate_warmup(self):
        """Calculate warmup days needed from strategy parameters"""
        max_period = 0

        lookback_params = [
            'trend_len', 'slow_len', 'fast_len', 'atr_len',
            'ma_period', 'sma_period', 'ema_period', 'rsi_period',
            'bb_period', 'macd_slow', 'lookback', 'period', 'length'
        ]

        for param_name, param_value in self.params.items():
            if any(key in param_name.lower() for key in lookback_params):
                if isinstance(param_value, (int, float)):
                    max_period = max(max_period, int(param_value))

        # Add small buffer for indicator stabilization (just need the period itself + a few extra)
        # Previously 1.5x was too aggressive and caused all bars to be skipped
        warmup = max_period + 10 if max_period > 0 else 50
        print(f"ℹ️  Chart warmup calculated: {warmup} days (from max period: {max_period})")
        return warmup

    def generate_chart(self, symbol, df, days=30, title_suffix=""):
        """
        Generate a technical chart for a symbol

        Args:
            symbol: Stock ticker
            df: DataFrame with OHLC data (must include warmup period)
            days: Number of days to show in chart
            title_suffix: Optional suffix for chart title (e.g., "30 Day" or "3 Month")

        Returns:
            BytesIO: PNG image buffer
        """
        # Ensure we have enough data: warmup + display days
        total_needed = days + self.warmup_days

        if len(df) < total_needed:
            print(f"⚠️  Insufficient data: have {len(df)}, need {total_needed} (warmup: {self.warmup_days}, display: {days})")
            # Use what we have, but warn
            if len(df) < self.warmup_days:
                print(f"⚠️  Not enough data for warmup, signals may be inaccurate")

        # Get last N days for display
        df_chart = df.tail(days).copy()

        if len(df_chart) == 0:
            return None

        # Calculate indicators using full available data for proper warmup
        # Use all data up to the end of the chart period for accurate signals
        indicators = self.strategy._calculate_indicators(df, self.params)

        # Get the indicators aligned with chart period
        # We need to get the last 'days' values of each indicator
        fast_sma = indicators.get('fast_sma', pd.Series())
        slow_sma = indicators.get('slow_sma', pd.Series())
        trend_ma = indicators.get('trend_ma', pd.Series())
        atr = indicators.get('atr', pd.Series())

        # Align indicators with chart dates
        if not fast_sma.empty:
            fast_sma = fast_sma.reindex(df_chart.index)
        if not slow_sma.empty:
            slow_sma = slow_sma.reindex(df_chart.index)
        if not trend_ma.empty:
            trend_ma = trend_ma.reindex(df_chart.index)
        if not atr.empty:
            atr = atr.reindex(df_chart.index)

        # Find buy/sell signals in this period
        buy_signals = []
        sell_signals = []

        # Track simulated positions to detect sells
        simulated_position = None  # {'entry_price': float, 'stop_loss': float, 'entry_date': datetime}

        # Get the start index of chart period in the full dataframe
        chart_start_idx = len(df) - len(df_chart)

        for i in range(len(df_chart)):
            # Get all data up to and including current bar (for signal calculation)
            current_end_idx = chart_start_idx + i + 1
            current_df = df.iloc[:current_end_idx].copy()

            # Need at least warmup days of data
            if len(current_df) < self.warmup_days:
                continue

            current_date = df_chart.index[i]
            current_price = df_chart['Close'].iloc[i]

            if simulated_position is None:
                # Not in a position - check for buy signal
                signal = self.strategy.get_entry_signal(current_df, self.params)
                if signal['signal']:
                    buy_signals.append((current_date, current_price))
                    # Start tracking simulated position
                    simulated_position = {
                        'entry_price': signal['price'],
                        'stop_loss': signal['stop_loss'],
                        'entry_date': current_date
                    }
            else:
                # In a position - check for exit signal
                exit_signal = self.strategy.get_exit_signal(
                    current_df,
                    self.params,
                    simulated_position['entry_price'],
                    simulated_position['stop_loss']
                )

                if exit_signal['signal']:
                    # Stop hit - record sell signal
                    sell_signals.append((current_date, current_price))
                    simulated_position = None  # Exit position
                else:
                    # Update trailing stop
                    simulated_position['stop_loss'] = exit_signal['new_stop']

        # Determine chart title
        if title_suffix:
            chart_title = f'{symbol} - {title_suffix}'
        else:
            chart_title = f'{symbol} - Last {days} Days'

        # Create figure with single plot
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 8), facecolor='#1e1e1e')

        # Style
        ax1.set_facecolor('#1e1e1e')
        ax1.tick_params(colors='white', which='both')
        ax1.spines['bottom'].set_color('#404040')
        ax1.spines['top'].set_color('#404040')
        ax1.spines['left'].set_color('#404040')
        ax1.spines['right'].set_color('#404040')

        # --- Main chart: Price and SMAs ---
        ax1.plot(df_chart.index, df_chart['Close'],
                 label='Close', color='#00d4ff', linewidth=2, zorder=5)

        if not fast_sma.empty:
            ax1.plot(df_chart.index, fast_sma,
                     label=f'Fast SMA ({self.params.get("fast_len", 7)})',
                     color='#ff6b6b', linewidth=1.5, alpha=0.8)

        if not slow_sma.empty:
            ax1.plot(df_chart.index, slow_sma,
                     label=f'Slow SMA ({self.params.get("slow_len", 50)})',
                     color='#00ff88', linewidth=1.5, alpha=0.8)

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

        ax1.set_title(chart_title,
                      color='white', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Price ($)', color='white', fontsize=12)
        ax1.set_xlabel('Date', color='white', fontsize=12)
        ax1.legend(loc='upper left', framealpha=0.9, facecolor='#2a2a2a',
                   edgecolor='#404040', labelcolor='white')
        ax1.grid(True, alpha=0.2, color='#404040')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

        # Format x-axis
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, facecolor='#1e1e1e',
                    edgecolor='none', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        return buf

    def generate_multi_timeframe_chart(self, symbol, chart_data_list):
        """
        Generate a stacked chart with multiple timeframes

        Args:
            symbol: Stock ticker
            chart_data_list: List of dicts with:
                - 'df': DataFrame with OHLC data
                - 'bars': Number of bars to display
                - 'title': Chart title (e.g., '30 Day (Hourly)')
                - 'interval': Bar interval for display (e.g., '1h', '1d')

        Returns:
            BytesIO: PNG image buffer with stacked charts
        """
        if not chart_data_list:
            return None

        num_charts = len(chart_data_list)

        # Create figure with stacked subplots
        fig, axes = plt.subplots(num_charts, 1, figsize=(14, 6 * num_charts), facecolor='#1e1e1e')

        # Handle single chart case
        if num_charts == 1:
            axes = [axes]

        for ax_idx, (ax, chart_config) in enumerate(zip(axes, chart_data_list)):
            df = chart_config['df']
            bars = chart_config['bars']
            title_suffix = chart_config['title']
            interval = chart_config.get('interval', '1d')

            if df is None or len(df) == 0:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                        color='white', fontsize=14, transform=ax.transAxes)
                ax.set_facecolor('#1e1e1e')
                continue

            # Adjust warmup based on interval (hourly needs less warmup in terms of bars)
            if interval == '1h':
                # For hourly, scale warmup: 200 days * ~6.5 trading hours = lots of bars
                # But we just need enough for indicators to stabilize
                effective_warmup = min(self.warmup_days * 7, len(df) - bars)  # ~7 hours per day
            else:
                effective_warmup = self.warmup_days

            # Ensure we don't request more bars than available
            bars = min(bars, len(df))

            # Get chart data
            df_chart = df.tail(bars).copy()

            if len(df_chart) == 0:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                        color='white', fontsize=14, transform=ax.transAxes)
                ax.set_facecolor('#1e1e1e')
                continue

            # Calculate indicators using full available data
            indicators = self.strategy._calculate_indicators(df, self.params)

            # Get and align indicators with chart dates
            fast_sma = indicators.get('fast_sma', pd.Series())
            slow_sma = indicators.get('slow_sma', pd.Series())
            trend_ma = indicators.get('trend_ma', pd.Series())

            if not fast_sma.empty:
                fast_sma = fast_sma.reindex(df_chart.index)
            if not slow_sma.empty:
                slow_sma = slow_sma.reindex(df_chart.index)
            if not trend_ma.empty:
                trend_ma = trend_ma.reindex(df_chart.index)

            # Find buy/sell signals in this period
            buy_signals = []
            sell_signals = []
            simulated_position = None

            chart_start_idx = len(df) - len(df_chart)

            for i in range(len(df_chart)):
                current_end_idx = chart_start_idx + i + 1
                current_df = df.iloc[:current_end_idx].copy()

                if len(current_df) < effective_warmup:
                    continue

                current_date = df_chart.index[i]
                current_price = df_chart['Close'].iloc[i]

                if simulated_position is None:
                    signal = self.strategy.get_entry_signal(current_df, self.params)
                    # Only process BUY signals (SELL signals don't have stop_loss)
                    is_buy = signal.get('signal_type', 'BUY') == 'BUY'
                    if signal['signal'] and is_buy and 'stop_loss' in signal:
                        buy_signals.append((current_date, current_price))
                        simulated_position = {
                            'entry_price': signal['price'],
                            'stop_loss': signal['stop_loss'],
                            'entry_date': current_date
                        }
                else:
                    exit_signal = self.strategy.get_exit_signal(
                        current_df,
                        self.params,
                        simulated_position['entry_price'],
                        simulated_position['stop_loss']
                    )

                    if exit_signal['signal']:
                        sell_signals.append((current_date, current_price))
                        simulated_position = None
                    else:
                        simulated_position['stop_loss'] = exit_signal['new_stop']

            # Style the subplot
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white', which='both')
            for spine in ax.spines.values():
                spine.set_color('#404040')

            # Plot price
            ax.plot(df_chart.index, df_chart['Close'],
                    label='Close', color='#00d4ff', linewidth=2, zorder=5)

            # Plot indicators
            if not fast_sma.empty:
                ax.plot(df_chart.index, fast_sma,
                        label=f'Fast SMA ({self.params.get("fast_len", 7)})',
                        color='#ff6b6b', linewidth=1.5, alpha=0.8)

            if not slow_sma.empty:
                ax.plot(df_chart.index, slow_sma,
                        label=f'Slow SMA ({self.params.get("slow_len", 50)})',
                        color='#00ff88', linewidth=1.5, alpha=0.8)

            if not trend_ma.empty:
                ax.plot(df_chart.index, trend_ma,
                        label=f'Trend MA ({self.params.get("trend_len", 200)})',
                        color='#ffd93d', linewidth=1.5, alpha=0.7, linestyle='--')

            # Plot buy signals
            if buy_signals:
                dates, prices = zip(*buy_signals)
                ax.scatter(dates, prices, color='#00ff00', marker='^',
                           s=150, label='BUY', zorder=10, edgecolors='white', linewidths=1.5)

            # Plot sell signals
            if sell_signals:
                dates, prices = zip(*sell_signals)
                ax.scatter(dates, prices, color='#ff0000', marker='v',
                           s=150, label='SELL', zorder=10, edgecolors='white', linewidths=1.5)

            # Labels and formatting
            ax.set_title(f'{symbol} - {title_suffix}',
                         color='white', fontsize=14, fontweight='bold', pad=10)
            ax.set_ylabel('Price ($)', color='white', fontsize=10)
            ax.legend(loc='upper left', framealpha=0.9, facecolor='#2a2a2a',
                      edgecolor='#404040', labelcolor='white', fontsize=8)
            ax.grid(True, alpha=0.2, color='#404040')

            # Format x-axis based on interval
            if interval == '1h':
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            else:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # Only show x-label on bottom chart
            if ax_idx == num_charts - 1:
                ax.set_xlabel('Date', color='white', fontsize=10)

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