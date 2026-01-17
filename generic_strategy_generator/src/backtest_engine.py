"""
Backtest Engine - Executes Strategies Against Historical Data
"""

import backtrader as bt
import yfinance as yf
import pandas as pd
from datetime import datetime
import os


class BacktestEngine:
    """Execute backtests using backtrader"""

    def __init__(self, config):
        """
        Initialize backtest engine

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.backtest_config = config.get('backtest', {})
        self.data_config = config.get('data', {})

        self.initial_capital = self.backtest_config.get('initial_capital', 100000)
        self.commission = self.backtest_config.get('commission', 0.001)
        self.slippage = self.backtest_config.get('slippage', 0.0005)
        self.position_size = self.backtest_config.get('position_size', 0.95)

        # Data cache
        self.data_cache = {}

    def load_stock_data(self, symbol):
        """
        Load historical data for a stock

        Args:
            symbol: Stock ticker symbol

        Returns:
            Backtrader data feed
        """
        # Check cache first
        if symbol in self.data_cache:
            return self.data_cache[symbol]

        source = self.data_config.get('source', 'yahoo')
        start_date = self.data_config.get('start_date', '2020-01-01')
        end_date = self.data_config.get('end_date', '2024-12-31')

        # Try CSV first if it exists (avoids rate limits)
        import os
        data_dir = self.data_config.get('data_dir', 'data/historical')
        csv_path = os.path.join(data_dir, f"{symbol}.csv")

        if os.path.exists(csv_path):
            data = self._load_from_csv(symbol)
        elif source == 'yahoo':
            data = self._load_from_yahoo(symbol, start_date, end_date)
        elif source == 'csv':
            data = self._load_from_csv(symbol)
        else:
            raise ValueError(f"Unknown data source: {source}")

        # Cache the data
        self.data_cache[symbol] = data
        return data

    def _load_from_yahoo(self, symbol, start_date, end_date):
        """Load data from Yahoo Finance with rate limit handling"""
        import time

        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                # Add small delay to avoid rate limiting
                if attempt > 0:
                    print(f"  Retry {attempt}/{max_retries} for {symbol} after {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    # Small delay on first attempt too
                    time.sleep(0.5)

                # Set user agent to avoid 401 errors
                import requests
                session = requests.Session()
                session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

                df = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    progress=False
                )

                if df.empty:
                    return None

                # Check for minimum data points (need at least 60 days for most indicators)
                if len(df) < 60:
                    print(f"Warning: {symbol} has only {len(df)} data points, skipping (need at least 60)")
                    return None

                # yfinance sometimes returns MultiIndex columns, flatten them
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Ensure column names are lowercase and standard
                df.columns = [col.lower() for col in df.columns]

                # Rename columns to match backtrader expectations
                column_mapping = {
                    'adj close': 'adjclose',
                }
                df = df.rename(columns=column_mapping)

                # Convert to backtrader data feed
                data = bt.feeds.PandasData(dataname=df)
                return data

            except Exception as e:
                error_msg = str(e)
                if "Rate" in error_msg or "429" in error_msg or "Too Many Requests" in error_msg:
                    if attempt < max_retries - 1:
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        print(f"Error loading {symbol}: Rate limited after {max_retries} attempts")
                        return None
                else:
                    print(f"Error loading {symbol} from Yahoo: {e}")
                    import traceback
                    traceback.print_exc()
            return None

    def _load_from_csv(self, symbol):
        """Load data from CSV file"""
        data_dir = self.data_config.get('data_dir', 'data/historical')
        csv_path = os.path.join(data_dir, f"{symbol}.csv")

        if not os.path.exists(csv_path):
            return None

        try:
            # yfinance saves with MultiIndex columns, so we need header=[0, 1]
            df = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)

            # Flatten MultiIndex columns - use the first level (Price type)
            df.columns = df.columns.get_level_values(0)

            # Normalize column names to lowercase
            df.columns = [col.lower() for col in df.columns]

            # Check for minimum data points
            if len(df) < 60:
                print(f"Warning: {symbol} has only {len(df)} data points, skipping (need at least 60)")
                return None

            data = bt.feeds.PandasData(dataname=df)
            return data
        except Exception as e:
            print(f"Error loading {symbol} from CSV: {e}")
            return None

    def run_backtest(self, strategy_class, symbol):
        """
        Run a single backtest

        Args:
            strategy_class: Backtrader strategy class
            symbol: Stock ticker symbol

        Returns:
            Dictionary with backtest results or None if failed
        """
        try:
            # Create cerebro instance
            cerebro = bt.Cerebro()

            # Add strategy
            try:
                cerebro.addstrategy(strategy_class)
            except Exception as e:
                import traceback
                print(f"Error in cerebro.addstrategy for {symbol}:")
                print(f"Strategy class: {strategy_class}")
                print(f"Strategy params: {strategy_class.params}")
                traceback.print_exc()
                raise

            # Load data
            data = self.load_stock_data(symbol)
            if data is None:
                return None

            # Add data to cerebro
            cerebro.adddata(data)

            # Set initial capital
            cerebro.broker.setcash(self.initial_capital)

            # Set commission
            cerebro.broker.setcommission(commission=self.commission)

            # Add position sizer - use percentage of capital
            cerebro.addsizer(bt.sizers.PercentSizer, percents=self.position_size * 100)

            # Add analyzers
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')

            # Run backtest
            starting_value = cerebro.broker.getvalue()
            results = cerebro.run()
            ending_value = cerebro.broker.getvalue()

            # Extract results
            strategy_result = results[0]

            # Extract trade count
            trades_analysis = strategy_result.analyzers.trades.get_analysis()
            total_trades = trades_analysis.get('total', {}).get('total', 0)

            return {
                'symbol': symbol,
                'starting_value': starting_value,
                'ending_value': ending_value,
                'pnl': ending_value - starting_value,
                'return_pct': ((ending_value - starting_value) / starting_value) * 100,
                'total_trades': total_trades,
                'trades': trades_analysis,
                'sharpe': strategy_result.analyzers.sharpe.get_analysis(),
                'drawdown': strategy_result.analyzers.drawdown.get_analysis(),
                'returns': strategy_result.analyzers.returns.get_analysis(),
                'sqn': strategy_result.analyzers.sqn.get_analysis(),
            }

        except Exception as e:
            import traceback
            print(f"Error running backtest for {symbol}: {e}")
            print("Full traceback:")
            traceback.print_exc()
            return None

    def run_backtest_multi_stock(self, strategy_class, symbols):
        """
        Run backtest across multiple stocks with capital split evenly

        Args:
            strategy_class: Backtrader strategy class
            symbols: List of stock ticker symbols

        Returns:
            List of backtest results
        """
        results = []

        # Split capital evenly across stocks
        num_stocks = len(symbols)
        capital_per_stock = self.initial_capital / num_stocks if num_stocks > 0 else self.initial_capital

        # Temporarily set capital for each individual backtest
        original_capital = self.initial_capital
        self.initial_capital = capital_per_stock

        for symbol in symbols:
            result = self.run_backtest(strategy_class, symbol)
            if result is not None:
                results.append(result)

        # Restore original capital
        self.initial_capital = original_capital

        return results

    def clear_cache(self):
        """Clear the data cache"""
        self.data_cache.clear()


class DataLoader:
    """Helper class for loading and managing stock data"""

    @staticmethod
    def load_stock_list(csv_path):
        """
        Load list of stock symbols from CSV

        Args:
            csv_path: Path to CSV file with 'symbol' column

        Returns:
            List of stock symbols
        """
        try:
            df = pd.read_csv(csv_path)
            if 'symbol' not in df.columns:
                raise ValueError("CSV must have 'symbol' column")
            # Strip whitespace from symbols
            return [s.strip() for s in df['symbol'].tolist()]
        except Exception as e:
            print(f"Error loading stock list: {e}")
            return []

    @staticmethod
    def download_historical_data(symbols, start_date, end_date, output_dir='data/historical'):
        """
        Download and save historical data for multiple stocks

        Args:
            symbols: List of stock symbols
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            output_dir: Directory to save CSV files
        """
        import time
        import requests

        # Set user agent globally
        session = requests.Session()
        session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

        os.makedirs(output_dir, exist_ok=True)

        for i, symbol in enumerate(symbols):
            try:
                print(f"Downloading {symbol}... ({i+1}/{len(symbols)})")

                # Add delay to avoid rate limiting (1 second between requests)
                if i > 0:
                    time.sleep(1)

                df = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    progress=False
                )

                if not df.empty:
                    output_path = os.path.join(output_dir, f"{symbol}.csv")
                    df.to_csv(output_path)
                    print(f"  ✓ Saved to {output_path}")
                else:
                    print(f"  ✗ No data for {symbol}")

            except Exception as e:
                print(f"  ✗ Error downloading {symbol}: {e}")
                # If we get a 401, add a longer delay
                if "401" in str(e) or "Unauthorized" in str(e):
                    print(f"  Waiting 5 seconds due to rate limit...")
                    time.sleep(5)