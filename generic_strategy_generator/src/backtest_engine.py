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
        
        if source == 'yahoo':
            data = self._load_from_yahoo(symbol, start_date, end_date)
        elif source == 'csv':
            data = self._load_from_csv(symbol)
        else:
            raise ValueError(f"Unknown data source: {source}")
        
        # Cache the data
        self.data_cache[symbol] = data
        return data
    
    def _load_from_yahoo(self, symbol, start_date, end_date):
        """Load data from Yahoo Finance"""
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                return None
            
            # Convert to backtrader data feed
            data = bt.feeds.PandasData(dataname=df)
            return data
        except Exception as e:
            print(f"Error loading {symbol} from Yahoo: {e}")
            return None
    
    def _load_from_csv(self, symbol):
        """Load data from CSV file"""
        data_dir = self.data_config.get('data_dir', 'data/historical')
        csv_path = os.path.join(data_dir, f"{symbol}.csv")
        
        if not os.path.exists(csv_path):
            return None
        
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
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
            cerebro.addstrategy(strategy_class)
            
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
            
            return {
                'symbol': symbol,
                'starting_value': starting_value,
                'ending_value': ending_value,
                'pnl': ending_value - starting_value,
                'return_pct': ((ending_value - starting_value) / starting_value) * 100,
                'trades': strategy_result.analyzers.trades.get_analysis(),
                'sharpe': strategy_result.analyzers.sharpe.get_analysis(),
                'drawdown': strategy_result.analyzers.drawdown.get_analysis(),
                'returns': strategy_result.analyzers.returns.get_analysis(),
                'sqn': strategy_result.analyzers.sqn.get_analysis(),
            }
        
        except Exception as e:
            print(f"Error running backtest for {symbol}: {e}")
            return None
    
    def run_backtest_multi_stock(self, strategy_class, symbols):
        """
        Run backtest across multiple stocks
        
        Args:
            strategy_class: Backtrader strategy class
            symbols: List of stock ticker symbols
            
        Returns:
            List of backtest results
        """
        results = []
        
        for symbol in symbols:
            result = self.run_backtest(strategy_class, symbol)
            if result is not None:
                results.append(result)
        
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
            return df['symbol'].tolist()
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
        os.makedirs(output_dir, exist_ok=True)
        
        for symbol in symbols:
            try:
                print(f"Downloading {symbol}...")
                df = yf.download(symbol, start=start_date, end=end_date, progress=False)
                
                if not df.empty:
                    output_path = os.path.join(output_dir, f"{symbol}.csv")
                    df.to_csv(output_path)
                    print(f"  Saved to {output_path}")
                else:
                    print(f"  No data for {symbol}")
            
            except Exception as e:
                print(f"  Error downloading {symbol}: {e}")
