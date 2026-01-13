# Generic Strategy Generator

A comprehensive framework for automatically generating, testing, and optimizing trading strategies using backtrader. The system tests all combinations of entry and exit indicators across multiple stocks to identify optimal trading strategies.

## Features

- 🎯 **Dynamic Strategy Generation**: Automatically creates strategy combinations from indicator definitions
- 📊 **Multi-Stock Testing**: Tests strategies across multiple stocks simultaneously
- 🔄 **Parallel Processing**: Utilizes multiple CPU cores for faster backtesting
- 📈 **Comprehensive Metrics**: Calculates 20+ performance metrics including Sharpe ratio, drawdown, profit factor, and more
- 🏆 **Multi-Objective Optimization**: Ranks strategies using weighted composite scoring
- 💾 **Results Database**: Stores all results in SQLite for later analysis
- 📑 **CSV Export**: Exports top strategies to CSV for easy review

## Quick Start

### Installation

```bash
# Install required packages
pip install backtrader yfinance pandas numpy pyyaml tqdm --break-system-packages
```

### Basic Usage

1. Define your indicators in CSV files:
   - `entry_indicators.csv`: Entry condition indicators
   - `exit_indicators.csv`: Exit condition indicators

2. List stocks to test in `stocks.csv`

3. Configure settings in `config.yaml`

4. Run the generator:

```bash
python main.py
```

### Advanced Usage

```bash
# Download data before running
python main.py --download-data

# Use custom configuration
python main.py --config my_config.yaml --stocks my_stocks.csv

# Export top 100 strategies with 8 parallel workers
python main.py --top-n 100 --parallel 8

# Clear database and start fresh
python main.py --clear-db
```

## File Structure

```
generic_strategy_generator/
├── config.yaml               # System configuration
├── entry_indicators.csv      # Entry indicator definitions
├── exit_indicators.csv       # Exit indicator definitions
├── stocks.csv               # List of stocks to test
├── main.py                  # Main orchestrator
├── ARCHITECTURE.md          # Detailed architecture documentation
├── src/
│   ├── indicator_library.py      # Indicator factory
│   ├── strategy_generator.py     # Strategy creation
│   ├── backtest_engine.py        # Backtest execution
│   ├── performance_analyzer.py   # Metrics calculation
│   ├── strategy_optimizer.py     # Strategy ranking
│   └── results_manager.py        # Results storage
├── data/
│   └── historical/          # Historical price data
└── results/
    ├── strategies.db        # SQLite database
    ├── top_strategies.csv   # Top strategies export
    └── reports/            # Generated reports
```

## Indicator CSV Format

### Entry Indicators (`entry_indicators.csv`)

```csv
sma,10
ema,20
ema,5
bb,17,3
atr,10,3
rsi,14
```

Format: `indicator_name,param1,param2,...`

### Supported Indicators

- **SMA**: Simple Moving Average (period)
- **EMA**: Exponential Moving Average (period)
- **BB**: Bollinger Bands (period, deviation)
- **ATR**: Average True Range (period, multiplier)
- **RSI**: Relative Strength Index (period)
- **MACD**: Moving Average Convergence Divergence (fast, slow, signal)
- **Stochastic**: Stochastic Oscillator (period, dfast, dslow)
- **CCI**: Commodity Channel Index (period)

## Strategy Types

### Entry Types

- **Crossover**: Price crosses above indicator
- **Threshold**: Indicator crosses threshold (for oscillators)
- **Breakout**: Price breaks out of bands

### Exit Types

- **Crossover**: Price crosses below indicator
- **Stop Loss**: Fixed percentage stop loss
- **Take Profit**: Fixed percentage take profit
- **Trailing Stop**: ATR-based trailing stop

## Configuration

Key configuration options in `config.yaml`:

```yaml
# Backtest Settings
backtest:
  initial_capital: 100000
  commission: 0.001
  
# Performance Filters
filters:
  min_trades: 10
  max_drawdown: 0.30
  min_sharpe: 0.5
  min_profit_factor: 1.2
  
# Optimization Weights
optimization:
  weights:
    total_return: 0.25
    sharpe_ratio: 0.20
    max_drawdown: 0.20
    profit_factor: 0.15
    win_rate: 0.10
    recovery_factor: 0.10
```

## Performance Metrics

The system calculates comprehensive metrics:

### Return Metrics
- Total Return (%)
- Average Return per Stock
- Median Return
- Return Standard Deviation

### Risk Metrics
- Maximum Drawdown
- Average Drawdown
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio

### Trade Metrics
- Total Trades
- Win Rate
- Profit Factor
- Expectancy
- Average Win/Loss Ratio

### Additional Metrics
- Largest Win/Loss
- Max Consecutive Wins/Losses
- Recovery Factor
- Number of Profitable Stocks

## Output Files

### `results/strategies.db`

SQLite database containing:
- All tested strategies with metrics
- Individual stock results for each strategy
- Timestamps for tracking

### `results/top_strategies.csv`

CSV export of top-ranked strategies with:
- Strategy description
- All performance metrics
- Entry/exit indicator details
- Composite score

### `results/reports/summary_report.txt`

Summary report showing:
- Total strategies tested
- Top 10 strategies with key metrics
- Execution statistics

## Example Output

```
================================================================================
STRATEGY OPTIMIZATION SUMMARY
================================================================================
Date: 2025-01-13 10:30:45
Total Strategies Analyzed: 36

TOP 10 STRATEGIES:
--------------------------------------------------------------------------------

Rank #1 (Score: 87.34)
  Description: Entry: CROSSOVER on EMA(5) | Exit: STOP_LOSS on SMA(10)
  Total Return: 45.23%
  Sharpe Ratio: 1.852
  Max Drawdown: -12.34%
  Profit Factor: 2.45
  Win Rate: 58.33%
  Total Trades: 47

Rank #2 (Score: 84.12)
  Description: Entry: THRESHOLD on RSI(14) | Exit: CROSSOVER on EMA(20)
  Total Return: 38.91%
  Sharpe Ratio: 1.723
  Max Drawdown: -15.67%
  Profit Factor: 2.12
  Win Rate: 55.00%
  Total Trades: 52
  
...
```

## Extending the System

### Adding New Indicators

Edit `src/indicator_library.py`:

```python
@staticmethod
def _create_custom_indicator(data, param1, param2):
    """Your custom indicator"""
    return bt.indicators.YourIndicator(data, param1=param1, param2=param2)
```

Add to factory method:

```python
elif indicator_name == 'custom':
    return IndicatorFactory._create_custom_indicator(data, *params)
```

### Custom Strategy Logic

Modify `src/strategy_generator.py` to implement custom entry/exit logic:

```python
def _check_entry_signal(self):
    # Your custom entry logic
    pass
```

### Custom Metrics

Add new metrics in `src/performance_analyzer.py`:

```python
def _calculate_custom_metric(self):
    # Your custom metric calculation
    pass
```

## Performance Tips

1. **Parallel Processing**: Use more workers for faster execution
   ```bash
   python main.py --parallel 8
   ```

2. **Data Caching**: Download data once, reuse for multiple runs
   ```bash
   python main.py --download-data
   # Then run without --download-data for subsequent tests
   ```

3. **Incremental Testing**: Test with a small stock universe first
   ```csv
   symbol
   AAPL
   MSFT
   ```

4. **Filter Early**: Set strict filters to reduce strategy count
   ```yaml
   filters:
     min_sharpe: 1.0  # Higher threshold
     min_profit_factor: 2.0
   ```

## Troubleshooting

### "No data for symbol"
- Check symbol spelling
- Verify date range
- Try downloading data manually with `--download-data`

### "Memory error"
- Reduce parallel workers
- Test fewer stocks
- Reduce strategy combinations

### "Database locked"
- Close other connections to database
- Increase save_interval in config

## Best Practices

1. **Start Small**: Test with 5-10 stocks first
2. **Validate Results**: Manually verify top strategies
3. **Avoid Overfitting**: Use out-of-sample testing
4. **Diversify Indicators**: Mix trend-following and mean-reversion
5. **Monitor Metrics**: Don't just optimize for returns

## Contributing

To add features or fix bugs:
1. Create a new branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

Built with:
- [Backtrader](https://www.backtrader.com/) - Backtesting framework
- [yfinance](https://github.com/ranaroussi/yfinance) - Market data
- [Pandas](https://pandas.pydata.org/) - Data manipulation

## Support

For issues or questions:
- Check ARCHITECTURE.md for detailed design documentation
- Review example outputs in results/
- Examine strategy_generator.py for strategy logic
