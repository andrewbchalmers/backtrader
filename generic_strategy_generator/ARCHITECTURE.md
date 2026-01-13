# Generic Strategy Generator - Architecture Design

## Overview
This system generates and tests trading strategies by combining entry and exit indicators across multiple stocks to find optimal strategy configurations.

## Core Components

### 1. Configuration Files
- **entry_indicators.csv**: Defines entry condition indicators and parameters
- **exit_indicators.csv**: Defines exit condition indicators and parameters
- **stocks.csv**: List of stock symbols to test against
- **config.yaml**: System configuration (data paths, date ranges, initial capital, etc.)

### 2. Indicator Library (indicator_library.py)
Factory pattern for creating indicators dynamically from CSV definitions:
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- BB (Bollinger Bands)
- ATR (Average True Range)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Extensible for additional indicators

### 3. Strategy Generator (strategy_generator.py)
Creates combinations of entry/exit indicators:
- Reads indicator CSVs
- Generates all valid strategy combinations
- Creates backtrader strategy classes dynamically
- Entry logic types: crossover, threshold, band breakout
- Exit logic types: crossover, stop loss, take profit, trailing stop

### 4. Backtest Engine (backtest_engine.py)
Runs backtests using backtrader:
- Loads historical data for each stock
- Executes strategy against stock data
- Collects performance metrics
- Handles multiple timeframes if needed

### 5. Performance Analyzer (performance_analyzer.py)
Calculates comprehensive metrics:
- **Return Metrics**: Total return, CAGR, sharpe ratio
- **Risk Metrics**: Max drawdown, volatility, downside deviation
- **Trade Metrics**: Win rate, profit factor, avg win/loss
- **Risk-Reward**: RR ratio, expectancy, recovery factor
- **Trade Analysis**: Max consecutive wins/losses, avg trade duration

### 6. Strategy Optimizer (strategy_optimizer.py)
Ranks and filters strategies:
- Multi-objective optimization (Pareto frontier)
- Customizable scoring function
- Filters: minimum trades, max drawdown threshold, min sharpe
- Exports top N strategies

### 7. Results Manager (results_manager.py)
Handles output and reporting:
- Strategy performance database (SQLite)
- CSV exports for top strategies
- Visualization generation (equity curves, drawdown plots)
- Summary reports

### 8. Main Orchestrator (main.py)
Coordinates the entire pipeline:
- Loads configurations
- Generates strategy combinations
- Distributes backtests (parallel processing)
- Aggregates results
- Produces final rankings

## Data Flow

1. Load entry_indicators.csv + exit_indicators.csv
2. Generate strategy combinations (entry × exit)
3. For each combination:
   - For each stock in stocks.csv:
     - Run backtest
     - Collect metrics
   - Aggregate cross-stock performance
4. Rank strategies by composite score
5. Export top strategies with detailed metrics

## Performance Metrics Priority

### Tier 1 (Critical)
- Total Return / CAGR
- Maximum Drawdown
- Sharpe Ratio
- Profit Factor

### Tier 2 (Important)
- Win Rate
- Risk-Reward Ratio
- Average Win/Loss Ratio
- Recovery Factor

### Tier 3 (Supporting)
- Number of Trades
- Max Consecutive Losses
- Calmar Ratio
- Sortino Ratio
- Expectancy

## Extensibility Points

1. **New Indicators**: Add to indicator_library.py
2. **Custom Entry/Exit Logic**: Extend strategy_generator.py
3. **Additional Metrics**: Update performance_analyzer.py
4. **Different Optimizers**: Swap strategy_optimizer.py implementation
5. **Alternative Data Sources**: Modify backtest_engine.py data loader

## File Structure
```
generic_strategy_generator/
├── README.md
├── config.yaml
├── entry_indicators.csv
├── exit_indicators.csv
├── stocks.csv
├── main.py
├── src/
│   ├── __init__.py
│   ├── indicator_library.py
│   ├── strategy_generator.py
│   ├── backtest_engine.py
│   ├── performance_analyzer.py
│   ├── strategy_optimizer.py
│   └── results_manager.py
├── data/
│   └── historical/  (stock price data)
├── results/
│   ├── strategies.db
│   ├── top_strategies.csv
│   └── reports/
└── tests/
    └── test_*.py
```

## Execution Flow Example

```bash
python main.py --stocks stocks.csv --top-n 50 --parallel 8
```

Output:
- `results/strategies.db`: All tested strategies with metrics
- `results/top_strategies.csv`: Top 50 ranked strategies
- `results/reports/strategy_001.html`: Detailed report for top strategy
- `results/reports/equity_curves.png`: Visual comparison

## Next Steps

1. Implement indicator_library.py
2. Build strategy_generator.py with dynamic strategy creation
3. Create backtest_engine.py with backtrader integration
4. Develop performance_analyzer.py with comprehensive metrics
5. Implement strategy_optimizer.py with multi-objective ranking
6. Build results_manager.py for output handling
7. Create main.py orchestrator
8. Add unit tests
9. Create sample stocks.csv and config.yaml
10. Document usage with examples
