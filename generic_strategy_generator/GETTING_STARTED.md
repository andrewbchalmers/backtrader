# Getting Started with Generic Strategy Generator

## Quick Setup (5 minutes)

### Step 1: Install Dependencies

```bash
cd generic_strategy_generator
pip install -r requirements.txt --break-system-packages
```

### Step 2: Verify Installation

```bash
python test_system.py
```

You should see:
```
✓ All tests passed!
✓ System is ready to run
✓ 36 strategies will be tested across 10 stocks
```

### Step 3: Download Historical Data (Optional but Recommended)

```bash
python main.py --download-data
```

This downloads historical data for all stocks in `stocks.csv` and saves them to `data/historical/`.

### Step 4: Run Your First Optimization

```bash
python main.py
```

This will:
1. Generate 36 strategy combinations
2. Test each strategy across 10 stocks (360 backtests total)
3. Calculate comprehensive performance metrics
4. Rank strategies by composite score
5. Export top 50 strategies to CSV
6. Generate summary report

Expected runtime: 2-5 minutes (depending on your system)

## Understanding the Output

### Console Output

You'll see progress bars and status messages:
```
================================================================================
GENERIC STRATEGY GENERATOR
================================================================================

✓ Loaded configuration from config.yaml
✓ Loaded 10 stocks from stocks.csv
✓ Loaded 6 entry indicators
✓ Loaded 6 exit indicators
✓ Generated 36 strategy combinations

Running backtests with 4 parallel workers...

Testing strategies: 100%|██████████| 36/36 [02:15<00:00,  3.76s/it]

✓ Completed 36 backtests in 135.4 seconds
  Average: 3.76 seconds per strategy
```

### Files Generated

1. **`results/top_strategies.csv`** - Your main results file
   - Ranked list of best strategies
   - All performance metrics
   - Strategy descriptions

2. **`results/strategies.db`** - SQLite database
   - Complete results for all strategies
   - Individual stock performance data
   - Query-able for custom analysis

3. **`results/reports/summary_report.txt`** - Summary report
   - Top 10 strategies with details
   - High-level statistics

## Customization

### Change Indicators

Edit `entry_indicators.csv` or `exit_indicators.csv`:

```csv
sma,10
sma,20
sma,50
ema,12
ema,26
rsi,14
rsi,21
bb,20,2
macd,12,26,9
```

### Change Stocks

Edit `stocks.csv`:

```csv
symbol
AAPL
MSFT
GOOGL
AMZN
TSLA
NVDA
```

### Adjust Settings

Edit `config.yaml`:

```yaml
# Test more strategies
optimization:
  top_n: 100

# Stricter filters
filters:
  min_sharpe: 1.0
  min_profit_factor: 2.0
  max_drawdown: 0.20

# More parallel workers
execution:
  parallel_workers: 8
```

## Example Workflow

### Scenario 1: Quick Test with 3 Stocks

1. Create minimal `stocks.csv`:
   ```csv
   symbol
   AAPL
   MSFT
   GOOGL
   ```

2. Run with 2 workers:
   ```bash
   python main.py --parallel 2
   ```

3. Review results in `results/top_strategies.csv`

### Scenario 2: Comprehensive Test with 50 Stocks

1. Create comprehensive `stocks.csv` with 50 stocks

2. Download data first:
   ```bash
   python main.py --download-data
   ```

3. Run with maximum parallelization:
   ```bash
   python main.py --parallel 8 --top-n 100
   ```

4. Analyze results in SQLite database:
   ```python
   import sqlite3
   import pandas as pd
   
   conn = sqlite3.connect('results/strategies.db')
   df = pd.read_sql_query("SELECT * FROM strategies WHERE sharpe_ratio > 1.5", conn)
   print(df.sort_values('score', ascending=False))
   ```

### Scenario 3: Test Specific Indicator Combination

1. Create focused indicator files:

   `entry_indicators.csv`:
   ```csv
   ema,5
   ema,10
   ```
   
   `exit_indicators.csv`:
   ```csv
   sma,20
   atr,14,2
   ```

2. Run optimization:
   ```bash
   python main.py
   ```

This generates 4 strategy combinations (2 entry × 2 exit).

## Interpreting Results

### Top Strategy Example

```csv
rank,strategy_id,description,score,total_return,sharpe_ratio,max_drawdown,profit_factor,win_rate
1,15,"Entry: CROSSOVER on EMA(5) | Exit: STOP_LOSS on SMA(10)",87.34,45.23,1.85,-12.34,2.45,58.33
```

**What this means:**
- **Score (87.34)**: Composite score out of 100 (higher is better)
- **Total Return (45.23%)**: Average return across all stocks
- **Sharpe Ratio (1.85)**: Risk-adjusted return (>1 is good, >2 is excellent)
- **Max Drawdown (-12.34%)**: Worst peak-to-trough decline
- **Profit Factor (2.45)**: Ratio of gross profit to gross loss
- **Win Rate (58.33%)**: Percentage of winning trades

### Good Strategy Characteristics

Look for strategies with:
- ✓ Sharpe ratio > 1.0
- ✓ Profit factor > 1.5
- ✓ Max drawdown < 20%
- ✓ Win rate > 45%
- ✓ Total trades > 20 (sufficient sample size)

## Troubleshooting

### Error: "No module named 'backtrader'"
```bash
pip install backtrader --break-system-packages
```

### Error: "No data for symbol XXXX"
- Symbol may be delisted or invalid
- Remove from `stocks.csv` or fix spelling
- Use `--download-data` to verify data availability

### Program runs very slowly
- Reduce number of stocks in `stocks.csv`
- Increase parallel workers: `--parallel 8`
- Reduce date range in `config.yaml`

### "Database locked" error
- Close other programs accessing the database
- Increase `save_interval` in `config.yaml`
- Use `--clear-db` to reset

## Next Steps

1. **Review Top Strategies**: Open `results/top_strategies.csv`

2. **Validate Winners**: Manually backtest top 3-5 strategies

3. **Paper Trade**: Test best strategy with paper trading

4. **Iterate**: Adjust indicators and filters based on results

5. **Out-of-Sample Test**: Test on different date ranges

## Advanced Usage

### Query Database Directly

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('results/strategies.db')

# Find strategies with >50% win rate and >2.0 profit factor
query = """
    SELECT description, total_return, sharpe_ratio, profit_factor, win_rate
    FROM strategies 
    WHERE win_rate > 0.5 AND profit_factor > 2.0
    ORDER BY score DESC
    LIMIT 10
"""

df = pd.read_sql_query(query, conn)
print(df)
```

### Export Specific Strategy Results

```python
from src.results_manager import ResultsManager

config = {...}  # Load config
rm = ResultsManager(config)

# Get specific strategy
strategy = rm.load_strategy_by_id(15)
print(strategy['description'])
print(strategy['stock_results'])
```

### Batch Processing Multiple Indicator Sets

```bash
# Test moving average strategies
cp ma_entry_indicators.csv entry_indicators.csv
cp ma_exit_indicators.csv exit_indicators.csv
python main.py --clear-db

# Test RSI strategies
cp rsi_entry_indicators.csv entry_indicators.csv
cp rsi_exit_indicators.csv exit_indicators.csv
python main.py --clear-db

# Compare results in database
```

## Support

- Review `ARCHITECTURE.md` for system design
- Check `README.md` for comprehensive documentation
- Examine source code in `src/` for implementation details

Happy strategy hunting! 🚀
