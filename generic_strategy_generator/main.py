"""
Main Orchestrator - Coordinates Strategy Generation and Testing Pipeline
"""

import argparse
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
import os
from src.indicator_library import parse_indicator_csv
from src.strategy_generator import StrategyGenerator, create_strategy_class
from src.backtest_engine import BacktestEngine, DataLoader
from src.performance_analyzer import PerformanceAnalyzer
from src.strategy_optimizer import StrategyOptimizer
from src.results_manager import ResultsManager


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_single_strategy(strategy_config, symbols, config):
    """
    Run a single strategy across all stocks

    Args:
        strategy_config: Strategy configuration dictionary
        symbols: List of stock symbols
        config: System configuration

    Returns:
        Tuple of (strategy_config, metrics) or (strategy_config, None) if failed
    """
    # Import inside function to ensure availability in worker processes
    from src.strategy_generator import create_strategy_class
    from src.backtest_engine import BacktestEngine
    from src.performance_analyzer import PerformanceAnalyzer

    try:
        # Create strategy class
        strategy_class = create_strategy_class(strategy_config)

        # Create backtest engine
        engine = BacktestEngine(config)

        # Run backtest across all stocks
        results = engine.run_backtest_multi_stock(strategy_class, symbols)

        if not results:
            return (strategy_config, None, None)

        # Calculate performance metrics
        analyzer = PerformanceAnalyzer(results)
        metrics = analyzer.calculate_metrics()

        return (strategy_config, metrics, results)

    except Exception as e:
        import traceback
        error_msg = f"Error running strategy {strategy_config['id']}: {e}\n"
        error_msg += traceback.format_exc()
        print(error_msg)
        return (strategy_config, None, None)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generic Strategy Generator')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--stocks', default='stocks.csv', help='Path to stocks CSV')
    parser.add_argument('--entry', default='entry_indicators_extended.csv', help='Path to entry indicators CSV')
    parser.add_argument('--exit', default='exit_indicators_extended.csv', help='Path to exit indicators CSV')
    parser.add_argument('--entry-filters', default=None, help='Path to entry filters CSV (default: <entry>_filters.csv)')
    parser.add_argument('--exit-filters', default=None, help='Path to exit filters CSV (default: <exit>_filters.csv)')
    parser.add_argument('--top-n', type=int, default=None, help='Number of top strategies to export')
    parser.add_argument('--parallel', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--clear-db', action='store_true', help='Clear database before running')
    parser.add_argument('--download-data', action='store_true', help='Download historical data first')
    args = parser.parse_args()

    print("=" * 80)
    print("GENERIC STRATEGY GENERATOR")
    print("=" * 80)
    print()

    # Load configuration
    config = load_config(args.config)
    print(f"✓ Loaded configuration from {args.config}")

    # Load stock list
    symbols = DataLoader.load_stock_list(args.stocks)
    print(f"✓ Loaded {len(symbols)} stocks from {args.stocks}")
    print(f"  Symbols: {', '.join(symbols)}")
    print()

    # Download data if requested
    if args.download_data:
        print("Downloading historical data...")
        start_date = config['data']['start_date']
        end_date = config['data']['end_date']
        DataLoader.download_historical_data(symbols, start_date, end_date)
        print("✓ Data download complete")
        print()

    # Load indicators (now handles both single and dual in same file)
    entry_indicators, entry_dual_from_main = parse_indicator_csv(args.entry)
    exit_indicators, exit_dual_from_main = parse_indicator_csv(args.exit)
    print(f"✓ Loaded {len(entry_indicators)} single entry indicators")
    print(f"✓ Loaded {len(exit_indicators)} single exit indicators")

    # Combine with explicit dual files if they exist
    entry_dual_indicators = entry_dual_from_main.copy()
    exit_dual_indicators = exit_dual_from_main.copy()

    dual_entry_path = args.entry.replace('.csv', '_dual.csv')
    if os.path.exists(dual_entry_path):
        from src.indicator_library import parse_dual_indicator_csv
        additional_dual = parse_dual_indicator_csv(dual_entry_path)
        entry_dual_indicators.extend(additional_dual)
        print(f"✓ Loaded {len(additional_dual)} additional dual entry indicators from {dual_entry_path}")

    dual_exit_path = args.exit.replace('.csv', '_dual.csv')
    if os.path.exists(dual_exit_path):
        from src.indicator_library import parse_dual_indicator_csv
        additional_dual = parse_dual_indicator_csv(dual_exit_path)
        exit_dual_indicators.extend(additional_dual)
        print(f"✓ Loaded {len(additional_dual)} additional dual exit indicators from {dual_exit_path}")

    # Load filter indicators
    from src.indicator_library import parse_filter_csv

    # Use explicit filter paths if provided, otherwise auto-detect
    entry_filter_path = args.entry_filters if args.entry_filters else args.entry.replace('.csv', '_filters.csv')
    exit_filter_path = args.exit_filters if args.exit_filters else args.exit.replace('.csv', '_filters.csv')

    entry_filters = parse_filter_csv(entry_filter_path)
    exit_filters = parse_filter_csv(exit_filter_path)

    if entry_dual_indicators:
        print(f"✓ Total dual entry indicators: {len(entry_dual_indicators)}")
    if exit_dual_indicators:
        print(f"✓ Total dual exit indicators: {len(exit_dual_indicators)}")
    if entry_filters:
        print(f"✓ Loaded {len(entry_filters)} entry filters")
    if exit_filters:
        print(f"✓ Loaded {len(exit_filters)} exit filters")

    print()

    # Generate strategies
    print("Generating strategy combinations...")
    generator = StrategyGenerator(
        entry_indicators,
        exit_indicators,
        config['strategy'],
        entry_dual_indicators,
        exit_dual_indicators,
        entry_filters,
        exit_filters
    )
    strategies = generator.generate_strategies()
    print(f"✓ Generated {len(strategies)} strategy combinations")
    print()

    # Initialize components
    optimizer = StrategyOptimizer(config)
    results_manager = ResultsManager(config)

    # Clear database if requested
    if args.clear_db:
        results_manager.clear_database()
        print("✓ Cleared database")
        print()

    # Determine number of workers
    n_workers = args.parallel or config['execution'].get('parallel_workers', 4)

    # Estimate completion time (rough estimate based on typical performance)
    avg_time_per_strategy = 2.0  # seconds (conservative estimate)
    total_estimated_time = (len(strategies) * avg_time_per_strategy) / n_workers
    est_minutes = int(total_estimated_time // 60)
    est_hours = est_minutes // 60
    est_minutes = est_minutes % 60

    print(f"{'='*80}")
    print(f"STARTING BACKTEST RUN")
    print(f"{'='*80}")
    print(f"Strategies to test: {len(strategies)}")
    print(f"Parallel workers: {n_workers}")
    print(f"Stocks per test: {len(symbols)}")
    if est_hours > 0:
        print(f"Estimated time: ~{est_hours}h {est_minutes}m (may vary)")
    else:
        print(f"Estimated time: ~{est_minutes}m (may vary)")
    print(f"{'='*80}")
    print()

    # Run backtests in parallel
    strategy_results = []
    save_interval = config['execution'].get('save_interval', 100)

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(run_single_strategy, strategy, symbols, config): strategy
            for strategy in strategies
        }

        # Process results as they complete
        with tqdm(total=len(strategies), desc="Testing strategies",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            for future in as_completed(futures):
                strategy_config, metrics, stock_results = future.result()

                if metrics is not None:
                    strategy_results.append((strategy_config, metrics))

                    # Save to database periodically
                    if len(strategy_results) % save_interval == 0:
                        results_manager.save_strategy(strategy_config, metrics)
                        if stock_results:
                            results_manager.save_stock_results(strategy_config['id'], stock_results)

                pbar.update(1)

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(int(elapsed_time), 60)
    hours, minutes = divmod(minutes, 60)

    print(f"\n{'='*80}")
    print(f"BACKTEST COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Completed {len(strategy_results)} backtests")
    print(f"  Total time: {hours}h {minutes}m {seconds}s" if hours > 0 else f"  Total time: {minutes}m {seconds}s")
    print(f"  Average: {elapsed_time/len(strategies):.2f} seconds per strategy")
    print(f"  Throughput: {len(strategies)/elapsed_time:.2f} strategies/second")
    print(f"{'='*80}")
    print()

    # Optimize and rank strategies
    print("Optimizing and ranking strategies...")
    top_n = args.top_n or config['optimization'].get('top_n', 50)
    top_strategies = optimizer.get_top_strategies(strategy_results, top_n)
    print(f"✓ Identified top {len(top_strategies)} strategies")
    print()

    # Save results
    print("Saving results...")
    for strategy_config, metrics, score in top_strategies:
        results_manager.save_strategy(strategy_config, metrics, score)

    results_manager.export_top_strategies(top_strategies)
    print(f"✓ Exported top strategies to {results_manager.csv_path}")
    print()

    # Generate summary report
    report = results_manager.generate_summary_report(top_strategies)
    print(report)

    # Save report to file
    report_path = f"{results_manager.reports_dir}/summary_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n✓ Saved summary report to {report_path}")

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()