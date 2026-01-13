"""
Quick Test Script - Verify System Functionality
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.indicator_library import parse_indicator_csv, IndicatorFactory, get_indicator_description
from src.strategy_generator import StrategyGenerator
import yaml


def test_indicator_parsing():
    """Test indicator CSV parsing"""
    print("Testing indicator parsing...")
    
    entry_indicators = parse_indicator_csv('entry_indicators.csv')
    exit_indicators = parse_indicator_csv('exit_indicators.csv')
    
    print(f"  ✓ Loaded {len(entry_indicators)} entry indicators")
    print(f"  ✓ Loaded {len(exit_indicators)} exit indicators")
    
    # Display indicators
    print("\n  Entry Indicators:")
    for name, params in entry_indicators:
        desc = get_indicator_description(name, params)
        print(f"    - {desc}")
    
    print("\n  Exit Indicators:")
    for name, params in exit_indicators:
        desc = get_indicator_description(name, params)
        print(f"    - {desc}")
    
    return entry_indicators, exit_indicators


def test_strategy_generation(entry_indicators, exit_indicators):
    """Test strategy generation"""
    print("\n\nTesting strategy generation...")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create generator
    generator = StrategyGenerator(
        entry_indicators,
        exit_indicators,
        config['strategy']
    )
    
    # Generate strategies
    strategies = generator.generate_strategies()
    
    print(f"  ✓ Generated {len(strategies)} strategy combinations")
    
    # Display first 5 strategies
    print("\n  Sample Strategies:")
    for i, strategy in enumerate(strategies[:5]):
        print(f"    {i+1}. {strategy['description']}")
    
    return strategies


def test_config_loading():
    """Test configuration loading"""
    print("\n\nTesting configuration loading...")
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("  ✓ Configuration loaded successfully")
    print(f"    - Initial capital: ${config['backtest']['initial_capital']:,}")
    print(f"    - Commission: {config['backtest']['commission']*100:.2f}%")
    print(f"    - Parallel workers: {config['execution']['parallel_workers']}")
    print(f"    - Top N strategies: {config['optimization']['top_n']}")
    
    return config


def test_stocks_loading():
    """Test stocks CSV loading"""
    print("\n\nTesting stocks loading...")
    
    from src.backtest_engine import DataLoader
    
    symbols = DataLoader.load_stock_list('stocks.csv')
    
    print(f"  ✓ Loaded {len(symbols)} stocks")
    print(f"    - Symbols: {', '.join(symbols)}")
    
    return symbols


def main():
    print("=" * 70)
    print("GENERIC STRATEGY GENERATOR - QUICK TEST")
    print("=" * 70)
    print()
    
    try:
        # Test 1: Indicator parsing
        entry_indicators, exit_indicators = test_indicator_parsing()
        
        # Test 2: Configuration loading
        config = test_config_loading()
        
        # Test 3: Stocks loading
        symbols = test_stocks_loading()
        
        # Test 4: Strategy generation
        strategies = test_strategy_generation(entry_indicators, exit_indicators)
        
        # Summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"✓ All tests passed!")
        print(f"✓ System is ready to run")
        print(f"✓ {len(strategies)} strategies will be tested across {len(symbols)} stocks")
        print(f"✓ Estimated total backtests: {len(strategies) * len(symbols)}")
        print("\nTo run the full optimization:")
        print("  python main.py")
        print("\nOr with data download:")
        print("  python main.py --download-data")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
