"""
Results Manager - Handle Strategy Results and Reporting
"""

import sqlite3
import pandas as pd
import json
import os
from datetime import datetime


class ResultsManager:
    """Manage strategy results and generate reports"""
    
    def __init__(self, config):
        """
        Initialize results manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_config = config.get('output', {})
        
        self.db_path = self.output_config.get('database', 'results/strategies.db')
        self.csv_path = self.output_config.get('top_strategies_csv', 'results/top_strategies.csv')
        self.reports_dir = self.output_config.get('reports_dir', 'results/reports')
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create strategies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY,
                strategy_id INTEGER,
                description TEXT,
                entry_indicator TEXT,
                entry_params TEXT,
                entry_type TEXT,
                exit_indicator TEXT,
                exit_params TEXT,
                exit_type TEXT,
                score REAL,
                total_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                profit_factor REAL,
                win_rate REAL,
                total_trades INTEGER,
                recovery_factor REAL,
                expectancy REAL,
                avg_win REAL,
                avg_loss REAL,
                win_loss_ratio REAL,
                timestamp TEXT
            )
        ''')
        
        # Create stock results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id INTEGER,
                symbol TEXT,
                return_pct REAL,
                trades INTEGER,
                sharpe REAL,
                max_drawdown REAL,
                timestamp TEXT,
                FOREIGN KEY (strategy_id) REFERENCES strategies (strategy_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_strategy(self, strategy_config, metrics, score=None):
        """
        Save a strategy to the database
        
        Args:
            strategy_config: Strategy configuration dictionary
            metrics: Performance metrics dictionary
            score: Composite score (optional)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO strategies (
                strategy_id, description,
                entry_indicator, entry_params, entry_type,
                exit_indicator, exit_params, exit_type,
                score, total_return, sharpe_ratio, max_drawdown,
                profit_factor, win_rate, total_trades,
                recovery_factor, expectancy,
                avg_win, avg_loss, win_loss_ratio,
                timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            strategy_config['id'],
            strategy_config['description'],
            strategy_config['entry_indicator'],
            json.dumps(strategy_config['entry_params']),
            strategy_config['entry_type'],
            strategy_config['exit_indicator'],
            json.dumps(strategy_config['exit_params']),
            strategy_config['exit_type'],
            score,
            metrics['total_return'],
            metrics['sharpe_ratio'],
            metrics['max_drawdown'],
            metrics['profit_factor'],
            metrics['win_rate'],
            metrics['total_trades'],
            metrics['recovery_factor'],
            metrics['expectancy'],
            metrics['avg_win'],
            metrics['avg_loss'],
            metrics['avg_win_loss_ratio'],
            timestamp
        ))
        
        conn.commit()
        conn.close()
    
    def save_stock_results(self, strategy_id, stock_results):
        """
        Save individual stock results
        
        Args:
            strategy_id: Strategy ID
            stock_results: List of stock result dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        for result in stock_results:
            cursor.execute('''
                INSERT INTO stock_results (
                    strategy_id, symbol, return_pct, trades,
                    sharpe, max_drawdown, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                strategy_id,
                result['symbol'],
                result['return_pct'],
                result['trades'].get('total', {}).get('total', 0),
                result['sharpe'].get('sharperatio', 0),
                result['drawdown'].get('max', {}).get('drawdown', 0),
                timestamp
            ))
        
        conn.commit()
        conn.close()
    
    def export_top_strategies(self, ranked_strategies):
        """
        Export top strategies to CSV
        
        Args:
            ranked_strategies: List of (strategy_config, metrics, score) tuples
        """
        data = []
        
        for strategy_config, metrics, score in ranked_strategies:
            row = {
                'rank': len(data) + 1,
                'strategy_id': strategy_config['id'],
                'description': strategy_config['description'],
                'score': round(score, 2),
                'total_return': round(metrics['total_return'], 2),
                'sharpe_ratio': round(metrics['sharpe_ratio'], 3),
                'max_drawdown': round(metrics['max_drawdown'], 2),
                'profit_factor': round(metrics['profit_factor'], 2),
                'win_rate': round(metrics['win_rate'] * 100, 2),
                'total_trades': metrics['total_trades'],
                'recovery_factor': round(metrics['recovery_factor'], 2),
                'expectancy': round(metrics['expectancy'], 2),
                'avg_win': round(metrics['avg_win'], 2),
                'avg_loss': round(metrics['avg_loss'], 2),
                'win_loss_ratio': round(metrics['avg_win_loss_ratio'], 2),
                'entry_indicator': strategy_config['entry_indicator'],
                'entry_params': ','.join(map(str, strategy_config['entry_params'])),
                'entry_type': strategy_config['entry_type'],
                'exit_indicator': strategy_config['exit_indicator'],
                'exit_params': ','.join(map(str, strategy_config['exit_params'])),
                'exit_type': strategy_config['exit_type'],
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(self.csv_path, index=False)
        print(f"Exported top strategies to {self.csv_path}")
    
    def generate_summary_report(self, ranked_strategies):
        """
        Generate a summary report
        
        Args:
            ranked_strategies: List of (strategy_config, metrics, score) tuples
            
        Returns:
            Report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("STRATEGY OPTIMIZATION SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Strategies Analyzed: {len(ranked_strategies)}")
        lines.append("")
        
        if ranked_strategies:
            lines.append("TOP 10 STRATEGIES:")
            lines.append("-" * 80)
            
            for i, (strategy_config, metrics, score) in enumerate(ranked_strategies[:10]):
                lines.append(f"\nRank #{i+1} (Score: {score:.2f})")
                lines.append(f"  Description: {strategy_config['description']}")
                lines.append(f"  Total Return: {metrics['total_return']:.2f}%")
                lines.append(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                lines.append(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
                lines.append(f"  Profit Factor: {metrics['profit_factor']:.2f}")
                lines.append(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
                lines.append(f"  Total Trades: {metrics['total_trades']}")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
    
    def load_all_strategies(self):
        """
        Load all strategies from database
        
        Returns:
            Pandas DataFrame
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM strategies ORDER BY score DESC", conn)
        conn.close()
        return df
    
    def load_strategy_by_id(self, strategy_id):
        """
        Load a specific strategy by ID
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Dictionary with strategy data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM strategies WHERE strategy_id = ?", (strategy_id,))
        row = cursor.fetchone()
        
        if row:
            columns = [description[0] for description in cursor.description]
            strategy = dict(zip(columns, row))
            
            # Load stock results
            cursor.execute("SELECT * FROM stock_results WHERE strategy_id = ?", (strategy_id,))
            stock_rows = cursor.fetchall()
            stock_columns = [description[0] for description in cursor.description]
            strategy['stock_results'] = [dict(zip(stock_columns, r)) for r in stock_rows]
        else:
            strategy = None
        
        conn.close()
        return strategy
    
    def clear_database(self):
        """Clear all data from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM strategies")
        cursor.execute("DELETE FROM stock_results")
        conn.commit()
        conn.close()
        print("Database cleared")
