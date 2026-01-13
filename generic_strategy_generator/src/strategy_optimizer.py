"""
Strategy Optimizer - Rank and Filter Strategies
"""

import numpy as np
import pandas as pd
from typing import List, Dict


class StrategyOptimizer:
    """Optimize and rank trading strategies"""
    
    def __init__(self, config):
        """
        Initialize strategy optimizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.optimization_config = config.get('optimization', {})
        self.filters = config.get('filters', {})
        
        # Scoring weights
        self.weights = self.optimization_config.get('weights', {
            'total_return': 0.25,
            'sharpe_ratio': 0.20,
            'max_drawdown': 0.20,
            'profit_factor': 0.15,
            'win_rate': 0.10,
            'recovery_factor': 0.10,
        })
    
    def filter_strategies(self, strategy_results):
        """
        Filter strategies based on minimum requirements
        
        Args:
            strategy_results: List of (strategy_config, metrics) tuples
            
        Returns:
            Filtered list of strategy results
        """
        filtered = []
        
        for strategy_config, metrics in strategy_results:
            if metrics is None:
                continue
            
            # Apply filters
            if not self._meets_requirements(metrics):
                continue
            
            filtered.append((strategy_config, metrics))
        
        return filtered
    
    def _meets_requirements(self, metrics):
        """Check if metrics meet minimum requirements"""
        # Minimum trades
        min_trades = self.filters.get('min_trades', 10)
        if metrics['total_trades'] < min_trades:
            return False
        
        # Maximum drawdown
        max_dd = self.filters.get('max_drawdown', 0.30)
        if abs(metrics['max_drawdown']) > max_dd * 100:
            return False
        
        # Minimum Sharpe ratio
        min_sharpe = self.filters.get('min_sharpe', 0.5)
        if metrics['sharpe_ratio'] < min_sharpe:
            return False
        
        # Minimum profit factor
        min_pf = self.filters.get('min_profit_factor', 1.2)
        if metrics['profit_factor'] < min_pf:
            return False
        
        # Minimum win rate
        min_wr = self.filters.get('min_win_rate', 0.35)
        if metrics['win_rate'] < min_wr:
            return False
        
        return True
    
    def calculate_composite_score(self, metrics):
        """
        Calculate composite score for a strategy
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Composite score (0-100)
        """
        # Normalize metrics to 0-1 scale
        normalized = self._normalize_metrics(metrics)
        
        # Calculate weighted score
        score = 0.0
        for metric, weight in self.weights.items():
            if metric in normalized:
                score += normalized[metric] * weight
        
        return score * 100  # Scale to 0-100
    
    def _normalize_metrics(self, metrics):
        """
        Normalize metrics to 0-1 scale
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Dictionary of normalized metrics
        """
        normalized = {}
        
        # Total return: normalize to 0-1 (assuming -50% to +100% range)
        total_return = metrics['total_return']
        normalized['total_return'] = np.clip((total_return + 50) / 150, 0, 1)
        
        # Sharpe ratio: normalize to 0-1 (assuming -1 to 3 range)
        sharpe = metrics['sharpe_ratio']
        normalized['sharpe_ratio'] = np.clip((sharpe + 1) / 4, 0, 1)
        
        # Max drawdown: invert and normalize (assuming 0% to 50% range)
        max_dd = abs(metrics['max_drawdown'])
        normalized['max_drawdown'] = np.clip(1 - (max_dd / 50), 0, 1)
        
        # Profit factor: normalize to 0-1 (assuming 0 to 3 range)
        pf = metrics['profit_factor']
        normalized['profit_factor'] = np.clip(pf / 3, 0, 1)
        
        # Win rate: already 0-1
        normalized['win_rate'] = metrics['win_rate']
        
        # Recovery factor: normalize to 0-1 (assuming 0 to 10 range)
        rf = metrics['recovery_factor']
        normalized['recovery_factor'] = np.clip(rf / 10, 0, 1)
        
        return normalized
    
    def rank_strategies(self, strategy_results):
        """
        Rank strategies by composite score
        
        Args:
            strategy_results: List of (strategy_config, metrics) tuples
            
        Returns:
            List of (strategy_config, metrics, score) tuples sorted by score
        """
        # Calculate scores
        ranked = []
        for strategy_config, metrics in strategy_results:
            score = self.calculate_composite_score(metrics)
            ranked.append((strategy_config, metrics, score))
        
        # Sort by score (descending)
        ranked.sort(key=lambda x: x[2], reverse=True)
        
        return ranked
    
    def get_top_strategies(self, strategy_results, top_n=None):
        """
        Get top N strategies
        
        Args:
            strategy_results: List of (strategy_config, metrics) tuples
            top_n: Number of top strategies to return (default from config)
            
        Returns:
            List of top strategy results
        """
        if top_n is None:
            top_n = self.optimization_config.get('top_n', 50)
        
        # Filter strategies
        filtered = self.filter_strategies(strategy_results)
        
        # Rank strategies
        ranked = self.rank_strategies(filtered)
        
        # Return top N
        return ranked[:top_n]
    
    def generate_pareto_frontier(self, strategy_results):
        """
        Generate Pareto frontier for multi-objective optimization
        
        Args:
            strategy_results: List of (strategy_config, metrics) tuples
            
        Returns:
            List of Pareto-optimal strategies
        """
        if not strategy_results:
            return []
        
        # Extract objectives (return and risk)
        objectives = []
        for strategy_config, metrics in strategy_results:
            # Maximize return, minimize drawdown
            return_obj = metrics['total_return']
            risk_obj = -abs(metrics['max_drawdown'])  # Negative for minimization
            objectives.append((return_obj, risk_obj))
        
        # Find Pareto optimal points
        pareto_indices = self._find_pareto_optimal(objectives)
        
        # Return Pareto optimal strategies
        pareto_strategies = [strategy_results[i] for i in pareto_indices]
        
        return pareto_strategies
    
    def _find_pareto_optimal(self, objectives):
        """
        Find Pareto optimal points
        
        Args:
            objectives: List of (obj1, obj2) tuples
            
        Returns:
            List of indices of Pareto optimal points
        """
        pareto_indices = []
        
        for i, obj_i in enumerate(objectives):
            is_pareto = True
            
            for j, obj_j in enumerate(objectives):
                if i == j:
                    continue
                
                # Check if j dominates i
                if (obj_j[0] >= obj_i[0] and obj_j[1] >= obj_i[1] and
                    (obj_j[0] > obj_i[0] or obj_j[1] > obj_i[1])):
                    is_pareto = False
                    break
            
            if is_pareto:
                pareto_indices.append(i)
        
        return pareto_indices
    
    def create_summary_dataframe(self, ranked_strategies):
        """
        Create a summary dataframe of ranked strategies
        
        Args:
            ranked_strategies: List of (strategy_config, metrics, score) tuples
            
        Returns:
            Pandas DataFrame
        """
        data = []
        
        for strategy_config, metrics, score in ranked_strategies:
            row = {
                'strategy_id': strategy_config['id'],
                'description': strategy_config['description'],
                'score': score,
                'total_return': metrics['total_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'profit_factor': metrics['profit_factor'],
                'win_rate': metrics['win_rate'],
                'total_trades': metrics['total_trades'],
                'recovery_factor': metrics['recovery_factor'],
                'expectancy': metrics['expectancy'],
                'avg_win': metrics['avg_win'],
                'avg_loss': metrics['avg_loss'],
                'win_loss_ratio': metrics['avg_win_loss_ratio'],
            }
            data.append(row)
        
        return pd.DataFrame(data)
