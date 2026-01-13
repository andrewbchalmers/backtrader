"""
Performance Analyzer - Calculate Comprehensive Trading Metrics
"""

import numpy as np
from datetime import datetime


class PerformanceAnalyzer:
    """Calculate comprehensive performance metrics from backtest results"""
    
    def __init__(self, results, risk_free_rate=0.02):
        """
        Initialize performance analyzer
        
        Args:
            results: List of backtest results (one per stock)
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.results = results
        self.risk_free_rate = risk_free_rate
    
    def calculate_metrics(self):
        """
        Calculate comprehensive performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.results:
            return None
        
        # Aggregate metrics
        metrics = {
            # Return metrics
            'total_return': self._calculate_total_return(),
            'avg_return_per_stock': self._calculate_avg_return(),
            'median_return': self._calculate_median_return(),
            'return_std': self._calculate_return_std(),
            
            # Risk metrics
            'max_drawdown': self._calculate_max_drawdown(),
            'avg_drawdown': self._calculate_avg_drawdown(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'sortino_ratio': self._calculate_sortino_ratio(),
            'calmar_ratio': self._calculate_calmar_ratio(),
            
            # Trade metrics
            'total_trades': self._calculate_total_trades(),
            'avg_trades_per_stock': self._calculate_avg_trades(),
            'win_rate': self._calculate_win_rate(),
            'profit_factor': self._calculate_profit_factor(),
            'expectancy': self._calculate_expectancy(),
            
            # Win/Loss metrics
            'avg_win': self._calculate_avg_win(),
            'avg_loss': self._calculate_avg_loss(),
            'avg_win_loss_ratio': self._calculate_avg_win_loss_ratio(),
            'largest_win': self._calculate_largest_win(),
            'largest_loss': self._calculate_largest_loss(),
            
            # Consecutive metrics
            'max_consecutive_wins': self._calculate_max_consecutive_wins(),
            'max_consecutive_losses': self._calculate_max_consecutive_losses(),
            
            # Recovery metrics
            'recovery_factor': self._calculate_recovery_factor(),
            
            # Stock count
            'stocks_tested': len(self.results),
            'profitable_stocks': self._count_profitable_stocks(),
        }
        
        return metrics
    
    def _calculate_total_return(self):
        """Calculate total return across all stocks"""
        returns = [r['return_pct'] for r in self.results]
        return np.mean(returns) if returns else 0.0
    
    def _calculate_avg_return(self):
        """Calculate average return per stock"""
        returns = [r['return_pct'] for r in self.results]
        return np.mean(returns) if returns else 0.0
    
    def _calculate_median_return(self):
        """Calculate median return"""
        returns = [r['return_pct'] for r in self.results]
        return np.median(returns) if returns else 0.0
    
    def _calculate_return_std(self):
        """Calculate standard deviation of returns"""
        returns = [r['return_pct'] for r in self.results]
        return np.std(returns) if len(returns) > 1 else 0.0
    
    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown across all stocks"""
        drawdowns = []
        for r in self.results:
            dd = r.get('drawdown', {})
            max_dd = dd.get('max', {}).get('drawdown', 0)
            drawdowns.append(max_dd)
        return max(drawdowns) if drawdowns else 0.0
    
    def _calculate_avg_drawdown(self):
        """Calculate average drawdown"""
        drawdowns = []
        for r in self.results:
            dd = r.get('drawdown', {})
            max_dd = dd.get('max', {}).get('drawdown', 0)
            drawdowns.append(max_dd)
        return np.mean(drawdowns) if drawdowns else 0.0
    
    def _calculate_sharpe_ratio(self):
        """Calculate average Sharpe ratio"""
        sharpes = []
        for r in self.results:
            sharpe = r.get('sharpe', {}).get('sharperatio', None)
            if sharpe is not None and not np.isnan(sharpe):
                sharpes.append(sharpe)
        return np.mean(sharpes) if sharpes else 0.0
    
    def _calculate_sortino_ratio(self):
        """Calculate Sortino ratio (return / downside deviation)"""
        returns = [r['return_pct'] for r in self.results]
        if not returns:
            return 0.0
        
        avg_return = np.mean(returns)
        downside_returns = [r for r in returns if r < 0]
        
        if not downside_returns:
            return float('inf') if avg_return > 0 else 0.0
        
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0
        
        return avg_return / downside_std
    
    def _calculate_calmar_ratio(self):
        """Calculate Calmar ratio (return / max drawdown)"""
        avg_return = self._calculate_avg_return()
        max_dd = self._calculate_max_drawdown()
        
        if max_dd == 0:
            return float('inf') if avg_return > 0 else 0.0
        
        return avg_return / abs(max_dd)
    
    def _calculate_total_trades(self):
        """Calculate total number of trades"""
        total = 0
        for r in self.results:
            trades = r.get('trades', {})
            total += trades.get('total', {}).get('total', 0)
        return total
    
    def _calculate_avg_trades(self):
        """Calculate average trades per stock"""
        trades = []
        for r in self.results:
            t = r.get('trades', {})
            trades.append(t.get('total', {}).get('total', 0))
        return np.mean(trades) if trades else 0.0
    
    def _calculate_win_rate(self):
        """Calculate overall win rate"""
        total_won = 0
        total_trades = 0
        
        for r in self.results:
            trades = r.get('trades', {})
            won = trades.get('won', {}).get('total', 0)
            total = trades.get('total', {}).get('total', 0)
            total_won += won
            total_trades += total
        
        return (total_won / total_trades) if total_trades > 0 else 0.0
    
    def _calculate_profit_factor(self):
        """Calculate profit factor (gross profit / gross loss)"""
        total_won_pnl = 0.0
        total_lost_pnl = 0.0
        
        for r in self.results:
            trades = r.get('trades', {})
            won_pnl = trades.get('won', {}).get('pnl', {}).get('total', 0.0)
            lost_pnl = abs(trades.get('lost', {}).get('pnl', {}).get('total', 0.0))
            total_won_pnl += won_pnl
            total_lost_pnl += lost_pnl
        
        return (total_won_pnl / total_lost_pnl) if total_lost_pnl > 0 else float('inf')
    
    def _calculate_expectancy(self):
        """Calculate expectancy per trade"""
        win_rate = self._calculate_win_rate()
        avg_win = self._calculate_avg_win()
        avg_loss = self._calculate_avg_loss()
        
        return (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
    
    def _calculate_avg_win(self):
        """Calculate average winning trade"""
        wins = []
        for r in self.results:
            trades = r.get('trades', {})
            avg_win = trades.get('won', {}).get('pnl', {}).get('average', 0.0)
            if avg_win > 0:
                wins.append(avg_win)
        return np.mean(wins) if wins else 0.0
    
    def _calculate_avg_loss(self):
        """Calculate average losing trade"""
        losses = []
        for r in self.results:
            trades = r.get('trades', {})
            avg_loss = trades.get('lost', {}).get('pnl', {}).get('average', 0.0)
            if avg_loss < 0:
                losses.append(avg_loss)
        return np.mean(losses) if losses else 0.0
    
    def _calculate_avg_win_loss_ratio(self):
        """Calculate average win/loss ratio"""
        avg_win = self._calculate_avg_win()
        avg_loss = abs(self._calculate_avg_loss())
        
        return (avg_win / avg_loss) if avg_loss > 0 else float('inf')
    
    def _calculate_largest_win(self):
        """Calculate largest winning trade"""
        wins = []
        for r in self.results:
            trades = r.get('trades', {})
            max_win = trades.get('won', {}).get('pnl', {}).get('max', 0.0)
            wins.append(max_win)
        return max(wins) if wins else 0.0
    
    def _calculate_largest_loss(self):
        """Calculate largest losing trade"""
        losses = []
        for r in self.results:
            trades = r.get('trades', {})
            max_loss = trades.get('lost', {}).get('pnl', {}).get('max', 0.0)
            losses.append(max_loss)
        return min(losses) if losses else 0.0
    
    def _calculate_max_consecutive_wins(self):
        """Calculate maximum consecutive wins"""
        max_consec = 0
        for r in self.results:
            trades = r.get('trades', {})
            streak = trades.get('streak', {}).get('won', {}).get('longest', 0)
            max_consec = max(max_consec, streak)
        return max_consec
    
    def _calculate_max_consecutive_losses(self):
        """Calculate maximum consecutive losses"""
        max_consec = 0
        for r in self.results:
            trades = r.get('trades', {})
            streak = trades.get('streak', {}).get('lost', {}).get('longest', 0)
            max_consec = max(max_consec, streak)
        return max_consec
    
    def _calculate_recovery_factor(self):
        """Calculate recovery factor (net profit / max drawdown)"""
        total_return = self._calculate_total_return()
        max_dd = abs(self._calculate_max_drawdown())
        
        if max_dd == 0:
            return float('inf') if total_return > 0 else 0.0
        
        return total_return / max_dd
    
    def _count_profitable_stocks(self):
        """Count number of profitable stocks"""
        return sum(1 for r in self.results if r['return_pct'] > 0)
    
    def format_metrics(self, metrics):
        """
        Format metrics for display
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Formatted string
        """
        if not metrics:
            return "No metrics available"
        
        formatted = []
        formatted.append("=" * 60)
        formatted.append("PERFORMANCE METRICS")
        formatted.append("=" * 60)
        
        formatted.append("\n--- RETURN METRICS ---")
        formatted.append(f"Total Return: {metrics['total_return']:.2f}%")
        formatted.append(f"Average Return per Stock: {metrics['avg_return_per_stock']:.2f}%")
        formatted.append(f"Median Return: {metrics['median_return']:.2f}%")
        formatted.append(f"Return Std Dev: {metrics['return_std']:.2f}%")
        
        formatted.append("\n--- RISK METRICS ---")
        formatted.append(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        formatted.append(f"Avg Drawdown: {metrics['avg_drawdown']:.2f}%")
        formatted.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        formatted.append(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
        formatted.append(f"Calmar Ratio: {metrics['calmar_ratio']:.3f}")
        
        formatted.append("\n--- TRADE METRICS ---")
        formatted.append(f"Total Trades: {metrics['total_trades']}")
        formatted.append(f"Avg Trades per Stock: {metrics['avg_trades_per_stock']:.1f}")
        formatted.append(f"Win Rate: {metrics['win_rate']*100:.2f}%")
        formatted.append(f"Profit Factor: {metrics['profit_factor']:.2f}")
        formatted.append(f"Expectancy: {metrics['expectancy']:.2f}")
        
        formatted.append("\n--- WIN/LOSS METRICS ---")
        formatted.append(f"Avg Win: ${metrics['avg_win']:.2f}")
        formatted.append(f"Avg Loss: ${metrics['avg_loss']:.2f}")
        formatted.append(f"Avg Win/Loss Ratio: {metrics['avg_win_loss_ratio']:.2f}")
        formatted.append(f"Largest Win: ${metrics['largest_win']:.2f}")
        formatted.append(f"Largest Loss: ${metrics['largest_loss']:.2f}")
        
        formatted.append("\n--- OTHER METRICS ---")
        formatted.append(f"Max Consecutive Wins: {metrics['max_consecutive_wins']}")
        formatted.append(f"Max Consecutive Losses: {metrics['max_consecutive_losses']}")
        formatted.append(f"Recovery Factor: {metrics['recovery_factor']:.2f}")
        formatted.append(f"Stocks Tested: {metrics['stocks_tested']}")
        formatted.append(f"Profitable Stocks: {metrics['profitable_stocks']}")
        
        formatted.append("=" * 60)
        
        return "\n".join(formatted)
