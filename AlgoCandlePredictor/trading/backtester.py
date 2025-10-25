import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
from core.logger import logger


class Backtester:
    """Backtest trading strategies with performance metrics"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades = []
        self.equity_curve = []
        logger.info(f"Backtester initialized with capital: {initial_capital}")
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0.0
        excess_returns = returns - (risk_free_rate / 252)
        if np.std(returns) == 0:
            return 0.0
        return np.sqrt(252) * np.mean(excess_returns) / np.std(returns)
    
    def calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(equity_curve) == 0:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd
    
    def calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate percentage"""
        if len(trades) == 0:
            return 0.0
        winning_trades = [t for t in trades if t['pnl'] > 0]
        return (len(winning_trades) / len(trades)) * 100
    
    def calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if len(trades) == 0:
            return 0.0
        gross_profit = sum([t['pnl'] for t in trades if t['pnl'] > 0])
        gross_loss = abs(sum([t['pnl'] for t in trades if t['pnl'] < 0]))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss
    
    def run_backtest(
        self,
        predictions: pd.DataFrame,
        prices: pd.DataFrame,
        position_size_func=None
    ) -> Dict:
        """
        Run backtest simulation
        
        Args:
            predictions: DataFrame with predictions and confidence
            prices: DataFrame with OHLCV data
            position_size_func: Function to calculate position size
            
        Returns:
            Dictionary with backtest results
        """
        try:
            logger.info("Starting backtest...")
            
            self.capital = self.initial_capital
            self.trades = []
            self.equity_curve = [self.capital]
            
            position = None
            
            for i in range(len(predictions)):
                current_price = prices.iloc[i]['close']
                pred = predictions.iloc[i]['prediction']
                confidence = predictions.iloc[i]['confidence']
                
                # Convert numeric predictions to strings if needed
                if isinstance(pred, (int, float, np.integer, np.floating)):
                    pred = 'UP' if pred == 1 else 'DOWN'
                
                # Entry logic
                if position is None and confidence >= 0.70:
                    # Calculate position size
                    if position_size_func:
                        pos_size = position_size_func(self.capital, confidence)
                    else:
                        pos_size = self.capital * 0.02  # 2% risk
                    
                    quantity = pos_size / current_price
                    
                    position = {
                        'entry_price': current_price,
                        'quantity': quantity,
                        'entry_time': i,
                        'direction': pred
                    }
                
                # Exit logic (simple: exit after 5 candles or stop loss)
                elif position is not None:
                    exit_condition = (i - position['entry_time'] >= 5)
                    
                    if exit_condition:
                        exit_price = current_price
                        
                        if position['direction'] == 'UP':
                            pnl = (exit_price - position['entry_price']) * position['quantity']
                        else:
                            pnl = (position['entry_price'] - exit_price) * position['quantity']
                        
                        self.capital += pnl
                        
                        self.trades.append({
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'quantity': position['quantity'],
                            'pnl': pnl,
                            'direction': position['direction']
                        })
                        
                        position = None
                
                self.equity_curve.append(self.capital)
            
            # Calculate metrics
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            
            metrics = {
                'initial_capital': self.initial_capital,
                'final_capital': self.capital,
                'total_return': ((self.capital - self.initial_capital) / self.initial_capital) * 100,
                'total_trades': len(self.trades),
                'win_rate': self.calculate_win_rate(self.trades),
                'profit_factor': self.calculate_profit_factor(self.trades),
                'sharpe_ratio': self.calculate_sharpe_ratio(returns),
                'max_drawdown': self.calculate_max_drawdown(self.equity_curve) * 100,
                'avg_trade_pnl': np.mean([t['pnl'] for t in self.trades]) if self.trades else 0
            }
            
            logger.info(f"\nBacktest Results:")
            logger.info(f"Total Return: {metrics['total_return']:.2f}%")
            logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
            logger.info(f"Win Rate: {metrics['win_rate']:.2f}%")
            logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}", exc_info=True)
            raise


backtester = Backtester()
