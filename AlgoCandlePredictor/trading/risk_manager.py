from typing import Dict, Optional
from datetime import datetime, date
from core.logger import logger
from core.config import config_manager


class RiskManager:
    """Risk management system for controlling position sizes and losses"""
    
    def __init__(self):
        self.daily_pnl: Dict[date, float] = {}
        self.active_positions: Dict[str, Dict] = {}
        self.trade_history: list = []
        logger.info("RiskManager initialized")
    
    @property
    def config(self):
        """Dynamically fetch current trading config to reflect real-time updates"""
        return config_manager.get_trading_config()
    
    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been reached"""
        today = date.today()
        daily_loss = abs(min(self.daily_pnl.get(today, 0.0), 0))
        
        if daily_loss >= self.config.max_daily_loss:
            logger.warning(f"Daily loss limit reached: {daily_loss} >= {self.config.max_daily_loss}")
            return False
        return True
    
    def calculate_position_size(self, symbol: str, entry_price: float, confidence: float) -> float:
        """Calculate appropriate position size based on risk parameters"""
        base_position_size = self.config.max_position_size
        
        risk_adjusted_size = base_position_size * confidence
        
        max_allowed = self.config.max_position_size
        position_size = min(risk_adjusted_size, max_allowed)
        
        logger.info(f"Calculated position size for {symbol}: {position_size} (confidence: {confidence})")
        return position_size
    
    def can_open_position(self, symbol: str, entry_price: float, confidence: float) -> tuple[bool, str]:
        """Check if a new position can be opened"""
        if not self.check_daily_loss_limit():
            return False, "Daily loss limit reached"
        
        if symbol in self.active_positions:
            return False, f"Position already exists for {symbol}"
        
        if confidence < self.config.confidence_threshold:
            return False, f"Confidence {confidence} below threshold {self.config.confidence_threshold}"
        
        position_size = self.calculate_position_size(symbol, entry_price, confidence)
        if position_size <= 0:
            return False, "Invalid position size"
        
        logger.info(f"Position check passed for {symbol}")
        return True, "OK"
    
    def open_position(self, symbol: str, side: str, entry_price: float, quantity: float) -> Dict:
        """Record a new position opening"""
        position = {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "quantity": quantity,
            "entry_time": datetime.now(),
            "pnl": 0.0
        }
        self.active_positions[symbol] = position
        logger.info(f"Position opened: {symbol} {side} @ {entry_price}, qty: {quantity}")
        return position
    
    def close_position(self, symbol: str, exit_price: float) -> Optional[Dict]:
        """Close an existing position and calculate PnL"""
        if symbol not in self.active_positions:
            logger.warning(f"No active position found for {symbol}")
            return None
        
        position = self.active_positions.pop(symbol)
        
        if position["side"].lower() == "buy":
            pnl = (exit_price - position["entry_price"]) * position["quantity"]
        else:
            pnl = (position["entry_price"] - exit_price) * position["quantity"]
        
        position["exit_price"] = exit_price
        position["exit_time"] = datetime.now()
        position["pnl"] = pnl
        
        today = date.today()
        self.daily_pnl[today] = self.daily_pnl.get(today, 0.0) + pnl
        self.trade_history.append(position)
        
        logger.info(f"Position closed: {symbol} @ {exit_price}, PnL: {pnl}")
        return position
    
    def update_position_pnl(self, symbol: str, current_price: float):
        """Update unrealized PnL for an active position"""
        if symbol not in self.active_positions:
            return
        
        position = self.active_positions[symbol]
        if position["side"].lower() == "buy":
            position["pnl"] = (current_price - position["entry_price"]) * position["quantity"]
        else:
            position["pnl"] = (position["entry_price"] - current_price) * position["quantity"]
    
    def get_active_positions(self) -> Dict[str, Dict]:
        """Get all active positions"""
        return self.active_positions
    
    def get_daily_pnl(self, day: Optional[date] = None) -> float:
        """Get PnL for a specific day"""
        day = day or date.today()
        return self.daily_pnl.get(day, 0.0)
    
    def get_statistics(self) -> Dict:
        """Get trading statistics"""
        total_trades = len(self.trade_history)
        winning_trades = [t for t in self.trade_history if t.get("pnl", 0) > 0]
        losing_trades = [t for t in self.trade_history if t.get("pnl", 0) < 0]
        
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        total_pnl = sum(t.get("pnl", 0) for t in self.trade_history)
        
        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "daily_pnl": round(self.get_daily_pnl(), 2),
            "active_positions_count": len(self.active_positions)
        }
    
    def reset(self):
        """Reset risk manager state"""
        self.active_positions.clear()
        self.daily_pnl.clear()
        self.trade_history.clear()
        logger.info("RiskManager reset completed")


risk_manager = RiskManager()
