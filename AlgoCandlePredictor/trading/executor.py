from typing import Dict, Optional
from datetime import datetime
from core.logger import logger
from trading.risk_manager import risk_manager


class TradeExecutor:
    """Trade execution engine - interfaces with broker APIs"""
    
    def __init__(self):
        self.is_connected = False
        self.broker_name = "MOCK_BROKER"
        self.execution_history = []
        logger.info("TradeExecutor initialized")
    
    def connect_broker(self) -> bool:
        """Connect to broker API"""
        try:
            logger.info(f"Connecting to {self.broker_name}...")
            self.is_connected = True
            logger.info(f"Successfully connected to {self.broker_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to broker: {e}", exc_info=True)
            self.is_connected = False
            return False
    
    def disconnect_broker(self):
        """Disconnect from broker API"""
        self.is_connected = False
        logger.info(f"Disconnected from {self.broker_name}")
    
    def execute_buy(self, symbol: str, quantity: float, price: float) -> Optional[Dict]:
        """Execute a buy order"""
        if not self.is_connected:
            logger.error("Cannot execute buy - not connected to broker")
            return None
        
        try:
            order = {
                "order_id": f"BUY_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "symbol": symbol,
                "side": "BUY",
                "quantity": quantity,
                "price": price,
                "status": "FILLED",
                "timestamp": datetime.now().isoformat(),
                "broker": self.broker_name
            }
            
            self.execution_history.append(order)
            logger.info(f"BUY order executed: {symbol} @ {price}, qty: {quantity}")
            
            risk_manager.open_position(symbol, "BUY", price, quantity)
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to execute buy order: {e}", exc_info=True)
            return None
    
    def execute_sell(self, symbol: str, quantity: float, price: float) -> Optional[Dict]:
        """Execute a sell order"""
        if not self.is_connected:
            logger.error("Cannot execute sell - not connected to broker")
            return None
        
        try:
            order = {
                "order_id": f"SELL_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "symbol": symbol,
                "side": "SELL",
                "quantity": quantity,
                "price": price,
                "status": "FILLED",
                "timestamp": datetime.now().isoformat(),
                "broker": self.broker_name
            }
            
            self.execution_history.append(order)
            logger.info(f"SELL order executed: {symbol} @ {price}, qty: {quantity}")
            
            risk_manager.close_position(symbol, price)
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to execute sell order: {e}", exc_info=True)
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol"""
        try:
            logger.debug(f"Fetching current price for {symbol}")
            mock_price = 2500.0
            return mock_price
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}", exc_info=True)
            return None
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get status of an order"""
        for order in self.execution_history:
            if order["order_id"] == order_id:
                return order
        return None
    
    def get_execution_history(self) -> list:
        """Get all execution history"""
        return self.execution_history
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        logger.info(f"Attempting to cancel order: {order_id}")
        order = self.get_order_status(order_id)
        if order and order["status"] != "FILLED":
            order["status"] = "CANCELLED"
            logger.info(f"Order cancelled: {order_id}")
            return True
        return False


trade_executor = TradeExecutor()
