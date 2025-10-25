from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from core.logger import logger
from trading.executor import trade_executor
from trading.risk_manager import risk_manager
from models.predict import prediction_service

router = APIRouter(prefix="/trading", tags=["Trading"])


class TradingSession:
    """Trading session state manager"""
    def __init__(self):
        self.is_active = False
        self.start_time: Optional[datetime] = None
        self.stop_time: Optional[datetime] = None
    
    def start(self):
        self.is_active = True
        self.start_time = datetime.now()
        self.stop_time = None
    
    def stop(self):
        self.is_active = False
        self.stop_time = datetime.now()


trading_session = TradingSession()


class TradeRequest(BaseModel):
    symbol: str
    side: str
    quantity: float
    price: Optional[float] = None


class CandleData(BaseModel):
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@router.post("/start")
async def start_trading():
    """Start trading session"""
    try:
        if trading_session.is_active:
            return {
                "status": "warning",
                "message": "Trading session already active",
                "session_start": trading_session.start_time.isoformat() if trading_session.start_time else None
            }
        
        logger.info("Starting trading session...")
        
        if not trade_executor.is_connected:
            trade_executor.connect_broker()
        
        if not prediction_service.model_loaded:
            prediction_service.load_model()
        
        trading_session.start()
        
        logger.info("Trading session started successfully")
        return {
            "status": "success",
            "message": "Trading session started",
            "session_start": trading_session.start_time.isoformat() if trading_session.start_time else None,
            "broker_connected": trade_executor.is_connected,
            "model_loaded": prediction_service.model_loaded
        }
    
    except Exception as e:
        logger.error(f"Failed to start trading session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_trading():
    """Stop trading session"""
    try:
        if not trading_session.is_active:
            return {
                "status": "warning",
                "message": "No active trading session"
            }
        
        logger.info("Stopping trading session...")
        
        trading_session.stop()
        
        active_positions = risk_manager.get_active_positions()
        if active_positions:
            logger.warning(f"Trading stopped with {len(active_positions)} active positions")
        
        logger.info("Trading session stopped successfully")
        return {
            "status": "success",
            "message": "Trading session stopped",
            "session_start": trading_session.start_time.isoformat() if trading_session.start_time else None,
            "session_end": trading_session.stop_time.isoformat() if trading_session.stop_time else None,
            "active_positions": len(active_positions)
        }
    
    except Exception as e:
        logger.error(f"Failed to stop trading session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def trading_status():
    """Get trading session status"""
    return {
        "is_active": trading_session.is_active,
        "session_start": trading_session.start_time.isoformat() if trading_session.start_time else None,
        "session_end": trading_session.stop_time.isoformat() if trading_session.stop_time else None,
        "active_positions": risk_manager.get_active_positions(),
        "statistics": risk_manager.get_statistics()
    }


@router.post("/execute")
async def execute_trade(trade: TradeRequest):
    """Manually execute a trade"""
    try:
        if not trading_session.is_active:
            raise HTTPException(status_code=400, detail="Trading session not active")
        
        logger.info(f"Manual trade execution requested: {trade.side} {trade.symbol}")
        
        price = trade.price or trade_executor.get_current_price(trade.symbol)
        if not price:
            raise HTTPException(status_code=400, detail="Could not determine price")
        
        if trade.side.upper() == "BUY":
            can_trade, message = risk_manager.can_open_position(trade.symbol, price, 0.75)
            if not can_trade:
                raise HTTPException(status_code=400, detail=message)
            
            order = trade_executor.execute_buy(trade.symbol, trade.quantity, price)
        elif trade.side.upper() == "SELL":
            order = trade_executor.execute_sell(trade.symbol, trade.quantity, price)
        else:
            raise HTTPException(status_code=400, detail="Invalid side, must be BUY or SELL")
        
        if not order:
            raise HTTPException(status_code=500, detail="Order execution failed")
        
        return {
            "status": "success",
            "message": "Trade executed",
            "order": order
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trade execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
async def get_prediction(candle: CandleData):
    """Get ML prediction for candle data"""
    try:
        logger.info(f"Prediction requested for {candle.symbol}")
        
        candle_dict = candle.model_dump()
        prediction, confidence = prediction_service.predict_next_candle(candle_dict)
        should_trade = prediction_service.should_trade(confidence)
        
        return {
            "symbol": candle.symbol,
            "prediction": prediction,
            "confidence": confidence,
            "should_trade": should_trade,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions")
async def get_positions():
    """Get all active positions"""
    positions = risk_manager.get_active_positions()
    return {
        "active_positions": positions,
        "count": len(positions),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/history")
async def get_trade_history(limit: int = 50):
    """Get trade execution history"""
    return {
        "trades": risk_manager.trade_history[-limit:],
        "total_trades": len(risk_manager.trade_history),
        "timestamp": datetime.now().isoformat()
    }
