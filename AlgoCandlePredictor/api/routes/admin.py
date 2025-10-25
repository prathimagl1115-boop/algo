from fastapi import APIRouter, HTTPException
from datetime import datetime
from pathlib import Path
from typing import Optional
from core.logger import logger
from trading.executor import trade_executor
from trading.risk_manager import risk_manager
from models.predict import prediction_service
from api.routes.trading import trading_session

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.post("/killswitch")
async def emergency_killswitch():
    """Emergency kill switch - stops all trading and closes positions"""
    try:
        logger.critical("EMERGENCY KILL SWITCH ACTIVATED")
        
        if trading_session.is_active:
            trading_session.stop()
        
        active_positions = risk_manager.get_active_positions()
        closed_positions = []
        
        for symbol, position in active_positions.items():
            logger.warning(f"Emergency closing position: {symbol}")
            current_price = trade_executor.get_current_price(symbol)
            if current_price:
                trade_executor.execute_sell(symbol, position["quantity"], current_price)
                closed_positions.append(symbol)
        
        if trade_executor.is_connected:
            trade_executor.disconnect_broker()
        
        logger.critical(f"Kill switch completed - closed {len(closed_positions)} positions")
        
        return {
            "status": "emergency_stop",
            "message": "All trading stopped, positions closed",
            "closed_positions": closed_positions,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.critical(f"Kill switch execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_system():
    """Reset entire trading system to initial state"""
    try:
        logger.warning("System reset initiated")
        
        if trading_session.is_active:
            trading_session.stop()
        
        risk_manager.reset()
        prediction_service.clear_history()
        trade_executor.execution_history.clear()
        
        if trade_executor.is_connected:
            trade_executor.disconnect_broker()
        
        logger.info("System reset completed successfully")
        
        return {
            "status": "success",
            "message": "System reset completed",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"System reset failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs")
async def get_logs(log_type: str = "trading", lines: int = 100):
    """Retrieve system logs"""
    try:
        log_dir = Path("logs")
        
        if log_type == "trading":
            log_pattern = f"trading_{datetime.now().strftime('%Y%m%d')}.log"
        elif log_type == "errors":
            log_pattern = f"errors_{datetime.now().strftime('%Y%m%d')}.log"
        else:
            raise HTTPException(status_code=400, detail="Invalid log_type. Use 'trading' or 'errors'")
        
        log_files = list(log_dir.glob(log_pattern))
        
        if not log_files:
            return {
                "log_type": log_type,
                "lines": [],
                "message": "No log file found for today"
            }
        
        log_file = log_files[0]
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return {
            "log_type": log_type,
            "log_file": str(log_file),
            "total_lines": len(all_lines),
            "returned_lines": len(recent_lines),
            "lines": [line.strip() for line in recent_lines],
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/list")
async def list_log_files():
    """List all available log files"""
    try:
        log_dir = Path("logs")
        if not log_dir.exists():
            return {"log_files": []}
        
        log_files = []
        for log_file in log_dir.glob("*.log"):
            stat = log_file.stat()
            log_files.append({
                "filename": log_file.name,
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        log_files.sort(key=lambda x: x["modified"], reverse=True)
        
        return {
            "log_files": log_files,
            "count": len(log_files)
        }
    
    except Exception as e:
        logger.error(f"Failed to list log files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/positions/{symbol}")
async def force_close_position(symbol: str, price: Optional[float] = None):
    """Force close a specific position"""
    try:
        logger.warning(f"Force close requested for {symbol}")
        
        if symbol not in risk_manager.active_positions:
            raise HTTPException(status_code=404, detail=f"No active position for {symbol}")
        
        position = risk_manager.active_positions[symbol]
        exit_price = price or trade_executor.get_current_price(symbol)
        
        if not exit_price:
            raise HTTPException(status_code=400, detail="Could not determine exit price")
        
        order = trade_executor.execute_sell(symbol, position["quantity"], exit_price)
        
        return {
            "status": "success",
            "message": f"Position closed for {symbol}",
            "order": order,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Force close failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/diagnostics")
async def system_diagnostics():
    """Get detailed system diagnostics"""
    try:
        diagnostics = {
            "broker": {
                "connected": trade_executor.is_connected,
                "broker_name": trade_executor.broker_name,
                "execution_count": len(trade_executor.execution_history)
            },
            "model": {
                "loaded": prediction_service.model_loaded,
                "version": prediction_service.model_version,
                "predictions_made": len(prediction_service.prediction_history)
            },
            "risk_manager": {
                "active_positions": len(risk_manager.active_positions),
                "total_trades": len(risk_manager.trade_history),
                "statistics": risk_manager.get_statistics()
            },
            "trading_session": {
                "is_active": trading_session.is_active,
                "start_time": trading_session.start_time.isoformat() if trading_session.start_time else None
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return diagnostics
    
    except Exception as e:
        logger.error(f"Diagnostics failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
