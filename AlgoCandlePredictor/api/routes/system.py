from fastapi import APIRouter, HTTPException
from datetime import datetime
from core.logger import logger
from core.config import config_manager
from trading.risk_manager import risk_manager
from trading.executor import trade_executor
from models.predict import prediction_service

router = APIRouter(prefix="/system", tags=["System"])


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.debug("Health check requested")
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "AlgoTrader API"
    }


@router.get("/status")
async def system_status():
    """Get comprehensive system status"""
    logger.info("System status requested")
    
    config = config_manager.get_trading_config()
    app_settings = config_manager.get_app_settings()
    
    status = {
        "system": {
            "app_name": app_settings.app_name,
            "version": app_settings.app_version,
            "debug_mode": app_settings.debug,
            "timestamp": datetime.now().isoformat()
        },
        "broker": {
            "connected": trade_executor.is_connected,
            "broker_name": trade_executor.broker_name
        },
        "model": {
            "loaded": prediction_service.model_loaded,
            "version": prediction_service.model_version
        },
        "trading": {
            "mode": config.mode,
            "symbols": config.symbols,
            "timeframe": config.timeframe,
            "confidence_threshold": config.confidence_threshold
        },
        "risk": {
            "active_positions": len(risk_manager.active_positions),
            "daily_pnl": risk_manager.get_daily_pnl(),
            "max_daily_loss": config.max_daily_loss
        }
    }
    
    return status


@router.get("/config")
async def get_configuration():
    """Get current trading configuration"""
    logger.info("Configuration requested")
    
    config = config_manager.get_trading_config()
    return {
        "config": config.model_dump(),
        "timestamp": datetime.now().isoformat()
    }


@router.post("/config")
async def update_configuration(updates: dict):
    """Update trading configuration"""
    try:
        logger.info(f"Configuration update requested: {updates}")
        updated_config = config_manager.update_trading_config(updates)
        config_manager.save_config()
        
        return {
            "status": "success",
            "message": "Configuration updated successfully",
            "config": updated_config.model_dump()
        }
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/metrics")
async def get_metrics():
    """Get trading metrics and statistics"""
    logger.info("Metrics requested")
    
    stats = risk_manager.get_statistics()
    prediction_stats = prediction_service.get_prediction_accuracy()
    
    return {
        "trading_stats": stats,
        "prediction_stats": prediction_stats,
        "timestamp": datetime.now().isoformat()
    }
