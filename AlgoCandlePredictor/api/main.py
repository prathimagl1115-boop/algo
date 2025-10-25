from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.logger import logger
from core.config import config_manager
from api.routes import system, trading, admin

app_settings = config_manager.get_app_settings()

app = FastAPI(
    title=app_settings.app_name,
    version=app_settings.app_version,
    description="AI-Powered Algorithmic Trading System with ML Predictions",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(trading.router)
app.include_router(admin.router)


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info(f"Starting {app_settings.app_name} v{app_settings.app_version}")
    logger.info(f"Debug mode: {app_settings.debug}")
    logger.info("API Documentation available at /docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down AlgoTrader API")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": app_settings.app_name,
        "version": app_settings.app_version,
        "status": "running",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=app_settings.host,
        port=app_settings.port,
        reload=app_settings.debug
    )
