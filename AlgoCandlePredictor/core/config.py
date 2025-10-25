import yaml
import os
from pathlib import Path
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


class TradingConfig(BaseModel):
    """Trading configuration model"""
    mode: str = Field(default="paper", description="Trading mode: paper or live")
    symbols: List[str] = Field(default=["RELIANCE"], description="List of symbols to trade")
    timeframe: str = Field(default="5m", description="Candle timeframe")
    max_position_size: float = Field(default=10000.0, description="Maximum position size in currency")
    max_loss_per_trade: float = Field(default=100.0, description="Maximum loss per trade")
    max_daily_loss: float = Field(default=500.0, description="Maximum daily loss limit")
    confidence_threshold: float = Field(default=0.70, description="Minimum prediction confidence to trade")
    

class AppSettings(BaseSettings):
    """Application settings from environment variables"""
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)
    
    app_name: str = "AlgoTrader"
    app_version: str = "1.0.0"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 5000


class ConfigManager:
    """Configuration manager for the trading system"""
    
    def __init__(self, config_path: str = "configs/config.yml"):
        self.config_path = Path(config_path)
        self.app_settings = AppSettings()
        self.trading_config = self._load_trading_config()
    
    def _load_trading_config(self) -> TradingConfig:
        """Load trading configuration from YAML file"""
        if not self.config_path.exists():
            return TradingConfig()
        
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}
        
        env_overrides = {
            'mode': os.getenv('TRADING_MODE', config_data.get('mode')),
            'symbols': os.getenv('TRADING_SYMBOLS', '').split(',') if os.getenv('TRADING_SYMBOLS') else config_data.get('symbols'),
            'timeframe': os.getenv('TRADING_TIMEFRAME', config_data.get('timeframe')),
            'max_position_size': float(os.getenv('MAX_POSITION_SIZE', config_data.get('max_position_size', 10000.0))),
            'max_loss_per_trade': float(os.getenv('MAX_LOSS_PER_TRADE', config_data.get('max_loss_per_trade', 100.0))),
            'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', config_data.get('max_daily_loss', 500.0))),
            'confidence_threshold': float(os.getenv('CONFIDENCE_THRESHOLD', config_data.get('confidence_threshold', 0.70))),
        }
        
        clean_config = {k: v for k, v in env_overrides.items() if v is not None}
        
        return TradingConfig(**clean_config)
    
    def get_trading_config(self) -> TradingConfig:
        """Get current trading configuration"""
        return self.trading_config
    
    def get_app_settings(self) -> AppSettings:
        """Get application settings"""
        return self.app_settings
    
    def update_trading_config(self, updates: Dict[str, Any]) -> TradingConfig:
        """Update trading configuration"""
        current_dict = self.trading_config.model_dump()
        current_dict.update(updates)
        self.trading_config = TradingConfig(**current_dict)
        return self.trading_config
    
    def save_config(self):
        """Save current configuration to YAML file"""
        self.config_path.parent.mkdir(exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.trading_config.model_dump(), f, default_flow_style=False)


config_manager = ConfigManager()
