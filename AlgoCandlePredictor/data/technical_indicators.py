import pandas as pd
import numpy as np
from ta import trend, momentum, volatility, volume
from typing import Dict
from core.logger import logger


class TechnicalIndicators:
    """Calculate technical indicators for candle data"""
    
    def __init__(self):
        logger.info("TechnicalIndicators initialized")
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a dataframe
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume
            
        Returns:
            DataFrame with all technical indicators added
        """
        try:
            df = df.copy()
            
            # Moving Averages
            df['sma_9'] = trend.sma_indicator(df['close'], window=9)
            df['sma_15'] = trend.sma_indicator(df['close'], window=15)
            df['sma_50'] = trend.sma_indicator(df['close'], window=50)
            df['ema_9'] = trend.ema_indicator(df['close'], window=9)
            df['ema_15'] = trend.ema_indicator(df['close'], window=15)
            df['ema_50'] = trend.ema_indicator(df['close'], window=50)
            
            # MACD
            macd = trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # RSI
            df['rsi_14'] = momentum.rsi(df['close'], window=14)
            
            # Bollinger Bands
            bollinger = volatility.BollingerBands(df['close'])
            df['bb_high'] = bollinger.bollinger_hband()
            df['bb_mid'] = bollinger.bollinger_mavg()
            df['bb_low'] = bollinger.bollinger_lband()
            df['bb_width'] = bollinger.bollinger_wband()
            df['bb_percent'] = bollinger.bollinger_pband()
            
            # ATR (Average True Range)
            df['atr_14'] = volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
            
            # Stochastic Oscillator
            stoch = momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # ADX (Average Directional Index)
            adx = trend.ADXIndicator(df['high'], df['low'], df['close'])
            df['adx'] = adx.adx()
            df['adx_pos'] = adx.adx_pos()
            df['adx_neg'] = adx.adx_neg()
            
            # Volume indicators
            df['volume_sma_20'] = volume.volume_weighted_average_price(
                df['high'], df['low'], df['close'], df['volume'], window=20
            )
            df['volume_change'] = df['volume'].pct_change()
            
            # Price change and returns
            df['price_change'] = df['close'].pct_change()
            df['price_change_2'] = df['close'].pct_change(periods=2)
            df['price_change_5'] = df['close'].pct_change(periods=5)
            
            # High-Low spread
            df['hl_spread'] = (df['high'] - df['low']) / df['close']
            
            # Candle structure
            df['candle_body'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
            df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
            
            # Target variable: next candle direction (1 for UP, 0 for DOWN)
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            
            # ✅ Clean up data (remove infinities and NaNs)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            
            # Drop rows still containing NaN if any (very rare)
            df.dropna(inplace=True)
            
            logger.info(f"Calculated {len(df.columns) - 6} technical indicators (cleaned and validated)")
            return df
        
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}", exc_info=True)
            raise
    
    def get_feature_names(self) -> list:
        """Get list of all feature names (excluding OHLCV and target)"""
        return [
            'sma_9', 'sma_15', 'sma_50',
            'ema_9', 'ema_15', 'ema_50',
            'macd', 'macd_signal', 'macd_diff',
            'rsi_14',
            'bb_high', 'bb_mid', 'bb_low', 'bb_width', 'bb_percent',
            'atr_14',
            'stoch_k', 'stoch_d',
            'adx', 'adx_pos', 'adx_neg',
            'volume_sma_20', 'volume_change',
            'price_change', 'price_change_2', 'price_change_5',
            'hl_spread',
            'candle_body', 'upper_shadow', 'lower_shadow'
        ]


# Singleton instance
technical_indicators = TechnicalIndicators()
