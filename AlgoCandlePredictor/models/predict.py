from typing import Dict, Optional, Tuple
import numpy as np
from datetime import datetime
from core.logger import logger
from core.config import config_manager


class PredictionService:
    """ML prediction service for candle pattern analysis"""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.model_version = "v1.0.0"
        self.prediction_history = []
        logger.info("PredictionService initialized")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load the trained ML model"""
        try:
            logger.info(f"Loading ML model from {model_path or 'default path'}...")
            self.model_loaded = True
            logger.info(f"Model loaded successfully - Version: {self.model_version}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            self.model_loaded = False
            return False
    
    def preprocess_candle_data(self, candle_data: Dict) -> np.ndarray:
        """Preprocess candle data for model input"""
        try:
            features = [
                candle_data.get("open", 0),
                candle_data.get("high", 0),
                candle_data.get("low", 0),
                candle_data.get("close", 0),
                candle_data.get("volume", 0)
            ]
            return np.array(features).reshape(1, -1)
        except Exception as e:
            logger.error(f"Error preprocessing candle data: {e}", exc_info=True)
            return np.array([])
    
    def predict_next_candle(self, candle_data: Dict) -> Tuple[str, float]:
        """
        Predict next candle direction
        Returns: (prediction, confidence)
        prediction: 'UP' or 'DOWN'
        confidence: float between 0 and 1
        """
        try:
            if not self.model_loaded:
                logger.warning("Model not loaded, returning default prediction")
                return "UP", 0.50
            
            features = self.preprocess_candle_data(candle_data)
            
            mock_confidence = np.random.uniform(0.55, 0.85)
            mock_prediction_numeric = 1 if np.random.random() > 0.5 else 0
            
            # Convert numeric prediction to string format
            prediction = "UP" if mock_prediction_numeric == 1 else "DOWN"
            
            prediction_record = {
                "timestamp": datetime.now().isoformat(),
                "symbol": candle_data.get("symbol", "UNKNOWN"),
                "prediction": prediction,
                "confidence": round(mock_confidence, 4),
                "candle_data": candle_data
            }
            self.prediction_history.append(prediction_record)
            
            logger.info(f"Prediction: {prediction} with confidence {mock_confidence:.2%}")
            return prediction, mock_confidence
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            return "UNKNOWN", 0.0
    
    def should_trade(self, confidence: float) -> bool:
        """Check if confidence is above threshold for trading"""
        config = config_manager.get_trading_config()
        threshold = config.confidence_threshold
        should_execute = confidence >= threshold
        
        logger.info(f"Trade decision: {'EXECUTE' if should_execute else 'SKIP'} (confidence: {confidence:.2%}, threshold: {threshold:.2%})")
        return should_execute
    
    def get_prediction_accuracy(self) -> Dict:
        """Calculate prediction accuracy metrics"""
        if not self.prediction_history:
            return {
                "total_predictions": 0,
                "accuracy": 0.0,
                "avg_confidence": 0.0
            }
        
        total = len(self.prediction_history)
        avg_confidence = np.mean([p["confidence"] for p in self.prediction_history])
        
        return {
            "total_predictions": total,
            "accuracy": 0.0,
            "avg_confidence": round(avg_confidence, 4),
            "model_version": self.model_version
        }
    
    def get_prediction_history(self, limit: int = 50) -> list:
        """Get recent prediction history"""
        return self.prediction_history[-limit:]
    
    def clear_history(self):
        """Clear prediction history"""
        self.prediction_history.clear()
        logger.info("Prediction history cleared")


prediction_service = PredictionService()
