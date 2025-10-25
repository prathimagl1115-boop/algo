#!/usr/bin/env python
"""
Simple training script for the algorithmic trading bot
Trains the Hybrid XGBoost-LSTM model on ULTRACEMCO 5-minute data
"""
import sys
from pathlib import Path
from models.train import HybridModelTrainer
from core.logger import logger

def main():
    """Main training function"""
    print("="*80)
    print("ALGORITHMIC TRADING BOT - MODEL TRAINING")
    print("="*80)
    print()
    print("Dataset: ULTRACEMCO 5-minute candles (194K+ rows, 2015-2025)")
    print("Model: Hybrid XGBoost-LSTM")
    print("Target: 70%+ directional accuracy")
    print()
    print("This will take approximately 10-20 minutes...")
    print("="*80)
    print()
    
    try:
        # Initialize trainer
        trainer = HybridModelTrainer()
        
        # Train the model on ULTRACEMCO data
        logger.info("Starting model training pipeline...")
        results = trainer.train_complete_pipeline(
            data_filepath='data_files/ULTRACEMCO_5minute_1761369726146.csv',
            symbol='ULTRACEMCO',
            sequence_length=60,
            top_n_features=15,
            epochs=50,
            batch_size=32
        )
        
        print()
        print("="*80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print()
        print("Test Set Performance:")
        print(f"  Accuracy: {results['test_metrics']['accuracy']:.2%}")
        print(f"  Directional Accuracy: {results['test_metrics']['directional_accuracy']:.2%}")
        print()
        print(f"Selected Top {len(results['selected_features'])} Features:")
        for i, feature in enumerate(results['feature_names'], 1):
            print(f"  {i}. {feature}")
        print()
        print("Trained models saved to: trained_models/")
        print("  - xgboost_ULTRACEMCO_latest.joblib")
        print("  - lstm_ULTRACEMCO_latest.keras")
        print("  - metadata_ULTRACEMCO_latest.json")
        print()
        print("Next steps:")
        print("  1. Test predictions via API: POST /trading/predict")
        print("  2. Start demo trading: POST /trading/start")
        print("  3. Monitor performance: GET /system/metrics")
        print("  4. Review backtest results for live trading readiness")
        print()
        print("Happy Trading! 🚀")
        print("="*80)
        
    except FileNotFoundError as e:
        print()
        print("ERROR: Data file not found!")
        print()
        print("Make sure the following file exists:")
        print("  data_files/ULTRACEMCO_5minute_1761369726146.csv")
        print()
        print("Available data files:")
        data_dir = Path("data_files")
        if data_dir.exists():
            for file in data_dir.glob("*.csv"):
                print(f"  - {file.name}")
        else:
            print("  (data_files directory not found)")
        sys.exit(1)
        
    except Exception as e:
        print()
        print("ERROR: Training failed!")
        print(f"Error: {e}")
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
