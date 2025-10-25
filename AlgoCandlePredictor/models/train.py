import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib
import json
from pathlib import Path
from datetime import datetime
from core.logger import logger
from data.preprocess import data_preprocessor


class HybridModelTrainer:
    """Train Hybrid XGBoost + LSTM model for candle prediction"""
    
    def __init__(self, model_save_dir: str = "trained_models"):
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)
        self.xgb_model = None
        self.lstm_model = None
        self.feature_importance = None
        self.selected_features = None
        logger.info("HybridModelTrainer initialized")
    
    def train_xgboost_feature_selector(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list,
        top_n_features: int = 15
    ) -> list:
        """
        Stage 1: Train XGBoost for feature selection
        
        Args:
            X_train, X_val: Training and validation features
            y_train, y_val: Training and validation targets
            feature_names: List of feature names
            top_n_features: Number of top features to select
            
        Returns:
            List of selected feature indices
        """
        try:
            logger.info("Training XGBoost for feature selection...")
            
            # Train XGBoost classifier
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                random_state=42,
                eval_metric='logloss'
            )
            
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=True
            )
            
            # Get feature importance
            self.feature_importance = self.xgb_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.feature_importance
            }).sort_values('importance', ascending=False)
            
            logger.info("\nTop 20 Features by Importance:")
            logger.info(feature_importance_df.head(20).to_string())
            
            # Select top N features
            top_indices = np.argsort(self.feature_importance)[-top_n_features:]
            self.selected_features = top_indices
            
            selected_feature_names = [feature_names[i] for i in top_indices]
            logger.info(f"\nSelected {top_n_features} features: {selected_feature_names}")
            
            # Evaluate XGBoost model
            y_pred_train = self.xgb_model.predict(X_train)
            y_pred_val = self.xgb_model.predict(X_val)
            
            train_acc = accuracy_score(y_train, y_pred_train)
            val_acc = accuracy_score(y_val, y_pred_val)
            
            logger.info(f"\nXGBoost Results:")
            logger.info(f"Training Accuracy: {train_acc:.4f}")
            logger.info(f"Validation Accuracy: {val_acc:.4f}")
            
            return top_indices
            
        except Exception as e:
            logger.error(f"Error training XGBoost: {e}", exc_info=True)
            raise
    
    def build_lstm_model(
        self,
        sequence_length: int,
        n_features: int
    ) -> Sequential:
        """
        Build LSTM neural network model
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of features
            
        Returns:
            Compiled LSTM model
        """
        try:
            model = Sequential([
                LSTM(units=128, return_sequences=True, input_shape=(sequence_length, n_features)),
                Dropout(0.2),
                BatchNormalization(),
                
                LSTM(units=64, return_sequences=True),
                Dropout(0.2),
                BatchNormalization(),
                
                LSTM(units=32, return_sequences=False),
                Dropout(0.2),
                
                Dense(units=16, activation='relu'),
                Dropout(0.1),
                
                Dense(units=1, activation='sigmoid')
            ])
            
            optimizer = Adam(learning_rate=0.001)
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("LSTM model built successfully")
            logger.info(f"Model summary: {model.summary()}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error building LSTM model: {e}", exc_info=True)
            raise
    
    def train_lstm_on_selected_features(
        self,
        X_train_seq: np.ndarray,
        y_train_seq: np.ndarray,
        X_val_seq: np.ndarray,
        y_val_seq: np.ndarray,
        selected_indices: list,
        epochs: int = 50,
        batch_size: int = 32
    ):
        """
        Stage 2: Train LSTM on selected features
        
        Args:
            X_train_seq, X_val_seq: Sequence data for training and validation
            y_train_seq, y_val_seq: Targets for training and validation
            selected_indices: Indices of selected features from XGBoost
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        try:
            logger.info("Training LSTM on selected features...")
            
            # Select only top features
            X_train_selected = X_train_seq[:, :, selected_indices]
            X_val_selected = X_val_seq[:, :, selected_indices]
            
            sequence_length = X_train_selected.shape[1]
            n_features = X_train_selected.shape[2]
            
            # Build LSTM model
            self.lstm_model = self.build_lstm_model(sequence_length, n_features)
            
            # Callbacks
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
            
            checkpoint = ModelCheckpoint(
                str(self.model_save_dir / 'lstm_best.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
            
            # Train model
            history = self.lstm_model.fit(
                X_train_selected, y_train_seq,
                validation_data=(X_val_selected, y_val_seq),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop, checkpoint, reduce_lr],
                verbose=1
            )
            
            # Evaluate
            train_loss, train_acc = self.lstm_model.evaluate(X_train_selected, y_train_seq, verbose=0)
            val_loss, val_acc = self.lstm_model.evaluate(X_val_selected, y_val_seq, verbose=0)
            
            logger.info(f"\nLSTM Results:")
            logger.info(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            logger.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            
            return history
            
        except Exception as e:
            logger.error(f"Error training LSTM: {e}", exc_info=True)
            raise
    
    def evaluate_hybrid_model(
        self,
        X_test_seq: np.ndarray,
        y_test_seq: np.ndarray,
        selected_indices: list
    ) -> dict:
        """
        Evaluate the complete hybrid model on test data
        
        Args:
            X_test_seq: Test sequence data
            y_test_seq: Test targets
            selected_indices: Selected feature indices
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            logger.info("Evaluating hybrid model on test data...")
            
            # Select features
            X_test_selected = X_test_seq[:, :, selected_indices]
            
            # Get predictions
            y_pred_proba = self.lstm_model.predict(X_test_selected)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_seq, y_pred)
            conf_matrix = confusion_matrix(y_test_seq, y_pred)
            class_report = classification_report(y_test_seq, y_pred)
            
            # Directional accuracy
            directional_acc = np.mean((y_pred == y_test_seq))
            
            logger.info(f"\nHybrid Model Test Results:")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Directional Accuracy: {directional_acc:.4f}")
            logger.info(f"\nConfusion Matrix:\n{conf_matrix}")
            logger.info(f"\nClassification Report:\n{class_report}")
            
            metrics = {
                'accuracy': float(accuracy),
                'directional_accuracy': float(directional_acc),
                'confusion_matrix': conf_matrix.tolist(),
                'classification_report': class_report
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating hybrid model: {e}", exc_info=True)
            raise
    
    def save_models(self, symbol: str):
        """Save trained models and metadata"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save XGBoost model
            xgb_path = self.model_save_dir / f'xgboost_{symbol}_{timestamp}.joblib'
            joblib.dump(self.xgb_model, xgb_path)
            logger.info(f"XGBoost model saved to {xgb_path}")
            
            # Save LSTM model
            lstm_path = self.model_save_dir / f'lstm_{symbol}_{timestamp}.keras'
            self.lstm_model.save(lstm_path)
            logger.info(f"LSTM model saved to {lstm_path}")
            
            # Save metadata
            metadata = {
                'symbol': symbol,
                'timestamp': timestamp,
                'selected_features': self.selected_features.tolist(),
                'xgb_model_path': str(xgb_path),
                'lstm_model_path': str(lstm_path)
            }
            
            metadata_path = self.model_save_dir / f'metadata_{symbol}_{timestamp}.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata saved to {metadata_path}")
            
            # Also save as latest
            joblib.dump(self.xgb_model, self.model_save_dir / f'xgboost_{symbol}_latest.joblib')
            self.lstm_model.save(self.model_save_dir / f'lstm_{symbol}_latest.keras')
            with open(self.model_save_dir / f'metadata_{symbol}_latest.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}", exc_info=True)
            raise
    
    def train_complete_pipeline(
        self,
        data_filepath: str,
        symbol: str,
        sequence_length: int = 60,
        top_n_features: int = 15,
        epochs: int = 50,
        batch_size: int = 32
    ) -> dict:
        """
        Complete training pipeline: preprocess -> XGBoost -> LSTM -> evaluate
        
        Args:
            data_filepath: Path to CSV data file
            symbol: Trading symbol
            sequence_length: Sequence length for LSTM
            top_n_features: Number of features to select
            epochs: LSTM training epochs
            batch_size: LSTM batch size
            
        Returns:
            Dictionary with training results and metrics
        """
        try:
            logger.info(f"Starting complete training pipeline for {symbol}")
            
            # Preprocess data
            data = data_preprocessor.prepare_for_training(
                data_filepath,
                sequence_length=sequence_length
            )
            
            # Stage 1: XGBoost feature selection
            selected_indices = self.train_xgboost_feature_selector(
                data['X_train'], data['y_train'],
                data['X_val'], data['y_val'],
                data['feature_names'],
                top_n_features=top_n_features
            )
            
            # Stage 2: LSTM training on selected features
            history = self.train_lstm_on_selected_features(
                data['X_train_seq'], data['y_train_seq'],
                data['X_val_seq'], data['y_val_seq'],
                selected_indices,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Evaluate on test set
            test_metrics = self.evaluate_hybrid_model(
                data['X_test_seq'], data['y_test_seq'],
                selected_indices
            )
            
            # Save models
            self.save_models(symbol)
            
            logger.info("Training pipeline completed successfully!")
            
            return {
                'test_metrics': test_metrics,
                'selected_features': selected_indices.tolist(),
                'feature_names': [data['feature_names'][i] for i in selected_indices]
            }
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}", exc_info=True)
            raise


# Entry point for training
if __name__ == "__main__":
    trainer = HybridModelTrainer()
    
    # Train on ULTRACEMCO data
    results = trainer.train_complete_pipeline(
        data_filepath='data_files/ULTRACEMCO_5minute_1761369726146.csv',
        symbol='ULTRACEMCO',
        sequence_length=60,
        top_n_features=15,
        epochs=50,
        batch_size=32
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nTest Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"Directional Accuracy: {results['test_metrics']['directional_accuracy']:.4f}")
    print(f"\nSelected Features: {results['feature_names']}")
