import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional
from pathlib import Path
from core.logger import logger
from data.technical_indicators import technical_indicators


class DataPreprocessor:
    """Preprocess candle data for ML model training"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        self.feature_names = []
        logger.info("DataPreprocessor initialized")
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load candle data from CSV file"""
        try:
            logger.info(f"Loading data from {filepath}")
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            logger.info(f"Loaded {len(df)} candles from {Path(filepath).name}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            raise
    
    def prepare_features(self, df: pd.DataFrame, calculate_indicators: bool = True) -> pd.DataFrame:
        """
        Prepare features from candle data
        """
        try:
            if calculate_indicators:
                df = technical_indicators.calculate_all_indicators(df)
            
            # Clean invalid values early
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Drop rows with NaN values (from indicator calculations)
            initial_len = len(df)
            df = df.dropna().reset_index(drop=True)
            dropped = initial_len - len(df)
            logger.info(f"Dropped {dropped} rows with NaN or inf values")
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}", exc_info=True)
            raise
    
    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets (time-series aware)"""
        try:
            assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1.0"
            
            n = len(df)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            
            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[train_end:val_end].copy()
            test_df = df.iloc[val_end:].copy()
            
            logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            
            return train_df, val_df, test_df
            
        except Exception as e:
            logger.error(f"Error splitting data: {e}", exc_info=True)
            raise
    
    def create_sequences_lstm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM model"""
        try:
            X_sequences = []
            y_sequences = []
            
            for i in range(sequence_length, len(X)):
                X_sequences.append(X[i-sequence_length:i])
                y_sequences.append(y[i])
            
            X_sequences = np.array(X_sequences)
            y_sequences = np.array(y_sequences)
            
            logger.info(f"Created {len(X_sequences)} sequences with length {sequence_length}")
            
            return X_sequences, y_sequences
            
        except Exception as e:
            logger.error(f"Error creating sequences: {e}", exc_info=True)
            raise
    
    def normalize_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: list
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize features using StandardScaler (fit on training data only)
        """
        try:
            # Extract feature data
            X_train = train_df[feature_cols].copy()
            X_val = val_df[feature_cols].copy()
            X_test = test_df[feature_cols].copy()

            # Clean infinities and NaNs before scaling
            for X in [X_train, X_val, X_test]:
                X.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Replace remaining NaNs with training column means
            X_train.fillna(X_train.mean(), inplace=True)
            X_val.fillna(X_train.mean(), inplace=True)
            X_test.fillna(X_train.mean(), inplace=True)

            # Optional: cap extreme outliers to avoid overflow
            X_train = X_train.clip(lower=-1e9, upper=1e9)
            X_val = X_val.clip(lower=-1e9, upper=1e9)
            X_test = X_test.clip(lower=-1e9, upper=1e9)

            # Debug check
            if np.isinf(X_train.values).any() or np.isnan(X_train.values).any():
                bad_cols = X_train.columns[np.isinf(X_train.values).any(axis=0) | np.isnan(X_train.values).any(axis=0)]
                logger.warning(f"⚠️ Columns with invalid values before scaling: {bad_cols.tolist()}")

            # Fit scaler on training data only
            self.scaler.fit(X_train.values)
            
            # Transform all sets
            X_train_scaled = self.scaler.transform(X_train.values)
            X_val_scaled = self.scaler.transform(X_val.values)
            X_test_scaled = self.scaler.transform(X_test.values)
            
            logger.info(f"Normalized {len(feature_cols)} features successfully")
            
            return X_train_scaled, X_val_scaled, X_test_scaled
            
        except Exception as e:
            logger.error(f"Error normalizing features: {e}", exc_info=True)
            raise
    
    def prepare_for_training(
        self,
        filepath: str,
        sequence_length: int = 60,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> dict:
        """Complete preprocessing pipeline for model training"""
        try:
            # Load and prepare data
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            logger.info(f"Loaded {len(df)} candles")
            
            # Calculate technical indicators
            df = self.prepare_features(df)
            
            # Split data
            train_df, val_df, test_df = self.split_data(df, train_ratio, val_ratio)
            
            # Get feature columns (all except date and target)
            feature_cols = technical_indicators.get_feature_names()
            self.feature_names = feature_cols
            
            # Normalize features (cleaned version)
            X_train, X_val, X_test = self.normalize_features(
                train_df, val_df, test_df, feature_cols
            )
            
            # Get targets
            y_train = train_df['target'].values
            y_val = val_df['target'].values
            y_test = test_df['target'].values
            
            # Create sequences for LSTM
            X_train_seq, y_train_seq = self.create_sequences_lstm(X_train, y_train, sequence_length)
            X_val_seq, y_val_seq = self.create_sequences_lstm(X_val, y_val, sequence_length)
            X_test_seq, y_test_seq = self.create_sequences_lstm(X_test, y_test, sequence_length)
            
            logger.info("Data preprocessing complete")
            
            return {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'X_train_seq': X_train_seq,
                'X_val_seq': X_val_seq,
                'X_test_seq': X_test_seq,
                'y_train_seq': y_train_seq,
                'y_val_seq': y_val_seq,
                'y_test_seq': y_test_seq,
                'feature_names': self.feature_names,
                'scaler': self.scaler
            }
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {e}", exc_info=True)
            raise


# Singleton instance
data_preprocessor = DataPreprocessor()
