import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridStockPredictor:
    def __init__(self, sequence_length=30, lstm_units=50, dropout_rate=0.2, learning_rate=0.001):
        """
        Initialize the hybrid model with configurable parameters.
        
        Args:
            sequence_length (int): Number of time steps to look back for LSTM
            lstm_units (int): Number of units in LSTM layer
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.scaler = MinMaxScaler()
        self.linear_model = LinearRegression()
        self.lstm_model: Optional[Sequential] = None
        self.feature_columns = None
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def prepare_data(self, df, feature_columns=None):
        """
        Prepare data for model training and prediction.
        
        Args:
            df (pd.DataFrame): Input dataframe with stock data
            feature_columns (list): List of feature columns to use
            
        Returns:
            tuple: (X_linear, X_lstm, y)
        """
        try:
            if feature_columns is None:
                feature_columns = ['Close', 'Volume', 'SMA_20', 'RSI', 'BB_Position']
            
            self.feature_columns = feature_columns
            
            # Create target variable (next day's close price)
            df['Target'] = df['Close'].shift(-1)
            df = df.dropna()
            
            # Scale features
            scaled_features = self.scaler.fit_transform(df[feature_columns])
            
            # Prepare data for linear regression
            X_linear = scaled_features
            
            # Prepare sequences for LSTM
            X_lstm = []
            for i in range(len(scaled_features) - self.sequence_length):
                X_lstm.append(scaled_features[i:(i + self.sequence_length)])
            
            X_lstm = np.array(X_lstm)
            y = df['Target'].values[self.sequence_length:]
            
            return X_linear[self.sequence_length:], X_lstm, y
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def build_lstm_model(self, input_shape):
        """
        Build and compile the LSTM model.
        
        Args:
            input_shape (tuple): Shape of input data
            
        Returns:
            keras.Model: Compiled LSTM model
        """
        try:
            model = Sequential([
                LSTM(self.lstm_units, input_shape=input_shape, return_sequences=True),
                Dropout(self.dropout_rate),
                LSTM(self.lstm_units),
                Dropout(self.dropout_rate),
                Dense(1)
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mean_squared_error',
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error building LSTM model: {str(e)}")
            raise
    
    def train(self, X_linear, X_lstm, y, validation_split=0.2, epochs=50, batch_size=32):
        """
        Train both linear regression and LSTM models.
        
        Args:
            X_linear (np.array): Features for linear regression
            X_lstm (np.array): Sequences for LSTM
            y (np.array): Target values
            validation_split (float): Fraction of data to use for validation
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            tuple: (training_history, model_metrics)
        """
        try:
            # Train linear regression
            self.linear_model.fit(X_linear, y)
            
            # Build and train LSTM
            self.lstm_model = self.build_lstm_model(input_shape=(X_lstm.shape[1], X_lstm.shape[2]))
            if self.lstm_model is None:
                raise ValueError("LSTM model failed to initialize")
            
            # Add early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            # Train LSTM model
            history = self.lstm_model.fit(
                X_lstm, y,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose='auto'
            )
            
            # Make predictions on training data
            combined_pred, linear_pred, lstm_pred = self.predict(X_linear, X_lstm)
            
            # Evaluate model
            metrics = self.evaluate(y, combined_pred)
            
            # Prepare training history for visualization
            training_history = {
                'epochs': range(1, len(history.history['loss']) + 1),
                'train_loss': history.history['loss'],
                'val_loss': history.history['val_loss']
            }
            
            return training_history, metrics
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
    
    def predict(self, X_linear, X_lstm):
        """
        Make predictions using both models and combine them.
        
        Args:
            X_linear (np.array): Features for linear regression
            X_lstm (np.array): Sequences for LSTM
            
        Returns:
            tuple: (combined_predictions, linear_predictions, lstm_predictions)
        """
        try:
            # Get predictions from both models
            linear_pred = self.linear_model.predict(X_linear)
            if self.lstm_model is None:
                raise ValueError("LSTM model has not been trained")
            lstm_pred = self.lstm_model.predict(X_lstm).flatten()
            
            # Combine predictions (simple average)
            combined_pred = (linear_pred + lstm_pred) / 2
            
            return combined_pred, linear_pred, lstm_pred
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def evaluate(self, y_true, y_pred):
        """
        Evaluate model performance using multiple metrics.
        
        Args:
            y_true (np.array): True values
            y_pred (np.array): Predicted values
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        try:
            metrics = {
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
                'Directional_Accuracy': np.mean(np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1])) * 100
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise 