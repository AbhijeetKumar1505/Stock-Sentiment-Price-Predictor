import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from typing import Optional
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    """PyTorch LSTM model for stock prediction."""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

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
        self.svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.lstm_model: Optional[LSTMModel] = None
        self.feature_columns = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
    
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
    
    def build_lstm_model(self, input_size):
        """
        Build the PyTorch LSTM model.
        
        Args:
            input_size (int): Number of input features
            
        Returns:
            LSTMModel: PyTorch LSTM model
        """
        try:
            model = LSTMModel(
                input_size=input_size,
                hidden_size=self.lstm_units,
                num_layers=2,
                dropout_rate=self.dropout_rate
            )
            
            return model.to(self.device)
            
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
            # Train SVR
            self.svr_model.fit(X_linear, y)
            # Train Random Forest
            self.rf_model.fit(X_linear, y)
            
            # Build LSTM model
            self.lstm_model = self.build_lstm_model(input_size=X_lstm.shape[2])
            if self.lstm_model is None:
                raise ValueError("LSTM model failed to initialize")
            
            # Prepare PyTorch tensors
            X_lstm_tensor = torch.FloatTensor(X_lstm).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            # Split data for validation
            split_idx = int(len(X_lstm) * (1 - validation_split))
            X_train = X_lstm_tensor[:split_idx]
            y_train = y_tensor[:split_idx]
            X_val = X_lstm_tensor[split_idx:]
            y_val = y_tensor[split_idx:]
            
            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Setup optimizer and loss function
            optimizer = optim.Adam(self.lstm_model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()
            
            # Training history
            train_losses = []
            val_losses = []
            
            # Training loop
            for epoch in range(epochs):
                # Training phase
                self.lstm_model.train()
                train_loss = 0.0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.lstm_model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation phase
                self.lstm_model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.lstm_model(batch_X).squeeze()
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                train_losses.append(train_loss / len(train_loader))
                val_losses.append(val_loss / len(val_loader))
                
                # Early stopping
                if epoch > 5 and val_losses[-1] > val_losses[-2]:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
            
            # Make predictions on training data
            combined_pred, linear_pred, lstm_pred, svr_pred, rf_pred = self.predict(X_linear, X_lstm)
            
            # Evaluate model
            metrics = self.evaluate(y, combined_pred)
            
            # Prepare training history for visualization
            training_history = {
                'epochs': range(1, len(train_losses) + 1),
                'train_loss': train_losses,
                'val_loss': val_losses
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
            # Get predictions from linear regression
            linear_pred = self.linear_model.predict(X_linear)
            # Get predictions from SVR
            svr_pred = self.svr_model.predict(X_linear)
            # Get predictions from Random Forest
            rf_pred = self.rf_model.predict(X_linear)
            
            # Get predictions from LSTM
            if self.lstm_model is None:
                raise ValueError("LSTM model has not been trained")
            
            self.lstm_model.eval()
            with torch.no_grad():
                X_lstm_tensor = torch.FloatTensor(X_lstm).to(self.device)
                lstm_pred = self.lstm_model(X_lstm_tensor).cpu().numpy().flatten()
            
            # Combine predictions (average of all four models)
            combined_pred = (linear_pred + lstm_pred + svr_pred + rf_pred) / 4
            
            return combined_pred, linear_pred, lstm_pred, svr_pred, rf_pred
            
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