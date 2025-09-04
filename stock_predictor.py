import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import os
import traceback
import glob
import tempfile
import time
import random
import requests
from pyrate_limiter import Duration, RequestRate, Limiter
from hybrid_model import HybridStockPredictor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tf_explain.core.grad_cam import GradCAM
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create caching directory
CACHE_DIR = os.path.join(tempfile.gettempdir(), "stock_predictor_cache")
try:
    os.makedirs(CACHE_DIR, exist_ok=True)
    logger.info(f"Cache directory created at: {CACHE_DIR}")
except Exception as e:
    logger.error(f"Failed to create cache directory: {str(e)}")
    CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    os.makedirs(CACHE_DIR, exist_ok=True)
    logger.info(f"Using fallback cache directory at: {CACHE_DIR}")

CACHE_DB = os.path.join(CACHE_DIR, "yfinance_cache")

# Set a user agent that rotates to avoid detection (not used for yfinance anymore)
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
]
# yf_session.headers['User-Agent'] = random.choice(USER_AGENTS)  # Removed

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# NewsAPI configuration
NEWS_API_KEY = os.getenv('NEWS_API_KEY', "faa59dc97acf42f1acdada2e9c9e4155")  # Use environment variable with fallback
if not NEWS_API_KEY:
    logger.warning("NEWS_API_KEY environment variable not set. Using default key which may have rate limits.")
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# Indian stock exchanges and their Yahoo Finance suffixes
INDIAN_EXCHANGES = {
    "NSE": ".NS",  # National Stock Exchange
    "BSE": ".BO",  # Bombay Stock Exchange
}

# Popular Indian stocks by sector
POPULAR_INDIAN_STOCKS = {
    "Banking": ["HDFCBANK.NS", "SBIN.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
    "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "TECHM.NS", "HCLTECH.NS"],
    "Automobile": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS"],
    "Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "BIOCON.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS", "POWERGRID.NS", "NTPC.NS", "BPCL.NS"],
    "Consumer Goods": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "MARICO.NS"],
    "Metals": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "COALINDIA.NS", "NMDC.NS"]
}

# Market indices for reference
MARKET_INDICES = {
    "USA": {
        "S&P 500": "^GSPC",
        "Dow Jones": "^DJI",
        "NASDAQ": "^IXIC"
    },
    "India": {
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN",
        "NIFTY Bank": "^NSEBANK"
    }
}

def clean_cache_files(max_age_days=7):
    """Delete cached stock data files that are older than specified days."""
    try:
        cache_pattern = os.path.join(CACHE_DIR, "*_data.csv")
        cache_files = glob.glob(cache_pattern)
        
        current_time = datetime.now()
        deleted_count = 0
        kept_count = 0
        
        for file_path in cache_files:
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if current_time - file_time > timedelta(days=max_age_days):
                os.remove(file_path)
                deleted_count += 1
            else:
                kept_count += 1
                
        logger.info(f"Cache cleanup: removed {deleted_count} old files, kept {kept_count} recent files")
        return deleted_count, kept_count
    except Exception as e:
        logger.error(f"Error cleaning cache files: {str(e)}")
        return 0, 0

def throttled_api_call(func, *args, **kwargs):
    """Utility function to throttle API calls and avoid rate limiting."""
    # Add a small random delay (100-500ms) to spread out requests
    time.sleep(random.uniform(0.1, 0.5))
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if "rate" in str(e).lower() or "limit" in str(e).lower():
            # If rate limited, add a longer delay before returning
            logger.warning(f"Rate limiting detected, adding delay: {str(e)}")
            time.sleep(random.uniform(1.0, 3.0))
        raise

def fetch_stock_data(ticker, period="1y", exchange=None, use_cache=True, max_retries=3, retry_delay=5):
    """Fetch stock data from Yahoo Finance using rate limit aware session."""
    # Format ticker with exchange suffix if Indian exchange is selected
    formatted_ticker = ticker
    if exchange and exchange in INDIAN_EXCHANGES:
        # Remove any existing .NS suffix before adding it
        if formatted_ticker.endswith('.NS'):
            formatted_ticker = formatted_ticker[:-3]
        formatted_ticker = f"{formatted_ticker}{INDIAN_EXCHANGES[exchange]}"
        logger.info(f"Formatted Indian stock ticker: {formatted_ticker}")
    
    cache_file = os.path.join(CACHE_DIR, f"{formatted_ticker}_{period}_data.csv")
    
    # Check if we have a cached version
    if use_cache and os.path.exists(cache_file):
        try:
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            # Increase cache duration to 7 days
            if datetime.now() - file_time < timedelta(days=7):
                logger.info(f"Using cached data for {formatted_ticker}")
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                return df
        except Exception as e:
            logger.error(f"Error reading cached data: {str(e)}")
            df = None
    else:
        df = None
    
    # Try to fetch data with retries
    retries = 0
    last_exception = None
    
    while retries < max_retries:
        try:
            logger.info(f"Fetching data for {formatted_ticker} (attempt {retries+1}/{max_retries})")
            
            # Add exponential backoff delay between retries
            if retries > 0:
                delay = retry_delay * (2 ** retries)  # Exponential backoff
                logger.info(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)
            else:
                # Initial delay to avoid rate limits
                time.sleep(10)
            
            # Use yf.download with proper parameters for Indian stocks
            try:
                df = yf.download(
                    tickers=[formatted_ticker],  # Pass as a list
                    period=period,
                    auto_adjust=True,
                    progress=False
                )
                
                logger.info(f"Downloaded data shape: {df.shape if df is not None else 'None'}")
                logger.info(f"Downloaded data columns: {df.columns if df is not None else 'None'}")
                
                if df is None or df.empty:
                    raise ValueError(f"No data found for ticker {formatted_ticker}")
                
                # Handle MultiIndex if present
                if isinstance(df.columns, pd.MultiIndex):
                    logger.info(f"Found MultiIndex columns: {df.columns}")
                    try:
                        # Convert MultiIndex to regular columns
                        df.columns = df.columns.get_level_values(0)
                        
                        logger.info(f"After MultiIndex handling, shape: {df.shape}")
                        logger.info(f"After MultiIndex handling, columns: {df.columns}")
                    except Exception as e:
                        logger.error(f"Error handling MultiIndex: {str(e)}")
                        # If all else fails, try to get the first column
                        df = df.iloc[:, 0]
                
                # Validate the data has required columns
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
                
                logger.info(f"Data validation passed. Final shape: {df.shape}")
                logger.info(f"Data sample:\n{df.head()}")
                
                # Only save to cache if caching is enabled
                if use_cache:
                    df.to_csv(cache_file, date_format='%Y-%m-%d')
                
                return df
                
            except Exception as e:
                logger.error(f"Error downloading data for {formatted_ticker}: {str(e)}")
                raise ValueError(f"Failed to fetch data for {formatted_ticker}. Error: {str(e)}")
            
        except Exception as e:
            last_exception = e
            logger.warning(f"Attempt {retries+1}/{max_retries} failed: {str(e)}")
            retries += 1
            
            # If it's a rate limit error, wait longer
            if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                time.sleep(60)  # Wait 60 seconds for rate limit to reset
    
    # All retries failed, check for cached data as fallback
    logger.error(f"Error fetching data for {formatted_ticker} after {max_retries} attempts: {str(last_exception)}")
    if use_cache and os.path.exists(cache_file):
        try:
            logger.warning(f"Falling back to cached data for {formatted_ticker}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df
        except Exception as e:
            logger.error(f"Error reading fallback cached data: {str(e)}")
    
    # If no cache or cache failed, raise a more user-friendly exception
    raise Exception(f"Could not fetch data for {formatted_ticker}. Yahoo Finance API may be rate-limited or the stock may not exist. Please try again later or enable caching.")

def fetch_index_data(index_ticker, period="7d", max_retries=3, retry_delay=2):
    """Fetch market index data with retry mechanism."""
    retries = 0
    last_exception = None
    
    while retries < max_retries:
        try:
            logger.info(f"Fetching index data for {index_ticker} (attempt {retries+1}/{max_retries})")
            index = yf.Ticker(index_ticker)
            df = index.history(period=period)
            
            # Return None if no data or not enough data points
            if df.empty or len(df) < 2:
                logger.warning(f"Not enough data points for {index_ticker}, got {len(df) if not df.empty else 0}")
                return None
            return df
        except Exception as e:
            last_exception = e
            logger.warning(f"Attempt {retries+1}/{max_retries} failed: {str(e)}")
            retries += 1
            
            # If it's not the last retry, wait before trying again
            if retries < max_retries:
                time.sleep(retry_delay * (retries + 1))  # Exponential backoff
    
    logger.error(f"Error fetching index data for {index_ticker} after {max_retries} attempts: {str(last_exception)}")
    return None

def get_sentiment_score(ticker, max_retries=2):
    """Fetch news articles and calculate sentiment score with retry mechanism."""
    for attempt in range(max_retries):
        try:
            news = newsapi.get_everything(
                q=ticker,
                language='en',
                sort_by='publishedAt',
                from_param=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                page_size=10  # Limit to fewer articles to avoid rate limits
            )
            
            if not news or 'articles' not in news:
                logger.warning(f"No news articles found for {ticker}")
                return 0.0
            
            if not news['articles']:
                logger.warning(f"No news articles found for {ticker}")
                return 0.0
            
            sentiment_scores = []
            for article in news['articles']:
                if article and 'title' in article and article['title']:
                    try:
                        sentiment = sentiment_analyzer.polarity_scores(article['title'])
                        sentiment_scores.append(sentiment['compound'])
                    except Exception as e:
                        logger.warning(f"Error analyzing sentiment for article: {str(e)}")
                        continue
            
            if not sentiment_scores:
                logger.warning(f"No valid sentiment scores calculated for {ticker}")
                return 0.0
                
            return np.mean(sentiment_scores)
        except Exception as e:
            logger.error(f"Error fetching sentiment for {ticker} (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
    
    # If all retries fail, return neutral sentiment
    logger.warning(f"Using neutral sentiment for {ticker} after {max_retries} failed attempts")
    return 0.0

def preprocess_data(df):
    """Preprocess stock data and calculate technical indicators."""
    logger.info(f"Preprocessing data. Initial shape: {df.shape}")
    logger.info(f"Initial columns: {df.columns}")
    
    # Convert all numeric columns to float
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            logger.info(f"Converted {col} to numeric. NaN count: {df[col].isna().sum()}")
    
    # Basic indicators
    df['Daily_Return'] = df['Close'].pct_change(fill_method=None)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    logger.info(f"Added basic indicators. Shape: {df.shape}")
    
    # Mean reversion indicators
    df['Price_SMA20_Ratio'] = df['Close'] / df['SMA_20']
    df['Price_SMA20_Pct_Diff'] = (df['Close'] - df['SMA_20']) / df['SMA_20'] * 100
    logger.info(f"Added mean reversion indicators. Shape: {df.shape}")
    
    # Calculate Bollinger Bands (useful for reversion analysis)
    df['SMA_20_Std'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (df['SMA_20_Std'] * 2)
    df['Lower_Band'] = df['SMA_20'] - (df['SMA_20_Std'] * 2)
    df['BB_Position'] = (df['Close'] - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band'])
    logger.info(f"Added Bollinger Bands. Shape: {df.shape}")
    
    # Calculate RSI (Relative Strength Index) - reversion tendency indicator
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    logger.info(f"Added RSI. Shape: {df.shape}")
    
    # Create reversion opportunity score
    # Higher scores indicate potential for mean reversion
    df['Reversion_Score'] = 0.0
    
    # If price is significantly above SMA, potential for downward reversion
    df.loc[df['Price_SMA20_Pct_Diff'] > 10, 'Reversion_Score'] += 1.0
    
    # If price is significantly below SMA, potential for upward reversion
    df.loc[df['Price_SMA20_Pct_Diff'] < -10, 'Reversion_Score'] += 1.0
    
    # If BB position is extreme, potential for reversion
    df.loc[df['BB_Position'] > 0.9, 'Reversion_Score'] += 1.0
    df.loc[df['BB_Position'] < 0.1, 'Reversion_Score'] += 1.0
    
    # If RSI is extreme, potential for reversion
    df.loc[df['RSI'] > 70, 'Reversion_Score'] += 1.0
    df.loc[df['RSI'] < 30, 'Reversion_Score'] += 1.0
    logger.info(f"Added reversion scores. Shape: {df.shape}")
    
    # Target variable: 1 if price goes up next day, 0 otherwise
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Drop NaN values
    df = df.dropna()
    logger.info(f"After dropping NaN values. Final shape: {df.shape}")
    logger.info(f"Final columns: {df.columns}")
    logger.info(f"Data sample:\n{df.head()}")
    
    return df

def train_model(df):
    """Train hybrid model on the data."""
    try:
        # Initialize hybrid model
        model = HybridStockPredictor(
            sequence_length=30,
            lstm_units=50,
            dropout_rate=0.2,
            learning_rate=0.001
        )
        
        # Prepare data
        X_linear, X_lstm, y = model.prepare_data(df)
        
        # Train model
        history = model.train(X_linear, X_lstm, y)
        
        # Make predictions on training data
        combined_pred, linear_pred, lstm_pred, svr_pred, rf_pred = model.predict(X_linear, X_lstm)
        
        # Evaluate model
        metrics = model.evaluate(y, combined_pred)
        
        # Return all predictions for further analysis
        return model, metrics, history, (combined_pred, linear_pred, lstm_pred, svr_pred, rf_pred)
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

def predict_trend(model, df, sentiment_score):
    """Make prediction for next day's trend."""
    try:
        # Prepare data for prediction
        X_linear, X_lstm, _ = model.prepare_data(df)
        
        # Get predictions
        combined_pred, linear_pred, lstm_pred, svr_pred, rf_pred = model.predict(X_linear[-1:], X_lstm[-1:])
        
        # Calculate confidence based on model agreement and sentiment
        preds = np.array([linear_pred[0], lstm_pred[0], svr_pred[0], rf_pred[0]])
        model_agreement = 1 - np.std(preds) / (np.abs(preds).mean() + 1e-8)
        confidence = 0.7 * model_agreement + 0.3 * (sentiment_score + 1) / 2
        
        # Determine trend (1 for up, 0 for down)
        prediction = 1 if combined_pred[0] > df['Close'].iloc[-1] else 0
        
        # Return all model predictions for display
        return prediction, confidence, {
            'combined': combined_pred[0],
            'linear': linear_pred[0],
            'lstm': lstm_pred[0],
            'svr': svr_pred[0],
            'rf': rf_pred[0]
        }
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise

def plot_trends(df):
    """Plot stock price, SMA, and Bollinger Bands."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price and indicators on upper chart
    ax1.plot(df.index, df['Close'], label='Stock Price')
    ax1.plot(df.index, df['SMA_20'], label='20-day SMA', alpha=0.7)
    
    # Plot Bollinger Bands if available
    if 'Upper_Band' in df.columns and 'Lower_Band' in df.columns:
        ax1.fill_between(df.index, df['Lower_Band'], df['Upper_Band'], alpha=0.2, color='gray')
        ax1.plot(df.index, df['Upper_Band'], 'g--', alpha=0.5)
        ax1.plot(df.index, df['Lower_Band'], 'r--', alpha=0.5)
    
    ax1.set_title('Stock Price, SMA and Bollinger Bands')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot RSI on lower chart if available
    if 'RSI' in df.columns:
        ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax2.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    return fig

def fetch_multiple_indian_stocks(stocks_list, start_date="2023-01-01", end_date="2025-05-03", export_csv=True):
    """
    Fetch historical data for multiple Indian stocks and optionally export to CSV.
    
    Args:
        stocks_list (list): List of stock tickers (without .NS suffix)
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        export_csv (bool): Whether to export data to CSV files
    
    Returns:
        dict: Dictionary containing DataFrames for each stock
    """
    # Add .NS suffix to all tickers
    formatted_stocks = [f"{stock}.NS" for stock in stocks_list]
    logger.info(f"Fetching data for {len(formatted_stocks)} Indian stocks")
    
    # Create directory for exported data if needed
    if export_csv:
        export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_data")
        os.makedirs(export_dir, exist_ok=True)
    
    # Dictionary to store all stock data
    stock_data = {}
    
    # Fetch data for each stock with improved error handling
    for stock in formatted_stocks:
        try:
            logger.info(f"Fetching data for {stock}")
            
            # Use yf.download with proper parameters and retry mechanism
            max_retries = 3
            retry_delay = 5
            retries = 0
            
            while retries < max_retries:
                try:
                    # Add delay between requests to avoid rate limiting
                    if retries > 0:
                        time.sleep(retry_delay * (2 ** retries))  # Exponential backoff
                    
                    df = yf.download(
                        tickers=[stock],  # Pass as a list
                        start=start_date,
                        end=end_date,
                        auto_adjust=True,
                        progress=False
                    )
                    
                    if df is None or df.empty:
                        logger.warning(f"No data found for {stock} on attempt {retries + 1}")
                        retries += 1
                        continue
                    
                    # Handle MultiIndex if present
                    if isinstance(df.columns, pd.MultiIndex):
                        try:
                            if stock in df.columns.levels[1]:
                                df = df[stock]
                            else:
                                df = df.xs(df.columns.levels[0][0], level=0, axis=1)
                        except Exception as e:
                            logger.error(f"Error handling MultiIndex for {stock}: {str(e)}")
                            df = df.iloc[:, 0]
                    
                    # Validate required columns
                    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if not all(col in df.columns for col in required_columns):
                        logger.warning(f"Missing required columns for {stock}")
                        retries += 1
                        continue
                    
                    # Store the data
                    stock_data[stock] = df
                    
                    # Export to CSV if requested
                    if export_csv:
                        csv_file = os.path.join(export_dir, f"{stock}_data.csv")
                        df.to_csv(csv_file)
                        logger.info(f"Exported data for {stock} to {csv_file}")
                    
                    # Calculate basic statistics
                    logger.info(f"Data summary for {stock}:")
                    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
                    logger.info(f"Number of trading days: {len(df)}")
                    logger.info(f"Average daily volume: {df['Volume'].mean():,.0f}")
                    
                    # Calculate returns
                    df['Daily_Return'] = df['Close'].pct_change()
                    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
                    
                    logger.info(f"Total return: {df['Cumulative_Return'].iloc[-1]:.2%}")
                    
                    # Break the retry loop if successful
                    break
                    
                except Exception as e:
                    logger.error(f"Error fetching data for {stock} (attempt {retries + 1}): {str(e)}")
                    retries += 1
                    if retries == max_retries:
                        logger.error(f"Failed to fetch data for {stock} after {max_retries} attempts")
            
            # Add a delay between stocks to avoid rate limiting
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Error processing {stock}: {str(e)}")
            continue
    
    if not stock_data:
        logger.error("No stock data was successfully fetched")
        return None
    
    return stock_data

def analyze_stock_data(stock_data):
    """
    Analyze the fetched stock data and generate insights.
    
    Args:
        stock_data (dict): Dictionary containing DataFrames for each stock
    """
    if not stock_data:
        logger.error("No stock data available for analysis")
        return None
    
    # Create a summary DataFrame
    summary_data = []
    
    for stock, df in stock_data.items():
        try:
            # Calculate key metrics
            total_return = df['Cumulative_Return'].iloc[-1]
            avg_volume = df['Volume'].mean()
            volatility = df['Daily_Return'].std() * np.sqrt(252)  # Annualized volatility
            
            # Calculate additional metrics
            max_drawdown = (df['Close'] / df['Close'].cummax() - 1).min()
            sharpe_ratio = (df['Daily_Return'].mean() * 252) / (df['Daily_Return'].std() * np.sqrt(252))
            
            summary_data.append({
                'Stock': stock,
                'Total_Return': total_return,
                'Avg_Volume': avg_volume,
                'Volatility': volatility,
                'Max_Drawdown': max_drawdown,
                'Sharpe_Ratio': sharpe_ratio,
                'Start_Price': df['Close'].iloc[0],
                'End_Price': df['Close'].iloc[-1]
            })
        except Exception as e:
            logger.error(f"Error analyzing data for {stock}: {str(e)}")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by total return
    summary_df = summary_df.sort_values('Total_Return', ascending=False)
    
    # Print summary
    logger.info("\nStock Performance Summary:")
    logger.info(summary_df.to_string())
    
    return summary_df

class KerasLSTMModel:
    def __init__(self, sequence_length=30, lstm_units=50, dropout_rate=0.2, learning_rate=0.001):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_columns = ['Close', 'Volume', 'SMA_20', 'RSI', 'BB_Position']

    def prepare_data(self, df):
        df = df.copy()
        df['Target'] = df['Close'].shift(-1)
        df = df.dropna()
        scaled_features = self.scaler.fit_transform(df[self.feature_columns])
        X = []
        for i in range(len(scaled_features) - self.sequence_length):
            X.append(scaled_features[i:(i + self.sequence_length)])
        X = np.array(X)
        y = df['Target'].values[self.sequence_length:]
        return X, y

    def build_model(self, input_shape):
        model = keras.Sequential([
            layers.LSTM(self.lstm_units, input_shape=input_shape, dropout=self.dropout_rate, return_sequences=False),
            layers.Dense(1)
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def train(self, df, epochs=50, batch_size=32, validation_split=0.2):
        X, y = self.prepare_data(df)
        self.model = self.build_model((X.shape[1], X.shape[2]))
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0)
        return history

    def predict(self, df):
        X, _ = self.prepare_data(df)
        return self.model.predict(X)

def compute_lstm_saliency_map(keras_model, df):
    # Prepare the last input sequence
    X, _ = keras_model.prepare_data(df)
    if len(X) == 0:
        return None, None
    last_input = X[-1:]
    input_tensor = tf.convert_to_tensor(last_input)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        pred = keras_model.model(input_tensor)
    grads = tape.gradient(pred, input_tensor)
    # Aggregate gradients (absolute value) over features
    saliency = tf.reduce_mean(tf.abs(grads), axis=2).numpy()[0]  # shape: (sequence_length,)
    # For heatmap, also return the input sequence for context
    return saliency, last_input[0]

# Helper functions for Streamlit integration

def train_keras_lstm_model(df, epochs=50, batch_size=32):
    keras_model = KerasLSTMModel()
    history = keras_model.train(df, epochs=epochs, batch_size=batch_size)
    return keras_model, history

def predict_with_keras_lstm(keras_model, df):
    preds = keras_model.predict(df)
    return preds

def main():
    try:
        st.set_page_config(page_title="Stock Price Predictor", layout="wide")
        
        st.title("📈 Stock Price Predictor")
        st.markdown("""
            <style>
            .big-metric {font-size: 2.2em; font-weight: bold;}
            .metric-up {color: #27ae60;}
            .metric-down {color: #c0392b;}
            .section-header {font-size: 1.3em; font-weight: 600; margin-top: 1.5em;}
            .divider {border-top: 1px solid #eee; margin: 1em 0;}
            </style>
        """, unsafe_allow_html=True)
        st.write("Predict whether a stock's price will rise or fall based on historical data and sentiment analysis.")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Sidebar for user input
        st.sidebar.header("Input Parameters")
        
        # Add tabs for different input methods
        input_method = st.sidebar.radio("Select Input Method", ["Enter Ticker", "Choose from Popular Indian Stocks"])
        
        if input_method == "Enter Ticker":
            ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, RELIANCE)", "AAPL").upper()
            if not ticker:
                st.error("Please enter a valid stock ticker")
                return
        else:
            # List popular Indian stocks by sector
            sector = st.sidebar.selectbox("Select Sector", list(POPULAR_INDIAN_STOCKS.keys()))
            stock_options = POPULAR_INDIAN_STOCKS[sector]
            ticker = st.sidebar.selectbox("Select Stock", stock_options)
            
            # Auto-select NSE for Indian stocks
            exchange = "NSE"
            st.sidebar.write(f"Exchange automatically set to: {exchange}")
        
        # Add option for selecting exchange for Indian stocks
        if input_method == "Enter Ticker":
            st.sidebar.subheader("Exchange Selection")
            exchange = st.sidebar.selectbox(
                "Select Exchange (for Indian stocks)",
                [None] + list(INDIAN_EXCHANGES.keys()),
                format_func=lambda x: "International" if x is None else x
            )
        
        # Add period selector
        period_options = {
            "1 Month": "1mo",
            "3 Months": "3mo",
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y",
            "5 Years": "5y"
        }
        selected_period = st.sidebar.selectbox("Select Time Period", list(period_options.keys()))
        period = period_options[selected_period]
        
        # Add caching option - now enabled by default
        use_cache = st.sidebar.checkbox("Use local cache", value=True, 
                                      help="Enable to cache data locally for faster loading and to avoid rate limiting. Disable to always fetch fresh data.")
        
        # Show cache info and add button to clean cache
        try:
            cache_files = glob.glob(os.path.join(CACHE_DIR, "*_data.csv"))
            if cache_files:
                oldest_file = min(cache_files, key=lambda x: os.path.getmtime(x))
                oldest_time = datetime.fromtimestamp(os.path.getmtime(oldest_file))
                newest_file = max(cache_files, key=lambda x: os.path.getmtime(x))
                newest_time = datetime.fromtimestamp(os.path.getmtime(newest_file))
                
                st.sidebar.markdown(f"**Cache Info:**")
                st.sidebar.markdown(f"• {len(cache_files)} cached stock files")
                st.sidebar.markdown(f"• Oldest: {oldest_time.strftime('%Y-%m-%d %H:%M')}")
                st.sidebar.markdown(f"• Newest: {newest_time.strftime('%Y-%m-%d %H:%M')}")
        except Exception as e:
            logger.error(f"Error displaying cache info: {str(e)}")
            st.sidebar.warning("Could not display cache information")
        
        if st.sidebar.button("Clear Cached Data"):
            try:
                deleted_count, kept_count = clean_cache_files()
                st.sidebar.success(f"Cache cleared successfully! Removed {deleted_count} old files, kept {kept_count} recent files")
                st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
            except Exception as e:
                st.sidebar.error(f"Error clearing cache: {str(e)}")
        
        # Rate limit warning
        st.sidebar.markdown("---")
        st.sidebar.warning("⚠️ Yahoo Finance enforces rate limits. The app now includes automatic rate limiting and caching to avoid disruptions.")
        
        # Main content area
        col1, col2 = st.columns([2, 1], gap="large")
        
        try:
            with st.spinner("Fetching stock data..."):
                try:
                    # Format ticker for display
                    formatted_ticker = ticker
                    if exchange and exchange in INDIAN_EXCHANGES:
                        # Remove any existing .NS suffix before adding it
                        if formatted_ticker.endswith('.NS'):
                            formatted_ticker = formatted_ticker[:-3]
                        formatted_ticker = f"{formatted_ticker}{INDIAN_EXCHANGES[exchange]}"
                        logger.info(f"Formatted Indian stock ticker: {formatted_ticker}")
                    
                    # Cache path for this specific request
                    cache_file = os.path.join(CACHE_DIR, f"{formatted_ticker}_{period}_data.csv")
                    
                    # Try to load from cache first if enabled
                    if use_cache and os.path.exists(cache_file):
                        try:
                            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                            if datetime.now() - file_time < timedelta(days=1):
                                logger.info(f"Using cached data for {formatted_ticker}")
                                # Read the CSV file with proper date parsing
                                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                                # Rename index to 'Date' for consistency
                                df.index.name = 'Date'
                        except Exception as e:
                            logger.error(f"Error reading cached data: {str(e)}")
                            df = None
                    else:
                        df = None
                    
                    if df is None:
                        # Use yf.download with the rate limited session
                        logger.info(f"Downloading data for {formatted_ticker} using rate-limited session")
                        try:
                            df = yf.download(
                                tickers=[formatted_ticker],  # Pass as a list
                                period=period,
                                auto_adjust=True,
                                progress=False
                            )
                            
                            logger.info(f"Downloaded data shape: {df.shape if df is not None else 'None'}")
                            logger.info(f"Downloaded data columns: {df.columns if df is not None else 'None'}")
                            
                            if df is None:
                                raise ValueError(f"Failed to download data for {formatted_ticker}")
                            
                            if df.empty:
                                raise ValueError(f"No data found for ticker {formatted_ticker}")
                            
                            # Handle MultiIndex if present
                            if isinstance(df.columns, pd.MultiIndex):
                                logger.info(f"Found MultiIndex columns: {df.columns}")
                                try:
                                    # Convert MultiIndex to regular columns
                                    df.columns = df.columns.get_level_values(0)
                                    
                                    logger.info(f"After MultiIndex handling, shape: {df.shape}")
                                    logger.info(f"After MultiIndex handling, columns: {df.columns}")
                                except Exception as e:
                                    logger.error(f"Error handling MultiIndex: {str(e)}")
                                    # If all else fails, try to get the first column
                                    df = df.iloc[:, 0]
                            
                            # Validate the data has required columns
                            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                            missing_columns = [col for col in required_columns if col not in df.columns]
                            if missing_columns:
                                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
                            
                            logger.info(f"Data validation passed. Final shape: {df.shape}")
                            logger.info(f"Data sample:\n{df.head()}")
                        except Exception as e:
                            logger.error(f"Error downloading data for {formatted_ticker}: {str(e)}")
                            raise ValueError(f"Failed to fetch data for {formatted_ticker}. Error: {str(e)}")
                        
                        # Save to cache if caching is enabled
                        if use_cache:
                            # Ensure index is named 'Date' before saving
                            df.index.name = 'Date'
                            df.to_csv(cache_file, date_format='%Y-%m-%d')
                    
                    # Preprocess data
                    df = preprocess_data(df)
                    
                    if df.empty:
                        st.error(f"No data found for ticker {formatted_ticker}")
                        return
                except Exception as e:
                    st.error(f"Error fetching stock data: {str(e)}")
                    st.warning("Try enabling caching or wait a few minutes before trying again.")
                    return
                
                try:
                    # Display stock info
                    stock = yf.Ticker(formatted_ticker)
                    try:
                        info = stock.info
                    except:
                        # Create a minimal info dict if we can't get the actual info
                        info = {}
                    
                    with col1:
                        st.subheader(f"Stock Information: {formatted_ticker}")
                        
                        # Only show available information
                        if 'longName' in info:
                            st.write(f"Company Name: {info.get('longName', 'N/A')}")
                        
                        # Handle different price fields for different markets
                        current_price = "N/A"
                        if 'currentPrice' in info:
                            current_price = info['currentPrice']
                        elif 'regularMarketPrice' in info:
                            current_price = info['regularMarketPrice']
                        elif len(df) > 0:
                            # Use the last close price from our data if available
                            current_price = df['Close'].iloc[-1]
                        
                        # Determine currency symbol based on exchange
                        currency_symbol = "₹" if exchange else "$"
                        
                        st.write(f"Current Price: {current_price if current_price == 'N/A' else f'{currency_symbol}{current_price:.2f}'}")
                        
                        if 'marketCap' in info:
                            st.write(f"Market Cap: {currency_symbol}{info.get('marketCap', 0):,.0f}")
                        
                        if 'fiftyTwoWeekHigh' in info:
                            st.write(f"52 Week High: {currency_symbol}{info.get('fiftyTwoWeekHigh'):.2f}")
                        
                        if 'fiftyTwoWeekLow' in info:
                            st.write(f"52 Week Low: {currency_symbol}{info.get('fiftyTwoWeekLow'):.2f}")
                        
                        # Display volume information if available
                        if 'volume' in info:
                            st.write(f"Volume: {info['volume']:,.0f}")
                        elif 'regularMarketVolume' in info:
                            st.write(f"Volume: {info['regularMarketVolume']:,.0f}")
                        
                        # Display exchange information
                        if exchange:
                            st.write(f"Exchange: {exchange}")
                        
                        # Try to display relevant market indices if possible
                        try:
                            st.subheader("Market Indices")
                            indices = MARKET_INDICES["India"] if exchange else MARKET_INDICES["USA"]
                            
                            # Use yf.download for batch downloading of indices
                            indices_tickers = list(indices.values())
                            indices_data_all = None
                            indices_cache_file = os.path.join(CACHE_DIR, "market_indices_data.csv")
                            # Try to load indices from cache or download
                            if use_cache and os.path.exists(indices_cache_file):
                                file_time = datetime.fromtimestamp(os.path.getmtime(indices_cache_file))
                                if datetime.now() - file_time < timedelta(hours=6):  # 6 hours for indices
                                    try:
                                        indices_data_all = pd.read_csv(indices_cache_file, index_col=0, parse_dates=True)
                                        indices_data_all.index = pd.to_datetime(indices_data_all.index)
                                        logger.info(f"Loaded indices data from cache. Shape: {indices_data_all.shape}")
                                    except Exception as e:
                                        logger.error(f"Error reading indices cache: {str(e)}")
                                        indices_data_all = None
                            if indices_data_all is None:
                                # Try batch download first
                                try:
                                    logger.info(f"Batch downloading indices: {indices_tickers}")
                                    batch_data = yf.download(
                                        tickers=indices_tickers,
                                        period="2d",
                                        auto_adjust=True,
                                        group_by='ticker',
                                        progress=False
                                    )
                                    # If batch_data is a MultiIndex DataFrame, reformat
                                    if isinstance(batch_data.columns, pd.MultiIndex):
                                        indices_data_all = batch_data
                                    else:
                                        # If only one index, wrap in MultiIndex
                                        indices_data_all = pd.concat({indices_tickers[0]: batch_data}, axis=1)
                                    logger.info(f"Batch download successful. Shape: {indices_data_all.shape}")
                                    # Save to cache if valid
                                    if use_cache and indices_data_all is not None and not indices_data_all.empty:
                                        indices_data_all.to_csv(indices_cache_file)
                                        logger.info(f"Saved indices data to cache: {indices_cache_file}")
                                except Exception as e:
                                    logger.error(f"Batch download failed: {str(e)}")
                                    indices_data_all = None
                                # If batch fails, try individual download with retry
                                if indices_data_all is None:
                                    indices_data_list = []
                                    for idx_ticker in indices_tickers:
                                        for attempt in range(3):
                                            try:
                                                logger.info(f"Downloading data for index {idx_ticker}, attempt {attempt+1}")
                                                idx_data = yf.download(
                                                    tickers=[idx_ticker],
                                                    period="2d",
                                                    auto_adjust=True,
                                                    progress=False
                                                )
                                                if idx_data is not None and not idx_data.empty:
                                                    idx_data.columns = pd.MultiIndex.from_product([[idx_ticker], idx_data.columns])
                                                    indices_data_list.append(idx_data)
                                                    logger.info(f"Successfully downloaded data for {idx_ticker}")
                                                    break
                                                else:
                                                    logger.warning(f"No data found for index {idx_ticker} on attempt {attempt+1}")
                                            except Exception as e:
                                                logger.error(f"Error downloading index {idx_ticker} on attempt {attempt+1}: {str(e)}")
                                                time.sleep(2)
                                                continue
                                    if indices_data_list:
                                        try:
                                            indices_data_all = pd.concat(indices_data_list, axis=1)
                                            logger.info(f"Combined indices data shape: {indices_data_all.shape}")
                                            if use_cache:
                                                indices_data_all.to_csv(indices_cache_file)
                                                logger.info(f"Saved indices data to cache: {indices_cache_file}")
                                        except Exception as e:
                                            logger.error(f"Error combining indices data: {str(e)}")
                                            indices_data_all = None
                            # Display indices data
                            indices_display = []
                            if indices_data_all is not None and not indices_data_all.empty:
                                for name, idx_ticker in indices.items():
                                    try:
                                        if idx_ticker in indices_data_all.columns.get_level_values(0):
                                            idx_data = indices_data_all[idx_ticker]
                                            if len(idx_data) >= 2:
                                                latest_price = idx_data['Close'].iloc[-1]
                                                prev_price = idx_data['Close'].iloc[-2]
                                                pct_change = ((latest_price - prev_price) / prev_price) * 100
                                                color = "green" if pct_change >= 0 else "red"
                                                indices_display.append({
                                                    "name": name,
                                                    "price": latest_price,
                                                    "change": pct_change,
                                                    "color": color
                                                })
                                                logger.info(f"Processed index {name} successfully")
                                    except Exception as e:
                                        logger.error(f"Error processing index {name}: {str(e)}")
                                        continue
                            # Create a small dataframe for the indices
                            if indices_display:
                                st.write("Latest market index values:")
                                for idx in indices_display:
                                    st.write(f"{idx['name']}: {idx['price']:.2f} ({idx['change']:.2f}%)")
                            else:
                                st.warning("Market index data currently unavailable. Please try again later. (Check your internet connection or Yahoo Finance availability.)")
                                logger.warning(f"No valid indices data to display. Last attempted tickers: {indices}")
                        except Exception as e:
                            logger.error(f"Error showing indices: {str(e)}")
                            st.warning("Could not load market indices data. Please try again later.")
                    
                    # Plot trends if possible
                    try:
                        fig = plot_trends(df)
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error plotting trends: {str(e)}")
                        st.warning("Could not generate price trend visualization")
                except Exception as e:
                    st.warning(f"Could not load complete stock information: {str(e)}")
                    st.info("Proceeding with available data for analysis.")
                
                # Get sentiment and make prediction
                try:
                    with st.spinner("Analyzing news sentiment..."):
                        try:
                            sentiment_score = get_sentiment_score(ticker)
                        except Exception as e:
                            logger.error(f"Error getting sentiment: {str(e)}")
                            sentiment_score = 0.0
                            st.warning("News sentiment analysis unavailable. Using neutral sentiment.")
                        
                        with st.spinner("Training prediction model..."):
                            try:
                                # Check if we have enough data for valid model training
                                if len(df) < 30:  # Require at least 30 days of data
                                    st.warning(f"Not enough historical data for reliable predictions. Found only {len(df)} data points, need at least 30.")
                                    st.info("Try selecting a longer time period or a different stock.")
                                    return
                                    
                                # Check if all rows have the same target value, which can cause issues
                                if df['Target'].nunique() == 1:
                                    st.warning(f"All target values are the same ({df['Target'].iloc[0]}). Model won't be able to learn patterns.")
                                    st.info("Try selecting a different time period or stock.")
                                    return
                                
                                model, metrics, training_history, predictions = train_model(df)
                                prediction, confidence, model_predictions = predict_trend(model, df, sentiment_score)
                                
                                # Train and predict with Keras LSTM
                                keras_model, keras_history = train_keras_lstm_model(df)
                                keras_preds = predict_with_keras_lstm(keras_model, df)
                                next_keras_pred = keras_preds[-1][0] if len(keras_preds) > 0 else None

                                # Display results
                                with col2:
                                    st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)
                                    # Prediction indicator
                                    prediction_color = "#27ae60" if prediction == 1 else "#c0392b"
                                    prediction_text = "UP 📈" if prediction == 1 else "DOWN 📉"
                                    st.markdown(f"<div class='big-metric' style='color:{prediction_color}'>{prediction_text}</div>", unsafe_allow_html=True)
                                    # Confidence meter
                                    st.markdown('<div class="section-header">Confidence Score</div>', unsafe_allow_html=True)
                                    st.progress(min(max(confidence, 0.0), 1.0))
                                    st.write(f"{confidence:.2%}")
                                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                                    # Show only combined and Keras LSTM predictions
                                    st.markdown('<div class="section-header">Model Predictions (next close)</div>', unsafe_allow_html=True)
                                    st.markdown(f"<b>Hybrid Combined:</b> <span style='color:#2980b9'>{model_predictions['combined']:.2f}</span>", unsafe_allow_html=True)
                                    if next_keras_pred is not None:
                                        st.markdown(f"<b>Keras LSTM:</b> <span style='color:#8e44ad'>{next_keras_pred:.2f}</span>", unsafe_allow_html=True)
                                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                                    # Model Performance
                                    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
                                    st.write(f"RMSE: {metrics['RMSE']:.2f}")
                                    st.write(f"MAPE: {metrics['MAPE']:.2f}%")
                                    st.write(f"Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%")
                                    st.write("Sentiment Score:", f"{sentiment_score:.2f}")
                                    # Sentiment interpretation (keep concise)
                                    if sentiment_score > 0.2:
                                        st.success("News Sentiment: Very Positive 😊")
                                    elif sentiment_score > 0:
                                        st.info("News Sentiment: Slightly Positive 🙂")
                                    elif sentiment_score < -0.2:
                                        st.error("News Sentiment: Very Negative 😟")
                                    elif sentiment_score < 0:
                                        st.warning("News Sentiment: Slightly Negative 😕")
                                    else:
                                        st.write("News Sentiment: Neutral 😐")
                                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                                    # Placeholder for tf-explain GradCAM visualization
                                    st.markdown('<div class="section-header">Model Explanation (Saliency Map)</div>', unsafe_allow_html=True)
                                    saliency, input_seq = compute_lstm_saliency_map(keras_model, df)
                                    if saliency is not None:
                                        fig, ax = plt.subplots(figsize=(8, 2))
                                        ax.plot(saliency, label='Saliency (importance)')
                                        ax.set_title('LSTM Saliency Map (last input sequence)')
                                        ax.set_xlabel('Time Step')
                                        ax.set_ylabel('Importance')
                                        ax.legend()
                                        st.pyplot(fig)
                                    else:
                                        st.info('Not enough data for saliency map or model not trained.')
                                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                                    # Mean Reversion Analysis
                                    st.markdown('<div class="section-header">Mean Reversion Analysis</div>', unsafe_allow_html=True)
                                    if 'Reversion_Score' in df.columns:
                                        reversion_score = df['Reversion_Score'].iloc[-1]
                                        st.write(f"Reversion Score: {reversion_score:.1f}/6.0")
                                        if reversion_score >= 4:
                                            st.success("Strong potential for price reversion 🔄")
                                        elif reversion_score >= 2:
                                            st.info("Moderate potential for price reversion ↔️")
                                        else:
                                            st.write("Low potential for price reversion ➡️")
                                    if 'RSI' in df.columns:
                                        rsi_value = df['RSI'].iloc[-1]
                                        st.write(f"RSI: {rsi_value:.1f}")
                                        if rsi_value > 70:
                                            st.warning("Overbought condition - potential downward reversion")
                                        elif rsi_value < 30:
                                            st.info("Oversold condition - potential upward reversion")
                                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                                    # Training History
                                    st.markdown('<div class="section-header">Model Training History</div>', unsafe_allow_html=True)
                                    if training_history and isinstance(training_history, dict):
                                        fig, ax = plt.subplots(figsize=(10, 4))
                                        epochs = training_history.get('epochs', [])
                                        train_loss = training_history.get('train_loss', [])
                                        val_loss = training_history.get('val_loss', [])
                                        if epochs and train_loss and val_loss:
                                            ax.plot(epochs, train_loss, label='Training Loss')
                                            ax.plot(epochs, val_loss, label='Validation Loss')
                                            ax.set_title('Model Loss Over Time')
                                            ax.set_xlabel('Epoch')
                                            ax.set_ylabel('Loss')
                                            ax.legend()
                                            st.pyplot(fig)
                            except ValueError as ve:
                                st.error(f"Value Error in model: {str(ve)}")
                                logger.error(f"Value Error in model: {str(ve)}\n{traceback.format_exc()}")
                            except IndexError as ie:
                                st.error(f"Index Error in model: {str(ie)}")
                                logger.error(f"Index Error in model: {str(ie)}\n{traceback.format_exc()}")
                            except Exception as e:
                                st.error(f"Error in model training or prediction: {str(e)}")
                                logger.error(f"Model error: {str(e)}\n{traceback.format_exc()}")
                except Exception as e:
                    st.error(f"Error in analysis: {str(e)}")
                    logger.error(f"Analysis error: {str(e)}\n{traceback.format_exc()}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
            return
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Application error: {str(e)}\n{traceback.format_exc()}")
        st.warning("Try enabling caching or wait a few minutes before trying again.")

if __name__ == "__main__":
    main() 