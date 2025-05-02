import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import os
import traceback
import glob
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# NewsAPI configuration
NEWS_API_KEY = "faa59dc97acf42f1acdada2e9c9e4155"  # Replace with your actual API key
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# Define a temporary directory for cache files
CACHE_DIR = os.path.join(tempfile.gettempdir(), "stock_predictor_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Indian stock exchanges and their Yahoo Finance suffixes
INDIAN_EXCHANGES = {
    "NSE": ".NS",  # National Stock Exchange
    "BSE": ".BO",  # Bombay Stock Exchange
}

# Popular Indian stocks by sector
POPULAR_INDIAN_STOCKS = {
    "Banking": ["HDFCBANK", "SBIN", "ICICIBANK", "KOTAKBANK", "AXISBANK"],
    "IT": ["TCS", "INFY", "WIPRO", "TECHM", "HCLTECH"],
    "Automobile": ["MARUTI", "TATAMOTORS", "M&M", "HEROMOTOCO", "BAJAJ-AUTO"],
    "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "BIOCON"],
    "Energy": ["RELIANCE", "ONGC", "POWERGRID", "NTPC", "BPCL"],
    "Consumer Goods": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "MARICO"],
    "Metals": ["TATASTEEL", "HINDALCO", "JSWSTEEL", "COALINDIA", "NMDC"]
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

def clean_cache_files():
    """Delete all cached stock data files that are older than 1 day."""
    try:
        cache_pattern = os.path.join(CACHE_DIR, "*_data.csv")
        cache_files = glob.glob(cache_pattern)
        
        current_time = datetime.now()
        deleted_count = 0
        
        for file_path in cache_files:
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if current_time - file_time > timedelta(days=1):
                os.remove(file_path)
                deleted_count += 1
                
        logger.info(f"Cleaned {deleted_count} old cache files")
    except Exception as e:
        logger.error(f"Error cleaning cache files: {str(e)}")

def fetch_stock_data(ticker, period="1y", exchange=None, use_cache=True):
    """Fetch stock data from Yahoo Finance and optionally cache it locally."""
    # Format ticker with exchange suffix if Indian exchange is selected
    formatted_ticker = ticker
    if exchange and exchange in INDIAN_EXCHANGES:
        formatted_ticker = f"{ticker}{INDIAN_EXCHANGES[exchange]}"
    
    cache_file = os.path.join(CACHE_DIR, f"{formatted_ticker}_data.csv")
    
    # Check if we have a cached version
    if use_cache and os.path.exists(cache_file):
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - file_time < timedelta(days=1):
            logger.info(f"Using cached data for {formatted_ticker}")
            return pd.read_csv(cache_file)
    
    try:
        logger.info(f"Fetching data for {formatted_ticker}")
        stock = yf.Ticker(formatted_ticker)
        df = stock.history(period=period)
        
        if df.empty:
            raise ValueError(f"No data found for ticker {formatted_ticker}")
        
        # Only save to cache if caching is enabled
        if use_cache:
            df.to_csv(cache_file)
        
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {formatted_ticker}: {str(e)}")
        if use_cache and os.path.exists(cache_file):
            logger.warning(f"Falling back to cached data for {formatted_ticker}")
            return pd.read_csv(cache_file)
        raise

def fetch_index_data(index_ticker, period="7d"):
    """Fetch market index data."""
    try:
        index = yf.Ticker(index_ticker)
        df = index.history(period=period)
        # Return None if no data or not enough data points
        if df.empty or len(df) < 2:
            logger.warning(f"Not enough data points for {index_ticker}, got {len(df) if not df.empty else 0}")
            return None
        return df
    except Exception as e:
        logger.error(f"Error fetching index data for {index_ticker}: {str(e)}")
        return None

def get_sentiment_score(ticker):
    """Fetch news articles and calculate sentiment score."""
    try:
        news = newsapi.get_everything(
            q=ticker,
            language='en',
            sort_by='publishedAt',
            from_param=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        )
        
        if not news['articles']:
            return 0.0
        
        sentiment_scores = []
        for article in news['articles']:
            if article['title']:
                sentiment = sentiment_analyzer.polarity_scores(article['title'])
                sentiment_scores.append(sentiment['compound'])
        
        return np.mean(sentiment_scores)
    except Exception as e:
        logger.error(f"Error fetching sentiment for {ticker}: {str(e)}")
        return 0.0

def preprocess_data(df):
    """Preprocess stock data and calculate technical indicators."""
    # Basic indicators
    df['Daily_Return'] = df['Close'].pct_change()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Mean reversion indicators
    df['Price_SMA20_Ratio'] = df['Close'] / df['SMA_20']
    df['Price_SMA20_Pct_Diff'] = (df['Close'] - df['SMA_20']) / df['SMA_20'] * 100
    
    # Calculate Bollinger Bands (useful for reversion analysis)
    df['SMA_20_Std'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (df['SMA_20_Std'] * 2)
    df['Lower_Band'] = df['SMA_20'] - (df['SMA_20_Std'] * 2)
    df['BB_Position'] = (df['Close'] - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band'])
    
    # Calculate RSI (Relative Strength Index) - reversion tendency indicator
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
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
    
    # Target variable: 1 if price goes up next day, 0 otherwise
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def train_model(df):
    """Train Random Forest model on the data."""
    # Include reversion indicators
    features = ['Daily_Return', 'SMA_20', 'Price_SMA20_Ratio', 'BB_Position', 
                'RSI', 'Price_SMA20_Pct_Diff', 'Reversion_Score']
    
    # Check if features exist in the dataframe
    available_features = [f for f in features if f in df.columns]
    X = df[available_features]
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'Feature': available_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return model, accuracy, feature_importance, available_features

def predict_trend(model, df, sentiment_score, feature_names):
    """Make prediction for next day's trend."""
    latest_data = df.iloc[-1]
    
    # Create input data as a DataFrame with proper column names
    feature_values = {name: latest_data[name] for name in feature_names}
    features_df = pd.DataFrame([feature_values])
    
    # Ensure the column order matches what the model was trained on
    features_df = features_df[feature_names]
    
    # Get the prediction and confidence
    prediction = model.predict(features_df)[0]
    
    # Safely get confidence scores
    try:
        prob_scores = model.predict_proba(features_df)
        if prob_scores.shape[1] >= 2:  # Check if we have at least 2 classes
            confidence = prob_scores[0][1]
        else:
            # If only one class in the output, use a default confidence
            confidence = 0.5
            logging.warning("Only one class in prediction probabilities. Using default confidence of 0.5")
    except Exception as e:
        logging.error(f"Error getting prediction probabilities: {str(e)}")
        confidence = 0.5  # Use a default if there's an error
    
    # Adjust confidence based on reversion score and sentiment
    reversion_confidence = latest_data.get('Reversion_Score', 0) / 6.0  # Normalize to [0,1]
    
    # Combine prediction confidence with reversion confidence and sentiment
    adjusted_confidence = 0.6 * confidence + 0.3 * reversion_confidence + 0.1 * (sentiment_score + 1) / 2
    
    return prediction, adjusted_confidence

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

def main():
    st.set_page_config(page_title="Stock Price Predictor", layout="wide")
    
    st.title("📈 Stock Price Predictor")
    st.write("Predict whether a stock's price will rise or fall based on historical data and sentiment analysis.")
    
    # Sidebar for user input
    st.sidebar.header("Input Parameters")
    
    # Add tabs for different input methods
    input_method = st.sidebar.radio("Select Input Method", ["Enter Ticker", "Choose from Popular Indian Stocks"])
    
    if input_method == "Enter Ticker":
        ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, RELIANCE)", "AAPL").upper()
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
        st.sidebar.subheader("For Indian Stocks")
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
    
    # Add caching option
    use_cache = st.sidebar.checkbox("Use local cache", value=False, 
                                  help="Enable to cache data locally for faster loading. Disable to always fetch fresh data.")
    
    # Add button to clean cache
    if st.sidebar.button("Clear Cached Data"):
        clean_cache_files()
        st.sidebar.success("Cache cleared successfully!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    try:
        with st.spinner("Fetching stock data..."):
            df = fetch_stock_data(ticker, period=period, exchange=exchange, use_cache=use_cache)
            df = preprocess_data(df)
            
            # Display stock info
            formatted_ticker = ticker
            if exchange and exchange in INDIAN_EXCHANGES:
                formatted_ticker = f"{ticker}{INDIAN_EXCHANGES[exchange]}"
            
            stock = yf.Ticker(formatted_ticker)
            info = stock.info
            
            with col1:
                st.subheader(f"Stock Information: {formatted_ticker}")
                st.write(f"Company Name: {info.get('longName', 'N/A')}")
                
                # Handle different price fields for different markets
                if 'currentPrice' in info:
                    current_price = info['currentPrice']
                elif 'regularMarketPrice' in info:
                    current_price = info['regularMarketPrice']
                else:
                    current_price = "N/A"
                
                # Determine currency symbol based on exchange
                currency_symbol = "₹" if exchange else "$"
                
                st.write(f"Current Price: {current_price if current_price == 'N/A' else f'{currency_symbol}{current_price:.2f}'}")
                st.write(f"Market Cap: {currency_symbol}{info.get('marketCap', 0):,.0f}")
                st.write(f"52 Week High: {currency_symbol}{info.get('fiftyTwoWeekHigh', 'N/A'):.2f}" if info.get('fiftyTwoWeekHigh') else "52 Week High: N/A")
                st.write(f"52 Week Low: {currency_symbol}{info.get('fiftyTwoWeekLow', 'N/A'):.2f}" if info.get('fiftyTwoWeekLow') else "52 Week Low: N/A")
                
                # Display volume information
                if 'volume' in info:
                    st.write(f"Volume: {info['volume']:,.0f}")
                elif 'regularMarketVolume' in info:
                    st.write(f"Volume: {info['regularMarketVolume']:,.0f}")
                
                # Display exchange information
                if exchange:
                    st.write(f"Exchange: {exchange}")
                    
                # Display relevant market indices
                st.subheader("Market Indices")
                indices = MARKET_INDICES["India"] if exchange else MARKET_INDICES["USA"]
                
                indices_data = []
                for name, idx_ticker in indices.items():
                    try:
                        idx_data = fetch_index_data(idx_ticker)
                        if idx_data is not None and len(idx_data) >= 2:
                            latest_price = idx_data['Close'].iloc[-1]
                            prev_price = idx_data['Close'].iloc[-2]
                            pct_change = ((latest_price - prev_price) / prev_price) * 100
                            color = "green" if pct_change >= 0 else "red"
                            indices_data.append({
                                "name": name,
                                "price": latest_price,
                                "change": pct_change,
                                "color": color
                            })
                        else:
                            logger.warning(f"Not enough data for index {name}, skipping")
                    except Exception as e:
                        logger.error(f"Error processing index {name}: {str(e)}")
                        continue
                
                # Create a small dataframe for the indices
                if indices_data:
                    st.write("Latest market index values:")
                    for idx in indices_data:
                        st.write(f"{idx['name']}: {idx['price']:.2f} ({idx['change']:.2f}%)")
                else:
                    st.write("Market index data currently unavailable")
            
            # Plot trends
            fig = plot_trends(df)
            st.pyplot(fig)
            
            # Get sentiment and make prediction
            with st.spinner("Analyzing news sentiment..."):
                sentiment_score = get_sentiment_score(ticker)
                
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
                        
                        model, accuracy, feature_importance, feature_names = train_model(df)
                        prediction, confidence = predict_trend(model, df, sentiment_score, feature_names)
                        
                        # Display results
                        with col2:
                            st.subheader("Prediction Results")
                            
                            # Prediction indicator
                            prediction_color = "green" if prediction == 1 else "red"
                            prediction_text = "UP 📈" if prediction == 1 else "DOWN 📉"
                            st.markdown(f"### <span style='color:{prediction_color}'>{prediction_text}</span>", unsafe_allow_html=True)
                            
                            # Confidence meter
                            st.write("Confidence Score:")
                            st.progress(min(max(confidence, 0.0), 1.0))  # Ensure confidence is between 0 and 1
                            st.write(f"{confidence:.2%}")
                            
                            # Additional metrics
                            st.write("Model Accuracy:", f"{accuracy:.2%}")
                            st.write("Sentiment Score:", f"{sentiment_score:.2f}")
                            
                            # Sentiment interpretation
                            if sentiment_score > 0.2:
                                st.write("News Sentiment: Very Positive 😊")
                            elif sentiment_score > 0:
                                st.write("News Sentiment: Slightly Positive 🙂")
                            elif sentiment_score < -0.2:
                                st.write("News Sentiment: Very Negative 😟")
                            elif sentiment_score < 0:
                                st.write("News Sentiment: Slightly Negative 😕")
                            else:
                                st.write("News Sentiment: Neutral 😐")
                            
                            # Mean Reversion Analysis
                            st.subheader("Mean Reversion Analysis")
                            
                            if 'Reversion_Score' in df.columns:
                                reversion_score = df['Reversion_Score'].iloc[-1]
                                st.write(f"Reversion Score: {reversion_score:.1f}/6.0")
                                
                                # Reversion strength interpretation
                                if reversion_score >= 4:
                                    st.write("Strong potential for price reversion 🔄")
                                elif reversion_score >= 2:
                                    st.write("Moderate potential for price reversion ↔️")
                                else:
                                    st.write("Low potential for price reversion ➡️")
                            
                            if 'RSI' in df.columns:
                                rsi_value = df['RSI'].iloc[-1]
                                st.write(f"RSI: {rsi_value:.1f}")
                                
                                if rsi_value > 70:
                                    st.write("Overbought condition - potential downward reversion")
                                elif rsi_value < 30:
                                    st.write("Oversold condition - potential upward reversion")
                            
                            # Feature importance
                            st.subheader("Feature Importance")
                            if not feature_importance.empty:
                                # Display top 5 features only
                                top_features = feature_importance.head(min(5, len(feature_importance)))
                                st.bar_chart(data=top_features.set_index('Feature')['Importance'])
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
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Application error: {str(e)}\n{traceback.format_exc()}")
        st.write("Please check if the ticker symbol is valid and try again.")

if __name__ == "__main__":
    main() 