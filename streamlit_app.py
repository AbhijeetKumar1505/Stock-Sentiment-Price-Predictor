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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# NewsAPI configuration
NEWS_API_KEY = "faa59dc97acf42f1acdada2e9c9e4155"  # Replace with your actual API key
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

def fetch_stock_data(ticker, period="1y"):
    """Fetch stock data from Yahoo Finance and cache it locally."""
    cache_file = f"{ticker}_data.csv"
    
    if os.path.exists(cache_file):
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - file_time < timedelta(days=1):
            return pd.read_csv(cache_file)
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        df.to_csv(cache_file)
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file)
        raise

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
    df['Daily_Return'] = df['Close'].pct_change()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    return df

def train_model(df):
    """Train Random Forest model on the data."""
    features = ['Daily_Return', 'SMA_20']
    X = df[features]
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

def predict_trend(model, df, sentiment_score):
    """Make prediction for next day's trend."""
    latest_data = df.iloc[-1]
    features = np.array([[latest_data['Daily_Return'], latest_data['SMA_20']]])
    
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0][1]
    
    return prediction, confidence

def plot_trends(df):
    """Plot stock price and 20-day SMA."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['Close'], label='Stock Price')
    ax.plot(df.index, df['SMA_20'], label='20-day SMA')
    ax.set_title('Stock Price and 20-day SMA')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    return fig

def main():
    st.set_page_config(page_title="Stock Price Predictor", layout="wide")
    
    st.title("📈 Stock Price Predictor")
    st.write("Predict whether a stock's price will rise or fall based on historical data and sentiment analysis.")
    
    # Sidebar for user input
    st.sidebar.header("Input Parameters")
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL").upper()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    try:
        with st.spinner("Fetching stock data..."):
            df = fetch_stock_data(ticker)
            df = preprocess_data(df)
            
            # Display stock info
            stock = yf.Ticker(ticker)
            info = stock.info
            
            with col1:
                st.subheader(f"Stock Information: {ticker}")
                st.write(f"Company Name: {info.get('longName', 'N/A')}")
                st.write(f"Current Price: ${info.get('currentPrice', 'N/A'):.2f}")
                st.write(f"Market Cap: ${info.get('marketCap', 0):,.0f}")
                st.write(f"52 Week High: ${info.get('fiftyTwoWeekHigh', 'N/A'):.2f}")
                st.write(f"52 Week Low: ${info.get('fiftyTwoWeekLow', 'N/A'):.2f}")
            
            # Plot trends
            fig = plot_trends(df)
            st.pyplot(fig)
            
            # Get sentiment and make prediction
            with st.spinner("Analyzing news sentiment..."):
                sentiment_score = get_sentiment_score(ticker)
                
                with st.spinner("Training prediction model..."):
                    model, accuracy = train_model(df)
                    prediction, confidence = predict_trend(model, df, sentiment_score)
            
            # Display results
            with col2:
                st.subheader("Prediction Results")
                
                # Prediction indicator
                prediction_color = "green" if prediction == 1 else "red"
                prediction_text = "UP 📈" if prediction == 1 else "DOWN 📉"
                st.markdown(f"### <span style='color:{prediction_color}'>{prediction_text}</span>", unsafe_allow_html=True)
                
                # Confidence meter
                st.write("Confidence Score:")
                st.progress(confidence)
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
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check if the ticker symbol is valid and try again.")

if __name__ == "__main__":
    main() 