# Stock Price Predictor

This project predicts whether a stock's price will rise or fall based on historical data and sentiment analysis. It uses machine learning (Random Forest Classifier) combined with technical indicators and news sentiment to make predictions.

## Features

- Historical stock data analysis using technical indicators
- News sentiment analysis using VADER
- Machine learning prediction using Random Forest
- Data caching to avoid API overuse
- Visualization of stock trends
- Error handling and logging
- Interactive web interface using Streamlit

## Prerequisites

- Python 3.8 or higher
- NewsAPI key (get one from [newsapi.org](https://newsapi.org))

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Replace the NewsAPI key in `streamlit_app.py`:
   ```python
   NEWS_API_KEY = "YOUR_API_KEY"  # Replace with your actual API key
   ```

## Usage

### Command Line Interface
Run the script from the command line:
```bash
python stock_predictor.py
```

### Web Interface (Recommended)
Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

The web interface provides:
1. A sidebar for entering the stock ticker
2. Real-time stock information display
3. Interactive price chart with 20-day SMA
4. Visual prediction results with confidence meter
5. Sentiment analysis with emoji indicators
6. Model accuracy metrics

## Output

The application provides:
- Prediction (Up/Down) with confidence score
- Model accuracy on test data
- Sentiment score from news analysis
- Visualization of stock price and 20-day SMA
- Additional stock information (market cap, 52-week high/low, etc.)

## Notes

- Stock data is cached locally to avoid API rate limits
- The model uses technical indicators (daily returns, 20-day SMA) and sentiment analysis
- Predictions are based on historical data and should not be used as the sole basis for investment decisions
- The NewsAPI has rate limits on the free tier
- The web interface provides a more user-friendly experience with real-time updates

## Error Handling

The application includes error handling for:
- Invalid stock tickers
- API rate limits
- Missing data
- Network issues

## License

This project is open source and available under the MIT License. 