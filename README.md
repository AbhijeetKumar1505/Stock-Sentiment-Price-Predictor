# Stock Price Predictor

This project predicts whether a stock's price will rise or fall based on historical data and sentiment analysis. It uses machine learning (Random Forest Classifier) combined with technical indicators, mean reversion analysis, and news sentiment to make predictions for both international and Indian stock markets.

## Features

* **Multi-Market Support**:
  * International stocks (US, Europe, etc.)
  * Indian stock markets (NSE and BSE)
  * Popular Indian stocks selection by sector
  * Exchange-specific formatting

* **Technical Analysis**:
  * Moving averages (SMA)
  * Bollinger Bands
  * RSI (Relative Strength Index)
  * Mean reversion indicators
  * Daily returns

* **Mean Reversion Analysis**:
  * Reversion score calculation
  * Overbought/oversold detection
  * Price deviation from moving averages
  * Reversion potential strength assessment

* **Sentiment Analysis**:
  * News sentiment analysis using VADER
  * Recent news article evaluation
  * Combined sentiment and technical indicators

* **Market Indices Tracking**:
  * US indices (S&P 500, Dow Jones, NASDAQ)
  * Indian indices (NIFTY 50, SENSEX, NIFTY Bank)
  * Percentage change monitoring

* **Data Management**:
  * Optional local caching system
  * Temporary directory storage
  * Cache cleanup functionality
  * Fresh data fetching option

* **Visualization**:
  * Interactive price charts
  * Technical indicator overlays
  * Bollinger Bands visualization
  * RSI charts
  * Feature importance graphs

* **Machine Learning**:
  * Random Forest prediction model
  * Feature importance analysis
  * Model accuracy metrics
  * Confidence scoring
  * Data validation

## Prerequisites

* Python 3.8 or higher
* NewsAPI key (get one from [newsapi.org](https://newsapi.org))

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Replace the NewsAPI key in the application:
   ```python
   NEWS_API_KEY = "YOUR_API_KEY"  # Replace with your actual API key
   ```

## Usage

### Web Interface (Recommended)
Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

### Web Interface Features

1. **Input Options**:
   * Enter ticker symbol manually
   * Select from popular Indian stocks by sector
   * Choose exchange (NSE/BSE for Indian stocks)
   * Select time period (1 month to 5 years)
   * Enable/disable local data caching

2. **Stock Information**:
   * Company name and current price
   * Market cap and volume
   * 52-week high and low
   * Current exchange

3. **Market Indices**:
   * Current values of relevant indices
   * Percentage changes
   * Market context

4. **Prediction Results**:
   * Up/Down forecast with confidence score
   * Model accuracy
   * News sentiment analysis with interpretation
   * Mean reversion analysis
   * Feature importance visualization

5. **Technical Charts**:
   * Price chart with SMA
   * Bollinger Bands
   * RSI indicator

## Mean Reversion Features

This application includes specialized mean reversion analysis to identify potential price correction opportunities:

* **Reversion Score**: Quantifies the likelihood of price reverting to the mean (0-6 scale)
* **RSI Analysis**: Identifies overbought (>70) and oversold (<30) conditions
* **Bollinger Band Position**: Measures where price is within the volatility bands
* **Price-SMA Deviation**: Calculates percentage deviation from 20-day moving average

## Indian Market Features

The application provides special support for Indian stock markets:

* **Exchange Selection**: Choose between NSE (.NS) and BSE (.BO)
* **Popular Stock Catalog**: Pre-populated lists of major Indian stocks by sector
* **Indian Currency**: Displays prices in ₹ for Indian stocks
* **Indian Indices**: Tracks NIFTY 50, SENSEX, and NIFTY Bank

## Data Caching System

The application includes an improved data management system:

* **Optional Caching**: Enable/disable local data storage
* **Temporary Storage**: Uses system temp directory instead of working directory
* **Cache Cleanup**: Button to manually clear cached data
* **Automatic Expiration**: Cached data expires after 1 day

## Notes

* Predictions are based on historical data and should not be used as the sole basis for investment decisions
* The NewsAPI has rate limits on the free tier
* For best results with Indian stocks, select the appropriate exchange suffix
* Mean reversion strategies work best in range-bound or highly volatile markets

## Error Handling

The application includes comprehensive error handling for:
* Invalid stock tickers
* Insufficient historical data
* API rate limits
* Network issues
* Model training edge cases
* Index data availability

## License

This project is open source and available under the MIT License. 