"""
Stock Price Predictor App - Streamlit Interface
This is the entry point for the stock prediction application.
"""
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the main application
try:
    from stock_predictor import main
    logger.info("Successfully imported stock_predictor module")
except ImportError as e:
    logger.error(f"Failed to import stock_predictor module: {str(e)}")
    print(f"Error: {str(e)}")
    print("Please make sure stock_predictor.py is in the same directory.")
    sys.exit(1)

if __name__ == "__main__":
    try:
        # Run the main application
        main()
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
        import traceback
        logger.error(traceback.format_exc()) 