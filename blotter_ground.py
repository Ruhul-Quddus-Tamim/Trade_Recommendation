"""
This script is used to check how accurate is the Trading Blotter Signals vs. Ground Truth Signals.

Trading Blotter suggest signals when to BUY, SELL and HOLD based on the RSI value.

In order to check whether the suggestion really helps user to BUY, SELL and HOLD or not? 
Therefore needs to created a true value called "Target" column and let the Trading Blotter generate signals. 
After generating the signals, then it is compared if the signal aligns with the Ground Truth value.
"""

import yaml
import logging
import pandas as pd
from traditional_trade import TradingEnvironmentwithBlotter
from feature_transformation import preprocess_and_transform
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Constants for Trading Environment
INITIAL_CASH = 10_000_000  # $10 million
WINDOW_SIZE = 60  # Example window size
DAILY_TRADING_LIMIT = 1000  # Example trading limit
TICKER = 'AAPL'  # Example ticker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from YAML file
try:
    with open("configs/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    logger.info("Configuration file loaded successfully.")
except FileNotFoundError:
    logger.error("Configuration file not found. Please ensure 'config.yaml' is in the 'configs' folder.")
    raise

# Preprocess the data to create the Target column
def preprocess_and_create_target(market_features_df, config):
    def determine_target(rsi):
        if rsi < 30:
            return 'Buy'
        elif rsi > 70:
            return 'Sell'
        else:
            return 'Hold'

    # Assuming the column with RSI values is named 'RSI'
    market_features_df['Target'] = market_features_df[config['data']['target_column']].apply(determine_target)
    return market_features_df

# Load and preprocess the data
try:
    logger.info("Preprocessing and transforming the data...")
    market_features_df = preprocess_and_transform(config['data']['file_path'])
    market_features_df = preprocess_and_create_target(market_features_df, config)
    logger.info("Data preprocessing and transformation completed successfully.")
except Exception as e:
    logger.error(f"Error during data transformation: {e}")
    raise

# Initialize the Trading Environment
env = TradingEnvironmentwithBlotter(market_features_df)

# Generate blotter signals
blotter_signals = []
ground_truth_signals = []

logger.info("Generating blotter signals and ground truth...")
for i in range(len(market_features_df)):
    trade_direction = env.step()  # Get the signal from the blotter
    blotter_signals.append(trade_direction)
    
    # Extract ground truth from the 'Target' column
    ground_truth_signals.append(market_features_df.iloc[i]['Target'])

# Convert blotter signals to numeric labels
signal_mapping = {'BUY': 0, 'SELL': 2, 'HOLD': 1}
blotter_signals_numeric = [signal_mapping[signal] for signal in blotter_signals]

# Encode the ground truth signals
label_encoder = LabelEncoder()
ground_truth_signals_numeric = label_encoder.fit_transform(ground_truth_signals)

# Evaluate the blotter signals against the ground truth
accuracy = accuracy_score(ground_truth_signals_numeric, blotter_signals_numeric)
precision = precision_score(ground_truth_signals_numeric, blotter_signals_numeric, average='weighted')
recall = recall_score(ground_truth_signals_numeric, blotter_signals_numeric, average='weighted')
f1 = f1_score(ground_truth_signals_numeric, blotter_signals_numeric, average='weighted')

# Log the evaluation metrics
logger.info("Trading Blotter vs Ground Truth Evaluation:")
logger.info(f"Accuracy: {accuracy:.4f}")
logger.info(f"Precision: {precision:.4f}")
logger.info(f"Recall: {recall:.4f}")
logger.info(f"F1 Score: {f1:.4f}")
