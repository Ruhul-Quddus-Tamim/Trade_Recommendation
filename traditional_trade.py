import pandas as pd
import yaml
import logging
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from helpers.feature_engineering import load_and_preprocess_data
from helpers.utils import preprocess_data_for_trading
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

# Trading Environment Setup
class TradingEnvironmentwithBlotter:
    def __init__(self, data, daily_trading_limit=DAILY_TRADING_LIMIT, window_size=WINDOW_SIZE):
        self.data = self.preprocess_data(data)
        self.daily_trading_limit = daily_trading_limit
        self.window_size = window_size
        self.reset()

    def preprocess_data(self, df):
        df['liquidity'] = df['bid_sz_00'] * df['bid_px_00'] + df['ask_sz_00'] * df['ask_px_00']
        return df

    def reset(self):
        self.current_step = 0
        self.balance = INITIAL_CASH
        self.shares_held = 0
        self.total_shares_traded = 0
        self.cumulative_reward = 0
        self.trades = []
        self.portfolio = {'cash': self.balance, 'holdings': {ticker: 0 for ticker in self.data['symbol'].unique()}}
        self.data['RSI'] = self.calculate_rsi(self.data['price'])
        self.data['pct_change'] = self.data['price'].pct_change()
        self.data['rolling_mean_vol'], self.data['rolling_std_vol'], self.data['rolling_mean_liq'], self.data['rolling_std_liq'] = self.calculate_vol_and_liquidity(self.data['price'], self.data['liquidity'], self.window_size)

    def calculate_rsi(self, data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_vol_and_liquidity(self, price_df, volume_df, window_size):
        rolling_mean_vol = price_df.pct_change().rolling(window=window_size).mean()
        rolling_std_vol = price_df.pct_change().rolling(window=window_size).std()
        rolling_mean_liq = volume_df.rolling(window=window_size).mean()
        rolling_std_liq = volume_df.rolling(window=window_size).std()
        return rolling_mean_vol, rolling_std_vol, rolling_mean_liq, rolling_std_liq

    def get_percentile(self, current_value, mean, std):
        if std > 0:
            z_score = (current_value - mean) / std
            percentile = norm.cdf(z_score)
        else:
            percentile = 0.5  # No variation
        return percentile

    def get_trade_price(self, base_price, current_vol, current_liq, mean_vol, std_vol, mean_liq, std_liq, trade_direction):
        vol_percentile = self.get_percentile(current_vol, mean_vol, std_vol)
        liq_percentile = self.get_percentile(current_liq, mean_liq, std_liq)

        # Define price adjustment scenarios based on market conditions
        if vol_percentile >= 0.9 and liq_percentile < 0.1:
            price_adjustment_percent = np.random.uniform(-0.25, -0.15)
        elif vol_percentile <= 0.1 and liq_percentile < 0.1:
            price_adjustment_percent = np.random.uniform(-0.10, -0.05)
        elif vol_percentile >= 0.9 and liq_percentile >= 0.9:
            price_adjustment_percent = np.random.uniform(-0.05, +0.10)
        else:
            price_adjustment_percent = np.random.uniform(-0.05, +0.05)  # Default for normal conditions

        # Adjust price based on trade direction
        if trade_direction == 'BUY':
            adjusted_price = base_price * (1 - price_adjustment_percent)
        else:  # SELL
            adjusted_price = base_price * (1 + price_adjustment_percent)

        return adjusted_price

    def step(self):
        row = self.data.iloc[self.current_step]
        current_price = row['price']
        current_rsi = row['RSI']
        current_vol = row['pct_change']
        current_liq = row['liquidity']
        mean_vol = row['rolling_mean_vol']
        std_vol = row['rolling_std_vol']
        mean_liq = row['rolling_mean_liq']
        std_liq = row['rolling_std_liq']

        if current_rsi < 30:  # Entry signal based on RSI
            trade_direction = 'BUY'
        elif current_rsi > 70:  # Exit signal based on RSI
            trade_direction = 'SELL'
        else:
            trade_direction = 'HOLD'

        return trade_direction

    def run(self):
        self.reset()
        for _ in range(len(self.data)):
            self.step()
        return self.cumulative_reward, self.trades

    def render(self):
        print(f'Cumulative reward: {self.cumulative_reward}')
        row = self.data.iloc[self.current_step]
        print(f'Total portfolio value: {self.portfolio["cash"] + self.portfolio["holdings"][row["symbol"]]*row["price"]}')
        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv('trades_blotter.csv', index=False)
        for trade in self.trades:
            print(f"Step: {trade['step']}, Timestamp: {trade['timestamp']}, Action: {trade['action']}, Price: {trade['price']}, Shares: {trade['shares']}, Symbol: {trade['symbol']}, Reward: {trade['reward']}, Transaction Cost: {trade['transaction_cost']}, Slippage: {trade['slippage']}, Time Penalty: {trade['time_penalty']}")


class TradingBlotterEvaluator:
    def __init__(self, market_features_df, X_test, model, config):
        self.market_features_df = market_features_df
        self.model = model
        self.X_test = X_test

        # Align the market features dataset with X_test by using the same rows/indices
        self.X_scaled, self.y_cleaned, self.label_encoder = preprocess_data_for_trading(market_features_df, config)

        # Ensure the data is aligned correctly
        self.model_signals = model.predict(self.X_scaled).argmax(axis=1)  # Model predictions on the scaled data
        self.blotter_signals = []

    def compare_signals(self):
        # Initialize the trading environment for the blotter signals
        env = TradingEnvironmentwithBlotter(self.market_features_df)

        # Generate blotter signals corresponding to the model signals
        for i in range(len(self.model_signals)):
            trade_direction = env.step()
            self.blotter_signals.append(trade_direction)

        # Convert blotter signals to numeric labels
        signal_mapping = {'BUY': 0, 'SELL': 2, 'HOLD': 1}
        blotter_signals_numeric = [signal_mapping[signal] for signal in self.blotter_signals]

        # Evaluate comparison between model predictions and blotter signals
        accuracy = accuracy_score(self.model_signals, blotter_signals_numeric)
        precision = precision_score(self.model_signals, blotter_signals_numeric, average='weighted')
        recall = recall_score(self.model_signals, blotter_signals_numeric, average='weighted')
        f1 = f1_score(self.model_signals, blotter_signals_numeric, average='weighted')

        # Log the evaluation metrics
        print("Comparison Metrics between Model and Blotter Signals:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Optionally, visualize the signals
        self.visualize_signals()

    def visualize_signals(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.model_signals, label="Model Signals", alpha=0.6)
        plt.plot(self.blotter_signals, label="Blotter Signals", alpha=0.6)
        plt.title("Model vs Blotter Signals")
        plt.xlabel("Time Step")
        plt.ylabel("Signal")
        plt.legend()
        plt.show()
