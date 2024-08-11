import yaml
import logging
from feature_transformation import preprocess_and_transform
from helpers.feature_engineering import load_and_preprocess_data
from scripts.train import ModelTrainer
from scripts.evaluate import ModelEvaluator
from traditional_trade import TradingEnvironmentwithBlotter, TradingBlotterEvaluator

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

# Step 1: Preprocess the data and add technical indicators
try:
    logger.info("Preprocessing and transforming the data...")
    market_features_df = preprocess_and_transform(config['data']['file_path'])
    logger.info("Data preprocessing and transformation completed successfully.")
except Exception as e:
    logger.error(f"Error during data transformation: {e}")
    raise

# Step 2: Load and preprocess the data for model training
try:
    logger.info("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, label_encoder = ModelTrainer.prepare_data(market_features_df, config)
    logger.info(f"Data preprocessing completed successfully. X_train shape: {X_train.shape}")
except Exception as e:
    logger.error(f"Error during data preprocessing: {e}")
    raise

# Step 3: Build and compile the model
try:
    logger.info("Building the model...")
    model = ModelTrainer.build_and_compile_model(config, input_shape=(X_train.shape[1],))
except Exception as e:
    logger.error(f"Error during model building: {e}")
    raise

# Step 4: Run Training
try:
    logger.info("Running Training...")
    ModelTrainer.train_model(model, X_train, y_train, config)
except Exception as e:
    logger.error(f"Error during training: {e}")
    raise

# Step 5: Run Evaluation
try:
    logger.info("Running Evaluation...")
    ModelEvaluator.evaluate_model(model, X_test, y_test, config, label_encoder)
except Exception as e:
    logger.error(f"Error during evaluation: {e}")
    raise

# Step 6: Save the Model
try:
    logger.info("Saving the model...")
    ModelEvaluator.save_model(model, config)
except Exception as e:
    logger.error(f"Error during model saving: {e}")
    raise

logger.info("Process completed successfully.")

# Step 7: Compare Model Signals with Blotter Signals
try:
    logger.info("Generating and comparing signals for blotter evaluation...")
    blotter_evaluator = TradingBlotterEvaluator(market_features_df, X_test, model, config)
    blotter_evaluator.compare_signals()
except Exception as e:
    logger.error(f"Error during signal comparison: {e}")
    raise

logger.info("Blotter evaluation and signal comparison completed successfully.")
