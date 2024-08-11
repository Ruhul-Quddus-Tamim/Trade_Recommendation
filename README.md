# Trade_Recommendation
LSTM based model for trading AAPL vs Traditional Trading AAPL

# Project Structure
```
Trade_Recommendation/
│
├── models/
│   ├── LSTM_model.py             # Implementation of the LSTM model
│
├── helpers/
│   ├── feature_engineering.py    # Feature engineering and data transformation scripts
│   └── utils.py                  # Utility functions (e.g., metrics, data splitting)
│
├── scripts/
│   ├── train.py                  # Script to train and fine-tune the model
│   ├── evaluate.py               # Script to evaluate the model and compare with traditional strategy
│
├── configs/
│   └── config.yaml               # Configuration file for hyperparameters and paths
│
├── feature_transformation.py     # Script for transforming features and preprocessing data
├── traditional_trade.py          # Script for simulating the Trading Blotter environement
├── blotter_ground.py             # Script to check how accurate is the Trading Blotter Signals vs. Ground Truth Signals.
├── main.py                       # Main entry point script to run the full process
├── README.md                     # Project overview and instructions
└── requirements.txt              # Project dependencies
```
