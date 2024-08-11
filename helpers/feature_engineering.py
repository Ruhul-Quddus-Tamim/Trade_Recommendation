import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(market_features_df, config):
    # Assuming the column with RSI values is named 'RSI', creating the target column
    def determine_target(rsi):
        if rsi < 30:
            return 'Buy'
        elif rsi > 70:
            return 'Sell'
        else:
            return 'Hold'

    market_features_df['Target'] = market_features_df[config['data']['target_column']].apply(determine_target)

    # Drop non-relevant or non-numeric columns (adjust as necessary)
    X = market_features_df.drop(columns=['Target', 'ts_recv', 'ts_event', 'action', 'side'])
    y = market_features_df['Target']

    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Convert categorical features to numeric using one-hot encoding
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Handle NaN and Inf values
    X_encoded.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
    X_encoded.fillna(method='ffill', inplace=True)

    # Drop any remaining rows with NaN or Inf values
    X_cleaned = X_encoded.replace([float('inf'), float('-inf')], float('nan')).dropna()
    y_cleaned = y_encoded[X_cleaned.index]

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cleaned)
    
    # After the encoding is done, you can check the mapping as follows:
    label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    print(label_mapping)

    
    return X_scaled, y_cleaned, label_encoder
