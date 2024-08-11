import tensorflow as tf

def build_lstm_model(input_shape, num_classes, lstm_units_1, lstm_units_2, dense_units, dropout_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((1, input_shape[0]), input_shape=input_shape),  # Reshape to (batch_size, 1, features)
        tf.keras.layers.LSTM(lstm_units_1, return_sequences=True),
        tf.keras.layers.LSTM(lstm_units_2),
        tf.keras.layers.Dense(dense_units, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_classes, activation='softmax')  # 3 classes: Buy, Hold, Sell
    ])
    
    return model

###########################################################################################################################
# def build_transformer_model(input_shape, num_classes, num_heads, key_dim, dense_units, dropout_rate, num_layers=2):
#     inputs = layers.Input(shape=input_shape)
    
#     # Positional Encoding
#     positions = tf.range(start=0, limit=input_shape[0], delta=1)
#     positions = layers.Embedding(input_dim=input_shape[0], output_dim=input_shape[1])(positions)
#     x = inputs + positions
    
#     # Stacked Transformer Encoders
#     for _ in range(num_layers):
#         x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
#         x = layers.LayerNormalization()(x)
    
#     # Global Average Pooling
#     x = layers.GlobalAveragePooling1D()(x)
    
#     # Fully connected layer
#     x = layers.Dense(dense_units, activation='relu')(x)
#     x = layers.Dropout(dropout_rate)(x)
    
#     # Output layer
#     outputs = layers.Dense(num_classes, activation='softmax')(x)
    
#     # Create model
#     model = models.Model(inputs, outputs)
#     return model
