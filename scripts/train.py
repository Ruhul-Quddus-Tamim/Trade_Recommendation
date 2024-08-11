import tensorflow as tf
from sklearn.model_selection import train_test_split
from helpers.feature_engineering import load_and_preprocess_data
from models.LSTM_model import build_lstm_model

class ModelTrainer:
    
    @staticmethod
    def prepare_data(df, config):
        # Load and preprocess data
        X_scaled, y_cleaned, label_encoder = load_and_preprocess_data(df, config)
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_cleaned, 
            test_size=config['data']['test_size'], 
            random_state=config['data']['random_state']
        )
        
        return X_train, X_test, y_train, y_test, label_encoder  # Return label_encoder as well

    @staticmethod
    def build_and_compile_model(config, input_shape):
        # Build the LSTM model with hyperparameters from config
        model = build_lstm_model(
            input_shape=input_shape,
            num_classes=config['model']['num_classes'],
            lstm_units_1=config['model']['lstm_units_1'],
            lstm_units_2=config['model']['lstm_units_2'],
            dense_units=config['model']['dense_units'],
            dropout_rate=config['model']['dropout_rate']
        )

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        
        return model


    @staticmethod
    def train_model(model, X_train, y_train, config):
        # Early Stopping and Learning Rate Scheduling
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config['training']['patience'], restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=config['training']['reduce_lr_factor'], 
                                                         patience=config['training']['reduce_lr_patience'], min_lr=config['training']['min_lr'])

        # Train the neural network model
        history = model.fit(
            X_train, y_train,
            epochs=config['training']['epochs'],
            batch_size=config['training']['batch_size'],
            validation_split=config['data']['validation_split'],
            callbacks=[early_stopping, reduce_lr]
        )
        
        return history
