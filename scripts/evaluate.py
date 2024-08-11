import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

class ModelEvaluator:

    @staticmethod
    def evaluate_model(model, X_test, y_test, config, label_encoder):
        # Evaluate the model
        evaluation = model.evaluate(X_test, y_test, return_dict=True)
        print("Evaluation Metrics:", evaluation)

        # Predict on the test set and generate a classification report
        y_pred = model.predict(X_test).argmax(axis=1)
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        print("Classification Report:\n", report)

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", conf_matrix)

    @staticmethod
    def save_model(model, config):
        # Save the trained model
        model.save(f"{config['training']['output_dir']}/lstm_model_final.keras")
        print(f"Model saved to {config['training']['output_dir']}/lstm_model_final.keras")
