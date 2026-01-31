import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
from typing import Tuple

def evaluate(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Generates predictions for the test set and calculates evaluation metrics.

    Args:
        model (tf.keras.Model): The trained neural network model.
        X_test (np.ndarray): Test feature data.
        y_test (np.ndarray): True labels for the test set.

    Returns:
        Tuple[np.ndarray, float]: A tuple containing the Confusion Matrix (np.ndarray) 
                                 and the Accuracy Score (float).
    """
    # Prediction on the Test set
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)
    
    # Calculate Metrics
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    return cm, acc
