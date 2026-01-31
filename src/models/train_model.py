import tensorflow as tf
import numpy as np

def train(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray, batch_size: int = 32, epochs: int = 100) -> tf.keras.callbacks.History:
    """
    Executes the training loop for the provided model.

    Args:
        model (tf.keras.Model): The neural network model to train.
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training target labels.
        batch_size (int, optional): Number of samples per gradient update. Defaults to 32.
        epochs (int, optional): Number of iterations over the entire dataset. Defaults to 100.

    Returns:
        tf.keras.callbacks.History: A record of training loss and metrics values.
    """
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    return history
