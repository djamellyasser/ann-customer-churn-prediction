import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Any
import numpy as np

def load_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the churn dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file containing the data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Returns feature matrix X and target vector y as numpy arrays.
    """
    dataset = pd.read_csv(filepath)
    # Extracts relevant features (omitting row number, customer id, and surname)
    X = dataset.iloc[:, 3:-1].values
    y = dataset.iloc[:, -1].values
    return X, y

def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the dataset into training and testing sets.

    Args:
        X (np.ndarray): The feature matrix.
        y (np.ndarray): The target vector.
        test_size (float): The proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int): Controls the shuffling applied to the data before the split. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
