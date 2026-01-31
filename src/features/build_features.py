import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from typing import Tuple

def encode_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, LabelEncoder, ColumnTransformer]:
    """
    Applies Label Encoding and One Hot Encoding to the features.
    Specifically encodes Gender (index 2) and Geography (index 1).

    Args:
        X_train (np.ndarray): Training feature matrix.
        X_test (np.ndarray): Testing feature matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray, LabelEncoder, ColumnTransformer]: 
        Processed X_train, X_test, the LabelEncoder for Gender, and the ColumnTransformer for Geography.
    """
    # Label Encoding Gender (index 2)
    le = LabelEncoder()
    X_train[:, 2] = le.fit_transform(X_train[:, 2])
    X_test[:, 2] = le.transform(X_test[:, 2])

    # One Hot Encoding Geography (index 1)
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    X_train = np.array(ct.fit_transform(X_train))
    X_test = np.array(ct.transform(X_test))
    
    return X_train, X_test, le, ct

def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Applies Standard Scaling to the features to ensure zero mean and unit variance.

    Args:
        X_train (np.ndarray): Training feature matrix.
        X_test (np.ndarray): Testing feature matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray, StandardScaler]: Processed X_train, X_test, and the fitted StandardScaler.
    """
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, sc
