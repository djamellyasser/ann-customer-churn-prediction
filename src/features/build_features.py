import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def encode_features(X_train, X_test):
    """
    Applies Label Encoding and One Hot Encoding to the features.
    Encodes Gender (col 2) and Geography (col 1).
    Returns X_train, X_test, le (LabelEncoder), ct (ColumnTransformer).
    """
    # Label Encoding Gender
    le = LabelEncoder()
    X_train[:, 2] = le.fit_transform(X_train[:, 2])
    X_test[:, 2] = le.transform(X_test[:, 2])

    # One Hot Encoding Geography
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    X_train = np.array(ct.fit_transform(X_train))
    X_test = np.array(ct.transform(X_test))
    
    return X_train, X_test, le, ct

def scale_features(X_train, X_test):
    """
    Applies Standard Scaling to the features.
    Returns X_train, X_test, sc (StandardScaler).
    """
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, sc
