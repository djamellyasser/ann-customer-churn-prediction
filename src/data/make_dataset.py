import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """
    Loads the dataset from the CSV file.
    """
    dataset = pd.read_csv(filepath)
    X = dataset.iloc[:, 3:-1].values
    y = dataset.iloc[:, -1].values
    return X, y

def split_data(X, y, test_size=0.2, random_state=0):
    """
    Splits the dataset into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
