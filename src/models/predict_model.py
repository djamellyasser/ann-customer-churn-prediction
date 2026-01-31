import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

def evaluate(model, X_test, y_test):
    """
    Evaluates the model on the test set.
    Returns Confusion Matrix and Accuracy Score.
    """
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)
    
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    return cm, acc
