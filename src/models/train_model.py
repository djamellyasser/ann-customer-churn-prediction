def train(model, X_train, y_train, batch_size=32, epochs=100):
    """
    Trains the model on the training set.
    """
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    return history
