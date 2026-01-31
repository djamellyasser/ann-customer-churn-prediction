import tensorflow as tf

def build_network() -> tf.keras.models.Sequential:
    """
    Constructs and compiles the Artificial Neural Network (ANN) architecture.
    
    The network consists of:
    1. Input layer and first hidden layer: 6 neurons, ReLU activation.
    2. Second hidden layer: 6 neurons, ReLU activation.
    3. Output layer: 1 neuron, Sigmoid activation (for binary classification).

    Returns:
        tf.keras.models.Sequential: The compiled Keras model.
    """
    ann = tf.keras.models.Sequential()
    
    # Adding the input layer and the first hidden layer
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    
    # Adding the second hidden layer
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    
    # Adding the output layer
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    
    # Compiling with Adam optimizer and binary crossentropy loss
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return ann
