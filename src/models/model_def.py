import tensorflow as tf

def build_network():
    """
    Builds and compiles the ANN model.
    """
    ann = tf.keras.models.Sequential()
    # Adding the input layer and the first hidden layer (input dim inferred on fit)
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    # Adding the second hidden layer
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    # Adding the output layer
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    
    # Compiling
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return ann
