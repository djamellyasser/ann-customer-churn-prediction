import numpy as np
from src.data.make_dataset import load_data, split_data
from src.features.build_features import encode_features, scale_features
from src.models.model_def import build_network
from src.models.train_model import train
from src.models.predict_model import evaluate

def main():
    print("Loading data...")
    X, y = load_data('data/raw/Churn_Modelling.csv')
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("Feature Engineering...")
    # Encoding
    X_train, X_test, le, ct = encode_features(X_train, X_test)
    # Scaling
    X_train, X_test, sc = scale_features(X_train, X_test)
    
    print("Building Model...")
    ann = build_network()
    
    print("Training Model...")
    train(ann, X_train, y_train, epochs=20, batch_size=32)
    
    print("Evaluating Model...")
    cm, acc = evaluate(ann, X_test, y_test)
    
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {acc}")
    
    print("Running Example Prediction...")
    # Example: 600, France, Male, 40, 3, 60000, 2, 1, 1, 50000
    sample = np.array([[600, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000]], dtype=object)
    
    # Preprocess sample
    try:
        sample[:, 2] = le.transform(sample[:, 2])
        sample = ct.transform(sample)
        sample = sc.transform(sample)
        
        prediction = ann.predict(sample) > 0.5
        print(f"Prediction (Will Exit?): {prediction[0][0]}")
    except Exception as e:
        print(f"Error during example prediction: {e}")

if __name__ == "__main__":
    main()
