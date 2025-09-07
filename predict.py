import joblib
import cv2
import numpy as np
from preprocess import extract_features

def predict_image(image_path, model_path='models/svm_model.pkl'):
    """
    Predict whether an image contains a cat or dog
    """
    # Load the trained model
    svm = joblib.load(model_path)
    
    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    
    # Extract features
    img_features = extract_features(np.array([img]))
    
    # Make prediction
    prediction = svm.predict(img_features)
    probability = svm.decision_function(img_features)
    
    # Convert to label
    label = "Cat" if prediction[0] == 0 else "Dog"
    confidence = abs(probability[0])
    
    return label, confidence

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    try:
        label, confidence = predict_image(image_path)
        print(f"Prediction: {label} (confidence: {confidence:.4f})")
    except Exception as e:
        print(f"Error: {e}")