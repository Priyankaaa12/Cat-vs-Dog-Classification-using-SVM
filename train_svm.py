import numpy as np
import joblib
import time
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess import load_dataset, extract_features

def train_svm():
    """
    Train SVM classifier on cat vs dog dataset
    """
    print("Loading dataset...")
    (X_train, y_train), (X_val, y_val) = load_dataset('data/processed')
    
    print("Extracting features...")
    X_train_features = extract_features(X_train)
    X_val_features = extract_features(X_val)
    
    print(f"Feature shape: {X_train_features.shape}")
    
    # Create SVM classifier
    print("Training SVM...")
    svm = SVC(kernel='rbf', random_state=42, verbose=True)
    
    # Train the model
    start_time = time.time()
    svm.fit(X_train_features, y_train)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on training set
    train_pred = svm.predict(X_train_features)
    train_accuracy = accuracy_score(y_train, train_pred)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    
    # Evaluate on validation set
    val_pred = svm.predict(X_val_features)
    val_accuracy = accuracy_score(y_val, val_pred)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_val, val_pred, target_names=['Cat', 'Dog']))
    
    # Save the model
    joblib.dump(svm, 'models/svm_model.pkl')
    print("Model saved as 'models/svm_model.pkl'")
    
    return svm, X_val_features, y_val

if __name__ == "__main__":
    train_svm()