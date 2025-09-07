import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from preprocess import load_dataset, extract_features

def evaluate_model():
    """
    Evaluate the trained SVM model
    """
    # Load the test dataset
    print("Loading test dataset...")
    # Note: You'll need to modify load_dataset to also load test data
    # For now, we'll use validation data for evaluation
    
    (X_train, y_train), (X_val, y_val) = load_dataset('data/processed')
    X_val_features = extract_features(X_val)
    
    # Load the trained model
    print("Loading trained model...")
    svm = joblib.load('models/svm_model.pkl')
    
    # Make predictions
    print("Making predictions...")
    y_pred = svm.predict(X_val_features)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Cat', 'Dog']))
    
    # Create confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('results/confusion_matrix.png')
    plt.show()
    
    # Save results to file
    with open('results/evaluation_results.txt', 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_val, y_pred, target_names=['Cat', 'Dog']))
    
    print("Evaluation results saved to 'results/evaluation_results.txt'")

if __name__ == "__main__":
    evaluate_model()