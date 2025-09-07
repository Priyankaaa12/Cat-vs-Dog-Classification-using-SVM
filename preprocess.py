import os
import cv2
import numpy as np
from sklearn.utils import shuffle

def load_images_from_folder(folder, label, img_size=(64, 64)):
    """
    Load images from a folder and assign labels
    """
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            # Read and preprocess image
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = img / 255.0  # Normalize to [0, 1]
            
            images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    return images, labels

def load_dataset(base_path, img_size=(64, 64)):
    """
    Load the entire dataset
    """
    # Load cats and dogs from train folder
    cat_train, cat_labels = load_images_from_folder(
        os.path.join(base_path, 'train/cats'), 0, img_size)
    dog_train, dog_labels = load_images_from_folder(
        os.path.join(base_path, 'train/dogs'), 1, img_size)
    
    # Load cats and dogs from validation folder
    cat_val, cat_val_labels = load_images_from_folder(
        os.path.join(base_path, 'val/cats'), 0, img_size)
    dog_val, dog_val_labels = load_images_from_folder(
        os.path.join(base_path, 'val/dogs'), 1, img_size)
    
    # Combine and shuffle
    X_train = np.array(cat_train + dog_train)
    y_train = np.array(cat_labels + dog_labels)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    
    X_val = np.array(cat_val + dog_val)
    y_val = np.array(cat_val_labels + dog_val_labels)
    X_val, y_val = shuffle(X_val, y_val, random_state=42)
    
    return (X_train, y_train), (X_val, y_val)

def extract_features(images):
    """
    Extract features from images for SVM (flatten and reduce dimensionality if needed)
    """
    # Flatten the images
    n_samples = len(images)
    flattened = images.reshape(n_samples, -1)
    
    # You could add feature extraction techniques here (HOG, etc.)
    return flattened

if __name__ == "__main__":
    # Test the functions
    (X_train, y_train), (X_val, y_val) = load_dataset('data/processed')
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Validation labels shape: {y_val.shape}")