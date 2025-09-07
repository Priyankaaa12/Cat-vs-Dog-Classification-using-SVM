import os
import shutil
import random
from sklearn.model_selection import train_test_split

def prepare_dataset(raw_data_path, output_path, train_ratio=0.8, val_ratio=0.1):
    """
    Prepare dataset by splitting into train, validation, and test sets
    """
    # Create output directories
    for folder in ['train/cats', 'train/dogs', 'val/cats', 'val/dogs', 'test/cats', 'test/dogs']:
        os.makedirs(os.path.join(output_path, folder), exist_ok=True)
    
    # Get all image files
    cat_images = [f for f in os.listdir(raw_data_path) if f.startswith('cat')]
    dog_images = [f for f in os.listdir(raw_data_path) if f.startswith('dog')]
    
    # Split data
    cat_train, cat_temp = train_test_split(cat_images, train_size=train_ratio, random_state=42)
    cat_val, cat_test = train_test_split(cat_temp, train_size=val_ratio/(1-train_ratio), random_state=42)
    
    dog_train, dog_temp = train_test_split(dog_images, train_size=train_ratio, random_state=42)
    dog_val, dog_test = train_test_split(dog_temp, train_size=val_ratio/(1-train_ratio), random_state=42)
    
    # Function to copy files
    def copy_files(files, source_dir, dest_dir):
        for file in files:
            shutil.copy2(os.path.join(source_dir, file), os.path.join(dest_dir, file))
    
    # Copy files to respective directories
    copy_files(cat_train, raw_data_path, os.path.join(output_path, 'train/cats'))
    copy_files(dog_train, raw_data_path, os.path.join(output_path, 'train/dogs'))
    copy_files(cat_val, raw_data_path, os.path.join(output_path, 'val/cats'))
    copy_files(dog_val, raw_data_path, os.path.join(output_path, 'val/dogs'))
    copy_files(cat_test, raw_data_path, os.path.join(output_path, 'test/cats'))
    copy_files(dog_test, raw_data_path, os.path.join(output_path, 'test/dogs'))
    
    print("Dataset preparation completed!")
    print(f"Training: {len(cat_train)} cats, {len(dog_train)} dogs")
    print(f"Validation: {len(cat_val)} cats, {len(dog_val)} dogs")
    print(f"Test: {len(cat_test)} cats, {len(dog_test)} dogs")

if __name__ == "__main__":
    prepare_dataset('data/raw', 'data/processed')