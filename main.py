import argparse
from scripts import prepare_dataset, preprocess, train_svm, evaluate, predict

def main():
    parser = argparse.ArgumentParser(description='Cat vs Dog Classification using SVM')
    parser.add_argument('--prepare', action='store_true', help='Prepare dataset')
    parser.add_argument('--train', action='store_true', help='Train SVM model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model')
    parser.add_argument('--predict', type=str, help='Predict image (provide path)')
    
    args = parser.parse_args()
    
    if args.prepare:
        print("Preparing dataset...")
        prepare_dataset.prepare_dataset('data/raw', 'data/processed')
    
    elif args.train:
        print("Training SVM model...")
        train_svm.train_svm()
    
    elif args.evaluate:
        print("Evaluating model...")
        evaluate.evaluate_model()
    
    elif args.predict:
        print(f"Predicting image: {args.predict}")
        try:
            label, confidence = predict.predict_image(args.predict)
            print(f"Prediction: {label} (confidence: {confidence:.4f})")
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        print("Please specify an action: --prepare, --train, --evaluate, or --predict <image_path>")

if __name__ == "__main__":
    main()