import json
import os
import joblib
import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from svm_model import train_svm, predict_svm
from evaluate import evaluate_model

# ชี้กลับไป 1 โฟลเดอร์เพื่อหาชุดข้อมูลเห็ด
DATA_PATH = "../MushroomImages" 
OUTPUT_DIR = "outputs"
IMG_SIZE = 100
TEST_SIZE = 0.2
MAX_PER_CLASS = 3000

def main():
    print("--" * 30)
    print("SVM Image Recognition: Edible vs Poisonous Mushrooms")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[Step 1] Loading dataset...")
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)

    np.save(f"{OUTPUT_DIR}/images.npy", images)
    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Total images : {len(images)}")
    print(f"Classes      : {classes}")

    print("\n[Step 2] Preprocess images...")
    X = to_features(images)
    y = labels
    print(f"Feature shape: {X.shape}")

    print("\n[Step 3] Splitting dataset...")
    X_train, X_test, y_train, y_test = split_dataset(X, y, TEST_SIZE)

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    print("\n[Step 4] Training SVM...")
    model, scaler = train_svm(X_train, y_train)

    joblib.dump(model, f"{OUTPUT_DIR}/svm_model.pkl")
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")
    print("SVM training completed.")

    print("\n[Step 5] Testing model...")
    predictions = predict_svm(model, scaler, X_test)

    print("\n[Step 6] Evaluating model...")
    evaluate_model(y_test, predictions, classes, save_path=f"{OUTPUT_DIR}/confusion_matrix.png")

if __name__ == "__main__":
    main()