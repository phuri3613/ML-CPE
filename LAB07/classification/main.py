import json
import os
import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from cnn_model import train_model, predict_model
from evaluate import evaluate_model, plot_history

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ชี้ไปยังโฟลเดอร์ MushroomImages ที่เก็บข้อมูลเห็ด
DATA_PATH = os.path.join(BASE_DIR, "..", "MushroomImages")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMG_SIZE =  224
TEST_SIZE = 0.2
VAL_SIZE = 0.1
MAX_PER_CLASS = 3000   # None = use all images
# จุดสำหรับเปลี่ยนค่าเพื่อทดลองตามใบงาน (เช่น 10, 30, 50)
EPOCHS = 30
BATCH_SIZE = 4      

def main():

    print("--" * 30)
    print("CNN Image Recognition: Edible vs Poisonous Mushrooms")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[Step 1] Loading dataset...")
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)

    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Total images : {len(images)}")
    print(f"Classes      : {classes}")

    print("\n[Step 2] Preprocessing images...")
    X = to_features(images)
    y = labels

    np.save(f"{OUTPUT_DIR}/features.npy", X)
    print(f"Feature shape: {X.shape}")

    print("\n[Step 3] Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, TEST_SIZE, VAL_SIZE
    )

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_val.npy", X_val)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_val.npy", y_val)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"Training samples  : {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples   : {len(X_test)}")

    print("\n[Step 4] Training model...")
    model, history = train_model(
        X_train, y_train, X_val, y_val, len(classes),
        OUTPUT_DIR, EPOCHS, BATCH_SIZE
    )

    print("Training completed.")

    print("\n[Step 5] Testing model...")
    predictions = predict_model(model, X_test)

    print("\n[Step 6] Evaluating model...")
    evaluate_model(y_test, predictions, classes,
                   save_path=f"{OUTPUT_DIR}/confusion_matrix.png")
    plot_history(history, f"{OUTPUT_DIR}/training_history.png")

if __name__ == "__main__":
    main()