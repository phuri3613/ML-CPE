import json
import os
import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from nn_model import train_model, predict_model
from evaluate import evaluate_model, plot_history

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ชี้โฟลเดอร์ไปยังข้อมูลเห็ดที่อยู่นอกโฟลเดอร์ classification
DATA_PATH = os.path.join(BASE_DIR, "..", "MushroomImages")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMG_SIZE = 100
TEST_SIZE = 0.2
VAL_SIZE = 0.1
MAX_PER_CLASS = 1000 # จำกัดจำนวนภาพต่อคลาสเพื่อให้เทรนได้เร็วขึ้น
BATCH_SIZE = 32

def main():
    print("--" * 30)
    print("Neural Network Image Recognition: Mushroom Classification")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[Step 1] Loading dataset...")
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)

    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\n[Step 2] Preprocessing images...")
    X = to_features(images)
    y = labels
    np.save(f"{OUTPUT_DIR}/features.npy", X)

    print("\n[Step 3] Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, TEST_SIZE, VAL_SIZE)

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_val.npy", X_val)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_val.npy", y_val)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    # วนลูปทดสอบการตั้งค่าตามใบงานที่ 6 (Configurations & Epochs)
    experiments = [
        {"config": "config_1", "epochs": 20},
        {"config": "config_1", "epochs": 50},
        {"config": "config_2", "epochs": 20},
        {"config": "config_2", "epochs": 50},
    ]

    results = []

    for exp in experiments:
        cfg = exp["config"]
        ep = exp["epochs"]
        print(f"\n{'='*50}")
        print(f"Running Experiment: {cfg} with {ep} Epochs")
        print(f"{'='*50}")

        model, history = train_model(
            X_train, y_train, X_val, y_val, len(classes),
            output_dir=OUTPUT_DIR, epochs=ep, batch_size=BATCH_SIZE, config_name=cfg
        )

        predictions = predict_model(model, X_test)
        
        conf_matrix_path = f"{OUTPUT_DIR}/confusion_matrix_{cfg}_{ep}ep.png"
        history_path = f"{OUTPUT_DIR}/training_history_{cfg}_{ep}ep.png"
        
        accuracy = evaluate_model(y_test, predictions, classes, save_path=conf_matrix_path)
        plot_history(history, history_path)

        results.append((cfg, ep, accuracy))

    print("\n" + "="*40)
    print("FINAL EXPERIMENT RESULTS")
    print("="*40)
    for cfg, ep, acc in results:
        print(f"Config: {cfg:<10} | Epochs: {ep:<3} | Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()