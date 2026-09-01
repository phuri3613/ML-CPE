import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# กำหนดชื่อโมเดลที่ต้องการโหลดมาทดสอบ (เลือก 1 ตัวจากการทดลองใน main.py)
TARGET_MODEL = "nn_model_config_2_50ep.keras"
N_SAMPLES = 4

def test_nn(n_samples=N_SAMPLES):
    model_path = f"{OUTPUT_DIR}/{TARGET_MODEL}"
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found. Please run main.py first.")
        return

    model = keras.models.load_model(model_path)
    X_test = np.load(f"{OUTPUT_DIR}/X_test.npy")
    y_test = np.load(f"{OUTPUT_DIR}/y_test.npy")
    with open(f"{OUTPUT_DIR}/classes.json") as f:
        classes = json.load(f)

    index = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[index]
    y_sample = y_test[index]

    probabilities = model.predict(X_sample, verbose=0)
    if probabilities.shape[-1] == 1:
        probabilities = probabilities.ravel()
        predictions = (probabilities > 0.5).astype(int)
        confidence = np.where(predictions == 1, probabilities, 1 - probabilities)
    else:
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)

    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 4.0 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        if i >= n_samples:
            ax.axis("off")
            continue

        pred = classes[predictions[i]]
        true = classes[y_sample[i]]
        correct = predictions[i] == y_sample[i]
        color = "green" if correct else "red"

        ax.imshow(X_sample[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Pred: {pred} ({confidence[i] * 100:.0f}%)\n"
                     f"True: {true}", color=color)

    correct_total = int((predictions == y_sample).sum())
    fig.suptitle(f"Prediction: {correct_total}/{n_samples} correct")
    fig.tight_layout()

    save_path = f"{OUTPUT_DIR}/prediction_sample.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    test_nn()