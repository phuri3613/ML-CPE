import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "outputs"
IMG_SIZE = 100
N_SAMPLES = 4 # กำหนดจำนวนรูปที่จะสุ่มมาเดโม่

def test_svm(n_samples=N_SAMPLES):
    # โหลด Model และ Scaler ที่เทรนไว้แล้วขึ้นมาใช้งาน (ไม่ต้องเทรนใหม่)
    model = joblib.load(f"{OUTPUT_DIR}/svm_model.pkl")
    scaler = joblib.load(f"{OUTPUT_DIR}/scaler.pkl")
    
    # โหลดชุดข้อมูล Test ที่เซฟเก็บไว้
    X_test = np.load(f"{OUTPUT_DIR}/X_test.npy")
    y_test = np.load(f"{OUTPUT_DIR}/y_test.npy")
    
    with open(f"{OUTPUT_DIR}/classes.json") as f:
        classes = json.load(f)

    # สุ่มเลือกตำแหน่งข้อมูล (Index) ขึ้นมา 4 ตัวโดยไม่ซ้ำกัน
    index = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[index]
    y_sample = y_test[index]

    # ส่งให้โมเดลทำนายผล
    predictions = model.predict(scaler.transform(X_sample))

    # คำนวณการจัดเรียงรูปในกราฟิก (Grid layout)
    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 4.0 * rows))
    axes = np.atleast_1d(axes).ravel()

    # วนลูปวาดรูปทีละภาพพร้อมแปะผลการทำนาย
    for i, ax in enumerate(axes):
        if i >= n_samples:
            ax.axis("off")
            continue

        pred = classes[predictions[i]]
        true = classes[y_sample[i]]
        
        # ตรวจสอบว่าโมเดลทายตรงกับความจริงหรือไม่
        correct = predictions[i] == y_sample[i]
        
        # ถ้าระบบทายถูก ตัวหนังสือจะเป็นสีเขียว ถ้าผิดจะเป็นสีแดง
        color = "green" if correct else "red"

        # วาดภาพเห็ด (คืนร่างจากอาร์เรย์ 1 มิติกลับเป็นภาพ 2 มิติ)
        ax.imshow(X_sample[i].reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Pred: {pred}\nTrue: {true}", color=color)

        print(f"[{i + 1}] Pred: {pred:<20} True: {true:<20} {'OK' if correct else 'WRONG'}")

    # สรุปผลว่าเดโม่ 4 ภาพ ทายถูกกี่ภาพ
    correct_total = int((predictions == y_sample).sum())
    print(f"\nCorrect: {correct_total}/{n_samples}")
    fig.suptitle(f"Prediction: {correct_total}/{n_samples} correct")
    fig.tight_layout()

    # บันทึกภาพเดโม่ลงเครื่อง
    save_path = f"{OUTPUT_DIR}/prediction_sample.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    test_svm()
