import json
import os
import joblib
import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from svm_model import train_svm, predict_svm
from evaluate import evaluate_model

# กำหนดเส้นทางโฟลเดอร์ข้อมูลและตัวแปรเริ่มต้น
DATA_PATH = "../MushroomImages" # ชี้กลับไป 1 โฟลเดอร์เพื่อหาชุดข้อมูลเห็ด
OUTPUT_DIR = "outputs"          # โฟลเดอร์สำหรับเก็บไฟล์ผลลัพธ์ (โมเดล, ตัวแปร)
IMG_SIZE = 100                  # กำหนดขนาดภาพ 100x100 พิกเซล
TEST_SIZE = 0.2                 # แบ่งข้อมูลเป็น Train 80% และ Test 20%
MAX_PER_CLASS = 3000            # จำกัดภาพสูงสุดคลาสละ 3,000 ภาพ ป้องกัน RAM เต็ม

def main():
    print("--" * 30)
    print("SVM Image Recognition: Edible vs Poisonous Mushrooms")
    print("--" * 30)

    # สร้างโฟลเดอร์ outputs ถ้ายังไม่มี
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[Step 1] Loading dataset...")
    # เรียกฟังก์ชันโหลดภาพเข้าสู่ระบบ
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)

    # บันทึกข้อมูลภาพและคลาสที่โหลดมาเป็นไฟล์ เพื่อลดเวลาโหลดในครั้งต่อไป
    np.save(f"{OUTPUT_DIR}/images.npy", images)
    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Total images : {len(images)}")
    print(f"Classes      : {classes}")

    print("\n[Step 2] Preprocess images...")
    # แปลงข้อมูลภาพ 2 มิติ ให้เป็นอาร์เรย์ 1 มิติ (Flatten) และ Normalize ค่าพิกเซล
    X = to_features(images)
    y = labels
    print(f"Feature shape: {X.shape}") # ดูรูปร่างของข้อมูลหลังแปลง

    print("\n[Step 3] Splitting dataset...")
    # แบ่งข้อมูลเป็นชุด Train และ Test ตามสัดส่วน
    X_train, X_test, y_train, y_test = split_dataset(X, y, TEST_SIZE)

    # เซฟชุดข้อมูลที่แบ่งแล้วเก็บไว้ สำหรับใช้ทดสอบทีหลัง
    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    print("\n[Step 4] Training SVM...")
    # เข้าสู่กระบวนการเทรนโมเดล (ทำ StandardScaler -> PCA -> SVM)
    model, scaler = train_svm(X_train, y_train)

    # บันทึกตัวโมเดล (Model) และตัวปรับสเกล (Scaler) เพื่อนำไปใช้งานจริง
    joblib.dump(model, f"{OUTPUT_DIR}/svm_model.pkl")
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")
    print("SVM training completed.")

    print("\n[Step 5] Testing model...")
    # นำข้อมูลชุด Test ไปทำนายผล
    predictions = predict_svm(model, scaler, X_test)

    print("\n[Step 6] Evaluating model...")
    # วัดผลความแม่นยำและสร้างภาพ Confusion Matrix
    evaluate_model(y_test, predictions, classes, save_path=f"{OUTPUT_DIR}/confusion_matrix.png")

if __name__ == "__main__":
    main()
