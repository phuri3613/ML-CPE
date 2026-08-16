
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CSV_PATH = Path(__file__).resolve().parent.parent / "data-mushroom" / "mushrooms.csv"
TARGET = "class"

def load_data(test_size=0.2, seed=42):

    # step 1 : read CSV
    df = pd.read_csv(CSV_PATH)
    df = df.dropna()            

    # step 2 : convert text to number
    # แยก Target และ Features ออกจากกัน
    y_text = df[TARGET]
    X_text = df[['gill-color', 'veil-color', 'cap-surface', 'habitat']]

    # แปลงผลลัพธ์ (target) เป็นตัวเลข : e (edible) -> 0, p (poisonous) -> 1
    class_names = sorted(y_text.unique())
    y = y_text.map({name: i for i, name in enumerate(class_names)})

    # แปลง Features ที่เป็น Text ให้เป็นตัวเลข (ใช้ Dummy เพราะเป็นข้อมูลที่ไม่มีลำดับความสำคัญ)
    X = pd.get_dummies(X_text)
    feature_names = list(X.columns)

    # บังคับชนิดข้อมูลเป็น 32-bit เพื่อประหยัด RAM และรองรับโมเดลขั้นสูง
    X = X.to_numpy(dtype="float32")
    y = y.to_numpy(dtype="int32")

    # step 3 : split data เป็น train 60 / validation 20 / test 20
    # แบ่ง Test ออกมาก่อน 20%
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    # นำ 80% ที่เหลือมาแบ่งเป็น Train (60%) และ Validation (20%) -> 20/80 = 0.25
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=seed, stratify=y_temp)

    # step 4 : Scaling 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_val = scaler.transform(X_val).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "class_names": class_names,
        "feature_names": feature_names,
        "n_rows": len(df),
    }

if __name__ == "__main__":
    # ทดสอบรันไฟล์นี้เพื่อเช็คขนาดของข้อมูล
    data = load_data()
    print("Train :", data["X_train"].shape)
    print("Val   :", data["X_val"].shape)
    print("Test  :", data["X_test"].shape)
    print("คลาส  :", data["class_names"])