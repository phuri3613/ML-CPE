from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def train_svm(X_train, y_train, pca_components=150):
    # ใช้ Pipeline สร้างท่อร้อยเรียงขั้นตอน ป้องกัน Data Leakage
    scaler = Pipeline([
        # 1. ปรับสเกลข้อมูลให้มี Mean=0 และ Variance=1
        ("scaler", StandardScaler()),
        
        # 2. ทำ PCA (Principal Component Analysis) 
        # เพื่อลดมิติภาพจาก 10,000 ฟีเจอร์ ให้เหลือเฉพาะฟีเจอร์สำคัญ 150 ตัว ช่วยลด Overfitting
        ("pca", PCA(n_components=min(pca_components, *X_train.shape),
                    whiten=True, random_state=42)),
    ])
    
    # คำนวณและปรับสเกลข้อมูล X_train
    X_train_scaled = scaler.fit_transform(X_train)

    # 3. กำหนดโมเดล SVM
    # ใช้ RBF Kernel สำหรับข้อมูลที่ไม่เป็นเส้นตรง
    # C=10 คือค่ายอมรับความผิดพลาด (ถ้า C สูง โมเดลจะพยายามจัดกลุ่มให้เป๊ะที่สุด)
    model = SVC(kernel="rbf", C=10, gamma="scale", cache_size=1000)
    
    # สั่งให้โมเดลเรียนรู้จากข้อมูลที่เตรียมไว้
    model.fit(X_train_scaled, y_train)

    # รีเทิร์นโมเดลที่เทรนเสร็จ และ Pipeline ของ scaler
    return model, scaler

def predict_svm(model, scaler, X_test):
    # นำข้อมูล Test มาผ่านกระบวนการลดมิติ (Scaler & PCA) แบบเดียวกับตอน Train
    X_test_scaled = scaler.transform(X_test)
    
    # ให้โมเดลทำนายผล
    predictions = model.predict(X_test_scaled)
    return predictions
