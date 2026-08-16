import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

CSV_PATH = Path(__file__).resolve().parent.parent / "data-mushroom" / "mushrooms.csv"
TARGET = "class"

def load_for_clustering():
    df = pd.read_csv(CSV_PATH)
    df = df.dropna()
    
    # 1. ซ่อนคำตอบ (TARGET) ไม่ให้ K-Means เห็น
    df_features = df.drop(columns=[TARGET])
    
    # 2. เลือก 4 ฟีเจอร์เพื่อให้กราฟ PCA 2D กระจายตัวแบ่งกลุ่มได้สวยงาม
    df_features = df_features[['gill-color', 'veil-color', 'cap-surface', 'habitat']]
    
    X = pd.get_dummies(df_features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, df