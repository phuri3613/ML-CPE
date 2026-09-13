import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset(X, y, test_size=0.2):
    y = np.asarray(y)
    
    # แบ่งข้อมูล X (Features) และ y (Labels) 
    # test_size=0.2 คือแบ่งเอาไว้ทดสอบ 20%
    # random_state=42 ฟิกค่าการสุ่มไว้ เพื่อให้รันกี่ครั้งข้อมูลก็ถูกแบ่งแบบเดิม
    # stratify=y บังคับให้สัดส่วนของคลาส (เห็ดพิษ/กินได้) กระจายตัวเท่าๆ กันในชุด Train และ Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=42,
        stratify=y
    )
    return X_train, X_test, y_train, y_test
