import cv2
import numpy as np

def preprocess_image(image, img_size=100):
    if image is None or image.size == 0:
        return None

    # หากภาพมี 3 แชนเนล (ภาพสี BGR) ให้แปลงเป็นขาวดำ (Grayscale) เพื่อลดขนาดข้อมูล
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ย่อ/ขยายภาพให้เป็นขนาด img_size x img_size ตามที่ระบบต้องการ
    image = cv2.resize(
        image,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA
    )
    return image

def to_features(images):
    # Flatten ข้อมูล: แปลงภาพ 2 มิติ (เช่น 100x100) ให้เป็นแถวเรียงยาว 1 มิติ (10,000 ฟีเจอร์)
    features = images.reshape(len(images), -1).astype(np.float32)
    
    # Normalization: หารด้วย 255 เพื่อให้ค่าพิกเซลทั้งหมดมีช่วงอยู่ระหว่าง 0 ถึง 1 
    # ช่วยให้โมเดล Machine Learning เทรนได้รวดเร็วและแม่นยำขึ้น
    features /= 255.0
    return features

def preprocess_images(images, img_size=100):
    # ฟังก์ชันเสริมสำหรับวนลูปจัดการหลายๆ ภาพพร้อมกัน
    processed = [preprocess_image(img, img_size) for img in images]
    processed = [img for img in processed if img is not None]
    return to_features(np.stack(processed))
