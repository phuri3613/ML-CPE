import os
import cv2
import numpy as np
from preprocessing import preprocess_image

# กำหนดนามสกุลไฟล์ภาพที่รองรับ
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

def load_data(data_path, img_size=100, max_per_class=None):
    images = []
    labels = []

    # ค้นหาโฟลเดอร์ย่อยใน data_path เพื่อกำหนดให้เป็นคลาส (เช่น Edible, Poisonous)
    classes = sorted([
        folder
        for folder in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, folder))
    ])
    print("Detected classes:", classes)

    # วนลูปเข้าไปอ่านภาพในแต่ละคลาส
    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_path, class_name)
        # กรองเอาเฉพาะไฟล์ที่มีนามสกุลตามที่กำหนด
        filenames = sorted(
            f for f in os.listdir(class_path)
            if f.lower().endswith(VALID_EXT)
        )

        loaded = 0
        skipped = 0
        for filename in filenames:
            # หยุดอ่านถ้าจำนวนภาพถึงขีดจำกัดที่ตั้งไว้
            if max_per_class and loaded >= max_per_class:
                break

            image_path = os.path.join(class_path, filename)
            # อ่านภาพด้วย OpenCV
            image = cv2.imread(image_path)
            # ส่งภาพไปปรับแต่ง (Grayscale และ Resize)
            image = preprocess_image(image, img_size)

            # ตรวจสอบว่าภาพเสียหรือไม่
            if image is None:
                skipped += 1
                continue

            images.append(image)
            labels.append(label)
            loaded += 1

        print(f"Loaded class {class_name}: {loaded} images ({skipped} skipped)")

    # แปลง List ให้เป็น Numpy Array เพื่อส่งไปคำนวณทางคณิตศาสตร์ต่อ
    return np.stack(images), np.array(labels), classes
