import os  
import cv2 
import numpy as np  
from preprocessing import preprocess_image  

# กำหนดนามสกุลไฟล์ภาพที่รองรับ (สร้างตัวแปรเก็บชุดข้อความนามสกุลไฟล์ภาพที่อนุญาตให้ระบบอ่านได้)
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

def load_data(data_path, img_size=100, max_per_class=None):
    # สร้างฟังก์ชันโดยรับค่า ที่อยู่โฟลเดอร์รูปภาพ (data_path), ขนาดภาพเป้าหมาย, และจำนวนรูปลิมิตต่อคลาส
    images = []  # สร้างกล่องลิสต์ว่างเพื่อรอเก็บข้อมูลรูปภาพที่โหลดมา
    labels = []  # สร้างกล่องลิสต์ว่างเพื่อรอเก็บตัวเลขระบุคลาสของภาพ (เช่น 0 = เห็ดกินได้, 1 = เห็ดพิษ)

    # ค้นหาโฟลเดอร์ย่อยใน data_path เพื่อกำหนดให้เป็นคลาส (เช่น Edible, Poisonous)
    classes = sorted([  # เริ่มสร้างลิสต์และเรียงลำดับตัวอักษร (A-Z) เพื่อเก็บรายชื่อโฟลเดอร์
        folder  # ดึงชื่อโฟลเดอร์มาเก็บไว้
        for folder in os.listdir(data_path)  # วนลูปอ่านรายชื่อไฟล์/โฟลเดอร์ทั้งหมด
        if os.path.isdir(os.path.join(data_path, folder))  # กรองเอาเฉพาะสิ่งที่เป็น "โฟลเดอร์" เท่านั้น
    ])
    print("Detected classes:", classes)  # พิมพ์รายชื่อคลาสที่ค้นเจอออกทางหน้าจอ

    # วนลูปเข้าไปอ่านภาพในแต่ละคลาส พร้อมกับสร้างตัวเลขอัตโนมัติ (label) และดึงชื่อโฟลเดอร์ (class_name)
    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_path, class_name)  # เอาเส้นทางหลักมาต่อกับชื่อโฟลเดอร์ ชี้ทางเข้าโฟลเดอร์ย่อย
        
        # กรองเอาเฉพาะไฟล์ที่มีนามสกุลตามที่กำหนด (โดยทำตัวพิมพ์เล็กก่อนเช็ค)
        filenames = sorted(
            f for f in os.listdir(class_path)
            if f.lower().endswith(VALID_EXT)
        )

        loaded = 0  # ตั้งตัวนับจำนวนรูปภาพที่โหลดสำเร็จเป็น 0
        skipped = 0  # ตั้งตัวนับจำนวนรูปภาพที่ไฟล์เสียหรืออ่านไม่ได้เป็น 0
        
        for filename in filenames:  # วนลูปเปิดดูทีละไฟล์ภาพ
            # หยุดอ่านถ้าจำนวนภาพถึงขีดจำกัดที่ตั้งไว้
            if max_per_class and loaded >= max_per_class:
                break  # ถ้าโหลดครบตามลิมิตแล้ว ให้หยุดวนลูปโฟลเดอร์นี้ทันที

            image_path = os.path.join(class_path, filename)  # สร้างเส้นทางเต็มไปยังตัวไฟล์รูปภาพ
            
            # อ่านภาพด้วย OpenCV
            image = cv2.imread(image_path)
            
            # ส่งภาพไปปรับแต่ง (Grayscale และ Resize) แล้วเก็บผลลัพธ์ทับตัวแปรเดิม
            image = preprocess_image(image, img_size)

            # ตรวจสอบว่าภาพเสียหรือไม่ (เช็คว่ารูปภาพมีปัญหาทำให้ได้ค่าว่างหรือไม่)
            if image is None:
                skipped += 1  # ถ้าภาพเสีย ให้นับจำนวนที่ข้ามเพิ่มขึ้น 1
                continue  # ข้ามการทำงานรอบนี้เพื่อไปโหลดภาพต่อไปทันที

            images.append(image)  # ถ้ารูปปกติ นำข้อมูลรูปภาพไปต่อท้ายในลิสต์ images
            labels.append(label)  # นำตัวเลขคลาสไปต่อท้ายในลิสต์ labels
            loaded += 1  # นับว่าโหลดภาพสำเร็จเพิ่มขึ้น 1

        # พิมพ์สรุปรายคลาสว่าโหลดสำเร็จกี่รูป และข้ามรูปพังไปกี่รูป
        print(f"Loaded class {class_name}: {loaded} images ({skipped} skipped)")

    # จับมัดรูปภาพทั้งหมด แปลง List ให้เป็น Numpy Array เพื่อส่งไปคำนวณทางคณิตศาสตร์ต่อ และส่งค่ากลับพร้อมรายชื่อคลาส
    return np.stack(images), np.array(labels), classes
