# Machine Learning Project: Edible and Poisonous Mushroom Image Classification (Convolutional Neural Network - CNN)

## ภาพรวมโปรเจกต์ (Project Overview)
โปรเจกต์นี้เป็นการประยุกต์ใช้เทคนิค Deep Learning โดยใช้โครงข่ายประสาทเทียมแบบคอนโวลูชัน (Convolutional Neural Network - CNN) ในการจำแนกประเภทของเห็ดว่า "มีพิษ" (Poisonous) หรือ "สามารถรับประทานได้" (Edible) จากข้อมูลภาพถ่าย (Image Dataset) โดยเน้นไปที่การออกแบบสถาปัตยกรรม CNN ที่มีประสิทธิภาพ การแก้ปัญหาข้อมูลภาพมีจำนวนน้อย (Small Dataset) ด้วยเทคนิค Data Augmentation และการปรับจูนพารามิเตอร์ (Hyperparameter Tuning) เช่น Batch Size และกลไก Early Stopping เพื่อให้โมเดลเกิดการเรียนรู้ที่เหมาะสมและไม่เกิดการ Overfitting เร็วเกินไป พร้อมทั้งปรับแต่งสภาพแวดล้อมให้รองรับการประมวลผลด้วยการ์ดจอ (GPU Acceleration)

## วัตถุประสงค์ (Objectives)
- จัดเตรียมชุดข้อมูลภาพถ่าย (Image Preprocessing) และเพิ่มความหลากหลายของข้อมูล (Data Augmentation) เพื่อแก้ปัญหาจำนวนข้อมูลที่มีจำกัด
- สร้างและออกแบบสถาปัตยกรรมโมเดล Convolutional Neural Network (CNN) สำหรับดึงคุณลักษณะเฉพาะ (Feature Extraction) จากรูปภาพ
- ฝึกสอนโมเดลโดยใช้เทคนิคควบคุมการเรียนรู้ (Early Stopping, ReduceLROnPlateau)
- ประเมินประสิทธิภาพของโมเดลผ่านค่าความแม่นยำ (Accuracy), ค่าความสูญเสีย (Loss) และวิเคราะห์ความถูกต้องผ่าน Confusion Matrix
- ตั้งค่า Environment เพื่อดึงพลังการประมวลผลจากการ์ดจอ NVIDIA GPU (CUDA/cuDNN) มาใช้ลดเวลาในการฝึกสอนโมเดล

## ชุดข้อมูล (Dataset)
- **ชื่อชุดข้อมูล:** Edible and Poisonous Mushroom Images
- **คำอธิบาย:** ชุดข้อมูลรูปภาพ (Image Data) ของเห็ด แบ่งย่อยเป็น 2 คลาส ได้แก่ `edible mushroom/` (เห็ดที่รับประทานได้) และ `poisonous mushroom/` (เห็ดมีพิษ)
- **ที่มา (Source):** Kaggle

## โครงสร้างโปรเจกต์ (Project Structure)
```text
LAB07/
├── MushroomImages/
│   ├── edible mushroom/
│   └── poisonous mushroom/
└── classification/
    ├── __pycache__/
    ├── outputs/
    │   ├── X_test.npy
    │   ├── X_train.npy
    │   ├── X_val.npy
    │   ├── classes.json
    │   ├── cnn_model.keras
    │   ├── confusion_matrix.png
    │   ├── features.npy
    │   ├── history.json
    │   ├── labels.npy
    │   ├── training_history.png
    │   ├── y_test.npy
    │   ├── y_train.npy
    │   └── y_val.npy
    ├── cnn_model.py
    ├── data_loader.py
    ├── evaluate.py
    ├── main.py
    ├── preprocessing.py
    ├── split_data.py
    ├── test_cnn.py
    └── README.md
```

## ขั้นตอนการทำงาน (Project Workflow)
1. **Data Loading & Preprocessing (`data_loader.py`, `preprocessing.py`, `split_data.py`):** โหลดภาพจากโฟลเดอร์ ทำการปรับขนาดภาพ (Resize) และแบ่งชุดข้อมูลออกเป็น Train, Validation และ Test
2. **Model Definition (`cnn_model.py`):** กำหนดโครงสร้างของ CNN ประกอบด้วย Layer ต่างๆ เช่น Rescaling, Data Augmentation, Conv2D, Batch Normalization, MaxPooling2D, GlobalAveragePooling2D และ Dense Layer พร้อม Dropout เพื่อลด Overfitting
3. **Model Training (`main.py`):** ฝึกสอนโมเดลด้วยค่าตัวแปรที่กำหนดเพื่อชดเชยจำนวนภาพที่มีน้อย มีการเรียกใช้ Callbacks (`EarlyStopping` และ `ReduceLROnPlateau`) โมเดลจะรันผ่านการประมวลผล GPU
4. **Model Evaluation (`evaluate.py`, `test_cnn.py`):** นำประวัติการฝึกมาประเมินประสิทธิภาพ วาดกราฟเปรียบเทียบ Accuracy/Loss และสร้าง Confusion Matrix

## การแสดงผลข้อมูลและการประเมินผล (Data Visualization & Model Evaluation)
จากการทดลองปรับแก้ค่าพารามิเตอร์เพื่อฝึกสอนโมเดล CNN ด้วยชุดข้อมูลขนาดเล็ก สามารถสรุปผลการทำงานผ่านกราฟผลลัพธ์ได้ดังนี้:

### 1. ประวัติการฝึกสอน (Training History)
![Training History](outputs/training_history.png)

- **เส้น Train (สีฟ้า):** มีความเหวี่ยงขึ้นลงอย่างเป็นธรรมชาติ (Fluctuations) มากขึ้นจากการปรับลด Batch Size ลง แสดงให้เห็นกระบวนการเรียนรู้และปรับตัวของโครงข่ายประสาทเทียม
- **เส้น Validation (สีส้ม):** กราฟมีลักษณะการกระโดดขึ้นเป็นขั้นบันได เนื่องจากชุดข้อมูล Validation มีขนาดเล็กมาก (หลักหน่วย) เมื่อโมเดลทายถูกเพิ่มเพียงไม่กี่ภาพ สัดส่วนเปอร์เซ็นต์จึงพุ่งสูงขึ้นอย่างรวดเร็ว
- **ภาพรวมการเทรน:** การยืดระยะเวลา Patience ของ Early Stopping ช่วยให้กราฟสามารถเทรนต่อเนื่องไปได้ไกลขึ้นและมีโอกาสลู่เข้า (Converge) ได้ดีกว่าเดิม

### 2. เมทริกซ์ความสับสน (Confusion Matrix)
![Confusion Matrix](outputs/confusion_matrix.png)

- *คำอธิบาย: เมทริกซ์แสดงผลการทดสอบการจำแนกประเภทเห็ดของโมเดลเทียบกับความจริง เพื่อตรวจสอบความเอนเอียง (Bias) และความแม่นยำในการแยกแยะคลาสที่มีพิษและไม่มีพิษ*

## ภาษาโปรแกรมและไลบรารี (Programming Language & Libraries)
- **ภาษาโปรแกรม:** Python 3.10
- **Environment:** Anaconda (Environment `ml_gpu`)
- **ไลบรารีหลัก:**
  - `tensorflow==2.10.0` / `keras` (รองรับ GPU Acceleration บน Windows)
  - `cudatoolkit=11.2`, `cudnn=8.1.0`
  - `opencv-python` (cv2), `scikit-learn`, `numpy`, `matplotlib`

## ผลลัพธ์ที่ได้ (Output)
หลังจากการรันระบบเสร็จสิ้นด้วย GPU โมเดลโครงข่ายประสาทเทียม (`cnn_model.keras`) ประวัติการทำงาน (`history.json`) ข้อมูลอาร์เรย์ (`.npy`) และรูปภาพกราฟผลการประเมินทั้งหมด จะถูกบันทึกและจัดเก็บโดยอัตโนมัติในโฟลเดอร์ `outputs/`

## เอกสารอ้างอิง (References)
- Dataset: Edible and Poisonous Mushroom Images Dataset: Kaggle https://www.kaggle.com/datasets/mdismielhossenabir/edible-and-poisonous-mushroom-images
