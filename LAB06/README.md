# Machine Learning Project: Edible and Poisonous Mushroom Image Classification (Neural Network)

## ภาพรวมโปรเจกต์ (Project Overview)
โปรเจกต์นี้เป็นการประยุกต์ใช้เทคนิค Machine Learning โดยใช้โครงข่ายประสาทเทียม (Neural Network - NN) ในการจำแนกประเภทของเห็ดว่า "มีพิษ" (Poisonous) หรือ "สามารถรับประทานได้" (Edible) จากข้อมูลภาพถ่าย (Image Dataset) โดยเน้นไปที่การเปรียบเทียบประสิทธิภาพของโมเดลเมื่อมีการปรับเปลี่ยนโครงสร้าง (Configurations) เช่น จำนวน Hidden Layers/Neurons และเปรียบเทียบผลลัพธ์จากการใช้จำนวนรอบการฝึกสอน (Epochs) ที่แตกต่างกัน

## วัตถุประสงค์ (Objectives)
- จัดเตรียม แย่งชุดข้อมูล (Train/Test Split) และทำ Standardization ข้อมูลรูปภาพเห็ดเพื่อใช้ฝึกสอนโมเดล
- สร้างและออกแบบสถาปัตยกรรมโมเดล Neural Network (NN)
- ฝึกสอนโมเดลและเปรียบเทียบประสิทธิภาพระหว่าง Configuration ที่ต่างกัน รวมถึงเปรียบเทียบผลการใช้จำนวน Epochs ที่ต่างกัน (เช่น 20 และ 50 Epochs)
- ประเมินประสิทธิภาพของแต่ละโมเดลผ่านค่าความแม่นยำ (Accuracy), ค่าความสูญเสีย (Loss) และวิเคราะห์ความถูกต้องผ่าน Confusion Matrix

## ชุดข้อมูล (Dataset)
- **ชื่อชุดข้อมูล:** Edible and Poisonous Mushroom Images
- **คำอธิบาย:** ชุดข้อมูลรูปภาพ (Image Data) ของเห็ด แบ่งย่อยเป็น 2 คลาส ได้แก่ `edible mushroom/` (เห็ดที่รับประทานได้) และ `poisonous mushroom/` (เห็ดมีพิษ)
- **ที่มา (Source):** [Kaggle](https://www.kaggle.com/datasets/mdismielhossenabir/edible-and-poisonous-mushroom-images)

## โครงสร้างโปรเจกต์ (Project Structure)
```text
LAB06/
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
    │   ├── confusion_matrix_config_1_20ep.png
    │   ├── confusion_matrix_config_1_50ep.png
    │   ├── confusion_matrix_config_2_20ep.png
    │   ├── confusion_matrix_config_2_50ep.png
    │   ├── features.npy
    │   ├── history_config_1_20ep.json
    │   ├── history_config_1_50ep.json
    │   ├── history_config_2_20ep.json
    │   ├── history_config_2_50ep.json
    │   ├── labels.npy
    │   ├── nn_model_config_1_20ep.keras
    │   ├── training_history_config_2_20ep.png
    │   ├── training_history_config_2_50ep.png
    │   ├── y_test.npy
    │   ├── y_train.npy
    │   └── y_val.npy
    ├── data_loader.py
    ├── evaluate.py
    ├── main.py
    ├── nn_model.py
    ├── preprocessing.py
    ├── split_data.py
    ├── test_nn.py
    ├── README.md
    ├── link-data.txt
    └── requirements.txt
```

## ขั้นตอนการทำงาน (Project Workflow)
1. **Data Loading & Preprocessing:** โหลดภาพ สกัดคุณลักษณะ และบันทึกข้อมูลเป็น `features.npy` และ `labels.npy` (`data_loader.py`)
2. **Model Definition:** กำหนดโครงสร้างของ Neural Network สำหรับการทดลองแต่ละ Configuration (`nn_model.py`)
3. **Model Training:** ฝึกสอนโมเดลตาม Configuration และจำนวน Epochs (20 และ 50) ที่กำหนด พร้อมเก็บบันทึกประวัติการฝึกเป็นไฟล์ `.json` และตัวโมเดลเป็นไฟล์ `.keras` (`main.py`)
4. **Model Evaluation:** นำประวัติการฝึกและผลการทำนายมาประเมินประสิทธิภาพ วาดกราฟเปรียบเทียบ Accuracy/Loss และ Confusion Matrix (`evaluate.py`)

## การแสดงผลข้อมูลและการประเมินผล (Data Visualization & Model Evaluation)
จากการทดลองฝึกสอนโมเดล Neural Network (ตัวอย่างจากการตั้งค่า Config 1 จำนวน 20 Epochs) สามารถสรุปและวิเคราะห์ประสิทธิภาพของโมเดลผ่านกราฟผลลัพธ์ได้ดังนี้:

## การแสดงผลข้อมูลและการประเมินผล (Data Visualization & Model Evaluation)
จากการทดลองฝึกสอนโมเดล Neural Network (ตัวอย่างจากการตั้งค่า Config 1 จำนวน 20 Epochs) สามารถสรุปและวิเคราะห์ประสิทธิภาพของโมเดลผ่านกราฟผลลัพธ์ได้ดังนี้:

### 1. ประวัติการฝึกสอน (Training History)

<div align="center">
  <img src="outputs/training_history_config_1_20ep.png" width="80%">
  <p><b>รูปที่ 1: Training History (Accuracy & Loss)</b></p>
</div>

* กราฟแสดงการเปลี่ยนแปลงของค่าความแม่นยำ (Accuracy) และค่าความสูญเสีย (Loss) ในระหว่างรอบการฝึกสอน 
* เส้นกราฟ Validation (สีส้ม) มีความผันผวน (Fluctuation) ค่อนข้างสูงมากเมื่อเทียบกับ Training (สีน้ำเงิน) บ่งชี้ว่าโมเดลอาจเผชิญกับภาวะความไม่เสถียรระหว่างการเรียนรู้ หรือชุดข้อมูล Validation อาจมีขนาดเล็กและไม่ครอบคลุมลักษณะข้อมูลทั้งหมด

### 2. เมทริกซ์ความสับสน (Confusion Matrix)

<div align="center">
  <img src="outputs/confusion_matrix_config_1_20ep.png" width="80%">
  <p><b>รูปที่ 2: Confusion Matrix (Config 1 - 20 Epochs)</b></p>
</div>

* เมทริกซ์แสดงให้เห็นว่าโมเดลสามารถทำนายคลาส `poisonous mushroom` ได้ถูกต้อง 8 ตัวอย่าง และทำนาย `edible mushroom` ถูกต้องเพียง 4 ตัวอย่าง
* โมเดลมีความผิดพลาดในการทำนายเห็ดที่กินได้ (True: edible) ว่าเป็นเห็ดมีพิษจำนวนถึง 4 ตัวอย่าง ซึ่งคิดเป็นสัดส่วนความผิดพลาดที่ค่อนข้างสูงสำหรับคลาสนี้

### 3. ตัวอย่างผลการทำนาย (Prediction Sample)

<div align="center">
  <img src="outputs/prediction_sample.png" width="80%">
  <p><b>รูปที่ 3: Prediction Sample</b></p>
</div>

* จากการสุ่มภาพมาทดสอบ 4 ภาพ โมเดลสามารถทำนายได้ถูกต้องเพียง 2 ภาพ (ความแม่นยำ 50% สำหรับกลุ่มตัวอย่างนี้)
* สิ่งที่น่าสังเกตจากภาพคือ **โมเดลเกิดความเอนเอียง (Model Bias) อย่างรุนแรง** โดยทำนายผลลัพธ์ของทั้ง 4 ภาพว่าเป็น `edible mushroom` ทั้งหมดด้วยค่าความมั่นใจ (Confidence) ที่สูงเกิน 88% ทุกภาพ แม้ว่าความจริงแล้วจะมีภาพเห็ดมีพิษปะปนอยู่ด้วยก็ตาม (ถือเป็น False Positive ที่อันตรายมากในบริบทของการจำแนกเห็ดพิษ)
## ภาษาโปรแกรมและไลบรารี (Programming Language & Libraries)
- **ภาษาโปรแกรม:** Python 3
- **ไลบรารี:** `tensorflow` / `keras` (สำหรับสร้าง Neural Network), `scikit-learn`, `numpy`, `matplotlib`

## ผลลัพธ์ที่ได้ (Output)
หลังจากการรันระบบเสร็จสิ้น โมเดล `.keras` ที่ถูกฝึกสอนสำเร็จ ประวัติการทำงาน (`.json`) ข้อมูลอาร์เรย์ และรูปภาพกราฟผลการประเมินทั้งหมด จะถูกบันทึกและจัดเก็บโดยอัตโนมัติในโฟลเดอร์ `classification/outputs/`


## เอกสารอ้างอิง (References)
* **Dataset:** Edible and Poisonous Mushroom Images Dataset: Kaggle https://www.kaggle.com/datasets/mdismielhossenabir/edible-and-poisonous-mushroom-images

* ## การแสดงผลข้อมูลและการประเมินผล (Data Visualization & Model Evaluation)
จากการทดลองฝึกสอนโมเดล Neural Network สามารถสรุปและวิเคราะห์ประสิทธิภาพของโมเดลผ่านกราฟผลลัพธ์ได้ดังนี้:

### 1. ประวัติการฝึกสอน (Training History)

<div align="center">
  <img src="outputs/training_history_config_2_20ep.png" width="80%">
  <p><b>รูปที่ 1: Training History (Accuracy & Loss)</b></p>
</div>

* กราฟแสดงการเปลี่ยนแปลงของค่าความแม่นยำ (Accuracy) และค่าความสูญเสีย (Loss) ในระหว่างรอบการฝึกสอน 
* เส้นกราฟ Validation (สีส้ม) มีความผันผวน (Fluctuation) ค่อนข้างสูงมากเมื่อเทียบกับ Training (สีน้ำเงิน) บ่งชี้ว่าโมเดลอาจเผชิญกับภาวะความไม่เสถียรระหว่างการเรียนรู้ หรือชุดข้อมูล Validation อาจมีขนาดเล็กและไม่ครอบคลุมลักษณะข้อมูลทั้งหมด

### 2. เมทริกซ์ความสับสน (Confusion Matrix)

<div align="center">
  <img src="outputs/confusion_matrix_config_1_20ep.png" width="80%">
  <p><b>รูปที่ 2: Confusion Matrix (Config 1 - 20 Epochs)</b></p>
</div>

* เมทริกซ์แสดงให้เห็นว่าโมเดลสามารถทำนายคลาส `poisonous mushroom` ได้ถูกต้อง 8 ตัวอย่าง และทำนาย `edible mushroom` ถูกต้องเพียง 4 ตัวอย่าง
* โมเดลมีความผิดพลาดในการทำนายเห็ดที่กินได้ (True: edible) ว่าเป็นเห็ดมีพิษจำนวนถึง 4 ตัวอย่าง ซึ่งคิดเป็นสัดส่วนความผิดพลาดที่ค่อนข้างสูงสำหรับคลาสนี้
