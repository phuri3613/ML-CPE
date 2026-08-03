#  NBA Player Statistics - Regression & Classification

##  ภาพรวมโปรเจกต์ (Project Overview)
โปรเจกต์นี้เป็นการทำ Data Preprocessing และประยุกต์ใช้ Machine Learning (Regression & Classification) กับชุดข้อมูลสถิตินักบาสเกตบอล NBA โดยมีจุดประสงค์เพื่อวิเคราะห์ตัวแปรทางสถิติที่มีผลต่อการทำคะแนน และการจำแนกตำแหน่งการเล่นของนักกีฬา กระบวนการทำงานครอบคลุมตั้งแต่การจัดการค่าว่างในข้อมูล (Missing Value Handling), การลดมิติข้อมูลด้วย PCA, การสร้างแบบจำลอง ไปจนถึงการประเมินประสิทธิภาพของโมเดลด้วยมาตรวัดทางสถิติและกราฟ

##  วัตถุประสงค์ (Objectives)
* จัดเตรียมข้อมูลและจัดการค่าที่ขาดหายไป (Missing Value Handling)
* ลดมิติข้อมูลที่มีความซับซ้อนโดยใช้ PCA (Principal Component Analysis)
* สร้างโมเดล Linear Regression เพื่อทำนาย **คะแนนเฉลี่ยต่อเกม (PTS)**
* สร้างโมเดล Logistic Regression เพื่อจำแนก **ตำแหน่งผู้เล่น (Guard vs Forward/Center)**
* ประเมินประสิทธิภาพของโมเดลและแสดงผลข้อมูล (Data Visualization)

##  ชุดข้อมูล (Dataset)
* **ชื่อชุดข้อมูล:** 2022-2023 NBA Player Stats - Regular
* **คำอธิบาย:** ชุดข้อมูลตาราง (Tabular Data) เก็บบันทึกสถิติการเล่นของนักบาสเกตบอล NBA ในฤดูกาลปกติ ประกอบด้วยคุณลักษณะต่างๆ เช่น อายุ, ตำแหน่ง, นาทีที่ลงเล่น, เปอร์เซ็นต์การชู้ต, การรีบาวด์ และการแอสซิสต์ ซึ่งเหมาะสำหรับการนำมาฝึกฝนการวิเคราะห์ข้อมูลและ Machine Learning

##  ขั้นตอนการทำงาน (Project Workflow)
1. **Load Dataset:** โหลดข้อมูลจากไฟล์ CSV
2. **Data Preprocessing:** ทำความสะอาดข้อมูลและแทนที่ค่าว่าง (NaN) ด้วย 0 เฉพาะคอลัมน์ตัวเลข
3. **Dimensionality Reduction:** ปรับสเกลข้อมูล (Standardization) และทำ PCA โดยคงความแปรปรวนไว้ที่ 95%
4. **Regression (LAB 1):** ฝึกสอนและทดสอบโมเดลทายผลคะแนนต่อเนื่อง
5. **Classification (LAB 2):** แปลงข้อมูลตำแหน่งให้เป็นหมวดหมู่ (Binary) และฝึกสอนโมเดลเพื่อจำแนกกลุ่ม
6. **Model Evaluation (LAB 3):** วัดผลโมเดลด้วยตัวชี้วัดประสิทธิภาพต่างๆ

##  การแสดงผลข้อมูล (Data Visualization)
โปรเจกต์นี้รวมการสร้างกราฟแสดงผลดังนี้:
* **Actual vs Predicted Points:** กราฟ Scatter Plot เปรียบเทียบคะแนนจริงและคะแนนที่โมเดลทำนายได้ (Linear Regression)
* **Confusion Matrix:** ตารางเมทริกซ์ความสับสนแสดงจำนวนการทายถูก/ผิด ในการจำแนกตำแหน่ง (Heatmap)
* **ROC Curve:** กราฟประเมินความสามารถในการแบ่งแยกคลาสของแบบจำลอง Classification พร้อมพื้นที่ใต้กราฟ (AUC)

---

##  ภาษาโปรแกรม (Programming Language)
* Python 3

##  ไลบรารี (Libraries)
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`

##  ผลลัพธ์ที่ได้ (Output)
เมื่อรันโปรแกรมสำเร็จ ระบบจะแสดงผลลัพธ์ดังนี้:
* ข้อมูลสรุปจำนวนแถวและคอลัมน์ที่โหลดสำเร็จ
* ผลการประเมิน Regression: ค่า R-squared ($R^2$) และ MSE
* กราฟเปรียบเทียบการทำนายคะแนน (Scatter Plot)
* กราฟตารางจำแนกตำแหน่ง (Confusion Matrix)
* ผลการประเมิน Classification: Accuracy, Precision, Recall และ F1-score
* กราฟ ROC Curve ประเมินประสิทธิภาพขั้นสูง

##  เอกสารอ้างอิง (References)
* **Dataset:** 
  joebeachcapital. *NBA Player Statistics*. 
  https://www.kaggle.com/datasets/joebeachcapital/nba-player-statistics
