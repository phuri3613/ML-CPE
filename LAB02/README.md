# AI Impact on Students - Data Preprocessing

## 📖 Project Overview | ภาพรวมโครงการ

### English
This project presents **Exploratory Data Analysis (EDA)** and **Data Preprocessing** on the **AI Impact on Students** dataset. The dataset contains information about students' academic performance, AI usage, study habits, stress levels, burnout risk, and learning behaviors. The objective of this project is to analyze the relationships among these factors and prepare the dataset for future machine learning applications. The workflow includes data exploration, visualization, missing value handling, duplicate removal, categorical feature encoding, and exporting the cleaned dataset. :contentReference[oaicite:0]{index=0}

### ภาษาไทย
โปรเจกต์นี้นำเสนอการวิเคราะห์ข้อมูลเบื้องต้น (Exploratory Data Analysis: EDA) และการเตรียมข้อมูล (Data Preprocessing) โดยใช้ชุดข้อมูล **AI Impact on Students** ซึ่งประกอบด้วยข้อมูลเกี่ยวกับผลการเรียน การใช้งาน AI พฤติกรรมการเรียน ระดับความเครียด ความเสี่ยงต่อภาวะหมดไฟ (Burnout Risk) และพฤติกรรมการเรียนรู้ของนักศึกษา จุดประสงค์ของโปรเจกต์คือการสำรวจความสัมพันธ์ของข้อมูลและเตรียมชุดข้อมูลให้พร้อมสำหรับการนำไปใช้ในงาน Machine Learning ต่อไป :contentReference[oaicite:1]{index=1}

---

## 🎯 Objectives | วัตถุประสงค์

- Perform Exploratory Data Analysis (EDA)
- Visualize numerical and categorical features
- Handle missing values
- Remove duplicate records
- Apply Label Encoding to categorical features
- Export a cleaned dataset for machine learning

### ภาษาไทย

- วิเคราะห์ข้อมูลเบื้องต้น (EDA)
- สร้างกราฟเพื่อศึกษาการกระจายและความสัมพันธ์ของข้อมูล
- จัดการข้อมูลที่หายไป (Missing Values)
- ลบข้อมูลซ้ำ (Duplicate Records)
- แปลงข้อมูลประเภทข้อความด้วย Label Encoding
- ส่งออกชุดข้อมูลที่ผ่านการเตรียมแล้ว

---

## 📂 Dataset

**Dataset Name**

AI Impact on Students

**Description**

The dataset contains student-related information including demographics, GPA, AI usage hours, traditional study hours, AI dependency, burnout risk, anxiety level, sleep duration, and other educational factors. It is designed for data analysis, visualization, and machine learning practice. :contentReference[oaicite:2]{index=2}

**ภาษาไทย**

ชุดข้อมูลประกอบด้วยข้อมูลของนักศึกษา เช่น ข้อมูลพื้นฐาน เกรดเฉลี่ย (GPA) ชั่วโมงการใช้ AI ชั่วโมงการเรียนแบบดั้งเดิม ระดับการพึ่งพา AI ความเสี่ยงต่อภาวะหมดไฟ (Burnout Risk) ระดับความวิตกกังวล ชั่วโมงการนอน และปัจจัยอื่น ๆ ที่เกี่ยวข้องกับการศึกษา เหมาะสำหรับการวิเคราะห์ข้อมูล การสร้างภาพข้อมูล และการฝึกสร้างโมเดล Machine Learning

---

## 📊 Project Workflow

- Load Dataset
- Exploratory Data Analysis (EDA)
- Data Visualization
- Missing Value Handling
- Duplicate Removal
- Label Encoding
- Export Clean Dataset

---

## 📈 Data Visualization

The project includes:

- Distribution of Numerical Features (Histogram)
- Correlation Heatmap
- Burnout Risk Level Distribution (Count Plot)

---

## 🛠 Programming Language

- Python 3.x

## 📚 Libraries

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 📤 Output

The program generates:

- Dataset Information
- Summary Statistics
- Missing Value Report
- Duplicate Record Report
- Histogram of Numerical Features
- Correlation Heatmap
- Burnout Risk Distribution
- Cleaned Dataset (`ai_cleaned_dataset_kaggle.csv`)

---

## 📚 References

### Dataset

Lavesh Jadon. **AI Impact on Students Dataset**.

https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students

### Notebook Reference

Rasya Arrafi. **Impact of AI on Students**.

https://www.kaggle.com/code/rasyaarrafi/impact-of-ai-on-students

