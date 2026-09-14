import matplotlib  
matplotlib.use("Agg")  
import matplotlib.pyplot as plt  
import numpy as np  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 

def evaluate_model(y_test, predictions, classes, save_path=None):
    # ฟังก์ชันสำหรับประเมินผลโมเดล รับค่า 4 อย่าง: ผลเฉลยจริง (y_test), คำตอบที่ AI ทาย (predictions), รายชื่อคลาส และที่อยู่สำหรับเซฟรูป
    labels = list(range(len(classes)))  # สร้างลิสต์ตัวเลขแทนคลาสทั้งหมด (เช่น [0, 1] สำหรับข้อมูล 2 คลาส)
    accuracy = accuracy_score(y_test, predictions)  # คำนวณค่าความแม่นยำ (Accuracy) ว่าทายถูกกี่เปอร์เซ็นต์

    print("\n------------ Evaluation ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")  # แสดงผลค่าความแม่นยำเป็นเปอร์เซ็นต์ (ทศนิยม 2 ตำแหน่ง)
    print("\nClassification Report:")
    
    # สร้างรายงานสรุปผลการประเมินเชิงลึก (Precision, Recall, F1-score) แยกรายคลาส
    report = classification_report(
        y_test, predictions, labels=labels, target_names=classes, zero_division=0
    )
    
    print(report)  # พิมพ์รายงานเชิงลึกออกทางหน้าจอ
    print("Confusion Matrix:")
    matrix = confusion_matrix(y_test, predictions, labels=labels)  # คำนวณตาราง Confusion Matrix ออกมาเป็นชุดตัวเลข
    print(matrix)  # พิมพ์ตารางตัวเลขออกทางหน้าจอ

    if save_path:  # ตรวจสอบว่ามีการระบุที่อยู่ไฟล์สำหรับเซฟรูปภาพมาด้วยหรือไม่
        plot_confusion_matrix(matrix, classes, save_path)  # เรียกใช้ฟังก์ชันด้านล่างเพื่อนำตัวเลขไปวาดเป็นกราฟตารางภาพ
        print(f"Saved: {save_path}")  # พิมพ์แจ้งเตือนว่าเซฟรูปภาพลงไฟล์เรียบร้อยแล้ว

    return accuracy  # ส่งค่าความแม่นยำกลับไปให้ระบบนำไปใช้งานต่อ

def plot_confusion_matrix(matrix, classes, save_path):
    # ฟังก์ชันสำหรับวาดรูปกราฟตาราง Confusion Matrix และเซฟเป็นไฟล์รูปภาพ
    fig, ax = plt.subplots(figsize=(5, 5))  # สร้างพื้นที่สำหรับวาดกราฟ (Figure) กำหนดขนาดไว้ที่ 5x5 นิ้ว
    ax.imshow(matrix, cmap="Blues")  # นำตัวเลขตารางมาพล็อตเป็นสี (Heatmap) โดยใช้โทนสีฟ้า (เลขมากสีเข้ม เลขน้อยสีอ่อน)

    ax.set_xticks(np.arange(len(classes)), classes)  # กำหนดชื่อคลาสในแกน X (แนวนอน)
    ax.set_yticks(np.arange(len(classes)), classes)  # กำหนดชื่อคลาสในแกน Y (แนวตั้ง)
    ax.set_xlabel("Predicted")  # ตั้งชื่อป้ายกำกับแกน X ว่า "Predicted" (คำตอบที่โมเดลทายออกมา)
    ax.set_ylabel("True")  # ตั้งชื่อป้ายกำกับแกน Y ว่า "True" (ความเป็นจริงตามผลเฉลย)
    ax.set_title("Confusion Matrix")  # ตั้งชื่อหัวข้อกราฟ

    threshold = matrix.max() / 2  # คำนวณค่ากึ่งกลางของตาราง เพื่อเอาไว้เป็นเกณฑ์เลือกสีตัวอักษรให้ตัดกับสีพื้นหลัง
    
    # วนลูปเข้าไปในตารางแต่ละช่อง เพื่อเขียนตัวเลขกำกับลงไปบนภาพ
    for i in range(len(classes)):
        for j in range(len(classes)):
            # เขียนตัวเลขลงกลางช่อง โดยถ้าพื้นหลังเป็นสีเข้ม (เลขมากกว่าเกณฑ์) ให้ใช้ฟอนต์สีขาว ถ้าพื้นอ่อนให้ใช้ฟอนต์สีดำ
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")

    fig.tight_layout()  # จัดระเบียบสัดส่วนของกราฟให้อัตโนมัติ ป้องกันไม่ให้ข้อความหรือขอบภาพขาดหาย
    fig.savefig(save_path, dpi=150)  # บันทึกรูปภาพลงเครื่องตามที่อยู่ save_path ด้วยความละเอียด 150 DPI
    plt.close(fig)  # ปิดการทำงานของกราฟนี้ทิ้ง เพื่อคืนหน่วยความจำ (RAM) ให้คอมพิวเตอร์
