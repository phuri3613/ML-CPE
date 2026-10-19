import torch
from pathlib import Path
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def main():
    device = 0 if torch.cuda.is_available() else 'cpu'

    # ค้นหาไฟล์โมเดลที่ดีที่สุด (best.pt) ในโฟลเดอร์ runs ให้อัตโนมัติ
    weights_paths = list((ROOT / 'runs').rglob('best.pt'))
    if not weights_paths:
        raise FileNotFoundError("Model not found: ไม่พบไฟล์ best.pt ในโฟลเดอร์ runs")
    
    # เลือกใช้ไฟล์ล่าสุดที่เจอ
    weights = weights_paths[-1]
    print(f"--> โหลดโมเดลจาก: {weights}")

    # โหลดโมเดล
    model = YOLO(str(weights))

    # เปิดกล้องเว็บแคม (index 0 คือกล้องตัวแรกของเครื่อง)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError('Cannot open webcam (camera index 0)')

    print("\n[INFO] เปิดกล้องสำเร็จ! นำชิ้นส่วนมาส่องหน้ากล้องได้เลย (กดปุ่ม 'q' เพื่อออก)")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # นำภาพจากกล้องไปประมวลผลผ่านโมเดล
            results = model.predict(
                source=frame,
                conf=0.25,
                imgsz=640,
                device=device,
                verbose=False # ปิดข้อความรกหน้าจอตอนรัน
            )

            # วาดกรอบสี่เหลี่ยมลงบนภาพ
            annotated_frame = results[0].plot()
            
            # แสดงหน้าต่าง
            cv2.imshow('PCB Component YOLO Real-time Detection', annotated_frame)

            # กด 'q' เพื่อปิดกล้อง
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()