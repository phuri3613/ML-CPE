import torch
from pathlib import Path
import random
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def main():
   
    device = 0 if torch.cuda.is_available() else 'cpu'
    
   
    weights_paths = list((ROOT / 'runs').rglob('best.pt'))
    if not weights_paths:
        raise FileNotFoundError("Model not found: ไม่พบไฟล์ best.pt ในโฟลเดอร์ runs")
    
    # เลือกใช้ไฟล์ล่าสุดที่เจอ
    weights = weights_paths[-1] 
    print(f"--> โหลดโมเดลจาก: {weights}")

    image_dir = ROOT / 'dataset' / 'test' / 'images'

    if not image_dir.exists():
        raise FileNotFoundError(f'Dataset not found: {image_dir}')

    # โหลดโมเดล
    model = YOLO(str(weights))

    extensions = {'.jpg', '.jpeg', '.png'}
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in extensions]

    if not images:
        raise RuntimeError('No test images found')

    # สุ่มเลือกรูปภาพจำนวน 5 รูป
    selected_images = random.sample(images, min(5, len(images)))

    for image in selected_images:
        results = model.predict(
            source=str(image),
            conf=0.25,
            imgsz=480,
            device=device,
            save=True,
            project=str(ROOT / 'runs'),
            name='image_test',
            exist_ok=True
        )

        for result in results:
            print(f'\nImage: {image.name}')
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                print(f'{model.names[class_id]}: {confidence:.2f}')

            # แสดงผลลัพธ์บนหน้าต่าง
            annotated = result.plot()
            cv2.imshow('PCB Component Detection', annotated)

            # รอผู้ใช้กดปุ่มใดๆ เพื่อดูรูปถัดไป หรือกด 'q' เพื่อออก
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print('\nImage testing completed!')

if __name__ == '__main__':
    main()