from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def main():
    # Load pretrained YOLO model
    model = YOLO("yolov8n.pt")

    # ----- Configuration 1: 10 Epochs -----
    print("=== Training Configuration 1 (10 Epochs, imgsz=480, batch=8) ===")
    model.train(
        data="dataset/data.yaml",
        epochs=10,
        imgsz=480,
        batch=8,
        device=0,
        workers=0,
        amp=True,
        cache=False,
        project="runs",
        name="pcb_config1_10epochs",
        exist_ok=True,
        save=True,
        plots=True
    )

    # ----- Configuration 2: 20 Epochs -----
    print("\n=== Training Configuration 2 (20 Epochs, imgsz=640, batch=16) ===")
    model_2 = YOLO("yolov8n.pt")
    model_2.train(
        data="dataset/data.yaml",
        epochs=20,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        amp=True,
        cache=False,
        project="runs",
        name="pcb_config2_20epochs",
        exist_ok=True,
        save=True,
        plots=True
    )

    print("\nTraining completed for both configurations!")

if __name__ == "__main__":
    main()