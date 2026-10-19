from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def evaluate_config(config_name):
    weights = ROOT / 'runs' / config_name / 'weights' / 'best.pt'

    if not weights.exists():
        print(f'Model not found: {weights}')
        return

    model = YOLO(str(weights))

    # Evaluate on test dataset
    metrics = model.val(
        data=str(ROOT / 'dataset' / 'data.yaml'),
        split='test',
        imgsz=480,
        batch=8,
        device=0,
        workers=0,
        plots=True,
        project=str(ROOT / 'runs'),
        name=f'eval_{config_name}',
        exist_ok=True,
    )

    # Display evaluation results
    print(f'\n===== Evaluation Results: {config_name} =====')
    print(f'Precision: {metrics.box.mp:.4f}')
    print(f'Recall:    {metrics.box.mr:.4f}')
    print(f'mAP50:     {metrics.box.map50:.4f}')
    print(f'mAP50-95:  {metrics.box.map:.4f}')

def main():
    evaluate_config('pcb_config1_10epochs')
    evaluate_config('pcb_config2_20epochs')

if __name__ == '__main__':
    main()