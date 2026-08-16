import os
from data_loader import load_data
from knn_tf import train_knn
from evaluate import evaluate_models

def main():
    out_dir = 'outputs'
    
    print(" Starting Classification Process...")
    
    # 1. โหลดข้อมูล
    data = load_data()
    
    # 2. ดึงข้อมูลตัวแปรออกมาใช้งาน
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    classes = data["class_names"]
    
    print(f"Dataset Info: Total {data['n_rows']} rows")
    print(f"Shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {data['X_test'].shape}")
    
    # 3. กำหนดค่า K ที่ต้องการทดสอบ
    k_values = [1, 3, 5, 7, 9, 11]
    
    print("Training KNN Models...")
    results, models = train_knn(X_train, y_train, X_val, y_val, k_values)
    
    for k, acc in results.items():
        print(f"Validation Accuracy (k={k}): {acc:.4f}")
        
    # 4. สร้างกราฟและบันทึกไฟล์ผลลัพธ์
    print("Generating Outputs...")
    best_k = evaluate_models(results, models, y_val, classes, out_dir)
    
    print(f"\nBest K Value based on Validation set: {best_k}")
    print(f"Results saved in {out_dir}/")

if __name__ == "__main__":
    main()