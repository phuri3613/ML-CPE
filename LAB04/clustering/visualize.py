import matplotlib.pyplot as plt
import os
import numpy as np

def save_clustering_outputs(inertias, best_model, X_scaled, original_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. วาดกราฟ Elbow Method
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(inertias) + 1), inertias, marker='o', color='blue')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.savefig(f"{out_dir}/01_elbow.png")
    plt.close()

    # 2. วาดกราฟ Cluster (แบบเดียวกับอาจารย์เป๊ะๆ)
    # เลือกคอลัมน์มาทำแกน X และ Y (เปลี่ยนชื่อคอลัมน์ตรงนี้ได้ตามต้องการ)
    x_col = 'gill-color'
    y_col = 'habitat'
    
    # แปลงข้อมูลข้อความเป็นตัวเลข (0, 1, 2,...) เพื่อให้พล็อตลงกราฟได้
    x_data = original_df[x_col].astype('category').cat.codes
    y_data = original_df[y_col].astype('category').cat.codes
    
    # ดึงป้ายกำกับกลุ่ม (Cluster) จากโมเดลที่รันเสร็จแล้ว
    clusters = best_model.labels_ if hasattr(best_model, 'labels_') else best_model.predict(X_scaled)
    unique_clusters = sorted(list(set(clusters)))
    
    # ชุดสีแบบเดียวกับกราฟของอาจารย์ (ฟ้า, ส้ม, เขียว, แดง)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    plt.figure(figsize=(10, 8))
    
    # พล็อตจุดลงกราฟทีละ Cluster เพื่อให้มี Legend
    for c in unique_clusters:
        idx = (clusters == c)
        # เติม noise (jitter) เล็กน้อย เพื่อไม่ให้จุดทับกันสนิทจนมองไม่เห็น
        x_jitter = x_data[idx] + np.random.uniform(-0.15, 0.15, size=sum(idx))
        y_jitter = y_data[idx] + np.random.uniform(-0.15, 0.15, size=sum(idx))
        
        plt.scatter(x_jitter, y_jitter, 
                    c=colors[c % len(colors)], 
                    label=f'Cluster {c}', 
                    alpha=0.6, edgecolors='w', s=50)

    plt.title('K-Means Clustering result')
    plt.xlabel(x_col.capitalize())
    plt.ylabel(y_col.capitalize())
    
    # ปรับชื่อแกนให้เป็นคำศัพท์เดิม จะได้อ่านง่าย
    plt.xticks(range(len(original_df[x_col].astype('category').cat.categories)), 
               original_df[x_col].astype('category').cat.categories)
    plt.yticks(range(len(original_df[y_col].astype('category').cat.categories)), 
               original_df[y_col].astype('category').cat.categories)
    
    # เพิ่มกล่อง Legend ไว้ที่มุมซ้ายล่าง แบบเดียวกับอาจารย์
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='-', alpha=0.3)
    
    plt.savefig(f"{out_dir}/02_clusters.png", bbox_inches='tight')
    plt.close()