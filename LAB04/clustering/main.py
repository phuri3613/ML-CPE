import os
from data_loader import load_for_clustering
from kmeans_tf import run_kmeans
from visualize import save_clustering_outputs

def main():
    out_dir = 'outputs'
    chosen_k = 3 
    
    print("Starting Clustering Process (Pro Version)...")
    
    
    X_scaled, original_df = load_for_clustering()
    
    print("Running K-Means...")
    inertias, models = run_kmeans(X_scaled, max_k=10)
    
    best_model = models[chosen_k]
    
    print("Generating Outputs...")
    save_clustering_outputs(inertias, best_model, X_scaled, original_df, out_dir)
    print(f"Clustering outputs saved in {out_dir}/")

if __name__ == "__main__":
    main()