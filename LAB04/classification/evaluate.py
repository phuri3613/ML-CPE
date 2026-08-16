import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import os

def evaluate_models(results, models, y_test, target_names, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    best_k = max(results, key=results.get)
    
    plt.figure(figsize=(8, 5))
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.title('Accuracy vs. K Value')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.savefig(f'{out_dir}/01_k_curve.png')
    plt.close()
    
    best_model, best_y_pred = models[best_k]
    cm = confusion_matrix(y_test, best_y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix (k={best_k})')
    plt.savefig(f'{out_dir}/02_confusion_matrix.png')
    plt.close()
    
    pd.DataFrame({'Actual': y_test, 'Predicted': best_y_pred}).to_csv(f'{out_dir}/predictions.csv', index=False)
    return best_k