
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# =========================================================
# Part 1: Exploratory Data Analysis (EDA)
# =========================================================
print("="*50)
print("Part 1: Exploratory Data Analysis (EDA)")
print("="*50)

# 1. Load Dataset
df = pd.read_csv('ai_student_impact_dataset (1).csv')

print("\n--- Displaying First 5 Rows ---")
print(df.head())

print("\n--- Displaying Last 5 Rows ---")
print(df.tail())

print("\n--- Dataset Information ---")
df.info()

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Duplicate Records ---")
print(f"Number of duplicate rows: {df.duplicated().sum()}")


# =========================================================
# Part 2: Data Visualization
# =========================================================
print("\n" + "="*50)
print("Part 2: Data Visualization")
print("="*50)
print("Displaying plots... (Please close the plot window to continue)")

# ---------------- Histogram ของทุกคอลัมน์ตัวเลข ----------------
numerical_cols = df.select_dtypes(include=['number']).columns

df[numerical_cols].hist(
    figsize=(12, 8),
    bins=30,
    edgecolor='black'
)

plt.suptitle("Distribution of Numerical Features", fontsize=14)
plt.tight_layout()
plt.show()

# ---------------- Correlation Heatmap ----------------
corr = df[numerical_cols].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(12, 8))
sns.heatmap(
    corr,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    mask=mask
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

if 'Burnout_Risk_Level' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=df,
        x='Burnout_Risk_Level',
        palette='viridis',
        order=['Low', 'Medium', 'High']
    )

    plt.title("Distribution of Burnout Risk Level")
    plt.xlabel("Burnout Risk")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
else:
    print("\nColumn 'Burnout_Risk_Level' not found. Available columns are:")
    print(df.columns.tolist())


# =========================================================
# Part 3: Data Cleaning & Feature Engineering
# =========================================================
print("\n" + "="*50)
print("Part 3: Data Cleaning & Feature Engineering")
print("="*50)

# Fill missing values (Numerical)
for col in df.select_dtypes(include=['number']).columns:
    df[col] = df[col].fillna(df[col].mean())

# Fill missing values (Categorical)
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Remove duplicate rows
df = df.drop_duplicates()

# Label Encoding
cat_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()

print("\n--- Applying Label Encoding ---")
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))
    print(f"Encoded column: {col}")

print("\n--- Final Preprocessed Dataset (First 5 Rows) ---")
print(df.head())


# =========================================================
# Part 4: Export Data
# =========================================================
df.to_csv('ai_cleaned_dataset_kaggle.csv', index=False)

print("\n" + "="*50)
print(" saved 'ai_cleaned_dataset_kaggle.csv'!")
print("="*50)