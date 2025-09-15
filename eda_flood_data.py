# backend/eda_flood_data.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")

csv_path = os.path.join(DATA_DIR, "historical_weather.csv")

# ------------------ Load Dataset ------------------
df = pd.read_csv(csv_path)

print("🔍 Dataset Shape:", df.shape)
print("\n📌 First 5 rows:\n", df.head())
print("\n📊 Info:\n")
print(df.info())
print("\n📈 Missing Values:\n", df.isnull().sum())

# ------------------ Basic Statistics ------------------
print("\n📊 Summary Statistics:\n", df.describe())

# ------------------ Distribution Plots ------------------
plt.figure(figsize=(12, 5))
sns.histplot(df["rainfall"], bins=30, kde=True)
plt.title("Rainfall Distribution")
plt.savefig(os.path.join(DATA_DIR, "rainfall_distribution.png"))
plt.close()

plt.figure(figsize=(12, 5))
sns.histplot(df["water_level"], bins=30, kde=True, color="blue")
plt.title("Water Level Distribution")
plt.savefig(os.path.join(DATA_DIR, "waterlevel_distribution.png"))
plt.close()

# ------------------ Risk Counts ------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="risk", data=df, palette="Set2")
plt.title("Flood Risk Distribution")
plt.savefig(os.path.join(DATA_DIR, "risk_counts.png"))
plt.close()

# ------------------ Correlation Heatmap ------------------
plt.figure(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(DATA_DIR, "correlation_heatmap.png"))
plt.close()

# ------------------ Scatter Plots ------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(x="rainfall", y="water_level", hue="risk", data=df, palette="coolwarm")
plt.title("Rainfall vs Water Level (Flood Risk)")
plt.savefig(os.path.join(DATA_DIR, "scatter_rainfall_waterlevel.png"))
plt.close()

print("\n✅ EDA Completed. Plots saved in /data folder.")
