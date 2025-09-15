# backend/train_model.py
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from data_processor import preprocess_flood_data
from utils.config import MODEL_DIR, FLOOD_MODEL_PATH
from utils.logger import logger

os.makedirs(MODEL_DIR, exist_ok=True)

def train_flood_model():
    """
    Train flood risk prediction model from dataset and save as .pkl
    """
    csv_path = os.path.join("data", "historical_weather.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Dataset not found at {csv_path}")

    # 1. Load dataset
    df = pd.read_csv(csv_path)
    logger.info(f"📂 Loaded dataset with shape {df.shape}")

    # 2. Ensure required columns exist
    required_cols = ["rainfall", "water_level", "risk"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Missing column: {col}")

    # 3. Preprocess data
    df = preprocess_flood_data(df)

    # 4. Split features/labels
    X = df[["rainfall", "water_level"]]
    y = df["risk"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Train model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    logger.info("✅ Model trained successfully")

    # 6. Evaluate
    y_pred = model.predict(X_test)
    logger.info("📊 Model Evaluation:\n" + classification_report(y_test, y_pred))

    # 7. Save model
    joblib.dump(model, FLOOD_MODEL_PATH)
    logger.info(f"💾 Model saved at {FLOOD_MODEL_PATH}")

    return model


if __name__ == "__main__":
    train_flood_model()
