# backend/prediction_model.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from data_processor import preprocess_flood_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "ml_models")

os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "flood_risk_model.pkl")

def train_and_save_model():
    """
    Train flood risk model from dataset and save it as pickle.
    Dataset must have ['rainfall', 'water_level', 'risk'] columns.
    """
    csv_path = os.path.join(DATA_DIR, "historical_weather.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset {csv_path} not found.")

    # Load dataset
    df = pd.read_csv(csv_path)

    # Check required columns
    required_cols = ["rainfall", "water_level", "risk"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataset.")

    # Preprocess data
    df = preprocess_flood_data(df)

    # Split features/target
    X = df[["rainfall", "water_level"]]
    y = df["risk"].astype(int)   # ensure int (0,1,2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("📊 Model Evaluation:")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved at {MODEL_PATH}")

    return model


def load_model():
    """
    Load trained flood risk model (train if not available).
    """
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print("✅ Loaded existing model.")
            return model
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}, retraining...")
            return train_and_save_model()
    else:
        return train_and_save_model()
