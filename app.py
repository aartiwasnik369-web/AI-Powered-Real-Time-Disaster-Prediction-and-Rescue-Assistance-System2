# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "ml_models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "flood_risk_model.pkl")

# ------------------ Load / Train Model ------------------

def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print("✅ Loaded existing model.")
            return model
        except Exception as e:
            print(f"⚠️ Error loading model: {e}. Will retrain.")

    csv_path = os.path.join(DATA_DIR, "historical_weather.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset {csv_path} not found.")

    df = pd.read_csv(csv_path)

    if "precipitation" in df.columns and "rainfall" not in df.columns:
        df.rename(columns={"precipitation": "rainfall"}, inplace=True)
    if "Water Level" in df.columns and "water_level" not in df.columns:
        df.rename(columns={"Water Level": "water_level"}, inplace=True)

    required_cols = ["rainfall", "water_level"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataset.")

    if "risk" not in df.columns:
        print("⚠️ 'risk' column not found. Generating automatically...")
        def calculate_risk(row):
            if row["rainfall"] > 200 or row["water_level"] > 5:
                return 2
            elif row["rainfall"] > 100 or row["water_level"] > 3:
                return 1
            else:
                return 0
        df["risk"] = df.apply(calculate_risk, axis=1)

    df = df.dropna(subset=["rainfall", "water_level", "risk"])

    X = df[["rainfall", "water_level"]]
    y = df["risk"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("✅ Trained new model.")

    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    return model

model = load_or_train_model()

# ------------------ API Endpoints ------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/api/predict_flood", methods=["POST"])
def predict_flood():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    if "rainfall" not in data or "water_level" not in data:
        return jsonify({"error": "Missing 'rainfall' or 'water_level' field"}), 400

    try:
        rf = float(data["rainfall"])
        wl = float(data["water_level"])
    except ValueError:
        return jsonify({"error": "rainfall and water_level must be numeric"}), 400

    X_new = pd.DataFrame([{"rainfall": rf, "water_level": wl}])

    try:
        pred_code = int(model.predict(X_new)[0])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    mapping = {0: "low", 1: "medium", 2: "high"}
    risk_str = mapping.get(pred_code, "unknown")

    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_new)[0]
        confidence = float(max(probs))
    else:
        confidence = 0.9

    return jsonify({
        "risk_level": risk_str,
        "risk_code": pred_code,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
