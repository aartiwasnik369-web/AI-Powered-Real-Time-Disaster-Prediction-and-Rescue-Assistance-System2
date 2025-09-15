# backend/app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import joblib
import os
import json

from data_processor import preprocess_flood_data
from prediction_model import train_flood_model, predict_flood_risk, run_object_detection_stub

app = Flask(__name__)
CORS(app)

# Ensure paths are relative to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "ml_models")

# ---------------- Load / train flood model (demo) ----------------
model_path = os.path.join(MODEL_DIR, "flood_prediction_model.pkl")
if os.path.exists(model_path):
    flood_model = joblib.load(model_path)
else:
    # Train a tiny demo model if not found
    dummy = pd.DataFrame({
        "rainfall":[5,10,20,35,50,70,5,15,45,60],
        "water_level":[0.5,1.0,2.1,3.0,3.8,4.5,0.3,1.4,3.3,4.8],
        "risk":[0,0,1,1,2,2,0,0,1,2]  # 0-low,1-medium,2-high
    })
    X = dummy[["rainfall","water_level"]]
    y = dummy["risk"]
    flood_model = train_flood_model(X, y)
    joblib.dump(flood_model, model_path)

# ---------------- Load data ----------------
def read_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

emdat_path = os.path.join(DATA_DIR, "emdat_data.csv")
weather_path = os.path.join(DATA_DIR, "historical_weather.csv")
sensor_path = os.path.join(DATA_DIR, "sensor_data.csv")
geojson_path = os.path.join(DATA_DIR, "geo_boundaries.json")

emdat_data = read_csv_safe(emdat_path)
historical_weather = read_csv_safe(weather_path)
sensor_data = read_csv_safe(sensor_path)

geo_boundaries = {}
try:
    with open(geojson_path, "r", encoding="utf-8") as f:
        geo_boundaries = json.load(f)
except Exception:
    geo_boundaries = {}

# ---------------- API routes ----------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status":"ok"})

@app.route("/api/predict_flood", methods=["POST"])
def predict_flood():
    """
    Body JSON: {"rainfall": float, "water_level": float}
    Returns: {"risk_level": "low|medium|high", "risk_code": 0|1|2, "confidence": float}
    """
    data = request.get_json(force=True, silent=True) or {}
    if not all(k in data for k in ("rainfall","water_level")):
        return jsonify({"error":"Missing rainfall/water_level"}), 400

    X = pd.DataFrame([[data["rainfall"], data["water_level"]]], columns=["rainfall","water_level"])
    X = preprocess_flood_data(X)
    pred_code = int(predict_flood_risk(flood_model, X)[0])

    mapping = {0:"low", 1:"medium", 2:"high"}
    # Simple confidence heuristic
    confidence = 0.9 if pred_code==2 else (0.8 if pred_code==1 else 0.85)

    return jsonify({
        "risk_level": mapping.get(pred_code, "low"),
        "risk_code": pred_code,
        "confidence": confidence
    })

@app.route("/api/detect_objects", methods=["POST"])
def detect_objects():
    """
    Accepts form-data with key 'image' (file). Returns stubbed detections.
    """
    if "image" not in request.files:
        return jsonify({"error":"No image provided"}), 400
    file = request.files["image"]
    # For a demo, we don't actually run a heavy model; we return a stub.
    detections = run_object_detection_stub(file)
    return jsonify({"detections": detections})

@app.route("/api/risk_areas", methods=["GET"])
def get_risk_areas():
    if not geo_boundaries:
        return jsonify({"error":"GeoJSON not available"}), 500
    return jsonify(geo_boundaries)

@app.route("/api/historical_disasters", methods=["GET"])
def get_historical_disasters():
    # Limit for demo
    return jsonify(emdat_data.head(100).to_dict(orient="records"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
