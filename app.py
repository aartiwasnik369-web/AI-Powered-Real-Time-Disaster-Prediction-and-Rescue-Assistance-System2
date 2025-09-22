import os
import traceback
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib


DATA_PATH = "flood_risk_dataset_india.csv"   #dataset on project root
MODEL_DIR = "ml_models"
MODEL_PATH = os.path.join(MODEL_DIR, "flood_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.pkl")

TARGET_COL = "Flood Occurred"

os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__)

def preprocess_df(df: pd.DataFrame, fit_encoders=False, encoders=None, scaler=None):

    df = df.copy()
    df.drop_duplicates(inplace=True)
    if fit_encoders:
        assert TARGET_COL in df.columns, f"{TARGET_COL} missing in dataset."

    if TARGET_COL in df.columns:
        y = df[TARGET_COL]
        X = df.drop(columns=[TARGET_COL])
    else:
        y = None
        X = df

    X.columns = [c.strip() for c in X.columns]

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    for col in X.columns:
        if col not in numeric_cols and col not in categorical_cols:
            try:
                X[col] = pd.to_numeric(X[col], errors="raise")
                if col not in numeric_cols:
                    numeric_cols.append(col)
                if col in categorical_cols:
                    categorical_cols.remove(col)
            except Exception:
                if col not in categorical_cols:
                    categorical_cols.append(col)

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col].fillna(X[col].mean(), inplace=True)
    for col in categorical_cols:
        X[col] = X[col].astype(str).fillna("missing")
        X[col].replace({"nan": "missing"}, inplace=True)

    encoders_out = {}
    if fit_encoders:
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = X[col].astype(str)
            le.fit(X[col].values)
            most_common = X[col].mode().iloc[0] if not X[col].mode().empty else (le.classes_[0] if len(le.classes_)>0 else "")
            encoders_out[col] = {"le": le, "most_common": most_common}

        scaler_out = StandardScaler()
        if numeric_cols:
            scaler_out.fit(X[numeric_cols])
            X[numeric_cols] = scaler_out.transform(X[numeric_cols])
        else:
            scaler_out = None

        for col in categorical_cols:
            X[col] = encoders_out[col]["le"].transform(X[col].astype(str))

        return X, y, encoders_out, scaler_out

    else:

        if encoders is None:
            encoders = {}

        for col in categorical_cols:
            X[col] = X[col].astype(str)
            if col in encoders:
                le = encoders[col]["le"]
                most_common = encoders[col].get("most_common", None)
                mapped = []
                for val in X[col].values:
                    if val in le.classes_:
                        mapped.append(int(le.transform([val])[0]))
                    else:
                        fallback = most_common if most_common is not None else (le.classes_[0] if len(le.classes_)>0 else "")
                        mapped.append(int(le.transform([fallback])[0]) if fallback in le.classes_ else 0)
                X[col] = mapped
            else:

                try:
                    X[col] = pd.to_numeric(X[col], errors="raise")
                except Exception:
                    # simple mapping
                    uniques = list(pd.Series(X[col].unique()))
                    map_dict = {k: i for i, k in enumerate(uniques)}
                    X[col] = X[col].map(map_dict).astype(int)

        # Scale numeric columns with provided scaler
        if scaler is not None and numeric_cols:
            X[numeric_cols] = scaler.transform(X[numeric_cols])

        return X

def train_and_save_model(csv_path=DATA_PATH, save_model=True):

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in CSV.")


    X, y, encoders, scaler = preprocess_df(df, fit_encoders=True)

    feature_list = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    rf = RandomForestClassifier(random_state=42)
    params = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5]
    }
    grid = GridSearchCV(rf, params, cv=3, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    if save_model:
        joblib.dump(best, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(encoders, ENCODERS_PATH)

    return {
        "best_params": grid.best_params_,
        "accuracy": float(acc),
        "features": feature_list
    }


def load_artifacts():
    model = None
    scaler = None
    encoders = None
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    if os.path.exists(ENCODERS_PATH):
        encoders = joblib.load(ENCODERS_PATH)
    return model, scaler, encoders


def predict_single(input_json):

    model, scaler, encoders = load_artifacts()
    if model is None or scaler is None or encoders is None:
        # If artifacts missing, attempt to train automatically
        train_info = train_and_save_model()
        model, scaler, encoders = load_artifacts()

    try:
        # Attempt build df directly from input_json
        df_input = pd.DataFrame([input_json])
    except Exception as e:
        raise ValueError("Input JSON could not be converted to DataFrame.") from e

    expected_cols = []
    if encoders is not None:
        expected_cols.extend(list(encoders.keys()))

    X_proc = preprocess_df(df_input, fit_encoders=False, encoders=encoders, scaler=scaler)

    model_input = X_proc.values
    if model_input.shape[1] != model.n_features_in_:
        raise ValueError(f"Model expects {model.n_features_in_} features but input has {model_input.shape[1]}. "
                         "Check that JSON keys match training feature names and order.")

    pred = model.predict(model_input)[0]
    prob = model.predict_proba(model_input)[0][1] if hasattr(model, "predict_proba") else None
    return int(pred), float(prob) if prob is not None else None

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flood Prediction API - running", "model_exists": os.path.exists(MODEL_PATH)})

@app.route("/train", methods=["POST"])
def train_route():

    try:
        payload = request.get_json(silent=True) or {}
        csv_path = payload.get("csv_path", DATA_PATH)
        result = train_and_save_model(csv_path=csv_path, save_model=True)
        return jsonify({"status": "trained", "result": result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 400

@app.route("/predict", methods=["POST"])
def predict_route():

    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No JSON body found"}), 400
        pred, prob = predict_single(data)
        return jsonify({"prediction": pred, "probability": prob})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@app.route("/predict_batch", methods=["POST"])
def predict_batch_route():

    try:
        if 'file' not in request.files:
            return jsonify({"error": "file (CSV) is required in form-data with key 'file'"}), 400
        file = request.files['file']
        df = pd.read_csv(file)
        # Preprocess batch using saved artifacts (train if missing)
        model, scaler, encoders = load_artifacts()
        if model is None or scaler is None or encoders is None:
            train_and_save_model()
            model, scaler, encoders = load_artifacts()

        X_proc = preprocess_df(df, fit_encoders=False, encoders=encoders, scaler=scaler)
        if X_proc.shape[1] != model.n_features_in_:
            return jsonify({"error": f"Model expects {model.n_features_in_} features, input has {X_proc.shape[1]}."}), 400

        preds = model.predict(X_proc.values)
        probs = model.predict_proba(X_proc.values)[:,1] if hasattr(model, "predict_proba") else [None]*len(preds)
        out = pd.DataFrame({
            "prediction": preds.astype(int),
            "probability": [float(x) for x in probs]
        })
        return out.to_json(orient="records")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@app.route("/model_info", methods=["GET"])
def model_info():
    try:
        model, scaler, encoders = load_artifacts()
        info = {"model_exists": model is not None, "scaler_exists": scaler is not None, "encoders_exist": encoders is not None}
        if encoders is not None:
            info["categorical_columns"] = list(encoders.keys())
        if model is not None:
            info["n_features_in_"] = int(model.n_features_in_)
        return jsonify(info)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


