# backend/prediction_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_flood_model(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    try:
        print(f"[train] accuracy={accuracy_score(y_test, y_pred):.3f}")
    except Exception:
        pass
    return model

def predict_flood_risk(model, X: pd.DataFrame):
    return model.predict(X)

def run_object_detection_stub(file_storage):
    """
    This is a lightweight stub to keep the demo fast.
    Replace with a real detector (e.g., Ultralytics YOLO) in production.
    """
    # Return fixed sample detections
    return [
        {"class":"human", "bbox":[48,60,160,220], "confidence":0.92},
        {"class":"animal", "bbox":[210,110,320,260], "confidence":0.87}
    ]
