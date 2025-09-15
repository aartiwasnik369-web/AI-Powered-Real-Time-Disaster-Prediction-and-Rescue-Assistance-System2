# backend/data_processor.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_flood_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realistic preprocessing pipeline for flood prediction datasets.
    - Handle missing values
    - Clip extreme outliers
    - Extract datetime features
    - Encode categorical variables
    - Scale numerical features
    """
    df = df.copy()

    # -------- Handle Missing Values --------
    df = df.fillna(0)

    # -------- Clip extreme values --------
    if "rainfall" in df.columns:
        df["rainfall"] = df["rainfall"].clip(lower=0, upper=500)
    if "water_level" in df.columns:
        df["water_level"] = df["water_level"].clip(lower=0, upper=20)

    # -------- Date Features --------
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["season"] = df["month"].map({
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Summer", 4: "Summer", 5: "Summer",
            6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
            10: "PostMonsoon", 11: "PostMonsoon"
        })
        df = df.drop(columns=["date"])

    # -------- Encode categorical columns --------
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # -------- Scale numerical features --------
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df
