# backend/data_processor.py
import pandas as pd

def preprocess_flood_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal demo preprocessing: fill NA and clip extremes.
    """
    df = df.copy()
    df = df.fillna(0)
    if "rainfall" in df.columns:
        df["rainfall"] = df["rainfall"].clip(lower=0, upper=500)
    if "water_level" in df.columns:
        df["water_level"] = df["water_level"].clip(lower=0, upper=20)
    return df
