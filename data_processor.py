import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os


class DataProcessor:
    def __init__(self, model_dir="ml_models"):
        self.model_dir = model_dir
        self.scaler = None
        self.encoders = {}
        self.numeric_cols = []
        self.categorical_cols = []
        self.feature_names = []

        os.makedirs(model_dir, exist_ok=True)

    def fit(self, df, target_col="Flood Occurred"):
        df_clean = df.copy()
        df_clean.drop_duplicates(inplace=True)

        if target_col in df_clean.columns:
            y = df_clean[target_col]
            X = df_clean.drop(columns=[target_col])
        else:
            X = df_clean

        X.columns = [str(col).strip() for col in X.columns]

        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

        for col in X.columns:
            if col not in self.numeric_cols and col not in self.categorical_cols:
                try:
                    X[col] = pd.to_numeric(X[col], errors='raise')
                    self.numeric_cols.append(col)
                except:
                    self.categorical_cols.append(col)

        for col in self.numeric_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            mean_val = X[col].mean()
            if pd.isna(mean_val):
                mean_val = 0
            X[col] = X[col].fillna(mean_val)

        for col in self.categorical_cols:
            X[col] = X[col].astype(str).fillna("missing")
            X[col] = X[col].replace('nan', 'missing')

        self.scaler = StandardScaler()
        if self.numeric_cols:
            X[self.numeric_cols] = self.scaler.fit_transform(X[self.numeric_cols])

        for col in self.categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.encoders[col] = le

        self.feature_names = X.columns.tolist()

        if target_col in df_clean.columns:
            return X, y
        return X

    def transform(self, df, target_col=None):
        df_clean = df.copy()
        df_clean.drop_duplicates(inplace=True)

        if target_col and target_col in df_clean.columns:
            y = df_clean[target_col]
            X = df_clean.drop(columns=[target_col])
        else:
            X = df_clean
            y = None

        X.columns = [str(col).strip() for col in X.columns]

        for col in self.numeric_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                mean_val = X[col].mean() if not X[col].isna().all() else 0
                X[col] = X[col].fillna(mean_val)
            else:
                X[col] = 0

        for col in self.categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype(str).fillna("missing")
                X[col] = X[col].replace('nan', 'missing')

                if col in self.encoders:
                    le = self.encoders[col]
                    valid_items = set(le.classes_)
                    X[col] = X[col].apply(lambda x: x if x in valid_items else le.classes_[0])
                    X[col] = le.transform(X[col])
                else:
                    unique_vals = X[col].unique()
                    mapping = {val: idx for idx, val in enumerate(unique_vals)}
                    X[col] = X[col].map(mapping)
            else:
                X[col] = 0

        missing_numeric = set(self.numeric_cols) - set(X.columns)
        for col in missing_numeric:
            X[col] = 0

        missing_categorical = set(self.categorical_cols) - set(X.columns)
        for col in missing_categorical:
            X[col] = 0

        X = X.reindex(columns=self.feature_names, fill_value=0)

        if self.numeric_cols and self.scaler:
            X[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])

        if y is not None:
            return X, y
        return X

    def save_processor(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(self.model_dir, "data_processor.pkl")

        processor_data = {
            'scaler': self.scaler,
            'encoders': self.encoders,
            'numeric_cols': self.numeric_cols,
            'categorical_cols': self.categorical_cols,
            'feature_names': self.feature_names
        }

        joblib.dump(processor_data, filepath)

    def load_processor(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(self.model_dir, "data_processor.pkl")

        if os.path.exists(filepath):
            processor_data = joblib.load(filepath)
            self.scaler = processor_data['scaler']
            self.encoders = processor_data['encoders']
            self.numeric_cols = processor_data['numeric_cols']
            self.categorical_cols = processor_data['categorical_cols']
            self.feature_names = processor_data['feature_names']
            return True
        return False


def preprocess_dataframe(df, target_col="Flood Occurred", fit=False, processor=None):
    if fit or processor is None:
        processor = DataProcessor()
        result = processor.fit(df, target_col)
        processor.save_processor()
        return result
    else:
        return processor.transform(df, target_col)