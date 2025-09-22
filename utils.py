import os
import json
import logging
import joblib
from datetime import datetime
from typing import Dict, Any, Optional


def setup_logging(log_file='flood_prediction.log'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path='config.json'):
    default_config = {
        "model_settings": {
            "test_size": 0.2,
            "random_state": 42,
            "cv_folds": 3
        },
        "training_params": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5]
        },
        "api_settings": {
            "host": "0.0.0.0",
            "port": 5000,
            "debug": True
        },
        "file_paths": {
            "data_path": "flood_risk_dataset_india.csv",
            "model_dir": "ml_models",
            "model_file": "flood_model.pkl",
            "scaler_file": "scaler.pkl",
            "encoder_file": "encoders.pkl",
            "processor_file": "data_processor.pkl"
        }
    }

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
            default_config.update(user_config)

    return default_config


def ensure_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        return True
    return False


def save_artifact(artifact, file_path):
    try:
        ensure_dir(os.path.dirname(file_path))
        joblib.dump(artifact, file_path)
        return True
    except Exception as e:
        logging.error(f"Error saving artifact to {file_path}: {e}")
        return False


def load_artifact(file_path):
    try:
        if os.path.exists(file_path):
            return joblib.load(file_path)
        return None
    except Exception as e:
        logging.error(f"Error loading artifact from {file_path}: {e}")
        return None


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_csv_file(file_path, required_columns=None):
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"

    try:
        df = pd.read_csv(file_path)
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                return False, f"Missing required columns: {missing_cols}"
        return True, df
    except Exception as e:
        return False, f"Error reading CSV: {e}"


def calculate_model_metrics(y_true, y_pred, y_prob=None):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0))
    }

    if y_prob is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
        except:
            metrics['roc_auc'] = 0.0

    return metrics


def format_prediction_response(prediction, probability, success=True, error_msg=None):
    response = {
        'success': success,
        'timestamp': get_timestamp()
    }

    if success:
        response.update({
            'prediction': int(prediction),
            'probability': float(probability) if probability is not None else None,
            'risk_level': 'high' if probability and probability > 0.7 else 'medium' if probability and probability > 0.4 else 'low'
        })
    else:
        response['error'] = error_msg

    return response


def check_model_health(model_paths):
    health_status = {}
    for name, path in model_paths.items():
        exists = os.path.exists(path)
        health_status[name] = {
            'exists': exists,
            'size': os.path.getsize(path) if exists else 0,
            'last_modified': datetime.fromtimestamp(os.path.getmtime(path)).isoformat() if exists else None
        }
    return health_status


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def clean_feature_names(feature_list):
    return [str(feature).strip().replace(' ', '_').lower() for feature in feature_list]


class ModelVersionManager:
    def __init__(self, model_dir="ml_models"):
        self.model_dir = model_dir
        self.versions_file = os.path.join(model_dir, "model_versions.json")
        ensure_dir(model_dir)

    def save_version_info(self, model_info):
        version_data = {
            'timestamp': get_timestamp(),
            'model_info': model_info,
            'version_id': len(self.get_all_versions()) + 1
        }

        all_versions = self.get_all_versions()
        all_versions.append(version_data)

        with open(self.versions_file, 'w') as f:
            json.dump(all_versions, f, indent=2)

        return version_data['version_id']

    def get_all_versions(self):
        if os.path.exists(self.versions_file):
            with open(self.versions_file, 'r') as f:
                return json.load(f)
        return []

    def get_latest_version(self):
        versions = self.get_all_versions()
        return versions[-1] if versions else None


def create_api_response(data, status=200, message="Success"):
    return {
        'status': status,
        'message': message,
        'data': data,
        'timestamp': get_timestamp()
    }


def handle_exception(e, context=""):
    error_info = {
        'error_type': type(e).__name__,
        'error_message': str(e),
        'context': context,
        'timestamp': get_timestamp()
    }
    logging.error(f"Exception in {context}: {e}")
    return error_info