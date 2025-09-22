# single_prediction.py
import requests
import json
import pandas as pd


def test_single_prediction():

    test_data = {
        "Latitude": 18.86166345,
        "Longitude": 78.83558374,
        "Rainfall (mm)": 218.9994933,
        "Temperature (°C)": 34.14433705,
        "Humidity (%)": 43.9129633,
        "River Discharge (m³/s)": 4236.182888,
        "Water Level (m)": 7.415552031,
        "Elevation (m)": 377.4654335,
        "Land Cover": "Water Body",
        "Soil Type": "Clay",
        "Population Density": 7276.742184,
        "Infrastructure": 1,
        "Historical Floods": 0
    }

    response = requests.post("http://localhost:5000/predict", json=test_data)
    result = response.json()

    print("Single Prediction Result:")
    print(f"Flood Prediction: {'YES' if result['prediction'] == 1 else 'NO'}")
    print(f"Probability: {result['probability']:.2%}")

    return result


# Run prediction
if __name__ == "__main__":
    test_single_prediction()

# batch_prediction.py
import requests
import pandas as pd
import json


def predict_batch_csv(csv_file_path):

    try:
        with open(csv_file_path, 'rb') as file:
            files = {'file': file}
            response = requests.post('http://localhost:5000/predict_batch', files=files)

        if response.status_code == 200:
            predictions = response.json()

            # Convert to DataFrame for better display
            pred_df = pd.DataFrame(predictions)

            print("Batch Predictions Completed!")
            print(f"Total records processed: {len(pred_df)}")
            print("\nPrediction Summary:")
            print(f"Flood Predicted: {pred_df['prediction'].sum()} records")
            print(f"No Flood Predicted: {len(pred_df) - pred_df['prediction'].sum()} records")

            # Save predictions to new CSV
            output_file = "predictions_with_results.csv"
            original_df = pd.read_csv(csv_file_path)
            result_df = pd.concat([original_df, pred_df], axis=1)
            result_df.to_csv(output_file, index=False)

            print(f"\nResults saved to: {output_file}")
            return pred_df
        else:
            print(f"Error: {response.json()}")

    except Exception as e:
        print(f"Error processing batch prediction: {e}")


# Test with your CSV
if __name__ == "__main__":
    predict_batch_csv("flood_risk_dataset_india.csv")

# prediction_dashboard.py
import requests
import pandas as pd
import json
from datetime import datetime


class FloodPredictionSystem:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url

    def get_model_info(self):
        """Model की information लें"""
        response = requests.get(f"{self.base_url}/model_info")
        return response.json()

    def predict_single(self, data):
        """Single prediction"""
        response = requests.post(f"{self.base_url}/predict", json=data)
        return response.json()

    def predict_batch(self, csv_path):
        """Batch prediction"""
        with open(csv_path, 'rb') as file:
            files = {'file': file}
            response = requests.post(f'{self.base_url}/predict_batch', files=files)
        return response.json()

    def analyze_prediction(self, prediction_result):

        pred = prediction_result['prediction']
        prob = prediction_result['probability']

        if pred == 1:
            risk_level = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.5 else "LOW"
            return f" FLOOD RISK: {risk_level} (Probability: {prob:.2%})"
        else:
            return f" NO FLOOD RISK (Probability: {prob:.2%})"


# Usage Example
if __name__ == "__main__":
    # Create prediction system instance
    predictor = FloodPredictionSystem()

    # Check model status
    print("=== MODEL INFORMATION ===")
    model_info = predictor.get_model_info()
    print(json.dumps(model_info, indent=2))

    # Test single prediction
    print("\n=== SINGLE PREDICTION TEST ===")
    test_data = {
        "Latitude": 18.86,
        "Longitude": 78.83,
        "Rainfall (mm)": 250.0,  # High rainfall
        "Temperature (°C)": 32.0,
        "Humidity (%)": 85.0,  # High humidity
        "River Discharge (m³/s)": 4500.0,  # High discharge
        "Water Level (m)": 8.0,  # High water level
        "Elevation (m)": 200.0,  # Low elevation
        "Land Cover": "Water Body",
        "Soil Type": "Clay",
        "Population Density": 8000.0,
        "Infrastructure": 1,
        "Historical Floods": 1
    }

    result = predictor.predict_single(test_data)
    analysis = predictor.analyze_prediction(result)
    print(analysis)
    print(f"Raw result: {result}")

# real_time_predictor.py
import requests
import json


def real_time_prediction_interface():
    """User-friendly prediction interface"""

    print(" FLOOD PREDICTION SYSTEM ")
    print("Enter location details for flood prediction:\n")

    # User input
    latitude = float(input("Latitude: "))
    longitude = float(input("Longitude: "))
    rainfall = float(input("Rainfall (mm): "))
    temperature = float(input("Temperature (°C): "))
    humidity = float(input("Humidity (%): "))
    river_discharge = float(input("River Discharge (m³/s): "))
    water_level = float(input("Water Level (m): "))
    elevation = float(input("Elevation (m): "))

    print("\nLand Cover Options: Water Body, Forest, Agricultural, Desert, Urban")
    land_cover = input("Land Cover: ")

    print("\nSoil Type Options: Clay, Peat, Loam, Sandy, Silt")
    soil_type = input("Soil Type: ")

    population_density = float(input("Population Density: "))
    infrastructure = int(input("Infrastructure (0/1): "))
    historical_floods = int(input("Historical Floods (0/1): "))

    # Prepare data
    prediction_data = {
        "Latitude": latitude,
        "Longitude": longitude,
        "Rainfall (mm)": rainfall,
        "Temperature (°C)": temperature,
        "Humidity (%)": humidity,
        "River Discharge (m³/s)": river_discharge,
        "Water Level (m)": water_level,
        "Elevation (m)": elevation,
        "Land Cover": land_cover,
        "Soil Type": soil_type,
        "Population Density": population_density,
        "Infrastructure": infrastructure,
        "Historical Floods": historical_floods
    }

    # Get prediction
    try:
        response = requests.post("http://localhost:5000/predict", json=prediction_data)
        result = response.json()

        print("\n" + "=" * 50)
        if result['prediction'] == 1:
            print("ALERT: FLOOD PREDICTED!")
            print(f"Probability: {result['probability']:.2%}")
            print("Take necessary precautions!")
        else:
            print("NO FLOOD RISK DETECTED")
            print(f"Safety Probability: {result['probability']:.2%}")
        print("=" * 50)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    real_time_prediction_interface()