# train_model.py

import requests
import json

try:
    # Training endpoint call
    training_url = "http://localhost:5000/train"
    response = requests.post(training_url)

    print("Training Status Code:", response.status_code)
    print("Training Response:")
    print(json.dumps(response.json(), indent=2))

    # Model information check (FIXED: it should be model_info, not host_info)
    info_url = "http://localhost:5000/model_info"
    info_response = requests.get(info_url)

    print("\nModel Information:")
    print(json.dumps(info_response.json(), indent=2))

    # Test prediction with CSV data
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

    prediction_url = "http://localhost:5000/predict"
    pred_response = requests.post(prediction_url, json=test_data)

    print("\nTest Prediction:")
    print(json.dumps(pred_response.json(), indent=2))

except requests.exceptions.ConnectionError:
    print("Error: Flask server is not running. Please run 'python app.py' first.")
except Exception as e:
    print(f"Error: {e}")