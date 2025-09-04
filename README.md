
# Flood & Disaster Predictor

End-to-end demo website + API:
- Predict flood risk from rainfall & water level
- Detect humans/animals in an image (stubbed demo)
- Visualize high/medium/low risk zones on a Leaflet map
- View a small sample of historical disasters

## Project Layout
```
flood_disaster_predictor/
├── backend/
│   ├── app.py
│   ├── data_processor.py
│   ├── prediction_model.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── data/
│   ├── emdat_data.csv
│   ├── historical_weather.csv
│   ├── sensor_data.csv
│   └── geo_boundaries.json
├── ml_models/
│   └── (flood_prediction_model.pkl will be auto-created on first run)
└── README.md
```

## Quick Start

### 1) Backend (Flask)
```bash
cd backend
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python app.py
```
This starts the API on http://localhost:5000

### 2) Frontend (no build tools needed)
Open `frontend/index.html` in your browser.
If you open it as a local file, most modern browsers will still allow calls to `http://localhost:5000`.
If CORS issues occur, use a simple static server, e.g.:
```bash
# Python 3
cd frontend
python -m http.server 5173
# Then open http://localhost:5173
```

## API Summary
- `GET /api/health` – quick status
- `POST /api/predict_flood` – body: `{"rainfall": float, "water_level": float}`
- `POST /api/detect_objects` – form-data: `image=<file>` (returns stubbed detections)
- `GET /api/risk_areas` – returns GeoJSON of high/medium/low risk zones
- `GET /api/historical_disasters` – returns a small sample

## Replace stubbed detector with YOLO (optional)
1. Install Ultralytics: `pip install ultralytics`
2. Load a model in `backend/prediction_model.py` and run inference on the uploaded image.
3. Parse detections into `{"class":"human"|"animal", "bbox":[x1,y1,x2,y2], "confidence":float}`.

## Notes
- This is a minimal reference app. Improve features by adding real-time feeds, authentication, database storage, etc.
- Risk polygons are a simplified demo. Replace with your own GeoJSON or dynamically generate from live predictions.
