const API_BASE = "http://localhost:5000";

// Model Training
document.getElementById("train-btn").addEventListener("click", async () => {
    const statusDiv = document.getElementById("training-status");
    statusDiv.textContent = "Training model...";
    
    try {
        const response = await fetch(`${API_BASE}/train`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({})
        });
        const data = await response.json();
        
        if (data.status === "trained") {
            statusDiv.innerHTML = `
                <strong>Training Successful!</strong><br>
                Accuracy: ${(data.result.accuracy * 100).toFixed(2)}%<br>
                Best Parameters: ${JSON.stringify(data.result.best_params)}
            `;
        } else {
            statusDiv.textContent = "Error: " + data.error;
        }
    } catch (err) {
        statusDiv.textContent = "Error: " + err.message;
    }
});

// Single Prediction Form
document.getElementById("prediction-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    
    const formData = {
        "Latitude": parseFloat(document.getElementById("latitude").value) || 0,
        "Longitude": parseFloat(document.getElementById("longitude").value) || 0,
        "Rainfall (mm)": parseFloat(document.getElementById("rainfall").value) || 0,
        "Temperature (°C)": parseFloat(document.getElementById("temperature").value) || 0,
        "Humidity (%)": parseFloat(document.getElementById("humidity").value) || 0,
        "River Discharge (m³/s)": parseFloat(document.getElementById("river_discharge").value) || 0,
        "Water Level (m)": parseFloat(document.getElementById("water_level").value) || 0,
        "Elevation (m)": parseFloat(document.getElementById("elevation").value) || 0,
        "Land Cover": document.getElementById("land_cover").value,
        "Soil Type": document.getElementById("soil_type").value,
        "Population Density": parseFloat(document.getElementById("population_density").value) || 0,
        "Infrastructure": parseInt(document.getElementById("infrastructure").value) || 0,
        "Historical Floods": parseInt(document.getElementById("historical_floods").value) || 0
    };

    const resultDiv = document.getElementById("prediction-result");
    resultDiv.innerHTML = "Predicting...";

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(formData)
        });
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        const predictionText = data.prediction === 1 ? "FLOOD RISK DETECTED! 🚨" : "No Flood Risk ✅";
        const probabilityPercent = (data.probability * 100).toFixed(2);
        
        resultDiv.innerHTML = `
            <div class="result-card ${data.prediction === 1 ? 'high-risk' : 'low-risk'}">
                <h3>${predictionText}</h3>
                <p>Confidence: ${probabilityPercent}%</p>
                <p>Risk Level: ${data.prediction === 1 ? 'HIGH' : 'LOW'}</p>
                ${data.prediction === 1 ? 
                    '<div class="alert">Immediate action recommended! Check rescue plan.</div>' : 
                    '<div class="safe">Area appears safe. Continue monitoring.</div>'
                }
            </div>
        `;
        
        // Generate rescue plan if flood detected
        if (data.prediction === 1) {
            generateRescuePlan(formData, data.probability);
        }
        
    } catch (err) {
        resultDiv.innerHTML = `<div class="error">Error: ${err.message}</div>`;
    }
});

// Batch Prediction
document.getElementById("batch-prediction-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById("csv-file");
    const resultDiv = document.getElementById("batch-result");
    
    if (!fileInput.files.length) {
        resultDiv.innerHTML = "<div class='error'>Please select a CSV file</div>";
        return;
    }
    
    resultDiv.innerHTML = "Processing batch prediction...";
    
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    
    try {
        const response = await fetch(`${API_BASE}/predict_batch`, {
            method: "POST",
            body: formData
        });
        
        const data = await response.json();
        
        if (response.status === 200) {
            displayBatchResults(data);
        } else {
            throw new Error(data.error || "Batch prediction failed");
        }
    } catch (err) {
        resultDiv.innerHTML = `<div class="error">Error: ${err.message}</div>`;
    }
});

function displayBatchResults(predictions) {
    const resultDiv = document.getElementById("batch-result");
    
    if (!Array.isArray(predictions)) {
        resultDiv.innerHTML = "<div class='error'>Invalid response format</div>";
        return;
    }
    
    const floodCount = predictions.filter(p => p.prediction === 1).length;
    const totalCount = predictions.length;
    
    let html = `
        <div class="batch-summary">
            <h3>Batch Prediction Results</h3>
            <p>Total Records: ${totalCount}</p>
            <p>Flood Predictions: ${floodCount}</p>
            <p>Safe Predictions: ${totalCount - floodCount}</p>
            <p>Flood Rate: ${((floodCount / totalCount) * 100).toFixed(2)}%</p>
        </div>
        <div class="results-table">
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Prediction</th>
                        <th>Probability</th>
                        <th>Risk Level</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    predictions.slice(0, 10).forEach((pred, index) => {
        html += `
            <tr>
                <td>${index + 1}</td>
                <td>${pred.prediction === 1 ? 'FLOOD 🚨' : 'Safe ✅'}</td>
                <td>${(pred.probability * 100).toFixed(2)}%</td>
                <td>${pred.prediction === 1 ? 'HIGH' : 'LOW'}</td>
            </tr>
        `;
    });
    
    html += `</tbody></table>`;
    
    if (predictions.length > 10) {
        html += `<p>... and ${predictions.length - 10} more records</p>`;
    }
    
    resultDiv.innerHTML = html;
}

// Rescue Plan Generation
async function generateRescuePlan(formData, probability) {
    const rescueDiv = document.getElementById("rescue-plan");
    rescueDiv.innerHTML = "Generating rescue plan...";
    
    try {
        const rescueData = {
            probability: probability,
            rainfall: formData["Rainfall (mm)"],
            water_level: formData["Water Level (m)"],
            population_density: formData["Population Density"]
        };
        
        // Simulate rescue plan generation (you can create a separate endpoint for this)
        const riskLevel = probability > 0.7 ? 'high' : probability > 0.4 ? 'medium' : 'low';
        
        const rescuePlan = {
            high: [
                "🚨 IMMEDIATE EVACUATION REQUIRED",
                "➡️ Move to higher ground immediately",
                "📞 Contact local authorities: NDRF - 1070",
                "⚡ Turn off electrical mains",
                "🎒 Take emergency kit and documents"
            ],
            medium: [
                "⚠️ PREPARE FOR POSSIBLE EVACUATION",
                "📊 Monitor weather updates regularly",
                "🎒 Prepare emergency bag",
                "📍 Identify safe evacuation routes",
                "📱 Keep communication devices charged"
            ],
            low: [
                "ℹ️ STAY ALERT AND MONITOR",
                "📡 Stay informed about weather conditions",
                "💧 Avoid low-lying areas",
                "📋 Keep emergency contacts handy",
                "🔍 Watch for changing conditions"
            ]
        };
        
        let html = `<div class="rescue-plan ${riskLevel}"><h3>Rescue Plan (${riskLevel.toUpperCase()} Risk)</h3><ul>`;
        rescuePlan[riskLevel].forEach(item => {
            html += `<li>${item}</li>`;
        });
        html += `</ul></div>`;
        
        rescueDiv.innerHTML = html;
    } catch (err) {
        rescueDiv.innerHTML = `<div class="error">Rescue plan error: ${err.message}</div>`;
    }
}

// Model Information
document.getElementById("model-info-btn").addEventListener("click", async () => {
    const infoDiv = document.getElementById("model-info");
    infoDiv.textContent = "Loading model information...";
    
    try {
        const response = await fetch(`${API_BASE}/model_info`);
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        infoDiv.innerHTML = `
            <div class="model-info">
                <h3>Model Information</h3>
                <p><strong>Model Exists:</strong> ${data.model_exists ? 'Yes ✅' : 'No ❌'}</p>
                <p><strong>Scaler Exists:</strong> ${data.scaler_exists ? 'Yes ✅' : 'No ❌'}</p>
                <p><strong>Encoders Exist:</strong> ${data.encoders_exist ? 'Yes ✅' : 'No ❌'}</p>
                ${data.n_features_in_ ? `<p><strong>Features:</strong> ${data.n_features_in_}</p>` : ''}
                ${data.categorical_columns ? `<p><strong>Categorical Columns:</strong> ${data.categorical_columns.join(', ')}</p>` : ''}
            </div>
        `;
    } catch (err) {
        infoDiv.textContent = "Error: " + err.message;
    }
});

// Initialize Map
function initializeMap() {
    const map = L.map('map').setView([20.5937, 78.9629], 5); // Center of India
    
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 18
    }).addTo(map);
    
    // Add sample markers (you can replace with actual data from your API)
    const sampleLocations = [
        { lat: 28.6139, lng: 77.2090, name: "Delhi", risk: "medium" },
        { lat: 19.0760, lng: 72.8777, name: "Mumbai", risk: "high" },
        { lat: 13.0827, lng: 80.2707, name: "Chennai", risk: "low" },
        { lat: 22.5726, lng: 88.3639, name: "Kolkata", risk: "medium" }
    ];
    
    sampleLocations.forEach(loc => {
        const marker = L.marker([loc.lat, loc.lng]).addTo(map);
        marker.bindPopup(`
            <b>${loc.name}</b><br>
            Risk Level: ${loc.risk.toUpperCase()}<br>
            <button onclick="viewLocationDetails(${loc.lat}, ${loc.lng})">View Details</button>
        `);
    });
    
    return map;
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    initializeMap();
    
    // Load initial model info
    document.getElementById("model-info-btn").click();
});

// Utility function for location details
function viewLocationDetails(lat, lng) {
    alert(`Location Details:\nLatitude: ${lat}\nLongitude: ${lng}\n\nThis would show detailed flood risk assessment.`);
}
