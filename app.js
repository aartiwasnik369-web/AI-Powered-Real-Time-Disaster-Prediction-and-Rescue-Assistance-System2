// frontend/app.js
const API_BASE = "http://localhost:5000";

// Flood form
document.getElementById("flood-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const rainfall = parseFloat(document.getElementById("rainfall").value || "0");
  const water_level = parseFloat(document.getElementById("water_level").value || "0");

  const resBox = document.getElementById("prediction-result");
  resBox.textContent = "Predicting...";

  try {
    const r = await fetch(`${API_BASE}/api/predict_flood`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({rainfall, water_level})
    });
    const data = await r.json();
    if (data.error) throw new Error(data.error);
    resBox.textContent = `Risk: ${data.risk_level.toUpperCase()} (code=${data.risk_code})\nConfidence: ${Math.round(data.confidence*100)}%`;
  } catch (err) {
    resBox.textContent = "Error: " + err.message;
  }
});

// Detection
document.getElementById("detect-btn").addEventListener("click", async () => {
  const fileInput = document.getElementById("image-input");
  const out = document.getElementById("detection-output");
  if (!fileInput.files.length) {
    out.textContent = "Please choose an image first.";
    return;
  }
  out.textContent = "Detecting...";

  const form = new FormData();
  form.append("image", fileInput.files[0]);
  try {
    const r = await fetch(`${API_BASE}/api/detect_objects`, { method:"POST", body: form });
    const data = await r.json();
    if (data.error) throw new Error(data.error);
    out.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    out.textContent = "Error: " + err.message;
  }
});

// Map + GeoJSON
const map = L.map("map").setView([21.15, 79.08], 6); // Central India-ish
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors"
}).addTo(map);

function styleByRisk(feature) {
  const risk = (feature.properties && feature.properties.risk) || "low";
  const colors = { high:"#ef4444", medium:"#eab308", low:"#22c55e" };
  return {
    color: colors[risk] || "#22c55e",
    weight: 2,
    fillOpacity: 0.3
  };
}

fetch(`${API_BASE}/api/risk_areas`)
  .then(r => r.json())
  .then(geo => {
    if (geo.error) throw new Error(geo.error);
    L.geoJSON(geo, { style: styleByRisk, onEachFeature: (f, layer) => {
      const risk = (f.properties && f.properties.risk) || "low";
      layer.bindPopup(`<b>Risk:</b> ${risk}`);
    }}).addTo(map);
  })
  .catch(err => console.error(err));

// Historical sample
document.getElementById("load-hist").addEventListener("click", async () => {
  const holder = document.getElementById("hist-table");
  holder.textContent = "Loading...";
  try {
    const r = await fetch(`${API_BASE}/api/historical_disasters`);
    const rows = await r.json();
    if (!Array.isArray(rows)) throw new Error("Unexpected response");
    const cols = ["dis_no","year","disaster_type","country","total_deaths","total_affected","total_damages"];
    const thead = `<thead><tr>${cols.map(c=>`<th>${c}</th>`).join("")}</tr></thead>`;
    const tbody = `<tbody>${rows.map(o=>`<tr>${cols.map(c=>`<td>${o[c] ?? ""}</td>`).join("")}</tr>`).join("")}</tbody>`;
    holder.innerHTML = `<table>${thead}${tbody}</table>`;
  } catch (err) {
    holder.textContent = "Error: " + err.message;
  }
});
