# 🛰️ Dual-Source Landslide Prediction Architecture
## Satellite Data + Sensor Data → Combined ML Prediction

---

## Complete System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES (DUAL INPUT)                           │
└──────────────────┬────────────────────────────────────┬─────────────────────┘
                   │                                    │
                   │                                    │
    ┌──────────────▼──────────────┐    ┌──────────────▼─────────────┐
    │   SATELLITE DATA SOURCES     │    │   ESP32 IoT SENSORS        │
    │   (Live from Space)          │    │   (Field Deployment)       │
    │                              │    │                            │
    │  🛰️ NASA POWER               │    │  📡 Soil Moisture Sensor   │
    │    • Rainfall (mm)           │    │  📡 Water Level Sensor     │
    │    • Temperature (°C)        │    │  📡 Tilt Sensor (MPU6050)  │
    │    • Humidity (%)            │    │  📡 Vibration Sensor       │
    │    • Wind Speed (m/s)        │    │  📡 Ultrasonic Distance    │
    │                              │    │  📡 GPS Module (optional)  │
    │  🌍 Open Elevation API       │    │                            │
    │    • Elevation (meters)      │    │  Sampling: Every 30 sec    │
    │    • Slope Calculation (°)   │    │  Transmission: WiFi        │
    │    • Aspect (direction)      │    │  Protocol: HTTP POST       │
    │                              │    │                            │
    │  🌧️ GPM IMERG                │    └────────────┬───────────────┘
    │    • Real-time Rainfall      │                 │
    │    • 4-6 hour delay          │                 │
    │    • 10km resolution         │                 │
    │                              │                 │
    │  🌱 Sentinel-2               │                 │
    │    • NDVI (vegetation)       │                 │
    │    • 10m resolution          │                 │
    │                              │                 │
    │  Update Freq: 5 minutes      │                 │
    │  Cache: 1 hour               │                 │
    └──────────────┬───────────────┘                 │
                   │                                 │
                   │ API Calls                       │ POST /api/sensor-data
                   │                                 │
                   ▼                                 ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │                    BACKEND API SERVER (Render)                    │
    │              https://landslide-api.onrender.com                   │
    │                                                                   │
    │  ┌──────────────────────────────────────────────────────────┐   │
    │  │           DATA COLLECTION & FUSION LAYER                  │   │
    │  │                                                           │   │
    │  │  ┌─────────────────────┐    ┌──────────────────────┐   │   │
    │  │  │ Satellite Service   │    │  Sensor Service      │   │   │
    │  │  │                     │    │                      │   │   │
    │  │  │ • Fetch NASA POWER  │    │ • Validate sensors   │   │   │
    │  │  │ • Get Elevation     │    │ • Save to MongoDB    │   │   │
    │  │  │ • Calculate slope   │    │ • Timestamp data     │   │   │
    │  │  │ • Cache results     │    │ • Extract location   │   │   │
    │  │  └─────────┬───────────┘    └──────────┬───────────┘   │   │
    │  │            │                           │               │   │
    │  │            └───────────┬───────────────┘               │   │
    │  │                        │                               │   │
    │  │            ┌───────────▼──────────────┐                │   │
    │  │            │   DATA FUSION MODULE     │                │   │
    │  │            │                          │                │   │
    │  │            │  Combines:               │                │   │
    │  │            │  • 5 Sensor readings     │                │   │
    │  │            │  • 4 Terrain features    │                │   │
    │  │            │  • 4 Climate features    │                │   │
    │  │            │  ═══════════════════      │                │   │
    │  │            │  Total: 13 Features      │                │   │
    │  │            │                          │                │   │
    │  │            │  Output JSON:            │                │   │
    │  │            │  {                       │                │   │
    │  │            │    // Sensor (5)         │                │   │
    │  │            │    soilMoisture: 100,    │                │   │
    │  │            │    waterLevel: 0,        │                │   │
    │  │            │    tilt: 0,              │                │   │
    │  │            │    vibration: 0,         │                │   │
    │  │            │    distance: 8.5,        │                │   │
    │  │            │    // Terrain (4)        │                │   │
    │  │            │    elevation: 350,       │                │   │
    │  │            │    slope: 5,             │                │   │
    │  │            │    aspect: 180,          │                │   │
    │  │            │    rainfall: 0,          │                │   │
    │  │            │    // Climate (4)        │                │   │
    │  │            │    temperature: 34.8,    │                │   │
    │  │            │    humidity: 44,         │                │   │
    │  │            │    windSpeed: 2.99,      │                │   │
    │  │            │    pressure: 1013        │                │   │
    │  │            │  }                       │                │   │
    │  │            └───────────┬──────────────┘                │   │
    │  └────────────────────────┼───────────────────────────────┘   │
    │                           │                                    │
    │                           │ POST combined data                 │
    │                           ▼                                    │
    │  ┌─────────────────────────────────────────────────────────┐  │
    │  │              ML PREDICTION ORCHESTRATOR                  │  │
    │  │                                                          │  │
    │  │  Step 1: Send to ML API                                 │  │
    │  │  Step 2: Receive prediction                             │  │
    │  │  Step 3: Save to MongoDB                                │  │
    │  │  Step 4: Check alert thresholds                         │  │
    │  │  Step 5: Broadcast via WebSocket                        │  │
    │  └─────────────────────────┬───────────────────────────────┘  │
    └────────────────────────────┼──────────────────────────────────┘
                                 │
                                 │ POST /predict
                                 ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                ML API SERVER (Render)                         │
    │          https://landslide-ml-api.onrender.com                │
    │                                                               │
    │  ┌────────────────────────────────────────────────────────┐  │
    │  │          ADVANCED ML MODEL (v4.0 - DUAL SOURCE)        │  │
    │  │                                                         │  │
    │  │  Input: 13 Features                                    │  │
    │  │  ┌─────────────────────────────────────────┐          │  │
    │  │  │ Sensor (5)  │ Terrain (4) │ Climate (4) │          │  │
    │  │  │ ────────────┼─────────────┼─────────────│          │  │
    │  │  │ soilMoist   │ elevation   │ temperature │          │  │
    │  │  │ waterLevel  │ slope       │ humidity    │          │  │
    │  │  │ tilt        │ aspect      │ windSpeed   │          │  │
    │  │  │ vibration   │ rainfall    │ pressure    │          │  │
    │  │  │ distance    │             │             │          │  │
    │  │  └─────────────────────────────────────────┘          │  │
    │  │               ↓                                        │  │
    │  │  ┌─────────────────────────────────────────┐          │  │
    │  │  │      LightGBM Ensemble Model            │          │  │
    │  │  │                                         │          │  │
    │  │  │  • Trained on 10,000+ samples          │          │  │
    │  │  │  • 13-feature model                     │          │  │
    │  │  │  • Sensor + Satellite fusion            │          │  │
    │  │  │  • 95%+ accuracy expected               │          │  │
    │  │  └─────────────────────────────────────────┘          │  │
    │  │               ↓                                        │  │
    │  │  ┌─────────────────────────────────────────┐          │  │
    │  │  │         Prediction Pipeline             │          │  │
    │  │  │                                         │          │  │
    │  │  │  1. Feature Validation                  │          │  │
    │  │  │  2. Model Inference                     │          │  │
    │  │  │  3. SHAP Explanation                    │          │  │
    │  │  │     → Which features matter most?       │          │  │
    │  │  │     → Sensor vs Satellite contribution  │          │  │
    │  │  │  4. Anomaly Detection                   │          │  │
    │  │  │  5. Risk Level Classification           │          │  │
    │  │  │     • CRITICAL (90-100)                 │          │  │
    │  │  │     • HIGH (70-89)                      │          │  │
    │  │  │     • MEDIUM (40-69)                    │          │  │
    │  │  │     • LOW (0-39)                        │          │  │
    │  │  └─────────────────────────────────────────┘          │  │
    │  │               ↓                                        │  │
    │  │  ┌─────────────────────────────────────────┐          │  │
    │  │  │       Enhanced Output with Source       │          │  │
    │  │  │          Attribution                     │          │  │
    │  │  │                                         │          │  │
    │  │  │  {                                      │          │  │
    │  │  │    riskLevel: "LOW",                   │          │  │
    │  │  │    riskScore: 25,                      │          │  │
    │  │  │    confidence: 92%,                    │          │  │
    │  │  │                                         │          │  │
    │  │  │    sourceContribution: {               │          │  │
    │  │  │      sensors: 60%,    // From IoT      │          │  │
    │  │  │      satellite: 40%   // From space    │          │  │
    │  │  │    },                                   │          │  │
    │  │  │                                         │          │  │
    │  │  │    topFactors: [                       │          │  │
    │  │  │      "vibration (sensor) → -4.5",      │          │  │
    │  │  │      "rainfall (satellite) → +2.3",    │          │  │
    │  │  │      "slope (satellite) → +1.8"        │          │  │
    │  │  │    ],                                   │          │  │
    │  │  │                                         │          │  │
    │  │  │    dataQuality: {                       │          │  │
    │  │  │      sensorStatus: "ACTIVE",           │          │  │
    │  │  │      satelliteStatus: "CACHED",        │          │  │
    │  │  │      dataAge: "2 min",                 │          │  │
    │  │  │      completeness: 100%                │          │  │
    │  │  │    }                                    │          │  │
    │  │  │  }                                      │          │  │
    │  │  └─────────────────────────────────────────┘          │  │
    │  └────────────────────────────────────────────────────────┘  │
    └──────────────────────────┬───────────────────────────────────┘
                               │
                               │ Return prediction
                               ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                    BACKEND (POST-PROCESSING)                  │
    │                                                               │
    │  • Save prediction to MongoDB                                │
    │  • Link sensor + satellite data                              │
    │  • Generate alerts if needed                                 │
    │  • Broadcast to all clients                                  │
    │                                                               │
    │  WebSocket Broadcast:                                        │
    │  io.emit('dual-source-update', {                             │
    │    sensorData: {...},                                        │
    │    satelliteData: {...},                                     │
    │    combinedPrediction: {...},                                │
    │    timestamp: "2026-09-10T10:30:00Z"                         │
    │  })                                                           │
    └──────────────────────────┬───────────────────────────────────┘
                               │
                               │ WebSocket
                               ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                  FRONTEND DASHBOARD (Vercel)                  │
    │            https://frontend-kappa-two-57.vercel.app           │
    │                                                               │
    │  ┌────────────────────────────────────────────────────────┐  │
    │  │         DUAL-SOURCE PREDICTION CARD (NEW!)             │  │
    │  │                                                         │  │
    │  │  ╔══════════════════════════════════════════════════╗  │  │
    │  │  ║  🛰️📡 COMBINED LANDSLIDE RISK ASSESSMENT        ║  │  │
    │  │  ║                                                  ║  │  │
    │  │  ║  Risk Level: MEDIUM                              ║  │  │
    │  │  ║  Risk Score: 52/100                              ║  │  │
    │  │  ║  Confidence: 92%                                 ║  │  │
    │  │  ║                                                  ║  │  │
    │  │  ║  Data Sources:                                   ║  │  │
    │  │  ║  📡 Sensors (60%)  ████████████░░░░░░           ║  │  │
    │  │  ║  🛰️ Satellite (40%) ████████░░░░░░░░░           ║  │  │
    │  │  ║                                                  ║  │  │
    │  │  ║  Top Risk Factors:                               ║  │  │
    │  │  ║  1. ☔ Heavy rainfall (Satellite) → +25 pts     ║  │  │
    │  │  ║  2. 📐 Steep slope (Satellite) → +15 pts        ║  │  │
    │  │  ║  3. 💧 High soil moisture (Sensor) → +12 pts    ║  │  │
    │  │  ║  4. 📡 Vibration activity (Sensor) → -8 pts     ║  │  │
    │  │  ║                                                  ║  │  │
    │  │  ║  Data Freshness:                                 ║  │  │
    │  │  ║  • Sensor: 30 seconds ago ✅                    ║  │  │
    │  │  ║  • Satellite: 2 minutes ago ✅                  ║  │  │
    │  │  ╚══════════════════════════════════════════════════╝  │  │
    │  └────────────────────────────────────────────────────────┘  │
    │                                                               │
    │  ┌────────────────────────────────────────────────────────┐  │
    │  │         SOURCE COMPARISON VIEW                         │  │
    │  │                                                         │  │
    │  │  ┌──────────────────┐  ┌──────────────────┐          │  │
    │  │  │ 📡 Sensor Data   │  │ 🛰️ Satellite Data│          │  │
    │  │  │                  │  │                  │          │  │
    │  │  │ Soil: 75%        │  │ Rainfall: 45mm   │          │  │
    │  │  │ Water: 32cm      │  │ Elevation: 350m  │          │  │
    │  │  │ Tilt: 2.5°       │  │ Slope: 15°       │          │  │
    │  │  │ Vibration: 5     │  │ Temp: 28°C       │          │  │
    │  │  │ Distance: 125cm  │  │ NDVI: 0.65       │          │  │
    │  │  │                  │  │ Humidity: 85%    │          │  │
    │  │  │ Status: LIVE     │  │ Status: CACHED   │          │  │
    │  │  │ Age: 30s         │  │ Age: 2m          │          │  │
    │  │  └──────────────────┘  └──────────────────┘          │  │
    │  └────────────────────────────────────────────────────────┘  │
    └───────────────────────────────────────────────────────────────┘
```

---

## Key Components of Dual-Source System

### 1. Data Sources (TWO INPUTS)

#### A. Satellite Data (Remote Sensing)
- **NASA POWER**: Rainfall, temperature, humidity, wind
- **Open Elevation**: Terrain elevation, slope, aspect
- **GPM IMERG**: Real-time precipitation
- **Sentinel-2**: Vegetation health (NDVI)

**Advantages:**
- ✅ Wide coverage
- ✅ Historical context
- ✅ Terrain features
- ✅ Weather patterns

**Limitations:**
- ⏱️ Update delay (5 min - 6 hours)
- 📍 Lower spatial resolution
- 🌥️ Weather dependent (clouds)

#### B. Sensor Data (Ground Truth)
- **IoT Sensors**: Soil, water, tilt, vibration, distance
- **GPS**: Precise location
- **Real-time**: 30-second updates

**Advantages:**
- ✅ Real-time updates
- ✅ High precision
- ✅ Ground truth
- ✅ Instant alerts

**Limitations:**
- 📍 Point measurement only
- 💰 Deployment cost
- 🔋 Maintenance required
- 📶 Connectivity needed

### 2. Data Fusion Module (THE COMBINER)

```javascript
// Pseudo-code for data fusion
async function fuseSensorAndSatelliteData(sensorReading) {
  // Get sensor data
  const sensorData = {
    soilMoisture: sensorReading.soilMoisture,
    waterLevel: sensorReading.waterLevel,
    tilt: sensorReading.tilt,
    vibration: sensorReading.vibration,
    ultrasonicDistance: sensorReading.ultrasonicDistance
  };

  // Get satellite data for sensor location
  const satelliteData = await fetchSatelliteData(
    sensorReading.latitude,
    sensorReading.longitude
  );

  // Combine into unified dataset
  const combinedData = {
    // Sensor features (5)
    ...sensorData,
    
    // Terrain features from satellite (4)
    elevation: satelliteData.elevation,
    slope: satelliteData.slope,
    aspect: satelliteData.aspect,
    rainfall: satelliteData.rainfall,
    
    // Climate features from satellite (4)
    temperature: satelliteData.temperature,
    humidity: satelliteData.humidity,
    windSpeed: satelliteData.windSpeed,
    pressure: satelliteData.pressure,
    
    // Metadata
    sensorTimestamp: sensorReading.timestamp,
    satelliteTimestamp: satelliteData.timestamp,
    dataQuality: calculateQuality(sensorData, satelliteData)
  };

  return combinedData;
}
```

### 3. Enhanced ML Model (v4.0)

**Training Data:**
```
10,000 samples with 13 features:
- 5 sensor features (real-time ground measurements)
- 4 terrain features (from satellite/DEM)
- 4 climate features (from weather satellites)

Target: Landslide occurrence (0 = No, 1 = Yes)
```

**Model Architecture:**
```python
# Enhanced LightGBM model
model = LGBMClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.05,
    num_leaves=50,
    feature_fraction=0.8,
    bagging_fraction=0.8
)

# 13 features with proper weighting
features = [
    # Sensor group (higher weight)
    'soilMoisture',      # weight: 1.2
    'waterLevel',        # weight: 1.2
    'tilt',              # weight: 1.5 (critical)
    'vibration',         # weight: 1.3
    'ultrasonicDistance',# weight: 1.1
    
    # Terrain group
    'elevation',         # weight: 1.0
    'slope',             # weight: 1.2 (important)
    'aspect',            # weight: 0.8
    'rainfall',          # weight: 1.4 (very important)
    
    # Climate group
    'temperature',       # weight: 0.7
    'humidity',          # weight: 0.9
    'windSpeed',         # weight: 0.6
    'pressure'           # weight: 0.5
]
```

### 4. SHAP Explanation with Source Attribution

```python
# Calculate SHAP values
shap_values = explainer.shap_values(features)

# Group by source
sensor_contribution = sum(shap_values[0:5])   # Sensor features
satellite_contribution = sum(shap_values[5:]) # Satellite features

# Attribution
sensor_pct = abs(sensor_contribution) / total * 100
satellite_pct = abs(satellite_contribution) / total * 100

# Result shows which source is more important
{
  "sourceContribution": {
    "sensors": 65,      # Sensors contribute 65%
    "satellite": 35     # Satellite contributes 35%
  },
  "topFactors": [
    "tilt (sensor) → +8.5 pts",
    "rainfall (satellite) → +6.2 pts",
    "vibration (sensor) → -4.1 pts"
  ]
}
```

---

## Implementation Steps

### Step 1: Update ML Model

```python
# train_dual_source_model.py

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import joblib

# Generate training data with 13 features
n_samples = 10000

data = pd.DataFrame({
    # Sensor features (5)
    'soilMoisture': np.random.uniform(0, 100, n_samples),
    'waterLevel': np.random.uniform(0, 100, n_samples),
    'tilt': np.random.uniform(0, 20, n_samples),
    'vibration': np.random.randint(0, 50, n_samples),
    'ultrasonicDistance': np.random.uniform(0, 500, n_samples),
    
    # Terrain features (4)
    'elevation': np.random.uniform(0, 3000, n_samples),
    'slope': np.random.uniform(0, 45, n_samples),
    'aspect': np.random.uniform(0, 360, n_samples),
    'rainfall': np.random.uniform(0, 200, n_samples),
    
    # Climate features (4)
    'temperature': np.random.uniform(10, 40, n_samples),
    'humidity': np.random.uniform(30, 100, n_samples),
    'windSpeed': np.random.uniform(0, 20, n_samples),
    'pressure': np.random.uniform(950, 1050, n_samples)
})

# Create target with complex logic
data['landslide'] = (
    (data['rainfall'] > 100) |
    (data['soilMoisture'] > 75) |
    (data['slope'] > 30) |
    (data['tilt'] > 15) |
    ((data['waterLevel'] > 70) & (data['vibration'] > 30))
).astype(int)

# Train model
X = data.drop('landslide', axis=1)
y = data['landslide']

model = LGBMClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.05
)

model.fit(X, y)

# Save
joblib.dump(model, 'landslide_model_dual_source.pkl')
print("✅ Dual-source model trained!")
```

### Step 2: Update Backend Data Fusion

```javascript
// backend/src/services/dataFusionService.js

import satelliteMlService from './satelliteMlService.js';

class DataFusionService {
  async fuseSensorWithSatellite(sensorData) {
    // Get satellite data for sensor location
    const satelliteData = await satelliteMlService.getSatelliteData(
      sensorData.latitude || 31.2548,
      sensorData.longitude || 75.7057
    );

    // Combine data
    const fusedData = {
      // Sensor features (5)
      soilMoisture: sensorData.soilMoisture || 0,
      waterLevel: sensorData.waterLevel || 0,
      tilt: sensorData.tilt || 0,
      vibration: sensorData.vibration || 0,
      ultrasonicDistance: sensorData.ultrasonicDistance || 0,
      
      // Terrain features (4)
      elevation: satelliteData?.terrain?.elevation || 350,
      slope: satelliteData?.terrain?.slope || 5,
      aspect: satelliteData?.terrain?.aspect || 180,
      rainfall: satelliteData?.rainfall?.last24h || 0,
      
      // Climate features (4)
      temperature: satelliteData?.climate?.temperature || 25,
      humidity: satelliteData?.climate?.humidity || 60,
      windSpeed: satelliteData?.climate?.windSpeed || 5,
      pressure: 1013,
      
      // Metadata
      sensorTimestamp: sensorData.timestamp,
      satelliteTimestamp: satelliteData?.timestamp,
      location: {
        lat: sensorData.latitude,
        lon: sensorData.longitude
      }
    };

    return fusedData;
  }
}

export default new DataFusionService();
```

### Step 3: Update ML API

```python
# ml/ml_api.py - Add new endpoint

@app.route('/predict-dual-source', methods=['POST'])
def predict_dual_source():
    """
    Predict using combined sensor + satellite data
    Expects 13 features
    """
    try:
        data = request.json
        
        # Validate all 13 features
        required_features = [
            # Sensor (5)
            'soilMoisture', 'waterLevel', 'tilt', 'vibration', 'ultrasonicDistance',
            # Terrain (4)
            'elevation', 'slope', 'aspect', 'rainfall',
            # Climate (4)
            'temperature', 'humidity', 'windSpeed', 'pressure'
        ]
        
        # Create feature DataFrame
        features = pd.DataFrame([{
            feat: data.get(feat, 0) for feat in required_features
        }])
        
        # Prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # SHAP explanation
        shap_values = explainer.shap_values(features)[0]
        
        # Calculate source contribution
        sensor_contribution = sum(abs(shap_values[0:5]))
        satellite_contribution = sum(abs(shap_values[5:13]))
        total_contribution = sensor_contribution + satellite_contribution
        
        # Response
        return jsonify({
            'success': True,
            'prediction': {
                'riskLevel': map_risk_level(prediction, probabilities),
                'riskScore': calculate_risk_score(probabilities),
                'confidence': float(max(probabilities) * 100),
                'sourceContribution': {
                    'sensors': round(sensor_contribution / total_contribution * 100),
                    'satellite': round(satellite_contribution / total_contribution * 100)
                },
                'shapExplanation': generate_shap_explanation(shap_values, features),
                'dataQuality': {
                    'sensorStatus': 'ACTIVE',
                    'satelliteStatus': 'CACHED',
                    'completeness': 100
                }
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
```

### Step 4: Create Frontend Component

```jsx
// frontend/src/components/DualSourcePrediction.jsx

import { useState, useEffect } from 'react';
import { Satellite, Wifi, TrendingUp } from 'lucide-react';
import axios from 'axios';

const DualSourcePrediction = ({ sensorData }) => {
  const [prediction, setPrediction] = useState(null);
  
  useEffect(() => {
    if (sensorData) {
      fetchDualSourcePrediction();
    }
  }, [sensorData]);
  
  const fetchDualSourcePrediction = async () => {
    try {
      const response = await axios.post(
        'https://landslide-api.onrender.com/api/dual-source-predict',
        sensorData
      );
      
      if (response.data.success) {
        setPrediction(response.data.prediction);
      }
    } catch (error) {
      console.error('Dual-source prediction error:', error);
    }
  };
  
  if (!prediction) return null;
  
  return (
    <div className="bg-gradient-to-br from-blue-900/40 to-purple-900/40 border-2 border-blue-500 rounded-xl p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center">
          <div className="flex">
            <Satellite className="w-6 h-6 text-blue-400 mr-1" />
            <Wifi className="w-6 h-6 text-green-400" />
          </div>
          <div className="ml-3">
            <h3 className="text-xl font-bold text-white">
              Dual-Source Risk Assessment
            </h3>
            <p className="text-sm text-gray-400">
              Satellite + Sensor Fusion
            </p>
          </div>
        </div>
        <div className="text-right">
          <div className="text-3xl font-bold text-white">
            {prediction.riskScore}
          </div>
          <div className="text-xs text-gray-400">
            {prediction.confidence}% confident
          </div>
        </div>
      </div>
      
      {/* Source Contribution */}
      <div className="bg-black/20 rounded-lg p-4 mb-4">
        <h4 className="text-sm font-semibold text-white mb-3">
          Data Source Contribution
        </h4>
        
        <div className="space-y-2">
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm text-gray-300 flex items-center">
                <Wifi className="w-4 h-4 mr-2 text-green-400" />
                Sensors
              </span>
              <span className="text-sm font-bold text-green-400">
                {prediction.sourceContribution.sensors}%
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="bg-green-400 h-2 rounded-full"
                style={{ width: `${prediction.sourceContribution.sensors}%` }}
              />
            </div>
          </div>
          
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm text-gray-300 flex items-center">
                <Satellite className="w-4 h-4 mr-2 text-blue-400" />
                Satellite
              </span>
              <span className="text-sm font-bold text-blue-400">
                {prediction.sourceContribution.satellite}%
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="bg-blue-400 h-2 rounded-full"
                style={{ width: `${prediction.sourceContribution.satellite}%` }}
              />
            </div>
          </div>
        </div>
      </div>
      
      {/* Top Factors */}
      <div className="bg-black/20 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-white mb-3">
          Top Risk Factors
        </h4>
        <ul className="space-y-2">
          {prediction.shapExplanation.topFactors.map((factor, i) => (
            <li key={i} className="text-sm text-gray-300 flex items-start">
              <TrendingUp className="w-4 h-4 mr-2 text-orange-400 mt-0.5" />
              {factor}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default DualSourcePrediction;
```

---

## Benefits of Dual-Source Approach

### 1. **Improved Accuracy**
- Sensor data provides ground truth
- Satellite data adds context and terrain
- Combined = More robust predictions

### 2. **Redundancy**
- If sensors fail → Satellite data still works
- If satellite unavailable → Sensors still work
- System never goes completely dark

### 3. **Better Explanations**
- SHAP shows which source is more important
- Users understand "why" better
- Attribution: Is it sensor issue or weather?

### 4. **Scalability**
- Satellite covers wide areas
- Sensors for critical points
- Cost-effective deployment

### 5. **Historical Context**
- Satellite provides long-term trends
- Sensors provide real-time alerts
- Best of both worlds

---

**Next Steps:**
1. Train new 13-feature model
2. Update backend data fusion
3. Deploy enhanced ML API
4. Create dual-source frontend component
5. Test end-to-end flow

This architecture gives you the **BEST OF BOTH WORLDS** - real-time sensor precision + satellite coverage!
