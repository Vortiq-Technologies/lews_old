#!/usr/bin/env python3
"""
ML Prediction API - Uses trained Random Forest model
Receives sensor data and returns landslide risk prediction with trend forecasting
"""

from flask import Flask, request, jsonify
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from trend_forecasting import TrendForecaster
import shap

# Custom JSON provider to handle numpy/pandas types
class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        return super().default(obj)

app = Flask(__name__)
app.json = NumpyJSONProvider(app)

# Enable CORS with explicit configuration
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://frontend-kappa-two-57.vercel.app",
            "https://*.vercel.app",
            "http://localhost:3000",
            "http://localhost:5173"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": False,
        "max_age": 3600
    }
})

# Helper function to convert numpy types recursively
def convert_to_python_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, (pd.DataFrame,)):
        return obj.to_dict('records')
    else:
        return obj

# Load the trained model
# Using simple 5-feature model that matches frontend sensor data
MODEL_PATH = 'landslide_model_simple.pkl'  # Simple 5-feature model (soilMoisture, waterLevel, tilt, vibration, ultrasonicDistance)
FALLBACK_MODEL_PATH = 'landslide_model.pkl'     # Fallback: Sensor-only model (99.4% accuracy)
ANOMALY_MODEL_PATH  = 'anomaly_model.pkl'
ANOMALY_SCALER_PATH = 'anomaly_scaler.pkl'
ANOMALY_THRESH_PATH = 'anomaly_thresholds.pkl'

model = None
anomaly_model = None
anomaly_scaler = None
anomaly_thresholds = None
forecaster = TrendForecaster()
shap_explainer = None  # SHAP explainer for model interpretability

def load_model():
    global model, anomaly_model, anomaly_scaler, anomaly_thresholds, shap_explainer

    model_loaded = False
    model_type = "Unknown"
    
    # Try loading historical model first (better for nationwide deployment)
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        model_type = "LightGBM (Historical India Data)"
        print(f"✅ Historical model loaded from {MODEL_PATH}")
        print(f"   Model: {model_type}")
        print(f"   Accuracy: 90.7%, F1: 86.4%, ROC-AUC: 97.6%")
        print(f"   Training: 5000 India landslide records (1998-2022)")
        print(f"   Coverage: 10 regions (Uttarakhand, Himachal, J&K, Kerala, Sikkim, etc.)")
        model_loaded = True
        
    # Fallback to sensor-only model if historical not found
    elif os.path.exists(FALLBACK_MODEL_PATH):
        model = joblib.load(FALLBACK_MODEL_PATH)
        model_type = "LightGBM (Sensor Data)"
        print(f"⚠️  Historical model not found, using fallback: {FALLBACK_MODEL_PATH}")
        print(f"   Model: {model_type}")
        print(f"   Accuracy: 99.4%, F1: 99.1%")
        print(f"   Training: 825 sensor records (Chandigarh)")
        model_loaded = True
    else:
        print(f"❌ No model files found!")
        print(f"   Tried: {MODEL_PATH}")
        print(f"   Tried: {FALLBACK_MODEL_PATH}")
        return False
    
    if model_loaded:
        # Initialize SHAP explainer for the loaded model
        try:
            print("🔍 Initializing SHAP explainer...")
            shap_explainer = shap.TreeExplainer(model)
            print("✅ SHAP explainer initialized successfully")
        except Exception as e:
            print(f"⚠️  SHAP explainer initialization failed: {e}")
            shap_explainer = None

    if os.path.exists(ANOMALY_MODEL_PATH):
        anomaly_model    = joblib.load(ANOMALY_MODEL_PATH)
        anomaly_scaler   = joblib.load(ANOMALY_SCALER_PATH)
        anomaly_thresholds = joblib.load(ANOMALY_THRESH_PATH)
        print(f"✅ Anomaly Detection model loaded")
    else:
        print(f"⚠️  Anomaly model not found — run train_anomaly_model.py")

    return True

@app.route('/', methods=['GET'])
def home():
    """Root endpoint - API info"""
    model_info = {
        'type': 'LightGBM (Historical India Data)',
        'accuracy': '90.7%',
        'f1_score': '86.4%',
        'roc_auc': '97.6%',
        'training_data': '5000 India landslide records (1998-2022)',
        'coverage': '10 regions across India',
        'features': '9 (terrain + weather + sensors)'
    }
    
    return jsonify({
        'name': 'Landslide ML Prediction API',
        'version': '3.0.0',
        'status': 'operational',
        'model': model_info,
        'explainability': 'SHAP (SHapley Additive exPlanations)',
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST)',
            'retrain': '/retrain (POST)'
        },
        'usage': {
            'example': 'POST /predict with JSON body',
            'required_fields': ['soilMoisture', 'waterLevel', 'tilt', 'vibration'],
            'optional_fields': ['ultrasonicDistance', 'history']
        },
        'features': [
            'Risk prediction (LOW/MEDIUM/HIGH/CRITICAL)',
            'Confidence score',
            'SHAP explainability - why this prediction?',
            'Feature importance analysis',
            'Anomaly detection',
            'Trend forecasting'
        ]
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'shap_enabled': shap_explainer is not None,
        'anomaly_model_loaded': anomaly_model is not None,
        'model_path': MODEL_PATH,
        'explainability': 'SHAP TreeExplainer' if shap_explainer else 'Not initialized'
    })

def detect_anomaly(current_data, history=None):
    """
    Run anomaly detection on current reading + optional history context.
    Returns anomaly score, severity label, and detected patterns.
    """
    if anomaly_model is None:
        return None

    # Build feature row (same as training)
    soil   = current_data.get('soilMoisture', 0)
    water  = current_data.get('waterLevel', 0)
    tilt   = current_data.get('tilt', 0)
    vib    = current_data.get('vibration', 0)
    dist   = current_data.get('ultrasonicDistance', 0)

    # Rate-of-change features from history
    soil_delta  = 0
    water_delta = 0
    tilt_delta  = 0
    soil_roll_dev  = 0
    water_roll_dev = 0

    if history and len(history) >= 2:
        prev = history[-1]
        soil_delta  = abs(soil  - prev.get('soilMoisture', soil))
        water_delta = abs(water - prev.get('waterLevel', water))
        tilt_delta  = abs(tilt  - prev.get('tilt', tilt))

    if history and len(history) >= 5:
        recent_soil  = [r.get('soilMoisture', 0) for r in history[-5:]]
        recent_water = [r.get('waterLevel', 0)   for r in history[-5:]]
        soil_roll_dev  = abs(soil  - np.mean(recent_soil))
        water_roll_dev = abs(water - np.mean(recent_water))

    combined_risk = (
        (soil  / 100) * 0.35 +
        (water / 100) * 0.35 +
        (tilt  / 90 ) * 0.20 +
        min(vib / 10, 1.0)    * 0.10
    )

    features = np.array([[
        soil, water, tilt, vib, dist,
        soil_delta, water_delta, tilt_delta,
        soil_roll_dev, water_roll_dev,
        combined_risk
    ]])

    features_scaled = anomaly_scaler.transform(features)
    prediction      = anomaly_model.predict(features_scaled)[0]   # 1=normal, -1=anomaly
    score           = float(anomaly_model.decision_function(features_scaled)[0])

    is_anomaly = (prediction == -1)

    # Severity based on calibrated thresholds
    if score < anomaly_thresholds['high']:
        severity = 'HIGH'
    elif score < anomaly_thresholds['medium']:
        severity = 'MEDIUM'
    elif score < anomaly_thresholds['low']:
        severity = 'LOW'
    else:
        severity = 'NORMAL'

    # Identify which patterns triggered the anomaly
    triggered_patterns = []
    if soil_delta > 15:
        triggered_patterns.append(f'Sudden soil moisture spike (+{soil_delta:.1f}%)')
    if water_delta > 20:
        triggered_patterns.append(f'Rapid water level change (+{water_delta:.1f}cm)')
    if tilt_delta > 3:
        triggered_patterns.append(f'Sudden tilt change (+{tilt_delta:.2f}°)')
    if soil_roll_dev > 20:
        triggered_patterns.append(f'Soil moisture {soil_roll_dev:.1f}% above rolling average')
    if water_roll_dev > 25:
        triggered_patterns.append(f'Water level {water_roll_dev:.1f}cm above rolling average')
    if tilt > 15:
        triggered_patterns.append(f'Tilt angle {tilt:.2f}° exceeds safe threshold')
    if vib > 5:
        triggered_patterns.append(f'High vibration count: {vib} events in window')
    if combined_risk > 0.7:
        triggered_patterns.append('Multi-sensor combined risk pattern elevated')

    return {
        'isAnomaly': bool(is_anomaly),
        'score': round(score, 4),
        'severity': severity,
        'patterns': triggered_patterns if triggered_patterns else (['Unusual sensor reading pattern detected'] if is_anomaly else []),
        'description': (
            'Unusual sensor pattern detected — conditions differ significantly from historical baseline.' if is_anomaly
            else 'Sensor readings within normal historical patterns.'
        )
    }

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict landslide risk from sensor data with trend forecasting
    
    Request body:
    {
        "soilMoisture": 10,
        "waterLevel": 22,
        "tilt": 0,
        "vibration": 0,
        "ultrasonicDistance": 376,
        "history": [...]  // Optional: for trend analysis
    }
    
    Response:
    {
        "success": true,
        "prediction": {
            "riskLevel": "LOW",
            "riskScore": 15,
            "confidence": 95.5,
            "features": {...},
            "trends": {...},
            "forecasts": [...],
            "warnings": [...]
        }
    }
    """
    try:
        # Log incoming request for debugging
        print(f"📥 Received prediction request from {request.headers.get('Origin', 'unknown')}")
        print(f"📦 Request data keys: {list(request.json.keys()) if request.json else 'None'}")
        
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        # Get sensor data from request
        data = request.json
        
        # Extract history for later use (don't delete it yet)
        history = data.get('history', [])
        
        # Validate required fields
        required_fields = ['soilMoisture', 'waterLevel', 'tilt', 'vibration']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Prepare features - ONLY 5 sensor features (matches simple model)
        features = pd.DataFrame([{
            'soilMoisture': data.get('soilMoisture', 0),
            'waterLevel': data.get('waterLevel', 0),
            'tilt': data.get('tilt', 0),
            'vibration': data.get('vibration', 0),
            'ultrasonicDistance': data.get('ultrasonicDistance', 0)
        }])
        
        print(f"✅ Features prepared: {list(features.columns)}, shape: {features.shape}")
        
        # Make current prediction
        prediction_class = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities) * 100)
        
        # Map numeric prediction to risk level
        # For binary classification: 0 = No Landslide (LOW), 1 = Landslide (HIGH)
        if prediction_class == 0:
            prediction = 'LOW'
        else:
            # Determine risk level based on probability
            landslide_prob = probabilities[1]  # Probability of landslide class
            if landslide_prob >= 0.8:
                prediction = 'CRITICAL'
            elif landslide_prob >= 0.6:
                prediction = 'HIGH'
            else:
                prediction = 'MEDIUM'
        
        # Calculate risk score (0-100)
        risk_score_map = {
            'LOW': 20,
            'MEDIUM': 50,
            'HIGH': 75,
            'CRITICAL': 95
        }
        base_score = risk_score_map.get(prediction, 20)
        
        # Adjust based on confidence
        risk_score = int(base_score * (confidence / 100))
        
        # Get feature importance
        feature_importance = {
            'soilMoisture': float(model.feature_importances_[0]),
            'waterLevel': float(model.feature_importances_[1]),
            'tilt': float(model.feature_importances_[2]),
            'vibration': float(model.feature_importances_[3]),
            'ultrasonicDistance': float(model.feature_importances_[4])
        }
        
        # Generate SHAP explanation
        shap_explanation = None
        if shap_explainer is not None:
            try:
                # Calculate SHAP values for this prediction
                shap_values = shap_explainer.shap_values(features)
                
                # For binary classification, shap_values is a list [class_0_values, class_1_values]
                # We want the values for the predicted class
                if isinstance(shap_values, list):
                    # Get SHAP values for the positive class (landslide risk)
                    shap_vals = shap_values[-1][0]  # Last class, first sample
                else:
                    shap_vals = shap_values[0]
                
                # Get base value (expected value)
                base_value = shap_explainer.expected_value
                if isinstance(base_value, list):
                    base_value = base_value[-1]
                
                # Create feature contributions dictionary
                feature_names = ['soilMoisture', 'waterLevel', 'tilt', 'vibration', 'ultrasonicDistance']
                contributions = {}
                for i, feature_name in enumerate(feature_names):
                    contributions[feature_name] = {
                        'value': float(features[feature_name].iloc[0]),
                        'contribution': float(shap_vals[i]),
                        'impact': 'increases' if shap_vals[i] > 0 else 'decreases' if shap_vals[i] < 0 else 'neutral'
                    }
                
                # Sort by absolute contribution
                sorted_features = sorted(
                    contributions.items(), 
                    key=lambda x: abs(x[1]['contribution']), 
                    reverse=True
                )
                
                # Generate explanation text
                top_factors = []
                for feature_name, contrib in sorted_features[:3]:  # Top 3 contributors
                    if abs(contrib['contribution']) > 0.01:  # Only significant contributions
                        impact_word = "increasing" if contrib['impact'] == 'increases' else "reducing"
                        # Format feature name for display
                        display_name = feature_name.replace('soilMoisture', 'Soil Moisture').replace('waterLevel', 'Water Level').replace('ultrasonicDistance', 'Distance')
                        top_factors.append(
                            f"{display_name} ({contrib['value']:.1f}) is {impact_word} risk"
                        )
                
                shap_explanation = {
                    'baseValue': float(base_value),
                    'contributions': contributions,
                    'topFactors': top_factors,
                    'explanation': f"Risk prediction driven by: {', '.join(top_factors[:2])}" if top_factors else "Multiple factors contributing equally"
                }
                
                print(f"✅ SHAP explanation generated: {top_factors}")
                
            except Exception as e:
                print(f"⚠️  SHAP explanation failed: {e}")
                shap_explanation = None
        
        # Convert all numeric values to native Python types
        safe_data = {k: int(v) if isinstance(v, (np.integer, np.int64, np.int32)) else 
                        float(v) if isinstance(v, (np.floating, np.float64, np.float32)) else v 
                     for k, v in data.items()}
        
        response = {
            'success': True,
            'prediction': {
                'riskLevel': prediction,  # Already a string (LOW/MEDIUM/HIGH/CRITICAL)
                'riskScore': int(risk_score),
                'confidence': round(float(confidence), 2),
                'features': safe_data,
                'featureImportance': feature_importance
            }
        }
        
        # Add SHAP explanation if available
        if shap_explanation:
            response['prediction']['shapExplanation'] = shap_explanation

        # Run anomaly detection
        current_data_for_anomaly = {
            'soilMoisture': data.get('soilMoisture', 0),
            'waterLevel': data.get('waterLevel', 0),
            'tilt': data.get('tilt', 0),
            'vibration': data.get('vibration', 0),
            'ultrasonicDistance': data.get('ultrasonicDistance', 0)
        }
        anomaly_result = detect_anomaly(current_data_for_anomaly, data.get('history', []))
        if anomaly_result:
            response['prediction']['anomaly'] = anomaly_result
            if anomaly_result['isAnomaly']:
                print(f"🚨 Anomaly detected! Severity: {anomaly_result['severity']}, Score: {anomaly_result['score']}")
        
        # Add trend analysis and forecasting if history provided
        history = data.get('history', [])
        if history and len(history) >= forecaster.min_data_points:
            # Parse timestamps
            for reading in history:
                if isinstance(reading.get('timestamp'), str):
                    reading['timestamp'] = datetime.fromisoformat(reading['timestamp'].replace('Z', '+00:00'))
            
            # Analyze trends
            trends = forecaster.analyze_sensor_trend(history)
            
            if trends:
                # Generate forecasts
                current_data = {
                    'soilMoisture': data.get('soilMoisture', 0),
                    'waterLevel': data.get('waterLevel', 0),
                    'tilt': data.get('tilt', 0),
                    'vibration': data.get('vibration', 0),
                    'ultrasonicDistance': data.get('ultrasonicDistance', 0)
                }
                forecasts = forecaster.forecast_risk(model, current_data, trends, forecast_minutes=[30, 60, 120, 180])
                
                # Generate warnings
                warnings = forecaster.generate_warning_message(prediction, forecasts, trends)
                
                response['prediction']['trends'] = trends
                response['prediction']['forecasts'] = forecasts
                if warnings:
                    response['prediction']['warnings'] = warnings
        
        # Convert all numpy types to Python natives before returning
        return jsonify(convert_to_python_types(response))
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"❌ Error in /predict: {str(e)}")
        print(error_traceback)
        
        # Return detailed error for debugging
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__,
            'traceback': error_traceback.split('\n')[-3:-1] if error_traceback else None
        }), 500

@app.route('/predict-dual-source', methods=['POST'])
def predict_dual_source():
    """
    Predict using combined sensor + satellite data (13 features)
    
    Request body:
    {
        "soilMoisture": 75,
        "waterLevel": 32,
        "tilt": 2.5,
        "vibration": 5,
        "ultrasonicDistance": 125,
        "elevation": 350,
        "slope": 15,
        "aspect": 180,
        "rainfall": 45,
        "temperature": 28,
        "humidity": 85,
        "windSpeed": 5,
        "pressure": 1013
    }
    """
    try:
        # Load dual-source model
        DUAL_MODEL_PATH = 'landslide_model_dual_source.pkl'
        if not os.path.exists(DUAL_MODEL_PATH):
            return jsonify({
                'success': False,
                'error': 'Dual-source model not found. Please train it first.'
            }), 404
        
        dual_model = joblib.load(DUAL_MODEL_PATH)
        
        data = request.json
        
        # Validate all 13 features
        required_features = [
            'soilMoisture', 'waterLevel', 'tilt', 'vibration', 'ultrasonicDistance',
            'elevation', 'slope', 'aspect', 'rainfall',
            'temperature', 'humidity', 'windSpeed', 'pressure'
        ]
        
        missing = [f for f in required_features if f not in data]
        if missing:
            return jsonify({
                'success': False,
                'error': f'Missing features: {", ".join(missing)}'
            }), 400
        
        # Create feature DataFrame
        features = pd.DataFrame([{
            feat: data.get(feat, 0) for feat in required_features
        }])
        
        # Make prediction
        prediction_class = dual_model.predict(features)[0]
        probabilities = dual_model.predict_proba(features)[0]
        
        # Calculate SHAP values
        dual_explainer = shap.TreeExplainer(dual_model)
        shap_values = dual_explainer.shap_values(features)
        
        if isinstance(shap_values, list):
            shap_vals = shap_values[-1][0]
        else:
            shap_vals = shap_values[0]
        
        # Calculate source contributions
        sensor_contribution = sum(abs(shap_vals[0:5]))
        terrain_contribution = sum(abs(shap_vals[5:9]))
        climate_contribution = sum(abs(shap_vals[9:13]))
        total_contribution = sensor_contribution + terrain_contribution + climate_contribution
        
        # Get top factors with source labels
        feature_names = required_features
        contributions = [(i, feature_names[i], shap_vals[i]) for i in range(len(shap_vals))]
        contributions.sort(key=lambda x: abs(x[2]), reverse=True)
        
        top_factors = []
        for i, feature_name, contrib in contributions[:5]:
            if abs(contrib) > 0.01:
                source = 'sensor' if i < 5 else 'satellite'
                impact = '+' if contrib > 0 else ''
                display_name = feature_name.replace('soilMoisture', 'Soil Moisture')\
                                         .replace('waterLevel', 'Water Level')\
                                         .replace('ultrasonicDistance', 'Distance')
                top_factors.append(f"{display_name} ({source}): {impact}{contrib:.2f}")
        
        # Map prediction to risk level
        if prediction_class == 0:
            risk_level = 'LOW'
            risk_score = 20
        else:
            landslide_prob = probabilities[1]
            if landslide_prob >= 0.8:
                risk_level = 'CRITICAL'
                risk_score = 90
            elif landslide_prob >= 0.6:
                risk_level = 'HIGH'
                risk_score = 75
            else:
                risk_level = 'MEDIUM'
                risk_score = 50
        
        confidence = float(max(probabilities) * 100)
        
        response = {
            'success': True,
            'prediction': {
                'riskLevel': risk_level,
                'riskScore': int(risk_score),
                'confidence': round(confidence, 2),
                'sourceContribution': {
                    'sensors': round(sensor_contribution / total_contribution * 100) if total_contribution > 0 else 50,
                    'terrain': round(terrain_contribution / total_contribution * 100) if total_contribution > 0 else 25,
                    'climate': round(climate_contribution / total_contribution * 100) if total_contribution > 0 else 25,
                    'satellite': round((terrain_contribution + climate_contribution) / total_contribution * 100) if total_contribution > 0 else 50
                },
                'topFactors': top_factors,
                'dataQuality': data.get('dataQuality', {
                    'sensorStatus': 'ACTIVE',
                    'satelliteStatus': 'CACHED',
                    'completeness': 100
                })
            }
        }
        
        return jsonify(convert_to_python_types(response))
        
    except Exception as e:
        import traceback
        print(f"❌ Dual-source prediction error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    """
    Retrain model with new data (future feature)
    """
    return jsonify({
        'success': False,
        'error': 'Retraining not yet implemented'
    }), 501

# Load model when app starts (for gunicorn)
print("🚀 Loading ML Model...")
if not load_model():
    print("⚠️  Warning: Model not loaded. Please train a model first.")
else:
    print("✅ Model loaded successfully!")
    print(f"📊 Model: LightGBM Historical (90.7% accuracy, 97.6% ROC-AUC)")
    print(f"🌏 Coverage: 10 India regions, 5000 historical patterns")

# For local development only
if __name__ == '__main__':
    print("🚀 Starting ML Prediction API (Development Mode)...")
    port = int(os.getenv('PORT', os.getenv('ML_API_PORT', 5001)))
    print(f"🌐 ML API listening on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
