#!/usr/bin/env python3
"""
Dual-Source Model Training
Combines 13 features: 5 sensor + 4 terrain + 4 climate
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from lightgbm import LGBMClassifier
import joblib
import json
from datetime import datetime

print("🚀 Training Dual-Source Landslide Model (13 Features)")
print("=" * 60)

# Generate realistic training data
np.random.seed(42)
n_samples = 10000

print(f"📊 Generating {n_samples} training samples...")

data = pd.DataFrame({
    # ========== SENSOR FEATURES (5) ==========
    'soilMoisture': np.random.uniform(0, 100, n_samples),
    'waterLevel': np.random.uniform(0, 100, n_samples),
    'tilt': np.random.uniform(0, 20, n_samples),
    'vibration': np.random.randint(0, 50, n_samples),
    'ultrasonicDistance': np.random.uniform(0, 500, n_samples),
    
    # ========== TERRAIN FEATURES (4) ==========
    'elevation': np.random.uniform(0, 3000, n_samples),
    'slope': np.random.uniform(0, 45, n_samples),
    'aspect': np.random.uniform(0, 360, n_samples),
    'rainfall': np.random.uniform(0, 200, n_samples),
    
    # ========== CLIMATE FEATURES (4) ==========
    'temperature': np.random.uniform(10, 40, n_samples),
    'humidity': np.random.uniform(30, 100, n_samples),
    'windSpeed': np.random.uniform(0, 20, n_samples),
    'pressure': np.random.uniform(950, 1050, n_samples)
})

print("\n📝 Creating target variable with multi-factor risk logic...")

# Complex landslide risk logic combining multiple sources
data['landslide'] = (
    # High risk conditions
    (data['rainfall'] > 100) |  # Heavy rainfall
    (data['soilMoisture'] > 75) |  # Saturated soil
    (data['slope'] > 30) |  # Steep terrain
    (data['tilt'] > 15) |  # Ground movement
    ((data['waterLevel'] > 70) & (data['vibration'] > 30)) |  # Water + vibration
    ((data['rainfall'] > 50) & (data['slope'] > 20)) |  # Rain + slope
    ((data['soilMoisture'] > 60) & (data['tilt'] > 10)) |  # Moisture + tilt
    ((data['humidity'] > 90) & (data['rainfall'] > 80))  # High humidity + rain
).astype(int)

print(f"✅ Target distribution: {data['landslide'].value_counts().to_dict()}")

# Prepare features and target
X = data.drop('landslide', axis=1)
y = data['landslide']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# Train LightGBM model with enhanced parameters
print("\n🤖 Training LightGBM Classifier...")
model = LGBMClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.05,
    num_leaves=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42,
    verbose=-1
)

model.fit(X_train, y_train)

# Evaluate model
print("\n📈 Evaluating model performance...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n✅ MODEL PERFORMANCE:")
print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   F1 Score:  {f1:.4f}")
print(f"   ROC-AUC:   {roc_auc:.4f}")

print(f"\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Risk', 'Landslide']))

# Feature importance
print(f"\n🔍 Feature Importance:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.iterrows():
    print(f"   {row['feature']:20s}: {row['importance']:.4f}")

# Save model
model_path = 'landslide_model_dual_source.pkl'
joblib.dump(model, model_path)
print(f"\n💾 Model saved: {model_path}")

# Save metadata
metadata = {
    "model_version": "4.0",
    "model_name": "Dual-Source Landslide Prediction Model",
    "training_date": datetime.now().isoformat(),
    "features": {
        "total": 13,
        "sensor": list(X.columns[:5]),
        "terrain": list(X.columns[5:9]),
        "climate": list(X.columns[9:13])
    },
    "performance": {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc)
    },
    "training_samples": n_samples,
    "model_type": "LightGBM",
    "description": "Combines sensor data + satellite terrain + climate for comprehensive landslide risk assessment"
}

metadata_path = 'model_metadata_dual_source.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"📄 Metadata saved: {metadata_path}")

print("\n" + "=" * 60)
print("✅ DUAL-SOURCE MODEL READY!")
print("=" * 60)
print("\nNext steps:")
print("1. Update ml_api.py to load this model")
print("2. Add /predict-dual-source endpoint")
print("3. Create backend data fusion service")
print("4. Deploy to Render")
