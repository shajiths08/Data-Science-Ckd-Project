"""
app.py
------
Flask Web Application for Chronic Kidney Disease (CKD) Early Risk Prediction.

Connects the trained Scikit-Learn Machine Learning Pipeline to an interactive
HTML/CSS/JavaScript Frontend Dashboard.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Ensure project root is in sys.path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.predict import load_pipeline, predict_patient, analyze_biomarkers
from src.data_preprocessing import ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, CLINICAL_RANGES

app = Flask(__name__)
app.config["SECRET_KEY"] = "ckd-medical-ai-2026-secret"

# Load the trained ML model ONCE when Flask server starts up
try:
    model_pipeline = load_pipeline()
    print("[+] Successfully loaded trained CKD ML Pipeline into Flask memory.")
except Exception as e:
    model_pipeline = None
    print(f"[!] Warning: ML model not found or failed to load ({e}). Run 'python src/train.py' first.")

# Standard Presets for Quick Testing (Viva Demos)
PRESETS = {
    "healthy": {
        "profile_name": "Healthy Individual (Low Risk)",
        "age": 32, "bp": 70.0, "sg": "1.025", "al": "0", "su": "0",
        "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
        "bgr": 95.0, "bu": 22.0, "sc": 0.8, "sod": 142.0, "pot": 4.1,
        "hemo": 15.5, "pcv": 46.0, "wbcc": 6800.0, "rbcc": 5.2,
        "htn": "no", "dm": "no", "cad": "no", "appet": "good", "pe": "no", "ane": "no"
    },
    "borderline": {
        "profile_name": "Borderline / Early Warning Patient",
        "age": 56, "bp": 85.0, "sg": "1.015", "al": "1", "su": "1",
        "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
        "bgr": 152.0, "bu": 48.0, "sc": 1.4, "sod": 136.0, "pot": 4.8,
        "hemo": 11.9, "pcv": 35.0, "wbcc": 8600.0, "rbcc": 4.0,
        "htn": "yes", "dm": "yes", "cad": "no", "appet": "good", "pe": "no", "ane": "no"
    },
    "severe": {
        "profile_name": "High-Risk CKD Patient",
        "age": 64, "bp": 90.0, "sg": "1.010", "al": "3", "su": "2",
        "rbc": "abnormal", "pc": "abnormal", "pcc": "present", "ba": "notpresent",
        "bgr": 210.0, "bu": 82.0, "sc": 4.1, "sod": 128.0, "pot": 5.7,
        "hemo": 8.6, "pcv": 27.0, "wbcc": 11200.0, "rbcc": 3.2,
        "htn": "yes", "dm": "yes", "cad": "yes", "appet": "poor", "pe": "yes", "ane": "yes"
    }
}


@app.route("/")
def index():
    """Renders the main diagnostic input form."""
    metrics_path = os.path.join(ROOT_DIR, "models", "metrics_summary.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    return render_template("index.html", metrics=metrics, clinical_ranges=CLINICAL_RANGES)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives clinical parameters from HTML form, validates types,
    runs ML prediction, and renders the result dashboard.
    """
    global model_pipeline
    if model_pipeline is None:
        try:
            model_pipeline = load_pipeline()
        except Exception as e:
            return f"Error: Trained model file not found ({e}). Please execute 'python src/train.py' first.", 500

    # Extract all features from HTML Form
    patient_data = {}
    for feat in ALL_FEATURES:
        val = request.form.get(feat, "").strip()
        if val == "" or val.lower() == "none":
            patient_data[feat] = np.nan
        else:
            if feat in NUMERIC_FEATURES:
                try:
                    patient_data[feat] = float(val)
                except ValueError:
                    patient_data[feat] = np.nan
            else:
                patient_data[feat] = val.lower()

    # Run inference through ML pipeline
    result = predict_patient(patient_data, pipeline=model_pipeline)

    return render_template(
        "result.html",
        result=result,
        patient_data=patient_data,
        clinical_ranges=CLINICAL_RANGES
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for programmatic or asynchronous prediction."""
    global model_pipeline
    if model_pipeline is None:
        try:
            model_pipeline = load_pipeline()
        except Exception as e:
            return jsonify({"error": f"Model not loaded: {e}"}), 500

    data = request.get_json(force=True)
    result = predict_patient(data, pipeline=model_pipeline)
    return jsonify(result)


@app.route("/sample/<profile_name>")
def get_sample(profile_name):
    """Returns sample patient data in JSON format for instant front-end autofill."""
    profile_key = profile_name.strip().lower()
    if profile_key in PRESETS:
        return jsonify(PRESETS[profile_key])
    return jsonify({"error": "Profile not found"}), 404


@app.route("/metrics")
def get_metrics():
    """Returns model benchmark scores."""
    metrics_path = os.path.join(ROOT_DIR, "models", "metrics_summary.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Metrics summary not found"}), 404


if __name__ == "__main__":
    # Host on all interfaces, port 5000
    print("[*] Starting Flask server on http://localhost:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=True)
