"""
app.py
======
Flask Backend for the Chronic Kidney Disease (CKD) Predictor.

This file is the HEART of your web application.
It connects your trained Machine Learning model to the outside world
so that a browser (HTML/JS) or any other tool can send patient data
and receive a CKD prediction.

How it works step by step:
  1. Flask starts and loads the saved ML pipeline from disk.
  2. A browser sends a POST request to /predict with patient JSON data.
  3. Flask validates and cleans the data.
  4. Flask builds a Pandas DataFrame (exact same format as training).
  5. The ML pipeline preprocesses + predicts automatically.
  6. Flask sends back a JSON response with the prediction result.

Run this file with:
    python app.py
Then open: http://localhost:5000
"""

# ============================================================
# STEP 1: IMPORT THE LIBRARIES WE NEED
# ============================================================

import os           # To work with file paths
import sys          # To modify the Python module search path
import json         # To read/write JSON data
import numpy as np  # For NaN values when data is missing
import pandas as pd # To create DataFrames (table of patient data)

# Flask is the web framework - it lets us create routes (URLs)
from flask import Flask, request, jsonify, render_template

# flask_cors allows our HTML/JavaScript frontend to talk to Flask
# Without CORS, browsers block cross-origin requests for security
from flask_cors import CORS

# ============================================================
# STEP 2: IMPORT OUR OWN MODULES
# ============================================================

# Make sure Python can find our src/ folder
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import the feature lists we defined during training
# NUMERIC_FEATURES  = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
# CATEGORICAL_FEATURES = ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
# ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES  (24 total)
from src.data_preprocessing import (
    ALL_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    CLINICAL_RANGES
)

# Import our prediction helper functions from src/predict.py
from src.predict import load_pipeline, predict_patient, analyze_biomarkers

# ============================================================
# STEP 3: CREATE THE FLASK APPLICATION
# ============================================================

app = Flask(__name__)

# Enable CORS for ALL routes.
# This means any webpage (HTML/JS) can send requests to this Flask server,
# even if it runs on a different port or domain.
# Without this line, browsers will block the request and show a CORS error.
CORS(app)

# A secret key is needed for sessions and security features.
# In a real production app, this should be a long random string stored in
# an environment variable - NOT written directly in code like this.
app.config["SECRET_KEY"] = "ckd-academic-project-2026"

# ============================================================
# STEP 4: LOAD THE TRAINED ML PIPELINE
# ============================================================

# We load the model ONCE when the server starts.
# This is important! Loading a model on every request would be very slow.
# The model is stored in a global variable so all routes can access it.

MODEL_PATH = os.path.join(ROOT_DIR, "models", "ckd_best_model.pkl")
FALLBACK_PATH = os.path.join(ROOT_DIR, "models", "ckd_pipeline.joblib")

model_pipeline = None  # We will fill this in below

try:
    model_pipeline = load_pipeline()  # Tries both .pkl and .joblib automatically
    print("[+] ML Pipeline loaded successfully.")
    print("[+] Server is ready to accept predictions.")
except FileNotFoundError:
    # If the model file is not found, the server still starts but
    # the /predict route will return an error message.
    print("[!] WARNING: Trained model file not found.")
    print("[!] Please run 'python src/train.py' first to train and save the model.")

# ============================================================
# STEP 5: DEFINE VALID CATEGORIES FOR CATEGORICAL FIELDS
# ============================================================

# These are the exact values that the model was trained on.
# If the user sends a different value (e.g. "Yes" instead of "yes"),
# we will try to fix it or return an error.

VALID_CATEGORIES = {
    "sg":    ["1.005", "1.010", "1.015", "1.020", "1.025"],
    "al":    ["0", "1", "2", "3", "4", "5"],
    "su":    ["0", "1", "2", "3", "4", "5"],
    "rbc":   ["normal", "abnormal"],
    "pc":    ["normal", "abnormal"],
    "pcc":   ["present", "notpresent"],
    "ba":    ["present", "notpresent"],
    "htn":   ["yes", "no"],
    "dm":    ["yes", "no"],
    "cad":   ["yes", "no"],
    "appet": ["good", "poor"],
    "pe":    ["yes", "no"],
    "ane":   ["yes", "no"]
}


# Clinical physiological boundaries (for input validation and catching negative/absurd values)
NUMERIC_BOUNDS = {
    "age":  (1.0, 120.0, "years"),
    "bp":   (20.0, 300.0, "mm/Hg"),
    "bgr":  (20.0, 1000.0, "mg/dL"),
    "bu":   (1.0, 500.0, "mg/dL"),
    "sc":   (0.1, 30.0, "mg/dL"),
    "sod":  (50.0, 200.0, "mEq/L"),
    "pot":  (1.0, 15.0, "mEq/L"),
    "hemo": (1.0, 25.0, "g/dL"),
    "pcv":  (5.0, 75.0, "%"),
    "wbcc": (500.0, 100000.0, "cells/µL"),
    "rbcc": (0.5, 10.0, "M/µL")
}


# ============================================================
# HELPER FUNCTION: BUILD PATIENT DATAFRAME
# ============================================================

def build_patient_dataframe(raw_data):
    """
    Converts raw JSON input (a Python dictionary) into a Pandas DataFrame
    with the EXACT same column names and order used during model training.

    This is critical!
    If the column order is wrong, the model gives wrong predictions.
    If a column is missing, we fill it with NaN so the imputer can handle it.

    Parameters:
    -----------
    raw_data : dict
        The JSON body sent by the user.

    Returns:
    --------
    patient_df : pd.DataFrame
        A one-row DataFrame ready to be passed to the model.

    errors : list
        A list of validation error messages (empty if everything is fine).
    """
    patient_row = {}  # We will build this dict feature by feature
    errors = []       # Collect all problems found in the input

    # --- Process NUMERIC features ---
    # These must be numbers (int or float).
    # If they are missing or cannot be converted, we use NaN.
    # The pipeline's SimpleImputer will fill NaN with the median later.
    for feature in NUMERIC_FEATURES:
        value = raw_data.get(feature)  # .get() returns None if key is missing

        if value is None or str(value).strip() == "":
            # Missing field - fill with NaN, let imputer handle it
            patient_row[feature] = np.nan
        else:
            try:
                # Convert to float (handles "1.2", 1, 1.2 etc.)
                num_val = float(value)
                # Check realistic clinical boundaries (prevent negative or absurd inputs)
                if feature in NUMERIC_BOUNDS:
                    min_b, max_b, unit = NUMERIC_BOUNDS[feature]
                    if num_val < min_b or num_val > max_b:
                        errors.append(
                            f"Field '{feature}' must be between {min_b} and {max_b} {unit}. Got: {num_val}"
                        )
                patient_row[feature] = num_val
            except (ValueError, TypeError):
                # User sent something like "abc" for a number field
                errors.append(
                    f"Field '{feature}' must be a number. Got: '{value}'"
                )
                patient_row[feature] = np.nan

    # --- Process CATEGORICAL features ---
    # These must be specific string values (e.g. "yes", "no", "normal").
    # We lowercase and strip whitespace to be forgiving with input.
    for feature in CATEGORICAL_FEATURES:
        value = raw_data.get(feature)

        if value is None or str(value).strip() == "":
            # Missing - let imputer handle it with most_frequent strategy
            patient_row[feature] = np.nan
        else:
            # Normalize: lowercase and remove extra spaces
            cleaned = str(value).strip().lower()
            patient_row[feature] = cleaned

    # --- Build the DataFrame with EXACT feature order ---
    # ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES (24 columns, fixed order)
    # We wrap patient_row in a list [{}] to make a single-row DataFrame
    patient_df = pd.DataFrame([patient_row], columns=ALL_FEATURES)

    # Final safety step: ensure numeric columns are float dtype
    for feature in NUMERIC_FEATURES:
        patient_df[feature] = pd.to_numeric(patient_df[feature], errors="coerce")

    return patient_df, errors


# ============================================================
# ROUTE 1: GET /
# Purpose: Home page / health check
# ============================================================

@app.route("/")
def index():
    """
    GET /
    -----
    This route serves the main HTML page.
    When you open http://localhost:5000 in your browser, this runs.

    It also reads the model performance metrics from a JSON file
    (if it exists) to display on the home page.
    """
    # Load model metrics for display on the home page (optional)
    metrics = {}
    metrics_path = os.path.join(ROOT_DIR, "models", "metrics_summary.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    # render_template() looks in the /templates folder for the HTML file
    return render_template(
        "index.html",
        metrics=metrics,
        clinical_ranges=CLINICAL_RANGES
    )


# ============================================================
# ROUTE 2: GET /health
# Purpose: Simple API health check
# ============================================================

@app.route("/health")
def health_check():
    """
    GET /health
    -----------
    A simple endpoint that returns the server status.
    Useful to quickly check if the server is running and if
    the model is loaded, without opening a browser.

    Test it: curl http://localhost:5000/health
    Or just visit: http://localhost:5000/health
    """
    return jsonify({
        "status": "ok",
        "message": "CKD Predictor API is running.",
        "model_loaded": model_pipeline is not None,
        "total_features": len(ALL_FEATURES),
        "features": ALL_FEATURES
    })


# ============================================================
# ROUTE 3: POST /predict
# Purpose: THE MAIN PREDICTION ROUTE
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    -------------
    This is the main prediction endpoint.

    How to call it from JavaScript:
        fetch("http://localhost:5000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ age: 45, bp: 80, sc: 1.2, ... })
        })

    How to call it from Python (for testing):
        import requests
        r = requests.post("http://localhost:5000/predict",
                         json={"age": 45, "bp": 80, "sc": 1.2, ...})
        print(r.json())

    Request body: JSON with any of the 24 CKD features.
    Missing fields are allowed - the model will handle them with imputation.

    Response JSON:
    {
        "prediction": "CKD" or "No CKD",
        "prediction_code": 1 or 0,
        "probability": 0.87,
        "risk_level": "High Risk",
        "message": "...",
        "abnormal_count": 3,
        "model_used": "Random Forest"
    }
    """
    # STEP A: Check if the model is loaded
    # ------------------------------------
    if model_pipeline is None:
        return jsonify({
            "error": "Model not loaded.",
            "message": "Please run 'python src/train.py' to train the model first.",
            "hint": "The file 'models/ckd_best_model.pkl' was not found."
        }), 503  # 503 = Service Unavailable


    # STEP B: Get JSON data from the request
    # ---------------------------------------
    # request.get_json() reads the JSON body sent by the client.
    # force=True means we accept JSON even if Content-Type header is wrong.
    # silent=True means we get None instead of an error if JSON is malformed.
    raw_data = request.get_json(force=True, silent=True)

    if raw_data is None:
        return jsonify({
            "error": "Invalid JSON.",
            "message": "Please send a valid JSON body with patient data.",
            "example": {
                "age": 45, "bp": 80, "sc": 1.2,
                "hemo": 14.0, "htn": "yes", "dm": "no"
            }
        }), 400  # 400 = Bad Request

    # Check that at least one patient biomarker or history field is actually provided
    non_empty_features = [
        k for k in ALL_FEATURES
        if raw_data.get(k) is not None and str(raw_data.get(k)).strip() != ""
    ]
    if len(non_empty_features) == 0:
        return jsonify({
            "error": "Empty patient data.",
            "message": "At least one patient biomarker or clinical parameter must be provided.",
            "hint": "Use the sample profile presets at /sample/healthy to quickly populate the form."
        }), 400  # 400 = Bad Request


    # STEP C: Build DataFrame and validate the input
    # -----------------------------------------------
    patient_df, validation_errors = build_patient_dataframe(raw_data)

    # If there were type errors (e.g. "abc" sent for a number field), return them
    if validation_errors:
        return jsonify({
            "error": "Validation failed.",
            "problems": validation_errors,
            "message": "Please fix the listed fields and try again."
        }), 422  # 422 = Unprocessable Entity


    # STEP D: Run the ML Pipeline
    # ----------------------------
    # predict_patient() is our helper function from src/predict.py
    # It handles:
    #   - Passing the DataFrame through the preprocessing pipeline
    #   - Getting the predicted class (0 or 1)
    #   - Getting the probability (0.0 to 1.0)
    #   - Checking clinical ranges for anomaly explanation
    try:
        result = predict_patient(raw_data, pipeline=model_pipeline)

    except Exception as e:
        # Catch any unexpected model errors and return a clean error message
        return jsonify({
            "error": "Prediction failed.",
            "message": str(e),
            "hint": "Check that all feature values are in the correct format."
        }), 500  # 500 = Internal Server Error


    # STEP E: Build and return the final JSON response
    # -------------------------------------------------
    # We return a clean, simple JSON that the frontend can easily read.

    prediction_label = "CKD" if result["prediction_code"] == 1 else "No CKD"
    probability      = result["ckd_probability_percent"] / 100.0  # Convert % back to 0-1

    # Write a plain English message based on the risk level
    if result["alert_level"] == "danger":
        message = (
            "The model predicts a HIGH likelihood of Chronic Kidney Disease. "
            "Immediate medical consultation is strongly advised."
        )
    elif result["alert_level"] == "warning":
        message = (
            "The model detects MODERATE risk indicators. "
            "Follow-up testing and monitoring are recommended."
        )
    else:
        message = (
            "The model predicts LOW likelihood of Chronic Kidney Disease. "
            "Maintain healthy habits and regular check-ups."
        )

    return jsonify({
        # Core prediction result
        "prediction":      prediction_label,
        "prediction_code": result["prediction_code"],      # 1 = CKD, 0 = No CKD
        "probability":     round(probability, 4),          # e.g. 0.87
        "probability_pct": result["ckd_probability_percent"],  # e.g. 87.0

        # Risk assessment
        "risk_level":  result["risk_level"],    # "High Risk" / "Moderate Risk" / "Low Risk"
        "alert_level": result["alert_level"],   # "danger" / "warning" / "success"

        # Plain English explanation
        "message": message,
        "recommendation": result["recommendation"],

        # Anomaly details
        "abnormal_count":      len(result["abnormal_biomarkers"]),
        "abnormal_biomarkers": result["abnormal_biomarkers"],

        # Metadata
        "model_used":    "Random Forest",
        "disclaimer":    "Academic prediction only. NOT a medical diagnosis."
    })


# ============================================================
# ROUTE 4: POST /api/predict
# Purpose: Alternate JSON-only endpoint (same logic, different URL)
# ============================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    POST /api/predict
    -----------------
    Identical to /predict.
    This is the "pure API" endpoint — it only returns JSON.
    Useful when you have a separate frontend that sends JSON
    and does NOT submit an HTML form.

    This is a standard REST API convention:
    - /predict    -> for HTML form submissions
    - /api/predict -> for JSON API calls from JavaScript fetch()
    """
    # Simply forward to the same predict() function
    return predict()


# ============================================================
# ROUTE 5: GET /sample/<profile>
# Purpose: Return demo patient data for testing
# ============================================================

# Three pre-defined test patients for viva demos and frontend autofill
SAMPLE_PROFILES = {
    "healthy": {
        "profile_name": "Healthy Adult (Low Risk)",
        "age": 32,    "bp": 70.0,   "sg": "1.025", "al": "0",    "su": "0",
        "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
        "bgr": 95.0,  "bu": 22.0,   "sc": 0.8,     "sod": 142.0, "pot": 4.1,
        "hemo": 15.5, "pcv": 46.0,  "wbcc": 6800.0, "rbcc": 5.2,
        "htn": "no",  "dm": "no",   "cad": "no",   "appet": "good", "pe": "no", "ane": "no"
    },
    "borderline": {
        "profile_name": "Borderline / Early Warning",
        "age": 56,    "bp": 85.0,   "sg": "1.015", "al": "1",    "su": "1",
        "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
        "bgr": 152.0, "bu": 48.0,   "sc": 1.4,     "sod": 136.0, "pot": 4.8,
        "hemo": 11.9, "pcv": 35.0,  "wbcc": 8600.0, "rbcc": 4.0,
        "htn": "yes", "dm": "yes",  "cad": "no",   "appet": "good", "pe": "no", "ane": "no"
    },
    "severe": {
        "profile_name": "High-Risk CKD Patient",
        "age": 64,    "bp": 90.0,   "sg": "1.010", "al": "3",    "su": "2",
        "rbc": "abnormal", "pc": "abnormal", "pcc": "present", "ba": "notpresent",
        "bgr": 210.0, "bu": 82.0,   "sc": 4.1,     "sod": 128.0, "pot": 5.7,
        "hemo": 8.6,  "pcv": 27.0,  "wbcc": 11200.0, "rbcc": 3.2,
        "htn": "yes", "dm": "yes",  "cad": "yes",  "appet": "poor", "pe": "yes", "ane": "yes"
    }
}


@app.route("/sample/<profile_name>")
def get_sample(profile_name):
    """
    GET /sample/<profile_name>
    --------------------------
    Returns a pre-filled sample patient as JSON.
    Used by the frontend JavaScript to autofill the form.

    Examples:
        GET /sample/healthy     -> returns healthy patient data
        GET /sample/borderline  -> returns borderline patient data
        GET /sample/severe      -> returns high-risk patient data
    """
    key = profile_name.strip().lower()
    if key in SAMPLE_PROFILES:
        return jsonify(SAMPLE_PROFILES[key])
    return jsonify({
        "error": f"Profile '{profile_name}' not found.",
        "available_profiles": list(SAMPLE_PROFILES.keys())
    }), 404  # 404 = Not Found


# ============================================================
# ROUTE 6: GET /metrics
# Purpose: Return model performance metrics as JSON
# ============================================================

@app.route("/metrics")
def get_metrics():
    """
    GET /metrics
    ------------
    Returns the saved model benchmark results as JSON.
    Useful for displaying model accuracy on the frontend.

    The metrics file is created by src/train.py after training.
    """
    metrics_path = os.path.join(ROOT_DIR, "models", "metrics_summary.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return jsonify(json.load(f))
    return jsonify({
        "error": "Metrics file not found.",
        "hint": "Run 'python src/train.py' to generate model metrics."
    }), 404


# ============================================================
# STEP 6: RUN THE FLASK SERVER
# ============================================================

if __name__ == "__main__":
    """
    This block only runs when you execute:
        python app.py

    It does NOT run when Flask is started by a production server like gunicorn.

    debug=True  -> Automatically reloads when you save the file.
                   Also shows detailed error pages in the browser.
                   NEVER use debug=True in production (security risk).

    host="0.0.0.0" -> Makes the server accessible from other devices
                      on the same WiFi network (useful for testing on phone).
                      Use "127.0.0.1" if you only want local access.

    port=5000   -> The port number. Access at http://localhost:5000
    """
    print("=" * 60)
    print("  Chronic Kidney Disease Predictor - Flask Backend")
    print("=" * 60)
    print("  [*] Open your browser at: http://localhost:5000")
    print("  [*] API endpoint:         http://localhost:5000/predict")
    print("  [*] Health check:         http://localhost:5000/health")
    print("  [*] Model metrics:        http://localhost:5000/metrics")
    print("  [*] Press CTRL+C to stop the server")
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True   # Set to False before deploying publicly
    )
