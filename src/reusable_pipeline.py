"""
reusable_pipeline.py
--------------------
Unified Scikit-Learn Pipeline for Chronic Kidney Disease (CKD) Prediction.

Combines Data Preprocessing (Imputation + Scaling + OneHotEncoding)
and the Champion Machine Learning Classifier (Random Forest) into
a SINGLE, self-contained pipeline object.

Key Features:
- Zero data leakage (fitted exclusively on training split)
- Saved via joblib together with the exact feature column order
- Reusable function: predict_ckd(patient_data)
- Beginner-friendly and viva-ready
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from src.data_preprocessing import (
    load_dataset, NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES
)

PIPELINE_PATH = os.path.join(ROOT_DIR, "models", "ckd_pipeline.joblib")
COLUMNS_PATH = os.path.join(ROOT_DIR, "models", "feature_columns.json")


# ==============================================================================
# 1. BUILD, TRAIN & SAVE THE UNIFIED PIPELINE
# ==============================================================================
def build_and_save_pipeline(data_path="data/raw/kidney_disease.csv"):
    """
    Constructs, trains, evaluates, and serializes the complete end-to-end pipeline.
    """
    print("=" * 70)
    print("BUILDING UNIFIED REUSABLE PIPELINE (PREPROCESSING + MODEL)")
    print("=" * 70)

    # 1. Load Raw Dataset
    X, y = load_dataset(os.path.join(ROOT_DIR, data_path))
    print(f"[+] Loaded dataset: {len(X)} patient records.")

    # 2. Strict Train-Test Split (80% Train, 20% Test) BEFORE fitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[+] Train/Test Split: {len(X_train)} Train, {len(X_test)} Test (Stratified).")

    # 3. Build Preprocessor using ColumnTransformer
    # Numerical Sub-Pipeline: Median Imputation -> Standardization
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical Sub-Pipeline: Most-Frequent (Mode) Imputation -> One-Hot Encoding
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    # 4. Combine Preprocessor and Model into ONE Unified Pipeline
    champion_model = RandomForestClassifier(
        n_estimators=100, max_depth=6, random_state=42
    )

    complete_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", champion_model)
    ])

    # 5. Train the Unified Pipeline on Training Data ONLY
    print("[*] Training unified pipeline on training partition...")
    complete_pipeline.fit(X_train, y_train)

    # 6. Evaluate on Unseen Test Data
    y_pred = complete_pipeline.predict(X_test)
    y_prob = complete_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n[+] Pipeline Evaluation on Unseen Test Split (80 patients):")
    print(f"    - Accuracy: {acc*100:.2f}%")
    print(f"    - Recall:   {rec:.4f} (Sensitivity)")
    print(f"    - F1-Score: {f1:.4f}")
    print(f"    - ROC-AUC:  {auc:.4f}")

    # 7. Save Complete Pipeline using joblib
    os.makedirs(os.path.dirname(PIPELINE_PATH), exist_ok=True)
    joblib.dump(complete_pipeline, PIPELINE_PATH)
    # Also save as ckd_best_model.pkl for standard naming
    joblib.dump(complete_pipeline, os.path.join(ROOT_DIR, "models", "ckd_best_model.pkl"))
    print(f"\n[+] Saved unified pipeline to: {PIPELINE_PATH}")

    # 8. Save Exact Feature Column Order
    feature_order = list(ALL_FEATURES)
    with open(COLUMNS_PATH, "w") as f:
        json.dump(feature_order, f, indent=2)
    print(f"[+] Saved exact feature column order ({len(feature_order)} features) to: {COLUMNS_PATH}")

    print("=" * 70)
    return complete_pipeline, feature_order


# Cached in-memory pipeline for ultra-fast prediction
_LOADED_PIPELINE = None
_LOADED_FEATURE_ORDER = None


# ==============================================================================
# 2. REUSABLE PREDICTION FUNCTION
# ==============================================================================
def predict_ckd(patient_data):
    """
    Accepts a single patient's clinical inputs (as a Python dictionary),
    applies the exact same preprocessing transformations automatically,
    and returns the predicted class and disease probability.

    Parameters:
    -----------
    patient_data : dict
        Dictionary containing clinical values (e.g., {'sc': 1.2, 'hemo': 14.5, ...})

    Returns:
    --------
    dict:
        {
            'prediction': 0 or 1,
            'label': 'Chronic Kidney Disease (CKD)' or 'No CKD (Healthy)',
            'probability': float (between 0.0 and 1.0),
            'probability_percent': float (0.0% to 100.0%)
        }
    """
    global _LOADED_PIPELINE, _LOADED_FEATURE_ORDER

    # 1. Load pipeline and feature order if not already in memory
    if _LOADED_PIPELINE is None:
        if not os.path.exists(PIPELINE_PATH):
            build_and_save_pipeline()
        _LOADED_PIPELINE = joblib.load(PIPELINE_PATH)

    if _LOADED_FEATURE_ORDER is None:
        if os.path.exists(COLUMNS_PATH):
            with open(COLUMNS_PATH) as f:
                _LOADED_FEATURE_ORDER = json.load(f)
        else:
            _LOADED_FEATURE_ORDER = ALL_FEATURES

    # 2. Convert dictionary into a single-row DataFrame with the EXACT column order
    # Any missing key is safely populated with NaN (imputer handles it)
    input_row = {col: [patient_data.get(col, np.nan)] for col in _LOADED_FEATURE_ORDER}
    input_df = pd.DataFrame(input_row)

    # Cast numeric columns to float to avoid datatype issues
    for col in NUMERIC_FEATURES:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    # Sanitize string categorical inputs
    for col in CATEGORICAL_FEATURES:
        if input_df[col].dtype == "object":
            input_df[col] = input_df[col].astype(str).str.strip().str.lower()
            input_df[col] = input_df[col].replace(["nan", "none", "?"], np.nan)

    # 3. Make Prediction using the unified pipeline
    # (Preprocessing is applied automatically inside the pipeline!)
    pred_class = int(_LOADED_PIPELINE.predict(input_df)[0])

    # 4. Extract Probability if predict_proba is supported
    if hasattr(_LOADED_PIPELINE, "predict_proba"):
        prob_array = _LOADED_PIPELINE.predict_proba(input_df)
        prob_ckd = float(prob_array[0, 1])
    else:
        prob_ckd = float(pred_class)

    label = "Chronic Kidney Disease (CKD)" if pred_class == 1 else "No CKD (Healthy)"

    return {
        "prediction": pred_class,
        "label": label,
        "probability": round(prob_ckd, 4),
        "probability_percent": round(prob_ckd * 100, 2)
    }


# ==============================================================================
# 3. TEST EXAMPLES (DEMO & VERIFICATION)
# ==============================================================================
if __name__ == "__main__":
    # Ensure pipeline is built and saved
    build_and_save_pipeline()

    print("\n" + "=" * 70)
    print("TESTING PREDICTION FUNCTION: predict_ckd(patient_data)")
    print("=" * 70)

    # Sample 1: Healthy Patient
    healthy_patient = {
        "age": 35, "bp": 70, "sg": "1.025", "al": "0", "su": "0",
        "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
        "bgr": 95, "bu": 24, "sc": 0.8, "sod": 140, "pot": 4.2,
        "hemo": 15.2, "pcv": 45, "wbcc": 6500, "rbcc": 5.1,
        "htn": "no", "dm": "no", "cad": "no", "appet": "good", "pe": "no", "ane": "no"
    }

    result_healthy = predict_ckd(healthy_patient)
    print("\n[Case 1] Healthy Patient Input:")
    print(f" -> Predicted Class:       {result_healthy['prediction']}")
    print(f" -> Predicted Label:       {result_healthy['label']}")
    print(f" -> Disease Probability:   {result_healthy['probability_percent']}%")

    # Sample 2: High-Risk CKD Patient
    ckd_patient = {
        "age": 64, "bp": 90, "sg": "1.010", "al": "3", "su": "2",
        "rbc": "abnormal", "pc": "abnormal", "pcc": "present", "ba": "notpresent",
        "bgr": 210, "bu": 82, "sc": 4.1, "sod": 128, "pot": 5.7,
        "hemo": 8.6, "pcv": 27, "wbcc": 11200, "rbcc": 3.2,
        "htn": "yes", "dm": "yes", "cad": "yes", "appet": "poor", "pe": "yes", "ane": "yes"
    }

    result_ckd = predict_ckd(ckd_patient)
    print("\n[Case 2] High-Risk CKD Patient Input:")
    print(f" -> Predicted Class:       {result_ckd['prediction']}")
    print(f" -> Predicted Label:       {result_ckd['label']}")
    print(f" -> Disease Probability:   {result_ckd['probability_percent']}%")
    print("=" * 70)
