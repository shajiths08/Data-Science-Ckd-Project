"""
predict.py
----------
Inference engine and clinical risk assessment utility for
Chronic Kidney Disease (CKD) prediction.
Provides both single-patient risk evaluation and batch inference.
"""

import os
import joblib
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.data_preprocessing import (
    ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    CLINICAL_RANGES, clean_raw_dataframe
)

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "ckd_pipeline.joblib")


def load_pipeline(model_path: str = DEFAULT_MODEL_PATH):
    """Loads the serialized scikit-learn champion pipeline."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Please execute 'python src/train.py' first."
        )
    return joblib.load(model_path)


def analyze_biomarkers(patient_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Compares patient lab values against standard clinical reference ranges
    to provide medical interpretability for healthcare providers and patients.
    """
    abnormalities = []

    for feat, info in CLINICAL_RANGES.items():
        if feat in patient_dict and patient_dict[feat] is not None:
            val = patient_dict[feat]
            try:
                num_val = float(val)
                min_norm = info["normal_min"]
                max_norm = info["normal_max"]

                if num_val > max_norm:
                    status = "High" if max_norm > 0 else "Present (Abnormal)"
                    abnormalities.append({
                        "feature": feat,
                        "name": info["name"],
                        "value": num_val,
                        "unit": info["unit"],
                        "normal_range": f"{min_norm} - {max_norm} {info['unit']}".strip(),
                        "status": status,
                        "description": info["desc"]
                    })
                elif num_val < min_norm and min_norm > 0:
                    abnormalities.append({
                        "feature": feat,
                        "name": info["name"],
                        "value": num_val,
                        "unit": info["unit"],
                        "normal_range": f"{min_norm} - {max_norm} {info['unit']}".strip(),
                        "status": "Low",
                        "description": info["desc"]
                    })
            except (ValueError, TypeError):
                pass

    # Qualitative clinical flags
    qualitative_flags = {
        "htn": ("Hypertension", "yes", "Chronic high blood pressure accelerates glomerular damage"),
        "dm": ("Diabetes Mellitus", "yes", "Diabetic nephropathy is a leading cause of CKD"),
        "cad": ("Coronary Artery Disease", "yes", "Cardiorenal syndrome interplay"),
        "appet": ("Appetite", "poor", "Uremic toxicity often leads to loss of appetite"),
        "pe": ("Pedal Edema", "yes", "Fluid retention and swelling caused by impaired kidney excretion"),
        "ane": ("Anemia", "yes", "Reduced erythropoietin production by damaged kidneys"),
        "rbc": ("Red Blood Cells in Urine", "abnormal", "Hematuria / glomerular filtration barrier damage"),
        "pc": ("Pus Cells in Urine", "abnormal", "Pyuria / urinary tract or renal parenchyma infection")
    }

    for feat, (label, bad_val, note) in qualitative_flags.items():
        if feat in patient_dict and str(patient_dict[feat]).strip().lower() == bad_val:
            abnormalities.append({
                "feature": feat,
                "name": label,
                "value": patient_dict[feat],
                "unit": "",
                "normal_range": f"Not {bad_val}",
                "status": "Abnormal / Present",
                "description": note
            })

    return abnormalities


def predict_patient(patient_data: Dict[str, Any], pipeline=None) -> Dict[str, Any]:
    """
    Performs comprehensive risk prediction and clinical assessment for a single patient record.
    """
    if pipeline is None:
        pipeline = load_pipeline()

    # Convert single patient record to DataFrame with expected columns
    input_row = {feat: [patient_data.get(feat, np.nan)] for feat in ALL_FEATURES}
    input_df = pd.DataFrame(input_row)

    # Cast numeric columns appropriately
    for num_col in NUMERIC_FEATURES:
        input_df[num_col] = pd.to_numeric(input_df[num_col], errors="coerce")

    # String sanitization for categorical features
    for cat_col in CATEGORICAL_FEATURES:
        if input_df[cat_col].dtype == object:
            input_df[cat_col] = input_df[cat_col].astype(str).str.strip().str.lower()
            input_df[cat_col] = input_df[cat_col].replace(["nan", "none", "?"], np.nan)

    # Predict class & probabilities
    pred_class = int(pipeline.predict(input_df)[0])
    has_proba = hasattr(pipeline, "predict_proba")
    ckd_probability = float(pipeline.predict_proba(input_df)[0, 1]) if has_proba else float(pred_class)

    # Determine risk category
    if ckd_probability >= 0.70:
        risk_category = "High Risk"
        alert_level = "danger"
        recommendation = (
            "High probability of Chronic Kidney Disease detected. Immediate nephrology consultation, "
            "comprehensive serum creatinine/eGFR testing, and strict blood pressure/glycemic control are strongly recommended."
        )
    elif ckd_probability >= 0.35:
        risk_category = "Moderate / Borderline Risk"
        alert_level = "warning"
        recommendation = (
            "Moderate risk indicators detected. Follow-up renal function panel (BUN, Creatinine, microalbuminuria) "
            "and active lifestyle/dietary interventions are advised."
        )
    else:
        risk_category = "Low Risk (Healthy Profile)"
        alert_level = "success"
        recommendation = (
            "Biomarkers indicate low probability of Chronic Kidney Disease. Maintain healthy hydration, "
            "balanced diet, and regular annual health screenings."
        )

    abnormal_biomarkers = analyze_biomarkers(patient_data)

    return {
        "prediction_code": pred_class,
        "prediction_label": "Chronic Kidney Disease (CKD)" if pred_class == 1 else "No CKD (Healthy)",
        "ckd_probability_percent": round(ckd_probability * 100, 2),
        "risk_level": risk_category,
        "alert_level": alert_level,
        "recommendation": recommendation,
        "abnormal_biomarkers": abnormal_biomarkers
    }


def predict_batch(df: pd.DataFrame, pipeline=None) -> pd.DataFrame:
    """Runs batch inference on a multi-row patient DataFrame."""
    if pipeline is None:
        pipeline = load_pipeline()

    # Pre-clean dataframe
    cleaned_df = clean_raw_dataframe(df)

    # Ensure required columns exist, imputing missing columns with NaN if necessary
    for feat in ALL_FEATURES:
        if feat not in cleaned_df.columns:
            cleaned_df[feat] = np.nan

    X_batch = cleaned_df[ALL_FEATURES]

    preds = pipeline.predict(X_batch)
    if hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba(X_batch)[:, 1]
    else:
        probas = preds.astype(float)

    result_df = df.copy()
    result_df["Prediction"] = ["Chronic Kidney Disease (CKD)" if p == 1 else "No CKD (Healthy)" for p in preds]
    result_df["CKD_Probability_%"] = np.round(probas * 100, 2)
    result_df["Risk_Level"] = [
        "High Risk" if pr >= 0.70 else ("Moderate Risk" if pr >= 0.35 else "Low Risk")
        for pr in probas
    ]
    return result_df


if __name__ == "__main__":
    # Smoke test sample
    sample = {
        "age": 48, "bp": 80, "sg": "1.020", "al": "1", "su": "0",
        "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
        "bgr": 121, "bu": 36, "sc": 1.2, "sod": 138, "pot": 4.5,
        "hemo": 15.4, "pcv": 44, "wbcc": 7800, "rbcc": 5.2,
        "htn": "yes", "dm": "yes", "cad": "no", "appet": "good", "pe": "no", "ane": "no"
    }
    print("Inference engine module ready.")
