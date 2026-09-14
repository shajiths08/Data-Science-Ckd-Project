"""
data_preprocessing.py
---------------------
Data cleaning, validation, imputation, and feature transformation pipelines
for Chronic Kidney Disease (CKD) prediction.
Follows strict featurization ordering (no data leakage).
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Canonical feature definitions based on UCI Clinical Dataset
NUMERIC_FEATURES: List[str] = [
    "age",    # Age in years
    "bp",     # Blood Pressure (mm/Hg)
    "bgr",    # Blood Glucose Random (mgs/dl)
    "bu",     # Blood Urea (mgs/dl)
    "sc",     # Serum Creatinine (mgs/dl)
    "sod",    # Sodium (mEq/L)
    "pot",    # Potassium (mEq/L)
    "hemo",   # Hemoglobin (gms)
    "pcv",    # Packed Cell Volume (%)
    "wbcc",   # White Blood Cell Count (cells/cumm)
    "rbcc"    # Red Blood Cell Count (millions/cmm)
]

CATEGORICAL_FEATURES: List[str] = [
    "sg",     # Specific Gravity (1.005, 1.010, 1.015, 1.020, 1.025)
    "al",     # Albumin (0, 1, 2, 3, 4, 5)
    "su",     # Sugar (0, 1, 2, 3, 4, 5)
    "rbc",    # Red Blood Cells (normal, abnormal)
    "pc",     # Pus Cell (normal, abnormal)
    "pcc",    # Pus Cell Clumps (present, notpresent)
    "ba",     # Bacteria (present, notpresent)
    "htn",    # Hypertension (yes, no)
    "dm",     # Diabetes Mellitus (yes, no)
    "cad",    # Coronary Artery Disease (yes, no)
    "appet",  # Appetite (good, poor)
    "pe",     # Pedal Edema (yes, no)
    "ane"     # Anemia (yes, no)
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "class"

# Standard Clinical Reference Ranges (for UI explainability and risk analysis)
CLINICAL_RANGES: Dict[str, Dict[str, Any]] = {
    "bp": {"name": "Blood Pressure", "unit": "mm/Hg", "normal_min": 60, "normal_max": 80, "desc": "Diastolic blood pressure"},
    "bgr": {"name": "Random Blood Glucose", "unit": "mg/dL", "normal_min": 70, "normal_max": 140, "desc": "Blood sugar concentration"},
    "bu": {"name": "Blood Urea", "unit": "mg/dL", "normal_min": 10, "normal_max": 50, "desc": "Waste product filtered by kidneys"},
    "sc": {"name": "Serum Creatinine", "unit": "mg/dL", "normal_min": 0.6, "normal_max": 1.2, "desc": "Critical kidney filtration marker (higher indicates dysfunction)"},
    "sod": {"name": "Sodium", "unit": "mEq/L", "normal_min": 135, "normal_max": 145, "desc": "Electrolyte balance"},
    "pot": {"name": "Potassium", "unit": "mEq/L", "normal_min": 3.5, "normal_max": 5.0, "desc": "Electrolyte regulated by kidneys"},
    "hemo": {"name": "Hemoglobin", "unit": "g/dL", "normal_min": 12.0, "normal_max": 17.5, "desc": "Oxygen-carrying protein (low in CKD anemia)"},
    "pcv": {"name": "Packed Cell Volume", "unit": "%", "normal_min": 36, "normal_max": 50, "desc": "Percentage of red blood cells in blood"},
    "wbcc": {"name": "White Blood Cell Count", "unit": "cells/µL", "normal_min": 4000, "normal_max": 11000, "desc": "Immune system activity marker"},
    "rbcc": {"name": "Red Blood Cell Count", "unit": "M/µL", "normal_min": 4.2, "normal_max": 5.9, "desc": "Red blood cell concentration"},
    "sg": {"name": "Specific Gravity", "unit": "", "normal_min": 1.015, "normal_max": 1.025, "desc": "Urine concentration ability of kidneys"},
    "al": {"name": "Albumin", "unit": "scale 0-5", "normal_min": 0, "normal_max": 0, "desc": "Protein leaking into urine (hallmark of kidney damage)"},
    "su": {"name": "Urine Sugar", "unit": "scale 0-5", "normal_min": 0, "normal_max": 0, "desc": "Glucose excreted in urine"}
}


def clean_raw_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans typos, anomalous string characters, and formatting artifacts
    present in the raw UCI dataset.
    """
    df = raw_df.copy()

    # Strip column name whitespace
    df.columns = [c.strip() for c in df.columns]

    # Replace '?' with NaN across entire dataset
    df = df.replace("?", np.nan).replace("\t?", np.nan)

    # Clean string columns: strip trailing/leading whitespace and lowercase
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace(["nan", "none", "?"], np.nan)

    # Convert columns that may be mistyped as object into numeric
    for num_col in ["pcv", "wbcc", "rbcc", "age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    # Specific typographical corrections for nominal variables in UCI CKD
    if "dm" in df.columns:
        df["dm"] = df["dm"].replace({"\tyes": "yes", " yes": "yes", "\tno": "no"})
    if "cad" in df.columns:
        df["cad"] = df["cad"].replace({"\tno": "no"})
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].replace({"ckd\t": "ckd", "\tckd": "ckd"})

    return df


def get_preprocessor() -> ColumnTransformer:
    """
    Constructs a robust scikit-learn preprocessing ColumnTransformer.
    - Numerical features: Median imputation followed by StandardScaler.
    - Categorical features: Most-frequent imputation followed by OneHotEncoder.
    This guarantees zero data leakage when fitted solely on the training split.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERIC_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor


def load_dataset(file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads and cleans raw data, returning feature matrix X and binary target y.
    y: 1 for CKD, 0 for notCKD.
    """
    raw_df = pd.read_csv(file_path)
    cleaned_df = clean_raw_dataframe(raw_df)

    # Ensure target column exists
    if TARGET_COLUMN not in cleaned_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    # Drop any row where target is missing
    cleaned_df = cleaned_df.dropna(subset=[TARGET_COLUMN])

    # Convert target to binary: 1 = ckd, 0 = notckd
    y = (cleaned_df[TARGET_COLUMN] == "ckd").astype(int)
    X = cleaned_df[ALL_FEATURES]

    return X, y
