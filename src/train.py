"""
train.py
--------
Model training, cross-validation, multi-model benchmarking,
feature importance extraction, and pipeline serialization for CKD prediction.
Adheres strictly to ML Best Practices (split before fitting, leak-free pipeline).
"""

import os
import sys
import json
import warnings
from typing import Dict, Any
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Models to benchmark
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.data_preprocessing import (
    load_dataset, get_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    clean_raw_dataframe
)


def get_models() -> Dict[str, Any]:
    """Returns candidate classifiers for benchmarking."""
    return {
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Support Vector Machine": SVC(probability=True, kernel="rbf", C=1.0, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42)
    }


def train_and_evaluate(data_path: str = "data/raw/kidney_disease.csv",
                       models_dir: str = "models",
                       processed_data_dir: str = "data/processed"):
    """
    Executes the full machine learning training & evaluation lifecycle:
    1. Loads and cleans raw data
    2. Performs stratified train-test split (80/20)
    3. Runs 5-Fold Stratified Cross-Validation on all candidate models
    4. Evaluates test set metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
    5. Saves champion model pipeline, metrics, feature importances, and sample data
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)

    print("=" * 65)
    print("CHRONIC KIDNEY DISEASE (CKD) PREDICTOR - MODEL TRAINING PIPELINE")
    print("=" * 65)

    # 1. Load data
    print(f"[*] Loading raw dataset from: {data_path}")
    X, y = load_dataset(data_path)
    total_samples = len(X)
    ckd_count = int(y.sum())
    notckd_count = total_samples - ckd_count
    print(f"    Total patient records: {total_samples}")
    print(f"    CKD Cases: {ckd_count} ({ckd_count / total_samples * 100:.1f}%)")
    print(f"    Non-CKD Cases: {notckd_count} ({notckd_count / total_samples * 100:.1f}%)")

    # Save cleaned full dataset for reference & inspection
    raw_df = pd.read_csv(data_path)
    clean_df = clean_raw_dataframe(raw_df)
    clean_csv_path = os.path.join(processed_data_dir, "ckd_cleaned.csv")
    clean_df.to_csv(clean_csv_path, index=False)
    print(f"[+] Saved cleaned dataset to: {clean_csv_path}")

    # 2. Strict Featurization Ordering: Split train/test before fitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[*] Data split into Train: {len(X_train)} samples, Test: {len(X_test)} samples (Stratified)")

    # 3. Benchmark Models
    candidate_models = get_models()
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    benchmark_results = {}
    fitted_pipelines = {}

    print("\n" + "-" * 75)
    print(f"{'Model':<24} | {'CV Acc':<8} | {'Test Acc':<9} | {'Precision':<9} | {'Recall':<7} | {'F1':<6} | {'ROC-AUC':<7}")
    print("-" * 75)

    best_model_name = None
    best_f1_score = -1.0

    for name, clf in candidate_models.items():
        preprocessor = get_preprocessor()
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        # Cross-validation on train split only
        cv_scores = cross_validate(
            pipeline, X_train, y_train, cv=cv_strategy,
            scoring=["accuracy", "f1", "roc_auc"], n_jobs=-1
        )
        mean_cv_acc = np.mean(cv_scores["test_accuracy"])

        # Fit on entire training set
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Evaluate on held-out test set
        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_proba))
        else:
            roc_auc = float(roc_auc_score(y_test, y_pred))

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        benchmark_results[name] = {
            "cv_accuracy_mean": round(float(mean_cv_acc), 4),
            "cv_accuracy_std": round(float(np.std(cv_scores["test_accuracy"])), 4),
            "test_accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": cm
        }

        print(f"{name:<24} | {mean_cv_acc:.4f}   | {acc:.4f}    | {prec:.4f}    | {rec:.4f} | {f1:.4f} | {roc_auc:.4f}")

        # Choose best model prioritizing F1 score and ROC-AUC
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model_name = name

    print("-" * 75)
    print(f"\n[*] Champion Model: {best_model_name} (F1: {best_f1_score:.4f})")

    # 4. Save Champion Pipeline
    champion_pipeline = fitted_pipelines[best_model_name]
    pipeline_save_path = os.path.join(models_dir, "ckd_pipeline.joblib")
    joblib.dump(champion_pipeline, pipeline_save_path)
    print(f"[+] Saved serialized pipeline to: {pipeline_save_path}")

    # 5. Extract Feature Importances from Random Forest
    rf_pipeline = fitted_pipelines["Random Forest"]
    fitted_preprocessor = rf_pipeline.named_steps["preprocessor"]
    fitted_rf = rf_pipeline.named_steps["classifier"]

    # Extract transformed feature names
    cat_encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_transformed_feature_names = NUMERIC_FEATURES + cat_feature_names

    importances = fitted_rf.feature_importances_
    feature_imp_list = [
        {"feature": feat, "importance": round(float(imp), 4)}
        for feat, imp in sorted(zip(all_transformed_feature_names, importances), key=lambda x: x[1], reverse=True)
    ]

    imp_save_path = os.path.join(models_dir, "feature_importance.json")
    with open(imp_save_path, "w") as f:
        json.dump(feature_imp_list, f, indent=2)
    print(f"[+] Saved feature importances to: {imp_save_path}")

    # 6. Save Benchmark Metrics Summary
    summary_data = {
        "champion_model": best_model_name,
        "dataset_summary": {
            "total_samples": total_samples,
            "ckd_cases": ckd_count,
            "notckd_cases": notckd_count,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        },
        "models": benchmark_results
    }
    metrics_save_path = os.path.join(models_dir, "metrics_summary.json")
    with open(metrics_save_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[+] Saved benchmark metrics to: {metrics_save_path}")

    # 7. Generate Sample Patient Profiles for testing & UI quick-selection
    generate_sample_patients(X_test, y_test, processed_data_dir)

    print("=" * 65)
    print("TRAINING & VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 65)


def generate_sample_patients(X_test: pd.DataFrame, y_test: pd.Series, output_dir: str):
    """Generates standard sample patient profiles representing Healthy, Mild Risk, and Severe CKD."""
    # Profile 1: Healthy adult
    healthy = {
        "profile_name": "Healthy Adult (Low Risk)",
        "age": 35.0, "bp": 70.0, "sg": "1.025", "al": "0", "su": "0",
        "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
        "bgr": 95.0, "bu": 24.0, "sc": 0.9, "sod": 140.0, "pot": 4.2,
        "hemo": 15.2, "pcv": 45.0, "wbcc": 6500.0, "rbcc": 5.1,
        "htn": "no", "dm": "no", "cad": "no", "appet": "good", "pe": "no", "ane": "no"
    }

    # Profile 2: Borderline / Early Warning signs
    borderline = {
        "profile_name": "Borderline / Early Warning",
        "age": 55.0, "bp": 85.0, "sg": "1.015", "al": "1", "su": "1",
        "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
        "bgr": 150.0, "bu": 48.0, "sc": 1.4, "sod": 136.0, "pot": 4.7,
        "hemo": 12.1, "pcv": 36.0, "wbcc": 8200.0, "rbcc": 4.1,
        "htn": "yes", "dm": "yes", "cad": "no", "appet": "good", "pe": "no", "ane": "no"
    }

    # Profile 3: Advanced CKD Patient
    severe = {
        "profile_name": "Severe CKD Patient (High Risk)",
        "age": 65.0, "bp": 90.0, "sg": "1.010", "al": "3", "su": "2",
        "rbc": "abnormal", "pc": "abnormal", "pcc": "present", "ba": "notpresent",
        "bgr": 210.0, "bu": 85.0, "sc": 4.2, "sod": 128.0, "pot": 5.6,
        "hemo": 8.4, "pcv": 26.0, "wbcc": 11500.0, "rbcc": 3.1,
        "htn": "yes", "dm": "yes", "cad": "yes", "appet": "poor", "pe": "yes", "ane": "yes"
    }

    sample_df = pd.DataFrame([healthy, borderline, severe])
    sample_csv_path = os.path.join(output_dir, "sample_patients.csv")
    sample_df.to_csv(sample_csv_path, index=False)
    print(f"[+] Saved sample patient profiles to: {sample_csv_path}")


if __name__ == "__main__":
    train_and_evaluate()
