"""
train.py
--------
Model Training and Benchmarking Pipeline for Chronic Kidney Disease (CKD) Prediction.

Evaluates 4 Classical ML Algorithms (as specified in academic curriculum):
1. Logistic Regression (Linear Baseline)
2. Decision Tree Classifier
3. Random Forest Classifier (Ensemble)
4. Support Vector Machine (SVM)

Metrics Computed:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.

Outputs:
- Serialized Champion Model: models/ckd_best_model.pkl
- Benchmark Metrics Summary: models/metrics_summary.json
- Feature Importance Ranking: models/feature_importance.json
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# Ensure project root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# The 4 Core Algorithms for College Project
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.data_preprocessing import (
    load_dataset, get_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    clean_raw_dataframe
)


def get_core_models():
    """
    Returns the 4 designated machine learning algorithms with standard hyperparameters.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "Support Vector Machine": SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
    }


def train_models():
    """
    Full training, 5-Fold cross-validation, evaluation, and serialization routine.
    """
    data_path = os.path.join(ROOT_DIR, "data", "raw", "kidney_disease.csv")
    models_dir = os.path.join(ROOT_DIR, "models")
    processed_dir = os.path.join(ROOT_DIR, "data", "processed")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    print("=" * 70)
    print("CHRONIC KIDNEY DISEASE (CKD) PREDICTOR - 4-MODEL BENCHMARK PIPELINE")
    print("=" * 70)

    # 1. Load Dataset
    print(f"[*] Loading raw dataset: {data_path}")
    X, y = load_dataset(data_path)
    total_samples = len(X)
    ckd_count = int(y.sum())
    notckd_count = total_samples - ckd_count
    print(f"    Total Patient Records: {total_samples}")
    print(f"    - CKD Positive: {ckd_count} ({ckd_count/total_samples*100:.1f}%)")
    print(f"    - Non-CKD Healthy: {notckd_count} ({notckd_count/total_samples*100:.1f}%)")

    # 2. Strict Featurization: Train/Test Split BEFORE fitting any transformer (Avoid Data Leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[*] Train/Test Split: {len(X_train)} Train samples, {len(X_test)} Test samples (80/20 Stratified)")

    # 3. Benchmark 4 Models with 5-Fold Cross Validation
    models = get_core_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics_summary = {}
    trained_pipelines = {}

    print("\n" + "-" * 78)
    print(f"{'Algorithm':<24} | {'CV Acc':<8} | {'Test Acc':<9} | {'Precision':<9} | {'Recall':<7} | {'F1':<6} | {'ROC-AUC':<7}")
    print("-" * 78)

    best_model_name = None
    best_f1 = -1.0

    for name, clf in models.items():
        # Build individual pipeline: Preprocessing (Median/Mode Imputer + Scaler/OneHot) -> Classifier
        pipe = Pipeline([
            ("preprocessor", get_preprocessor()),
            ("classifier", clf)
        ])

        # 5-Fold CV on training split
        cv_scores = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=["accuracy", "f1", "roc_auc"], n_jobs=-1
        )
        mean_cv_acc = np.mean(cv_scores["test_accuracy"])

        # Train on full training split
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        # Evaluate on unseen test split
        y_pred = pipe.predict(X_test)
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_proba))
        else:
            roc_auc = float(roc_auc_score(y_test, y_pred))

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        metrics_summary[name] = {
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

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    print("-" * 78)
    print(f"\n[*] Champion Model Selected: {best_model_name} (F1 Score: {best_f1:.4f})")

    # 4. Save the Champion Model to models/ckd_best_model.pkl
    champion_pipeline = trained_pipelines[best_model_name]
    best_model_path = os.path.join(models_dir, "ckd_best_model.pkl")
    joblib.dump(champion_pipeline, best_model_path)
    # Also save as ckd_pipeline.joblib for compatibility
    joblib.dump(champion_pipeline, os.path.join(models_dir, "ckd_pipeline.joblib"))
    print(f"[+] Saved champion pipeline to: {best_model_path}")

    # 5. Extract Feature Importances from Random Forest
    rf_pipe = trained_pipelines["Random Forest"]
    fitted_preprocessor = rf_pipe.named_steps["preprocessor"]
    fitted_rf = rf_pipe.named_steps["classifier"]

    cat_encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_feature_names = NUMERIC_FEATURES + cat_feature_names

    importances = fitted_rf.feature_importances_
    feature_imp_list = [
        {"feature": feat, "importance": round(float(imp), 4)}
        for feat, imp in sorted(zip(all_feature_names, importances), key=lambda x: x[1], reverse=True)
    ]

    imp_save_path = os.path.join(models_dir, "feature_importance.json")
    with open(imp_save_path, "w") as f:
        json.dump(feature_imp_list, f, indent=2)
    print(f"[+] Saved feature importances to: {imp_save_path}")

    # 6. Save Metrics Summary JSON
    summary_report = {
        "champion_model": best_model_name,
        "dataset_summary": {
            "total_samples": total_samples,
            "ckd_cases": ckd_count,
            "notckd_cases": notckd_count,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        },
        "models": metrics_summary
    }
    metrics_path = os.path.join(models_dir, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"[+] Saved metrics summary to: {metrics_path}")

    print("=" * 70)
    print("MODEL TRAINING & SERIALIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    train_models()
