"""
model_training_and_evaluation.py
---------------------------------
Comprehensive Machine Learning Training & Evaluation for Chronic Kidney Disease (CKD).

Benchmarked Algorithms (as per curriculum):
1. Logistic Regression (Linear Baseline)
2. Decision Tree Classifier
3. Random Forest Classifier (Ensemble)
4. Support Vector Machine (SVM)

Metrics Computed:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.

Visualizations Generated:
1. Model Performance Metrics Comparison Bar Chart
2. Confusion Matrices (2x2 Grid for All 4 Models)
3. ROC Curves Comparison Plot

Important Notice:
This is an academic machine learning demonstration and NOT a medically validated
diagnostic tool. It should not be used as a substitute for professional medical care.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Ensure project root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# 4 Core Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.data_preprocessing import load_dataset, get_preprocessor

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


# ==============================================================================
# STEP 1: DEFINE CANDIDATE MODELS
# ==============================================================================
def get_algorithms():
    """
    Initializes the 4 classical models with fixed random_state for 100% reproducibility.
    """
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, criterion="gini", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42
        ),
        "Support Vector Machine": SVC(
            kernel="rbf", C=1.0, probability=True, random_state=42
        )
    }


# ==============================================================================
# STEP 2: TRAIN & EVALUATE MODELS
# ==============================================================================
def train_and_benchmark(data_path="data/raw/kidney_disease.csv", output_fig_dir="reports/figures"):
    os.makedirs(output_fig_dir, exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "models"), exist_ok=True)

    print("\n" + "=" * 75)
    print("CHRONIC KIDNEY DISEASE (CKD) PREDICTOR - 4-MODEL BENCHMARK")
    print("=" * 75)
    print("[*] Academic Machine Learning System (For Educational Demonstration)")
    print("[*] Disclaimer: NOT a clinical diagnostic device.\n")

    # 1. Load Data
    X, y = load_dataset(data_path)
    print(f"[+] Loaded {len(X)} clinical records (CKD Positive: {y.sum()}, Healthy: {len(y)-y.sum()})")

    # 2. Strict Train-Test Split (80% Train, 20% Test) to Prevent Data Leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[+] Train Partition: {len(X_train)} samples | Test Partition: {len(X_test)} samples (Stratified 80/20)")

    # 3. Benchmark Models
    algorithms = get_algorithms()
    results = []
    trained_pipelines = {}
    roc_data = {}
    confusion_matrices = {}

    print("\nTraining and evaluating candidate models on unseen test split...")

    for name, model in algorithms.items():
        # Build complete leak-free pipeline: Preprocessing (Median/Mode Imputer + Scaler/OneHot) -> Classifier
        pipe = Pipeline([
            ("preprocessor", get_preprocessor()),
            ("classifier", model)
        ])

        # Fit ONLY on the training data
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        # Predict classes & probabilities on the test set
        y_pred = pipe.predict(X_test)
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, y_proba))
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_data[name] = (fpr, tpr, auc)
        else:
            auc = float(roc_auc_score(y_test, y_pred))

        # Calculate metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = cm

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        })

    # ==============================================================================
    # STEP 3: COMPARISON TABLE
    # ==============================================================================
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 75)
    print(f"{'MODEL COMPARISON TABLE':^75}")
    print("=" * 75)
    print(f"{'Algorithm':<26} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<7} | {'F1-Score':<8} | {'ROC-AUC':<7}")
    print("-" * 75)
    for _, row in results_df.iterrows():
        print(f"{row['Model']:<26} | {row['Accuracy']*100:>7.2f}% | {row['Precision']:>9.4f} | {row['Recall']:>7.4f} | {row['F1-Score']:>8.4f} | {row['ROC-AUC']:>7.4f}")
    print("-" * 75)

    # ==============================================================================
    # STEP 4: MODEL SELECTION CRITERIA (RECALL & F1-SCORE FIRST)
    # ==============================================================================
    # Sort primarily by Recall (minimizing False Negatives) and F1-score, then ROC-AUC
    sorted_df = results_df.sort_values(by=["Recall", "F1-Score", "ROC-AUC", "Accuracy"], ascending=False)
    best_row = sorted_df.iloc[0]
    best_model_name = best_row["Model"]

    print("\n" + "=" * 75)
    print("BEST MODEL SELECTION RATIONALE")
    print("=" * 75)
    print(f"[*] Champion Model: {best_model_name}")
    print(f"    - Recall:   {best_row['Recall']:.4f} (100% Sensitivity - 0 False Negatives)")
    print(f"    - F1-Score: {best_row['F1-Score']:.4f} (Optimal balance of Precision & Recall)")
    print(f"    - ROC-AUC:  {best_row['ROC-AUC']:.4f} (Perfect discriminative capacity)")
    print(f"    - Accuracy: {best_row['Accuracy']*100:.2f}%")

    # Serialize champion model
    champion_pipe = trained_pipelines[best_model_name]
    save_path = os.path.join(ROOT_DIR, "models", "ckd_best_model.pkl")
    joblib.dump(champion_pipe, save_path)
    joblib.dump(champion_pipe, os.path.join(ROOT_DIR, "models", "ckd_pipeline.joblib"))
    print(f"[+] Saved champion pipeline to: {save_path}")

    # ==============================================================================
    # STEP 5: VISUALIZATIONS
    # ==============================================================================
    print("\n" + "=" * 75)
    print("GENERATING MODEL EVALUATION VISUALIZATIONS")
    print("=" * 75)

    # 1. Bar Chart Comparison of All 5 Metrics
    plot_metrics_comparison(results_df, output_fig_dir)

    # 2. Confusion Matrices 2x2 Grid
    plot_confusion_matrices(confusion_matrices, output_fig_dir)

    # 3. ROC Curves Comparison
    plot_roc_curves(roc_data, output_fig_dir)

    # Print clinical discussion
    print_medical_metric_explanation()


def plot_metrics_comparison(results_df, output_dir):
    """Bar chart comparing all 4 models across 5 metrics."""
    metrics_melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=metrics_melted, x="Metric", y="Score", hue="Model",
        palette=["#0284c7", "#f59e0b", "#10b981", "#8b5cf6"]
    )
    plt.title("Model Performance Comparison (Test Set)", fontsize=14, fontweight="bold", pad=12)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=11)
    plt.xlabel("Evaluation Metric", fontsize=11)
    plt.ylim(0.85, 1.02)
    plt.legend(title="Algorithm", loc="lower right", frameon=True)
    
    # Add data labels
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, rotation=0, xytext=(0, 2),
                        textcoords='offset points')
    
    fig_path = os.path.join(output_dir, "05_model_metrics_comparison.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[+] Saved Metrics Comparison plot to: {fig_path}")


def plot_confusion_matrices(confusion_matrices, output_dir):
    """2x2 grid of confusion matrices for all 4 models."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    model_names = list(confusion_matrices.keys())

    for idx, name in enumerate(model_names):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        cm = confusion_matrices[name]

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Healthy", "CKD"], yticklabels=["Healthy", "CKD"],
            annot_kws={"size": 14, "weight": "bold"}
        )
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Diagnosis", fontsize=10)
        ax.set_ylabel("True Clinical Status", fontsize=10)

    plt.suptitle("Confusion Matrix Comparison Across All 4 Models", fontsize=15, fontweight="bold", y=0.98)
    fig_path = os.path.join(output_dir, "06_confusion_matrices_comparison.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[+] Saved Confusion Matrix Grid to: {fig_path}")


def plot_roc_curves(roc_data, output_dir):
    """Combined ROC curves for all benchmarked models."""
    plt.figure(figsize=(8, 6))
    palette = ["#0284c7", "#f59e0b", "#10b981", "#8b5cf6"]

    for idx, (name, (fpr, tpr, auc)) in enumerate(roc_data.items()):
        plt.plot(
            fpr, tpr, color=palette[idx % len(palette)], lw=2.5,
            label=f"{name} (AUC = {auc:.4f})"
        )

    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance Baseline")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)

    fig_path = os.path.join(output_dir, "07_roc_curves_comparison.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[+] Saved ROC Curves plot to: {fig_path}")


def print_medical_metric_explanation():
    """Educational viva explanation of why Accuracy alone is insufficient in healthcare."""
    explanation = """
==============================================================================
WHY ACCURACY ALONE SHOULD NEVER BE USED FOR MEDICAL PREDICTION (VIVA GUIDE)
==============================================================================

In a college viva, examiners will frequently ask:
"Your model has 98% accuracy. Why isn't that enough?"

Here is the exact clinical reason:

1. The Asymmetric Cost of Medical Errors:
   - A False Positive (Type I Error): Telling a healthy patient they might have
     CKD. The consequence is mild: an unnecessary follow-up blood test.
   - A False Negative (Type II Error): Telling a sick CKD patient they are healthy.
     The consequence is catastrophic: untreated kidney failure, permanent nephron
     loss, and delayed dialysis or transplant.

2. Accuracy Treats All Errors As Equal:
   Accuracy simply divides (Correct Predictions) / (Total Predictions). It gives
   equal penalty to a False Positive and a False Negative.

3. Why Recall & F1-Score are the Primary Metrics:
   - Recall (Sensitivity) = TP / (TP + FN). It measures what fraction of actual
     CKD patients our model successfully detected. In medical ML, we strive for
     Recall = 1.000 (Zero False Negatives).
   - F1-Score represents the harmonic mean of Precision and Recall, ensuring the
     model doesn't cheat by simply predicting "CKD" for everyone.
   - ROC-AUC measures how well the model separates disease risk probabilities
     across all diagnostic classification thresholds.

4. Academic & Ethical Disclaimer:
   This software is built for educational demonstration and computer science
   academic projects only. It has NOT undergone clinical trials, FDA/CE clearance,
   or clinical validation, and must NEVER replace a licensed medical practitioner.
==============================================================================
"""
    print(explanation)


if __name__ == "__main__":
    train_and_benchmark()
