"""
eda_and_preprocessing.py
------------------------
Step-by-Step Data Preprocessing and Exploratory Data Analysis (EDA)
for Chronic Kidney Disease (CKD) Prediction.

Designed for a 2nd-year College Student:
- Clear, readable, beginner-friendly Python code
- Step-by-step explanations for each stage
- Zero Data Leakage (Train/Test isolation)
- Saves plots and exports reusable preprocessor objects
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Set styling for plots
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


# ==============================================================================
# STEP 1: LOAD THE DATASET
# ==============================================================================
def step_1_load_data(file_path):
    print("\n" + "=" * 70)
    print("STEP 1: LOADING THE DATASET")
    print("=" * 70)
    
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(file_path)
    print(f"[+] Successfully loaded dataset from: {file_path}")
    return df


# ==============================================================================
# STEP 2: DISPLAY SHAPE, COLUMNS, AND SAMPLE RECORDS
# ==============================================================================
def step_2_inspect_data(df):
    print("\n" + "=" * 70)
    print("STEP 2: BASIC INSPECTION OF DATA")
    print("=" * 70)
    
    # Number of rows and columns
    rows, cols = df.shape
    print(f"Number of Rows (Patients):   {rows}")
    print(f"Number of Columns (Features): {cols}")
    
    print("\nColumn Names:")
    print(list(df.columns))
    
    print("\nFirst 5 Sample Records:")
    print(df.head())


# ==============================================================================
# STEP 3: CLEAN INCONSISTENT VALUES & DATA TYPES
# ==============================================================================
def step_3_clean_raw_values(df):
    """
    Real-world clinical datasets often contain typos from manual entry:
    - '?' or '\t?' used instead of missing values (NaN)
    - Extra whitespace like '\tckd' or ' yes'
    - Numeric values accidentally stored as strings
    """
    print("\n" + "=" * 70)
    print("STEP 3: CLEANING INCONSISTENT CHARACTERS & TYPOS")
    print("=" * 70)
    
    df_clean = df.copy()

    # 1. Clean column names (strip leading/trailing whitespace)
    df_clean.columns = [col.strip() for col in df_clean.columns]

    # 2. Replace '?' and tab symbols '\t?' with standard Pandas NaN
    df_clean = df_clean.replace("?", np.nan).replace("\t?", np.nan)

    # 3. Strip extra spaces and convert text to lowercase across all text columns
    for col in df_clean.columns:
        if df_clean[col].dtype == "object":
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
            # Restore proper NaN for string 'nan'
            df_clean[col] = df_clean[col].replace(["nan", "none", "?"], np.nan)

    # 4. Fix specific typos common in the UCI CKD dataset
    if "dm" in df_clean.columns:
        # Sometimes recorded as '\tyes' or ' yes'
        df_clean["dm"] = df_clean["dm"].replace({"\tyes": "yes", " yes": "yes", "\tno": "no"})
    
    if "cad" in df_clean.columns:
        df_clean["cad"] = df_clean["cad"].replace({"\tno": "no"})

    if "class" in df_clean.columns:
        # Target column sometimes has 'ckd\t'
        df_clean["class"] = df_clean["class"].replace({"ckd\t": "ckd", "\tckd": "ckd"})

    # 5. Convert numeric columns that got misread as strings/objects to numeric float
    numeric_cols = ["age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    print("[+] Finished cleaning typos and standardized datatypes.")
    return df_clean


# ==============================================================================
# STEP 4: IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
# ==============================================================================
def step_4_identify_columns(df):
    print("\n" + "=" * 70)
    print("STEP 4: IDENTIFYING NUMERICAL & CATEGORICAL COLUMNS")
    print("=" * 70)
    
    # Define expected lists based on medical nature of features
    num_cols = ["age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"]
    cat_cols = ["sg", "al", "su", "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
    
    # Ensure they exist in the dataframe
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]
    target_col = "class"

    print(f"Numerical Features ({len(num_cols)}):")
    print(f" -> {num_cols}")
    
    print(f"\nCategorical / Discrete Features ({len(cat_cols)}):")
    print(f" -> {cat_cols}")
    
    print(f"\nTarget Variable:")
    print(f" -> '{target_col}' (ckd = 1, notckd = 0)")

    return num_cols, cat_cols, target_col


# ==============================================================================
# STEP 5: CHECK MISSING VALUES
# ==============================================================================
def step_5_check_missing_values(df):
    print("\n" + "=" * 70)
    print("STEP 5: CHECKING MISSING VALUES (NULLS)")
    print("=" * 70)
    
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_table = pd.DataFrame({
        "Missing Count": missing,
        "Percentage (%)": missing_percent.round(2)
    })
    
    # Show only features that have missing values
    missing_features = missing_table[missing_table["Missing Count"] > 0].sort_values(by="Missing Count", ascending=False)
    print(missing_features)
    
    print("\nViva Note:")
    print("- Dropping rows with missing values would discard more than half the clinical records.")
    print("- Therefore, proper statistical Imputation is required.")


# ==============================================================================
# STEP 6: EXPLORATORY DATA ANALYSIS (EDA) & PLOTTING
# ==============================================================================
def step_6_generate_eda_plots(df, num_cols, output_dir="reports/figures"):
    print("\n" + "=" * 70)
    print("STEP 6: GENERATING USEFUL EDA PLOTS")
    print("=" * 70)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Target Class Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="class", palette=["#0284c7", "#10b981"])
    plt.title("Target Distribution: CKD vs. Healthy", fontsize=12, fontweight="bold")
    plt.xlabel("Clinical Diagnosis")
    plt.ylabel("Patient Count")
    class_plot_path = os.path.join(output_dir, "01_class_distribution.png")
    plt.tight_layout()
    plt.savefig(class_plot_path, dpi=150)
    plt.close()
    print(f"[+] Saved Class Distribution plot to: {class_plot_path}")

    # 2. Serum Creatinine by Diagnosis (Key Kidney Waste Marker)
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="class", y="sc", palette=["#ef4444", "#10b981"])
    plt.title("Serum Creatinine Levels (sc) by Disease Status", fontsize=12, fontweight="bold")
    plt.xlabel("Diagnosis")
    plt.ylabel("Serum Creatinine [mg/dL]")
    plt.ylim(0, 10)  # Zoom for visibility
    sc_plot_path = os.path.join(output_dir, "02_serum_creatinine_boxplot.png")
    plt.tight_layout()
    plt.savefig(sc_plot_path, dpi=150)
    plt.close()
    print(f"[+] Saved Serum Creatinine boxplot to: {sc_plot_path}")

    # 3. Hemoglobin vs Packed Cell Volume (PCV) Scatterplot
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=df, x="hemo", y="pcv", hue="class",
        palette={"ckd": "#ef4444", "notckd": "#10b981"}, alpha=0.8, s=60
    )
    plt.title("Hemoglobin vs Packed Cell Volume (Anemia Indication)", fontsize=12, fontweight="bold")
    plt.xlabel("Hemoglobin [g/dL]")
    plt.ylabel("Packed Cell Volume (PCV) [%]")
    hemo_plot_path = os.path.join(output_dir, "03_hemo_vs_pcv_anemia.png")
    plt.tight_layout()
    plt.savefig(hemo_plot_path, dpi=150)
    plt.close()
    print(f"[+] Saved Hemoglobin vs PCV plot to: {hemo_plot_path}")

    # 4. Correlation Heatmap among Numerical Biomarkers
    plt.figure(figsize=(9, 7))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Correlation Heatmap of Clinical Biomarkers", fontsize=12, fontweight="bold")
    corr_plot_path = os.path.join(output_dir, "04_correlation_heatmap.png")
    plt.tight_layout()
    plt.savefig(corr_plot_path, dpi=150)
    plt.close()
    print(f"[+] Saved Correlation Heatmap to: {corr_plot_path}")


# ==============================================================================
# STEP 7: SEPARATE X AND y & TRAIN-TEST SPLIT (PREVENT DATA LEAKAGE)
# ==============================================================================
def step_7_split_data(df, num_cols, cat_cols, target_col):
    print("\n" + "=" * 70)
    print("STEP 7: SEPARATING FEATURES (X) & TARGET (y) + TRAIN-TEST SPLIT")
    print("=" * 70)
    
    # Filter rows where target might be missing
    clean_df = df.dropna(subset=[target_col]).copy()

    # Convert target to binary: 1 = ckd, 0 = notckd
    y = (clean_df[target_col] == "ckd").astype(int)
    X = clean_df[num_cols + cat_cols].copy()

    # Train/Test Split BEFORE scaling or imputing to prevent Data Leakage!
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Feature Matrix X shape: {X.shape}")
    print(f"Target Vector y shape:   {y.shape}")
    print(f" -> X_train: {X_train.shape} (80% training data)")
    print(f" -> X_test:  {X_test.shape} (20% test data)")

    # Class distribution in train and test
    print("\nClass Distribution:")
    print(f" - Train Split: {y_train.sum()} CKD cases ({(y_train.sum()/len(y_train))*100:.1f}%), {len(y_train)-y_train.sum()} Healthy")
    print(f" - Test Split:  {y_test.sum()} CKD cases ({(y_test.sum()/len(y_test))*100:.1f}%), {len(y_test)-y_test.sum()} Healthy")

    return X_train, X_test, y_train, y_test


# ==============================================================================
# STEP 8: HANDLE MISSING VALUES & ENCODE (FIT ON TRAIN ONLY)
# ==============================================================================
def step_8_impute_and_encode(X_train, X_test, num_cols, cat_cols):
    """
    Crucial Rule:
    Fit imputer and encoder ONLY on X_train.
    Transform X_train and X_test using the fitted statistics.
    """
    print("\n" + "=" * 70)
    print("STEP 8: IMPUTATION & CATEGORICAL ENCODING (LEAK-FREE)")
    print("=" * 70)
    
    X_train = X_train.copy()
    X_test = X_test.copy()

    # 1. Numerical Imputation using MEDIAN
    num_imputer = SimpleImputer(strategy="median")
    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])
    print("[+] Numerical missing values imputed with Median (fitted on train split).")

    # 2. Categorical Imputation using MODE (most frequent)
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols].astype(str))
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols].astype(str))
    print("[+] Categorical missing values imputed with Mode (fitted on train split).")

    # 3. Simple Binary Encoding for 2-level categorical features
    # Map 'yes' -> 1, 'no' -> 0; 'normal' -> 0, 'abnormal' -> 1; 'present' -> 1, 'notpresent' -> 0; 'good' -> 0, 'poor' -> 1
    binary_maps = {
        "rbc": {"normal": 0, "abnormal": 1},
        "pc": {"normal": 0, "abnormal": 1},
        "pcc": {"notpresent": 0, "present": 1},
        "ba": {"notpresent": 0, "present": 1},
        "htn": {"no": 0, "yes": 1},
        "dm": {"no": 0, "yes": 1},
        "cad": {"no": 0, "yes": 1},
        "appet": {"good": 0, "poor": 1},
        "pe": {"no": 0, "yes": 1},
        "ane": {"no": 0, "yes": 1}
    }

    for col, mapping in binary_maps.items():
        if col in X_train.columns:
            X_train[col] = X_train[col].map(mapping).fillna(0).astype(int)
            X_test[col] = X_test[col].map(mapping).fillna(0).astype(int)

    # Convert multi-level categoricals (sg, al, su) to numeric float
    for col in ["sg", "al", "su"]:
        if col in X_train.columns:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce").fillna(0)
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce").fillna(0)

    print("[+] Categorical variables encoded into clean numerical representations.")
    return X_train, X_test, num_imputer, cat_imputer


# ==============================================================================
# STEP 9: FEATURE SCALING (APPLY ONLY WHERE APPROPRIATE)
# ==============================================================================
def step_9_scale_features(X_train, X_test, num_cols):
    """
    Apply StandardScaler only on numerical columns:
    z = (x - mean) / std
    Fit on X_train ONLY, then transform X_train and X_test.
    """
    print("\n" + "=" * 70)
    print("STEP 9: FEATURE SCALING (STANDARD SCALER)")
    print("=" * 70)
    
    scaler = StandardScaler()
    
    # Scale only continuous numerical measurements (e.g. age, blood pressure, creatinine)
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    print("[+] Standardized numerical features (mean = 0, std = 1).")
    print(f"Sample scaled values for Serum Creatinine ('sc') in Train:")
    print(X_train_scaled["sc"].head(3))

    return X_train_scaled, X_test_scaled, scaler


# ==============================================================================
# STEP 10: PRINT FINAL SUMMARY & EXPORT CLEAN DATA
# ==============================================================================
def step_10_summary_and_export(X_train_scaled, X_test_scaled, y_train, y_test, output_dir="data/processed"):
    print("\n" + "=" * 70)
    print("STEP 10: PREPROCESSING SUMMARY & DATA EXPORT")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)

    print("Final Verification Checklist:")
    print(f"1. Training Set Features shape: {X_train_scaled.shape}")
    print(f"2. Test Set Features shape:     {X_test_scaled.shape}")
    print(f"3. Any Remaining Nulls in Train: {X_train_scaled.isnull().sum().sum()}")
    print(f"4. Any Remaining Nulls in Test:  {X_test_scaled.isnull().sum().sum()}")
    print(f"5. All columns numerical:       {all(X_train_scaled.dtypes != 'object')}")

    # Export preprocessed CSV files for modeling
    train_export = X_train_scaled.copy()
    train_export["target"] = y_train.values
    test_export = X_test_scaled.copy()
    test_export["target"] = y_test.values

    train_export.to_csv(os.path.join(output_dir, "ckd_train_preprocessed.csv"), index=False)
    test_export.to_csv(os.path.join(output_dir, "ckd_test_preprocessed.csv"), index=False)
    print(f"[+] Saved preprocessed datasets to: {output_dir}/")
    print("=" * 70)
    print("PREPROCESSING & EDA COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")


# ==============================================================================
# MAIN EXECUTION ROUTINE
# ==============================================================================
def main():
    data_file = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "kidney_disease.csv")
    if not os.path.exists(data_file):
        print(f"Error: {data_file} does not exist.")
        return

    # Step 1 & 2
    raw_df = step_1_load_data(data_file)
    step_2_inspect_data(raw_df)

    # Step 3, 4, 5
    clean_df = step_3_clean_raw_values(raw_df)
    num_cols, cat_cols, target_col = step_4_identify_columns(clean_df)
    step_5_check_missing_values(clean_df)

    # Step 6: Visualizations
    step_6_generate_eda_plots(clean_df, num_cols)

    # Step 7: Train-test split (Strict isolation)
    X_train, X_test, y_train, y_test = step_7_split_data(clean_df, num_cols, cat_cols, target_col)

    # Step 8: Imputation & Encoding
    X_train_enc, X_test_enc, num_imp, cat_imp = step_8_impute_and_encode(X_train, X_test, num_cols, cat_cols)

    # Step 9: Feature Scaling
    X_train_scaled, X_test_scaled, scaler = step_9_scale_features(X_train_enc, X_test_enc, num_cols)

    # Step 10: Final summary and export
    step_10_summary_and_export(X_train_scaled, X_test_scaled, y_train, y_test)


if __name__ == "__main__":
    main()
