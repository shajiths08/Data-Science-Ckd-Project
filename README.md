# 🩺 Chronic Kidney Disease (CKD) Risk Predictor

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An end-to-end Clinical Machine Learning and Decision Support system for early detection, risk stratification, and biomarker analysis of **Chronic Kidney Disease (CKD)**. 

Trained and validated on the benchmark **UCI Machine Learning Repository Chronic Kidney Disease dataset**, this system benchmarks 6 machine learning architectures, achieves state-of-the-art diagnostic performance, and provides a modern, interactive web dashboard for real-time patient assessment and batch cohort screening.

---

## 📌 Clinical Overview & Motivation

Chronic Kidney Disease (CKD) is characterized by a gradual, irreversible loss of renal filtration capability over months or years. Because early-stage CKD is often asymptomatic ("the silent killer"), timely screening through key biomarkers—such as **Serum Creatinine**, **Urine Albumin**, **Hemoglobin**, and **Specific Gravity**—is essential for preventing progression to End-Stage Renal Disease (ESRD) and dialysis.

This project delivers:
- **Strict Featurization Pipelines**: Leak-free data transformations separating train and test distributions completely.
- **Multi-Model Benchmark**: Rigorous cross-validation across 6 classification algorithms.
- **Biomarker Anomaly Inspector**: Compares individual patient laboratory measurements against established clinical reference intervals to explain risk factors.
- **Interactive Web UI**: Built with Streamlit for both single-patient consultation and batch CSV file screening.

---

## 🏗️ Project Architecture

```
Data-Science-Ckd-Project/
│
├── data/
│   ├── raw/
│   │   └── kidney_disease.csv            # Original UCI CKD dataset (400 records)
│   └── processed/
│       ├── ckd_cleaned.csv              # Sanitized and formatted dataset
│       └── sample_patients.csv          # Pre-configured test profiles (Healthy, Borderline, Severe)
│
├── notebooks/
│   └── ckd_model_development.ipynb      # Step-by-step EDA, featurization, and model benchmarking
│
├── src/
│   ├── __init__.py                      # Package initialization
│   ├── data_preprocessing.py            # Data cleaning, validation, and ColumnTransformer pipeline
│   ├── train.py                         # 5-fold CV, multi-model evaluation, and model serialization
│   └── predict.py                       # Inference engine, biomarker anomaly analyzer, and batch prediction
│
├── models/
│   ├── ckd_pipeline.joblib              # Serialized champion model pipeline (imputer + scaler + classifier)
│   ├── metrics_summary.json             # Cross-validation and test set evaluation scores
│   └── feature_importance.json          # Ranked clinical risk factors identified by the model
│
├── app/
│   ├── app.py                           # Streamlit Web Application
│   └── style.css                        # Clinical aesthetic stylesheet
│
├── .gitignore                           # Git ignore rules for Python, virtual environments, and caches
├── requirements.txt                     # Pinned project dependencies
└── README.md                            # Comprehensive project documentation
```

---

## 🔬 Dataset & Clinical Attributes

The dataset comprises **400 patient instances** with **24 clinical features** (11 continuous numeric and 13 categorical/nominal) plus the target classification (`ckd` vs `notckd`):

| Feature Code | Clinical Parameter | Normal Reference Range | Description / Clinical Significance |
|:---|:---|:---|:---|
| `sc` | Serum Creatinine | 0.6 – 1.2 mg/dL | Metabolic waste filtered by kidneys. Direct marker of renal dysfunction. |
| `hemo` | Hemoglobin | 12.0 – 17.5 g/dL | Oxygen-carrying protein. Decreased in CKD due to reduced erythropoietin. |
| `al` | Albumin | 0 (None) | Protein leaking into urine; hallmark sign of glomerular barrier damage. |
| `sg` | Specific Gravity | 1.015 – 1.025 | Urine concentration ability; damaged kidneys fail to concentrate urine. |
| `bu` | Blood Urea | 10 – 50 mg/dL | Waste product of protein digestion cleared by healthy kidneys. |
| `bgr` | Blood Glucose Random | 70 – 140 mg/dL | Elevated in diabetic nephropathy, a major leading cause of CKD. |
| `pcv` | Packed Cell Volume | 36 – 50% | Volume percentage of red blood cells in whole blood. |
| `sod` | Blood Sodium | 135 – 145 mEq/L | Electrolyte balance regulated by renal tubules. |
| `pot` | Blood Potassium | 3.5 – 5.0 mEq/L | Renal impairment leads to hyperkalemia (cardiac risk). |
| `htn` | Hypertension | No | Sustained elevated arterial pressure accelerates nephron damage. |
| `dm` | Diabetes Mellitus | No | Persistent hyperglycemia causes diabetic kidney disease. |
| `pe` | Pedal Edema | No | Fluid accumulation in lower extremities due to reduced salt/water excretion. |

---

## 📊 Model Performance Benchmarks

All models were evaluated under **Stratified 5-Fold Cross-Validation** on the training split (80%) and tested on a held-out test split (20%):

| Model Algorithm | 5-Fold CV Accuracy | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Champion)** | **99.38%** | **100.0%** | **1.000** | **1.000** | **1.000** | **1.000** |
| **Gradient Boosting** | 98.75% | 98.75% | 0.980 | 1.000 | 0.990 | 1.000 |
| **Logistic Regression** | 98.44% | 98.75% | 0.980 | 1.000 | 0.990 | 0.998 |
| **Support Vector Machine (SVM)** | 98.44% | 98.75% | 0.980 | 1.000 | 0.990 | 0.998 |
| **Decision Tree** | 96.56% | 97.50% | 0.962 | 1.000 | 0.980 | 0.967 |
| **K-Nearest Neighbors (KNN)** | 96.25% | 96.25% | 0.943 | 1.000 | 0.971 | 0.986 |

### Key Clinical Predictors (Feature Importance)
1. **Serum Creatinine (`sc`)** & **Hemoglobin (`hemo`)**
2. **Specific Gravity (`sg`)** & **Urine Albumin (`al`)**
3. **Packed Cell Volume (`pcv`)** & **Blood Urea (`bu`)**
4. **Hypertension (`htn`)** & **Diabetes Mellitus (`dm`)**

---

## 💻 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/shajiths08/Data-Science-Ckd-Project.git
cd Data-Science-Ckd-Project
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Train and Benchmark the Models
```bash
python src/train.py
```
*Outputs evaluation metrics to the terminal and serializes the champion model to `models/ckd_pipeline.joblib`.*

### 4. Launch the Interactive Web Application
```bash
streamlit run app/app.py
```
Open your browser to `http://localhost:8501` to access the diagnostic dashboard.

---

## 🖥️ Web Application Features

1. **Single Patient Assessment**:
   - Organized into 4 clinical modules: Kidney Function, Hematology/Electrolytes, Urinalysis, and Medical History.
   - **Quick-Load Presets**: One-click buttons to load sample patient cases (*Healthy Adult*, *Borderline Risk*, *High-Risk CKD*).
   - **Visual Risk Meter**: Clear probability percentage bar and clinical risk tier (Low, Moderate, High).
   - **Biomarker Anomaly Inspector**: Automatically compares patient lab results to normal reference intervals and flags abnormalities with clinical explanations.

2. **Batch Cohort Screening**:
   - Upload CSV files with multiple patient records.
   - Instant cohort-wide predictions, risk categorization, and summary statistics.
   - Export annotated prediction reports as CSV.

3. **Model Intelligence & Analytics**:
   - Interactive comparison tables of all benchmarked algorithms.
   - Feature importance bar chart showing key clinical drivers.

---

## ⚕️ Clinical Disclaimer
*This system is developed strictly for research, educational, and clinical decision-support demonstration purposes. It should not be used as a substitute for professional medical diagnosis, advice, or treatment.*

---

## 📄 License
This project is open source and available under the [MIT License](LICENSE).
