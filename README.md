# 🩺 Chronic Kidney Disease (CKD) Risk Predictor

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning and Clinical Decision Support system for early detection, risk assessment, and biomarker anomaly analysis of **Chronic Kidney Disease (CKD)**.

Designed with a modular architecture that is **simple to understand, easy to maintain, and ready for college viva defense**.

---

## 📌 Project Overview

Chronic Kidney Disease (CKD) is characterized by gradual loss of kidney function over time. Early stages often produce no symptoms, making it a "silent killer". This project uses machine learning trained on real patient data (UCI Machine Learning Repository) to identify patients at risk using routine laboratory and clinical tests.

### Key Highlights
- **Leak-Free ML Pipeline**: Strictly fits preprocessing transformations on the training partition only.
- **4-Model Benchmark**: Rigorous cross-validation across **Logistic Regression**, **Decision Trees**, **Random Forest**, and **Support Vector Machines (SVM)**.
- **Biomarker Anomaly Analysis**: Evaluates patient test values against normal medical reference ranges to explain why a patient is flagged.
- **Dual Web Interface**: Complete **Flask + HTML/CSS/JS** web dashboard, plus an interactive **Streamlit** dashboard.

---

## 🏗️ Project Architecture & Folder Structure

```text
Chronic-Kidney-Disease-Predictor/
│
├── data/
│   ├── raw/
│   │   └── kidney_disease.csv          # Original UCI dataset (400 patient records)
│   └── processed/
│       ├── ckd_cleaned.csv            # Cleaned, standardized dataset
│       └── sample_patients.csv        # Pre-configured test profiles (Healthy, Borderline, Severe)
│
├── notebooks/
│   └── ckd_model_development.ipynb    # Step-by-step EDA, correlations, and ML benchmarking
│
├── src/
│   ├── __init__.py                    # Package initializer
│   ├── data_preprocessing.py          # Data cleaning, median/mode imputation, and ColumnTransformer
│   ├── train.py                       # 5-Fold Stratified CV, 4-model evaluation, and serialization
│   └── predict.py                     # Inference engine & biomarker anomaly checker
│
├── models/
│   ├── ckd_best_model.pkl             # Serialized champion model pipeline (Joblib/Pickle)
│   ├── metrics_summary.json           # Evaluation scores across all benchmarked algorithms
│   └── feature_importance.json        # Gini feature importances
│
├── templates/
│   ├── index.html                     # Clinical input form with quick-fill demo presets
│   └── result.html                    # Diagnostic report card with anomaly inspector
│
├── static/
│   ├── css/
│   │   └── style.css                  # Clean, responsive medical aesthetic styling
│   └── js/
│       └── main.js                    # Client-side validation and quick-fill autofill
│
├── app.py                             # Main Flask Web Application Server
├── requirements.txt                   # Project dependencies
└── README.md                          # Comprehensive documentation
```

---

## 🔬 Clinical Input Features (24 Parameters)

Features are organized into 4 intuitive clinical panels:

1. **Renal Function & Urinalysis Chemistry**:
   - `sc` (Serum Creatinine, normal: 0.6 – 1.2 mg/dL) — *Direct filtration waste marker*
   - `bu` (Blood Urea, normal: 10 – 50 mg/dL)
   - `sg` (Specific Gravity, normal: 1.015 – 1.025)
   - `al` (Albumin in urine: 0 to 5) — *Proteinuria indicates glomerular filter damage*
   - `su` (Sugar in urine: 0 to 5)
   - `bgr` (Random Blood Glucose, normal: 70 – 140 mg/dL)
2. **Complete Blood Count & Electrolytes**:
   - `hemo` (Hemoglobin, normal: 12.0 – 17.5 g/dL) — *Low in renal anemia due to erythropoietin deficiency*
   - `pcv` (Packed Cell Volume, normal: 36 – 50%)
   - `rbcc` (Red Blood Cell Count, normal: 4.2 – 5.9 M/µL)
   - `wbcc` (White Blood Cell Count, normal: 4,000 – 11,000 cells/µL)
   - `sod` (Sodium, normal: 135 – 145 mEq/L)
   - `pot` (Potassium, normal: 3.5 – 5.0 mEq/L)
3. **Urine Microscopic & Physical Signs**:
   - `rbc` (Red Blood Cells in Urine: normal / abnormal)
   - `pc` (Pus Cells in Urine: normal / abnormal)
   - `pcc` (Pus Cell Clumps: present / not present)
   - `ba` (Bacteria: present / not present)
   - `pe` (Pedal Edema: yes / no) — *Swelling in ankles/feet from fluid retention*
4. **Vitals & Medical History**:
   - `age` (Years) & `bp` (Blood Pressure in mm/Hg)
   - `htn` (Hypertension: yes / no) & `dm` (Diabetes Mellitus: yes / no)
   - `cad` (Coronary Artery Disease: yes / no)
   - `appet` (Appetite: good / poor) & `ane` (Anemia: yes / no)

---

## 📊 Model Benchmark Results (4 Core Algorithms)

Evaluated under **Stratified 5-Fold Cross-Validation** on the training partition (80%) and tested on unseen held-out records (20%):

| Algorithm | 5-Fold CV Accuracy | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Champion)** | **98.44%** | **100.0%** | **1.000** | **1.000** | **1.000** | **1.000** |
| **Logistic Regression** | 99.06% | 98.75% | 1.000 | 0.980 | 0.9899 | 0.9987 |
| **Support Vector Machine (SVM)** | 100.0% | 98.75% | 1.000 | 0.980 | 0.9899 | 0.9993 |
| **Decision Tree** | 95.94% | 97.50% | 1.000 | 0.960 | 0.9796 | 0.9800 |

### Top Predictive Biomarkers (Feature Importance)
1. **Hemoglobin (`hemo`)** & **Packed Cell Volume (`pcv`)**
2. **Red Blood Cell Count (`rbcc`)**
3. **Serum Creatinine (`sc`)**
4. **Urine Albumin (`al`)**
5. **Hypertension (`htn`)**

---

## 💻 Quickstart & Setup

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/shajiths08/Data-Science-Ckd-Project.git
cd Data-Science-Ckd-Project

# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.\.venv\Scripts\activate
# Activate environment (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Machine Learning Models
```bash
python src/train.py
```
*Trains all 4 algorithms, computes cross-validation scores, and saves `models/ckd_best_model.pkl`.*

### 3. Run the Flask Web Application
```bash
python app.py
```
Open **`http://localhost:5000`** in your browser.

---

## 🎓 Viva & Exam Q&A Reference (College Level)

**Q1: Why is Random Forest preferred over a single Decision Tree?**  
*Answer:* A single Decision Tree tends to overfit training data easily. Random Forest builds an ensemble of multiple decorrelated trees trained on bootstrap samples and averages their votes, significantly reducing variance and boosting test accuracy.

**Q2: Why must we handle missing values with Median instead of Mean?**  
*Answer:* Clinical lab tests often contain extreme outliers (e.g., a patient with acute kidney failure might have a Serum Creatinine of 24 mg/dL while normal is 1.0 mg/dL). The Mean is skewed heavily by outliers, whereas the Median is robust and reflects typical central tendency.

**Q3: What is Data Leakage and how do we prevent it?**  
*Answer:* Data Leakage occurs when information from the test dataset is inadvertently used to train the model (e.g., fitting a scaler or imputer on the full dataset before splitting). We prevent it by using Scikit-Learn `Pipeline` and `ColumnTransformer`, which fit strictly on `X_train` and only transform `X_test`.

**Q4: Why is Recall more important than Accuracy here?**  
*Answer:* In healthcare, a **False Negative** means diagnosing a sick patient as healthy, which can result in fatal lack of treatment. Maximizing Recall minimizes False Negatives.

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
