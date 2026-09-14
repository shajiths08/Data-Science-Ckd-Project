"""
app.py
------
Streamlit Web Application for Chronic Kidney Disease (CKD) Early Risk Prediction.
Includes:
- Interactive Single-Patient Assessment with clinical biomarker range checks
- Quick-Load Clinical Presets (Healthy, Early Risk, Severe CKD)
- Batch Patient CSV Screening with report download
- Benchmark Model Comparison & Feature Importance Visualizations
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st

# Ensure root directory is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.predict import load_pipeline, predict_patient, predict_batch, DEFAULT_MODEL_PATH
from src.data_preprocessing import ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, CLINICAL_RANGES

st.set_page_config(
    page_title="CKD Risk Predictor | Clinical AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def get_cached_pipeline():
    """Loads and caches the champion ML pipeline."""
    if os.path.exists(DEFAULT_MODEL_PATH):
        return load_pipeline(DEFAULT_MODEL_PATH)
    return None


@st.cache_data
def get_cached_metrics():
    """Loads metrics summary if available."""
    metrics_path = os.path.join(ROOT_DIR, "models", "metrics_summary.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return None


@st.cache_data
def get_cached_feature_importances():
    """Loads feature importance list if available."""
    imp_path = os.path.join(ROOT_DIR, "models", "feature_importance.json")
    if os.path.exists(imp_path):
        with open(imp_path) as f:
            return json.load(f)
    return None


pipeline = get_cached_pipeline()
metrics_data = get_cached_metrics()
feature_imp_data = get_cached_feature_importances()

# Presets for Quick Selection
PRESETS = {
    "Healthy Adult": {
        "age": 32, "bp": 70.0, "sg": "1.025", "al": "0", "su": "0",
        "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
        "bgr": 95.0, "bu": 22.0, "sc": 0.8, "sod": 142.0, "pot": 4.1,
        "hemo": 15.5, "pcv": 46.0, "wbcc": 6800.0, "rbcc": 5.2,
        "htn": "no", "dm": "no", "cad": "no", "appet": "good", "pe": "no", "ane": "no"
    },
    "Borderline / Early Warning": {
        "age": 56, "bp": 85.0, "sg": "1.015", "al": "1", "su": "1",
        "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
        "bgr": 152.0, "bu": 48.0, "sc": 1.4, "sod": 136.0, "pot": 4.8,
        "hemo": 11.9, "pcv": 35.0, "wbcc": 8600.0, "rbcc": 4.0,
        "htn": "yes", "dm": "yes", "cad": "no", "appet": "good", "pe": "no", "ane": "no"
    },
    "High Risk CKD Case": {
        "age": 64, "bp": 90.0, "sg": "1.010", "al": "3", "su": "2",
        "rbc": "abnormal", "pc": "abnormal", "pcc": "present", "ba": "notpresent",
        "bgr": 210.0, "bu": 82.0, "sc": 4.1, "sod": 128.0, "pot": 5.7,
        "hemo": 8.6, "pcv": 27.0, "wbcc": 11200.0, "rbcc": 3.2,
        "htn": "yes", "dm": "yes", "cad": "yes", "appet": "poor", "pe": "yes", "ane": "yes"
    }
}

# Session State Initialization for Form Inputs
if "patient_data" not in st.session_state:
    st.session_state.patient_data = PRESETS["Healthy Adult"].copy()

# Header Section
st.markdown("""
<div class="hero-container">
    <span class="badge">Machine Learning Clinical Decision Support</span>
    <h1 class="hero-title">🩺 Chronic Kidney Disease (CKD) Risk Predictor</h1>
    <p class="hero-subtitle">
        An intelligent diagnostic platform trained on validated patient cohorts to facilitate early risk identification,
        biomarker anomaly profiling, and personalized clinical guidance.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Patient Profiles")
    st.markdown("Pre-fill the diagnostic form with representative clinical scenarios:")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🟢 Healthy", use_container_width=True):
            st.session_state.patient_data = PRESETS["Healthy Adult"].copy()
            st.rerun()
    with col_btn2:
        if st.button("🟡 Borderline", use_container_width=True):
            st.session_state.patient_data = PRESETS["Borderline / Early Warning"].copy()
            st.rerun()

    if st.button("🔴 High-Risk CKD Case", use_container_width=True):
        st.session_state.patient_data = PRESETS["High Risk CKD Case"].copy()
        st.rerun()

    st.markdown("---")
    st.header("📊 Model Metrics")
    if metrics_data:
        champ = metrics_data.get("champion_model", "Random Forest")
        champ_metrics = metrics_data.get("models", {}).get(champ, {})
        st.success(f"**Active Model:** {champ}")
        st.metric("Test Accuracy", f"{champ_metrics.get('test_accuracy', 1.0) * 100:.1f}%")
        st.metric("ROC-AUC Score", f"{champ_metrics.get('roc_auc', 1.0):.3f}")
        st.metric("F1-Score", f"{champ_metrics.get('f1_score', 1.0):.3f}")
    else:
        st.info("Train the model via `python src/train.py` to see live benchmarks.")

    st.markdown("---")
    st.caption("🔬 Built with scikit-learn & Streamlit • UCI Clinical Dataset")

# Navigation Tabs
tab_single, tab_batch, tab_analytics = st.tabs([
    "👤 Individual Patient Assessment",
    "📁 Batch Screening (CSV)",
    "📈 Model Intelligence & Benchmarks"
])

# -------------------------------------------------------------
# TAB 1: INDIVIDUAL PATIENT ASSESSMENT
# -------------------------------------------------------------
with tab_single:
    curr = st.session_state.patient_data

    st.subheader("📋 Enter Clinical & Laboratory Parameters")

    with st.form("ckd_diagnostic_form"):
        # Section 1: Kidney Function Tests
        with st.expander("🧪 1. Renal Function & Urinalysis Chemistry", expanded=True):
            r1c1, r1c2, r1c3 = st.columns(3)
            with r1c1:
                sc = st.number_input(
                    "Serum Creatinine (sc) [mg/dL]",
                    min_value=0.2, max_value=25.0, value=float(curr.get("sc", 1.0)), step=0.1,
                    help="Normal reference: 0.6 - 1.2 mg/dL. Primary marker of kidney filtration."
                )
                bu = st.number_input(
                    "Blood Urea (bu) [mg/dL]",
                    min_value=1.0, max_value=400.0, value=float(curr.get("bu", 30.0)), step=1.0,
                    help="Normal reference: 10 - 50 mg/dL. Waste product excreted by kidneys."
                )
            with r1c2:
                sg = st.selectbox(
                    "Specific Gravity (sg)",
                    options=["1.005", "1.010", "1.015", "1.020", "1.025"],
                    index=["1.005", "1.010", "1.015", "1.020", "1.025"].index(str(curr.get("sg", "1.020"))),
                    help="Urine concentration capability. Normal: 1.015 - 1.025."
                )
                al = st.selectbox(
                    "Albumin (al)",
                    options=["0", "1", "2", "3", "4", "5"],
                    index=["0", "1", "2", "3", "4", "5"].index(str(curr.get("al", "0"))),
                    help="Protein in urine. Normal: 0. Higher indicates kidney filtration barrier leakage."
                )
            with r1c3:
                su = st.selectbox(
                    "Sugar in Urine (su)",
                    options=["0", "1", "2", "3", "4", "5"],
                    index=["0", "1", "2", "3", "4", "5"].index(str(curr.get("su", "0"))),
                    help="Glucose excretion in urine. Normal: 0."
                )
                bgr = st.number_input(
                    "Random Blood Glucose (bgr) [mg/dL]",
                    min_value=40.0, max_value=500.0, value=float(curr.get("bgr", 110.0)), step=5.0,
                    help="Normal reference: 70 - 140 mg/dL."
                )

        # Section 2: Complete Blood Count & Electrolytes
        with st.expander("🩸 2. Hematology & Electrolyte Panel", expanded=True):
            r2c1, r2c2, r2c3 = st.columns(3)
            with r2c1:
                hemo = st.number_input(
                    "Hemoglobin (hemo) [g/dL]",
                    min_value=3.0, max_value=20.0, value=float(curr.get("hemo", 14.0)), step=0.1,
                    help="Normal reference: 12.0 - 17.5 g/dL. Low values signify renal anemia."
                )
                pcv = st.number_input(
                    "Packed Cell Volume (pcv) [%]",
                    min_value=10.0, max_value=60.0, value=float(curr.get("pcv", 42.0)), step=1.0,
                    help="Normal reference: 36 - 50%."
                )
            with r2c2:
                rbcc = st.number_input(
                    "Red Blood Cell Count (rbcc) [M/µL]",
                    min_value=1.5, max_value=8.0, value=float(curr.get("rbcc", 4.8)), step=0.1,
                    help="Normal reference: 4.2 - 5.9 M/µL."
                )
                wbcc = st.number_input(
                    "White Blood Cell Count (wbcc) [cells/µL]",
                    min_value=1500.0, max_value=30000.0, value=float(curr.get("wbcc", 7500.0)), step=100.0,
                    help="Normal reference: 4,000 - 11,000 cells/µL."
                )
            with r2c3:
                sod = st.number_input(
                    "Sodium (sod) [mEq/L]",
                    min_value=90.0, max_value=180.0, value=float(curr.get("sod", 140.0)), step=1.0,
                    help="Normal reference: 135 - 145 mEq/L."
                )
                pot = st.number_input(
                    "Potassium (pot) [mEq/L]",
                    min_value=1.5, max_value=10.0, value=float(curr.get("pot", 4.4)), step=0.1,
                    help="Normal reference: 3.5 - 5.0 mEq/L."
                )

        # Section 3: Urine Microscopic & Physical Findings
        with st.expander("🔬 3. Urine Microscopic Findings & Physical Signs", expanded=False):
            r3c1, r3c2, r3c3 = st.columns(3)
            with r3c1:
                rbc = st.selectbox(
                    "Red Blood Cells in Urine (rbc)",
                    options=["normal", "abnormal"],
                    index=0 if curr.get("rbc", "normal") == "normal" else 1
                )
                pc = st.selectbox(
                    "Pus Cells in Urine (pc)",
                    options=["normal", "abnormal"],
                    index=0 if curr.get("pc", "normal") == "normal" else 1
                )
            with r3c2:
                pcc = st.selectbox(
                    "Pus Cell Clumps (pcc)",
                    options=["notpresent", "present"],
                    index=0 if curr.get("pcc", "notpresent") == "notpresent" else 1
                )
                ba = st.selectbox(
                    "Bacteria (ba)",
                    options=["notpresent", "present"],
                    index=0 if curr.get("ba", "notpresent") == "notpresent" else 1
                )
            with r3c3:
                pe = st.selectbox(
                    "Pedal Edema (pe)",
                    options=["no", "yes"],
                    index=0 if curr.get("pe", "no") == "no" else 1,
                    help="Swelling in feet or ankles due to fluid retention."
                )

        # Section 4: Patient Vitals & Medical History
        with st.expander("📋 4. Demographics, Vitals & Medical History", expanded=False):
            r4c1, r4c2, r4c3 = st.columns(3)
            with r4c1:
                age = st.number_input("Age (years)", min_value=1, max_value=110, value=int(curr.get("age", 45)))
                bp = st.number_input("Blood Pressure (bp) [mm/Hg]", min_value=40.0, max_value=200.0, value=float(curr.get("bp", 80.0)), step=5.0)
            with r4c2:
                htn = st.selectbox("Hypertension (htn)", options=["no", "yes"], index=0 if curr.get("htn", "no") == "no" else 1)
                dm = st.selectbox("Diabetes Mellitus (dm)", options=["no", "yes"], index=0 if curr.get("dm", "no") == "no" else 1)
            with r4c3:
                cad = st.selectbox("Coronary Artery Disease (cad)", options=["no", "yes"], index=0 if curr.get("cad", "no") == "no" else 1)
                appet = st.selectbox("Appetite", options=["good", "poor"], index=0 if curr.get("appet", "good") == "good" else 1)
                ane = st.selectbox("Anemia", options=["no", "yes"], index=0 if curr.get("ane", "no") == "no" else 1)

        submitted = st.form_submit_button("🔍 Run Diagnostic Prediction", use_container_width=True, type="primary")

    if submitted:
        active_pipeline = get_cached_pipeline()
        if active_pipeline is None:
            st.error("⚠️ Model pipeline not trained yet. Please run `python src/train.py`.")
        else:
            patient_record = {
                "age": age, "bp": bp, "sg": sg, "al": al, "su": su,
                "rbc": rbc, "pc": pc, "pcc": pcc, "ba": ba,
                "bgr": bgr, "bu": bu, "sc": sc, "sod": sod, "pot": pot,
                "hemo": hemo, "pcv": pcv, "wbcc": wbcc, "rbcc": rbcc,
                "htn": htn, "dm": dm, "cad": cad, "appet": appet, "pe": pe, "ane": ane
            }

            result = predict_patient(patient_record, pipeline=active_pipeline)
            prob = result["ckd_probability_percent"]
            alert_lvl = result["alert_level"]

            st.markdown("---")
            st.subheader("🎯 Assessment Results")

            card_class = "danger" if alert_lvl == "danger" else ("warning" if alert_lvl == "warning" else "success")
            icon = "🚨" if alert_lvl == "danger" else ("⚠️" if alert_lvl == "warning" else "✅")

            st.markdown(f"""
            <div class="result-card {card_class}">
                <h3>{icon} {result['prediction_label']}</h3>
                <div class="metric-container">
                    <span class="metric-value">{prob}%</span>
                    <span>estimated risk probability</span>
                </div>
                <p><strong>Clinical Risk Tier:</strong> {result['risk_level']}</p>
                <p>{result['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Progress Bar for Visual Risk Representation
            st.progress(float(prob) / 100.0)

            # Biomarker Anomaly Inspector
            abnormalities = result["abnormal_biomarkers"]
            st.markdown("### 🔬 Clinical Biomarker Anomaly Analysis")
            if abnormalities:
                st.warning(f"Detected {len(abnormalities)} parameter(s) outside standard reference values:")
                anom_df = pd.DataFrame(abnormalities)
                st.dataframe(
                    anom_df[["name", "value", "unit", "normal_range", "status", "description"]].rename(
                        columns={
                            "name": "Biomarker", "value": "Observed Value", "unit": "Unit",
                            "normal_range": "Normal Reference", "status": "Status", "description": "Clinical Implication"
                        }
                    ),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.success("✅ All evaluated biomarkers fall within standard reference intervals.")

# -------------------------------------------------------------
# TAB 2: BATCH SCREENING (CSV)
# -------------------------------------------------------------
with tab_batch:
    st.subheader("📁 Batch Patient Screening via CSV")
    st.markdown("Upload a patient cohort file containing clinical parameters to obtain batch predictions.")

    sample_csv_path = os.path.join(ROOT_DIR, "data", "processed", "sample_patients.csv")
    if os.path.exists(sample_csv_path):
        with open(sample_csv_path, "rb") as f:
            st.download_button(
                "📥 Download Sample CSV Template",
                f,
                file_name="sample_ckd_patients_template.csv",
                mime="text/csv"
            )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded **{len(batch_df)}** records:")
            st.dataframe(batch_df.head(5), use_container_width=True)

            if st.button("🚀 Process Batch Predictions", type="primary"):
                active_pipeline = get_cached_pipeline()
                if active_pipeline is None:
                    st.error("Trained model pipeline not available.")
                else:
                    processed_results = predict_batch(batch_df, pipeline=active_pipeline)
                    st.success("Batch processing complete!")

                    # Quick summary metrics
                    total_p = len(processed_results)
                    ckd_p = (processed_results["Prediction"].str.contains("CKD", regex=False)).sum()
                    healthy_p = total_p - ckd_p

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Patients Screened", total_p)
                    c2.metric("Predicted CKD Cases", ckd_p, delta=f"{ckd_p/total_p*100:.1f}%", delta_color="inverse")
                    c3.metric("Predicted Healthy", healthy_p, delta=f"{healthy_p/total_p*100:.1f}%")

                    st.dataframe(processed_results, use_container_width=True)

                    # Export button
                    csv_export = processed_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Annotated Predictions CSV",
                        data=csv_export,
                        file_name="ckd_batch_predictions.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# -------------------------------------------------------------
# TAB 3: MODEL INTELLIGENCE & BENCHMARKS
# -------------------------------------------------------------
with tab_analytics:
    st.subheader("📈 Machine Learning Benchmarks & Model Explainability")

    if metrics_data:
        models_dict = metrics_data.get("models", {})
        bench_list = []
        for m_name, m_stats in models_dict.items():
            bench_list.append({
                "Model": m_name,
                "5-Fold CV Accuracy": f"{m_stats['cv_accuracy_mean']*100:.1f}% ± {m_stats['cv_accuracy_std']*100:.1f}%",
                "Test Accuracy": f"{m_stats['test_accuracy']*100:.1f}%",
                "Precision": f"{m_stats['precision']:.3f}",
                "Recall": f"{m_stats['recall']:.3f}",
                "F1-Score": f"{m_stats['f1_score']:.3f}",
                "ROC-AUC": f"{m_stats['roc_auc']:.3f}"
            })
        bench_df = pd.DataFrame(bench_list)
        st.dataframe(bench_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("🎯 Top Decisive Clinical Biomarkers (Feature Importance)")
        if feature_imp_data:
            top_features = pd.DataFrame(feature_imp_data[:12])
            st.bar_chart(top_features.set_index("feature")["importance"])
            st.caption("Feature importances derived from the ensemble model showing primary predictive drivers.")
    else:
        st.info("Execute `python src/train.py` to populate performance analytics and feature importances.")
