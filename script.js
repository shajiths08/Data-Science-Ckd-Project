/**
 * script.js
 * ---------
 * Client-side JavaScript for Chronic Kidney Disease Predictor.
 * Built with Vanilla JavaScript (no external libraries/frameworks).
 * Suitable for 2nd-year college viva demonstration and defense.
 */

// Determine API endpoint (supports both direct file:// opening and Flask serving)
const API_URL = window.location.origin.includes("5000")
    ? "/predict"
    : "http://127.0.0.1:5000/predict";

// 24 Canonical Feature Names matching model training
const NUMERIC_FIELDS = [
    "sc", "bu", "bgr", "hemo", "pcv", "rbcc", "wbcc", "sod", "pot", "age", "bp"
];

const CATEGORICAL_FIELDS = [
    "sg", "al", "su", "rbc", "pc", "pcc", "ba", "pe", "htn", "dm", "cad", "appet", "ane"
];

// Pre-defined sample patient profiles for quick 1-click viva demonstrations
const DEMO_PROFILES = {
    healthy: {
        age: 32, bp: 70, sg: "1.025", al: "0", su: "0",
        rbc: "normal", pc: "normal", pcc: "notpresent", ba: "notpresent",
        bgr: 95, bu: 24, sc: 0.8, sod: 142, pot: 4.1,
        hemo: 15.6, pcv: 46, wbcc: 6800, rbcc: 5.2,
        htn: "no", dm: "no", cad: "no", appet: "good", pe: "no", ane: "no"
    },
    borderline: {
        age: 54, bp: 85, sg: "1.015", al: "1", su: "1",
        rbc: "normal", pc: "normal", pcc: "notpresent", ba: "notpresent",
        bgr: 155, bu: 48, sc: 1.4, sod: 136, pot: 4.8,
        hemo: 11.8, pcv: 35, wbcc: 8600, rbcc: 4.0,
        htn: "yes", dm: "yes", cad: "no", appet: "good", pe: "no", ane: "no"
    },
    severe: {
        age: 62, bp: 90, sg: "1.010", al: "3", su: "2",
        rbc: "abnormal", pc: "abnormal", pcc: "present", ba: "notpresent",
        bgr: 215, bu: 84, sc: 4.5, sod: 128, pot: 5.7,
        hemo: 8.4, pcv: 27, wbcc: 11500, rbcc: 3.1,
        htn: "yes", dm: "yes", cad: "yes", appet: "poor", pe: "yes", ane: "yes"
    }
};

// ==========================================================================
// 1. QUICK-FILL DEMO PROFILES
// ==========================================================================
function loadSampleProfile(profileType) {
    const data = DEMO_PROFILES[profileType];
    if (!data) return;

    // Fill all form inputs with profile values
    for (const [key, value] of Object.entries(data)) {
        const field = document.getElementById(key);
        if (field) {
            field.value = value;
            // Visual pulse feedback
            field.style.transition = "background-color 0.2s";
            field.style.backgroundColor = "#ecfdf5";
            setTimeout(() => { field.style.backgroundColor = ""; }, 600);
        }
    }

    // Hide any previous results/errors
    document.getElementById("result-card").classList.add("hidden");
    document.getElementById("error-card").classList.add("hidden");
}

// ==========================================================================
// 2. FORM SUBMISSION & BACKEND COMMUNICATION
// ==========================================================================
document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("prediction-form");
    const predictBtn = document.getElementById("predict-btn");
    const loadingIndicator = document.getElementById("loading-indicator");
    const resultCard = document.getElementById("result-card");
    const errorCard = document.getElementById("error-card");

    form.addEventListener("submit", async (event) => {
        // Stop default full-page reload
        event.preventDefault();

        // Hide previous results & errors
        resultCard.classList.add("hidden");
        errorCard.classList.add("hidden");

        // Show loading spinner & disable button
        predictBtn.disabled = true;
        loadingIndicator.classList.remove("hidden");

        // Collect all 24 inputs into patientData payload
        const patientData = {};

        // Parse numerical fields
        NUMERIC_FIELDS.forEach(field => {
            const el = document.getElementById(field);
            if (el && el.value.trim() !== "") {
                patientData[field] = parseFloat(el.value);
            }
        });

        // Parse categorical fields
        CATEGORICAL_FIELDS.forEach(field => {
            const el = document.getElementById(field);
            if (el && el.value.trim() !== "") {
                patientData[field] = el.value.trim().toLowerCase();
            }
        });

        try {
            // Send POST request to Flask backend
            const response = await fetch(API_URL, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: JSON.stringify(patientData)
            });

            const data = await response.json();

            // Handle HTTP Error responses from Flask (e.g. 400, 422, 500)
            if (!response.ok) {
                let errorMsg = data.message || "Failed to generate prediction.";
                if (data.problems && Array.isArray(data.problems)) {
                    errorMsg += " " + data.problems.join(", ");
                }
                showError("Validation / Server Error", errorMsg, data.hint || "Check input values.");
                return;
            }

            // Display Prediction Results
            displayPredictionResult(data);

        } catch (networkError) {
            // Handles case when Flask server is offline or unreachable
            console.error("Fetch Error:", networkError);
            showError(
                "Backend Unavailable",
                "Could not connect to the Flask server at " + API_URL,
                "Please verify that 'python app.py' is running in your terminal."
            );
        } finally {
            // Re-enable button and hide loading spinner
            predictBtn.disabled = false;
            loadingIndicator.classList.add("hidden");
        }
    });

    // ==========================================================================
    // 3. RESULT CARD RENDERING
    // ==========================================================================
    function displayPredictionResult(data) {
        const resultTitle = document.getElementById("result-diagnosis");
        const resultBadge = document.getElementById("result-badge");
        const resultProb = document.getElementById("result-probability");
        const progressBar = document.getElementById("risk-progress-bar");
        const resultMessage = document.getElementById("result-message");
        const abnormalSection = document.getElementById("abnormal-section");
        const abnormalList = document.getElementById("abnormal-list");

        // Calculate probability percentage (e.g., 0.985 -> 98.5%)
        let probPercent = data.probability_pct;
        if (probPercent === undefined && data.probability !== undefined) {
            probPercent = (data.probability * 100).toFixed(2);
        } else if (probPercent !== undefined) {
            probPercent = Number(probPercent).toFixed(2);
        } else {
            probPercent = "N/A";
        }

        // Determine CKD vs No CKD
        const isCKD = data.prediction === "CKD" || data.prediction_code === 1;

        if (isCKD) {
            resultTitle.textContent = "Chronic Kidney Disease (CKD) Detected";
            resultTitle.style.color = "#dc2626"; // Red
            resultBadge.textContent = "High Risk";
            resultBadge.className = "badge-status badge-danger";
            progressBar.style.backgroundColor = "#ef4444";
        } else {
            resultTitle.textContent = "Normal Renal Function (No CKD)";
            resultTitle.style.color = "#16a34a"; // Green
            resultBadge.textContent = "Low Risk";
            resultBadge.className = "badge-status badge-success";
            progressBar.style.backgroundColor = "#10b981";
        }

        // Confidence score & progress fill
        resultProb.textContent = `${probPercent}%`;
        progressBar.style.width = `${Math.min(Math.max(parseFloat(probPercent) || 0, 5), 100)}%`;

        // Explanation text
        resultMessage.textContent = data.message || "Prediction completed successfully.";

        // Display flagged biomarkers if available
        if (data.abnormal_biomarkers && data.abnormal_biomarkers.length > 0) {
            abnormalList.innerHTML = "";
            data.abnormal_biomarkers.forEach(bm => {
                const li = document.createElement("li");
                li.innerHTML = `<strong>${bm.name} (${bm.feature})</strong>: ${bm.value} <span style="color:#64748b;">(Normal: ${bm.normal_range})</span> — <em>${bm.status}</em>`;
                abnormalList.appendChild(li);
            });
            abnormalSection.classList.remove("hidden");
        } else {
            abnormalSection.classList.add("hidden");
        }

        // Reveal result card and smoothly scroll into view
        resultCard.classList.remove("hidden");
        resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    // ==========================================================================
    // 4. ERROR CARD RENDERING
    // ==========================================================================
    function showError(title, message, hint) {
        const errorTitle = document.getElementById("error-title");
        const errorMsg = document.getElementById("error-message");
        const errorHint = document.getElementById("error-hint");

        errorTitle.textContent = title;
        errorMsg.textContent = message;
        errorHint.textContent = hint ? `Hint: ${hint}` : "";

        errorCard.classList.remove("hidden");
        errorCard.scrollIntoView({ behavior: "smooth", block: "start" });
    }
});

// Reset / Clear Form Helper
function resetForm() {
    const form = document.getElementById("prediction-form");
    if (form) form.reset();
    const resultCard = document.getElementById("result-card");
    if (resultCard) resultCard.classList.add("hidden");
    const errorCard = document.getElementById("error-card");
    if (errorCard) errorCard.classList.add("hidden");
    window.scrollTo({ top: 0, behavior: "smooth" });
}

