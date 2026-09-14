/**
 * main.js
 * -------
 * Client-side interactivity for the CKD Predictor web app.
 * Handles:
 *   1. Quick-fill preset profiles (Healthy / Borderline / Severe)
 *   2. Form submission loading feedback
 *   3. Smooth scroll progress bar animation on the result page
 */

// ============================================================
// 1. PRESET AUTOFILL
// ============================================================
async function loadPreset(profileName) {
    try {
        const response = await fetch(`/sample/${profileName}`);
        if (!response.ok) throw new Error("Network error: failed to fetch sample data.");
        const data = await response.json();

        // Populate every form field that matches a key in the response
        for (const [key, value] of Object.entries(data)) {
            const el = document.getElementById(key);
            if (!el) continue;

            // Set value
            el.value = value;

            // Flash green highlight so user can see what changed
            el.style.transition = "background-color 0.2s";
            el.style.backgroundColor = "#ecfdf5";
            setTimeout(() => { el.style.backgroundColor = ""; }, 800);
        }

        // Update the status label next to the preset buttons
        const label = document.getElementById("active-preset-label");
        if (label) {
            label.textContent = "Loaded: " + (data.profile_name || profileName);
            label.style.color = "#059669";
        }
    } catch (err) {
        console.error("Preset load error:", err);
        alert("Could not load preset profile. Please check your connection.");
    }
}

// ============================================================
// 2. FORM SUBMIT FEEDBACK
// ============================================================
document.addEventListener("DOMContentLoaded", () => {

    // Index page: show loading state on submit
    const form = document.getElementById("ckd-form");
    if (form) {
        form.addEventListener("submit", () => {
            const btn = document.getElementById("submit-btn");
            if (btn) {
                btn.disabled = true;
                btn.textContent = "Analyzing patient data...";
            }
        });
    }

    // ============================================================
    // 3. RESULT PAGE: Animate the progress bar on page load
    // ============================================================
    const progressFill = document.querySelector(".progress-bar-fill");
    if (progressFill) {
        // The width is already set inline via Jinja; just trigger a CSS repaint
        // so the transition plays when the page loads (not instantly)
        const finalWidth = progressFill.style.width;
        progressFill.style.width = "0%";
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                progressFill.style.width = finalWidth;
            });
        });
    }

    // ============================================================
    // 4. RESULT PAGE: Highlight high-risk rows in anomaly table
    // ============================================================
    document.querySelectorAll(".anomaly-table tbody tr").forEach(row => {
        const badge = row.querySelector(".badge-danger");
        if (badge) {
            row.style.backgroundColor = "#fff5f5";
        }
    });
});
