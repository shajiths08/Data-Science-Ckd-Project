/**
 * main.js
 * Handles preset demographic autofill and client-side interactions.
 */

async function loadPreset(profileName) {
    try {
        const response = await fetch(`/sample/${profileName}`);
        if (!response.ok) {
            throw new Error("Failed to fetch sample data.");
        }
        const data = await response.json();

        // Populate form fields
        for (const [key, value] of Object.entries(data)) {
            const el = document.getElementById(key);
            if (el) {
                el.value = value;
                // Trigger visual highlight
                el.style.backgroundColor = "#f0fdf4";
                setTimeout(() => {
                    el.style.backgroundColor = "";
                }, 600);
            }
        }

        // Show brief status notification
        const label = document.getElementById("active-preset-label");
        if (label) {
            label.textContent = `Loaded: ${data.profile_name}`;
            label.style.color = "#059669";
        }
    } catch (err) {
        console.error("Error loading preset:", err);
        alert("Could not load preset profile.");
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("ckd-form");
    if (form) {
        form.addEventListener("submit", () => {
            const submitBtn = document.getElementById("submit-btn");
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.textContent = "⏳ Analyzing Patient Data...";
            }
        });
    }
});
