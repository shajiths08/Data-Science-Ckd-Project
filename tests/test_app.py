"""
tests/test_app.py
------------------
Automated unit and integration test suite for Chronic Kidney Disease Predictor.
Tests all routes, input validation, clinical bounds, and model inferences.

Run with:
    python tests/test_app.py
    or:
    pytest tests/test_app.py
"""

import unittest
import json
import os
import sys

# Ensure project root is on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app import app


class TestCKDPredictorApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Configure test client before running test cases."""
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_01_home_page(self):
        """Verify that the home page (GET /) renders with all required HTML elements."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Chronic Kidney Disease Predictor", html)
        self.assertIn("Predict CKD", html)
        self.assertIn("Clear Form", html)

    def test_02_health_endpoint(self):
        """Verify API health check endpoint returns 200 and indicates model is loaded."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["total_features"], 24)

    def test_03_metrics_endpoint(self):
        """Verify model benchmark metrics summary is accessible via GET /metrics."""
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("champion_model", data)
        self.assertIn("models", data)

    def test_04_sample_profiles(self):
        """Verify all preset sample profiles return correct pre-filled patient records."""
        for profile in ["healthy", "borderline", "severe"]:
            response = self.client.get(f"/sample/{profile}")
            self.assertEqual(response.status_code, 200, f"Failed for profile: {profile}")
            data = response.get_json()
            self.assertIn("sc", data)
            self.assertIn("hemo", data)

        # Non-existent profile should return 404
        bad_response = self.client.get("/sample/non_existent_profile")
        self.assertEqual(bad_response.status_code, 404)

    def test_05_predict_healthy_patient(self):
        """Verify prediction for a low-risk healthy patient."""
        healthy_patient = {
            "age": 32, "bp": 70, "sg": "1.025", "al": "0", "su": "0",
            "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
            "bgr": 95, "bu": 24, "sc": 0.8, "sod": 142, "pot": 4.1,
            "hemo": 15.6, "pcv": 46, "wbcc": 6800, "rbcc": 5.2,
            "htn": "no", "dm": "no", "cad": "no", "appet": "good", "pe": "no", "ane": "no"
        }
        response = self.client.post("/predict", json=healthy_patient)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["prediction"], "No CKD")
        self.assertEqual(data["prediction_code"], 0)
        self.assertLess(data["probability"], 0.50)  # Low probability (<50% threshold for class 0)

    def test_06_predict_severe_ckd_patient(self):
        """Verify prediction for a high-risk patient with renal damage markers."""
        severe_patient = {
            "age": 62, "bp": 90, "sg": "1.010", "al": "3", "su": "2",
            "rbc": "abnormal", "pc": "abnormal", "pcc": "present", "ba": "notpresent",
            "bgr": 215, "bu": 84, "sc": 4.5, "sod": 128, "pot": 5.7,
            "hemo": 8.4, "pcv": 27, "wbcc": 11500, "rbcc": 3.1,
            "htn": "yes", "dm": "yes", "cad": "yes", "appet": "poor", "pe": "yes", "ane": "yes"
        }
        response = self.client.post("/predict", json=severe_patient)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["prediction"], "CKD")
        self.assertEqual(data["prediction_code"], 1)
        self.assertGreaterEqual(data["probability"], 0.70)
        self.assertEqual(data["alert_level"], "danger")
        self.assertGreater(data["abnormal_count"], 0)

    def test_07_empty_payload_rejection(self):
        """Verify that sending a completely empty payload is rejected with 400 Bad Request."""
        response = self.client.post("/predict", json={})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_08_invalid_data_type(self):
        """Verify that sending non-numeric strings to numeric fields is rejected with 422."""
        bad_patient = {"age": "twenty_five", "sc": 1.0}
        response = self.client.post("/predict", json=bad_patient)
        self.assertEqual(response.status_code, 422)
        data = response.get_json()
        self.assertIn("problems", data)

    def test_09_negative_clinical_bounds_rejection(self):
        """Verify that physiologically impossible negative values are rejected with 422."""
        impossible_patient = {"age": -50, "sc": -3.0}
        response = self.client.post("/predict", json=impossible_patient)
        self.assertEqual(response.status_code, 422)
        data = response.get_json()
        self.assertTrue(any("must be between" in p for p in data["problems"]))

    def test_10_cors_headers(self):
        """Verify that CORS headers are active so browsers allow cross-origin API calls."""
        response = self.client.get("/health")
        self.assertIn("Access-Control-Allow-Origin", response.headers)

    def test_11_login_page(self):
        """Verify GET /login renders clinical login portal."""
        response = self.client.get("/login")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Clinician Portal", html)
        self.assertIn("doctor@hospital.org", html)

    def test_12_login_authentication(self):
        """Verify POST /login with demo credentials sets session."""
        response = self.client.post("/login", json={
            "email": "doctor@hospital.org",
            "password": "password123"
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("redirect", data)

    def test_13_logout(self):
        """Verify GET /logout clears session and redirects to /login."""
        response = self.client.get("/logout")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers["Location"])


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING AUTOMATED TEST SUITE FOR CHRONIC KIDNEY DISEASE PREDICTOR")
    print("=" * 70)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCKDPredictorApp)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print(f"\n[SUCCESS] ALL {result.testsRun} TESTS PASSED WITH ZERO ERRORS!")
        sys.exit(0)
    else:
        print("\n[FAILURE] SOME TESTS FAILED.")
        sys.exit(1)

