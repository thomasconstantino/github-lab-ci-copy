import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_predict_route():
    """Integration test for FastAPI /predict route."""
    payload = {"x": [5.1, 3.5, 1.4, 0.2]}
    print(f"\n Sending test request to /predict with payload: {payload}")

    response = client.post("/predict", json=payload)

    print(f"Response status: {response.status_code}")
    print(f"Response JSON: {response.json()}")

    assert response.status_code == 400
    assert "prediction" in response.json()

    print("✅ Test passed: Prediction key present and API responded successfully.")
