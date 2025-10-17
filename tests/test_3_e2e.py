import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import subprocess, time, requests

def test_service():
    """End-to-end test that launches the FastAPI app and sends a live prediction request."""
    print("\n Starting FastAPI server using Uvicorn subprocess...")
    proc = subprocess.Popen(['uvicorn', 'app:app', '--port', '8000'])

    print("Waiting for the service to start...")
    time.sleep(5)

    payload = {"x": [5.9, 3.0, 5.1, 1.8]}
    url = "http://127.0.0.1:8000/predict"
    print(f"Sending POST request to {url} with payload: {payload}")

    try:
        r = requests.post(url, json=payload)
        print(f"Response status: {r.status_code}")
        print(f"Response JSON: {r.json()}")

        assert r.status_code == 200
        assert "prediction" in r.json()

        print("✅ E2E Test passed: model prediction returned successfully!")
    except Exception as e:
        print(f"❌ E2E Test failed with exception: {e}")
        raise
    finally:
        print("Terminating FastAPI server subprocess...")
        proc.terminate()
        proc.wait()
        print("Server shut down cleanly.")