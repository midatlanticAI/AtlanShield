import subprocess
import time
import requests
import sys
import os

def test_api():
    print("Starting Atlan API Server...")
    # Start server in background
    # Use cwd to ensure it finds static/ files
    cwd = os.path.dirname(os.path.abspath(__file__))
    server_process = subprocess.Popen(
        [sys.executable, "main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd
    )
    
    # Wait for server to start
    time.sleep(5)
    
    base_url = "http://localhost:8000"
    
    try:
        # 1. Check Health
        print("\n1. Checking Health...")
        resp = requests.get(f"{base_url}/")
        print(f"Status: {resp.status_code}, Body: {resp.json()}")
        if resp.status_code != 200:
            raise Exception("Health check failed")
            
        # 2. Baseline Analysis (Great Job)
        print("\n2. Baseline Analysis ('Great Job')...")
        payload = {"text": "Great Job"}
        resp = requests.post(f"{base_url}/analyze", json=payload)
        data = resp.json()
        print(f"Response: {data}")
        if data.get("resonance_score", 0) < 1.0:
            print("WARNING: Baseline resonance low (expected > 1.0)")
            
        # 3. Set Mood to Failure (622Hz)
        print("\n3. Setting Mood to FAILURE (622Hz)...")
        resp = requests.post(f"{base_url}/set_mood", json={"frequency": 622.0})
        print(f"Response: {resp.json()}")
        
        # 4. Sarcasm Analysis (Great Job + Failure Mood)
        print("\n4. Sarcasm Analysis ('Great Job' + Failure Mood)...")
        resp = requests.post(f"{base_url}/analyze", json=payload)
        data = resp.json()
        print(f"Response: {data}")
        
        if "Sarcastic" in data.get("interpretation", ""):
            print("PASS: API detected Sarcasm.")
        else:
            print("FAIL: API did not detect Sarcasm.")
            
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        print("\nStopping Server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    test_api()
