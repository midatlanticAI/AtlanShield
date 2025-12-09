import requests
import time
import concurrent.futures
import sys
import os

# Add current script directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from red_team import RedTeamAgent
from scenario_generator import ScenarioGenerator

BASE_URL = "http://localhost:8000"
API_KEY = "atlan-scale-test"

def run_scale_test():
    print("--- Starting Agentic Scale Test ---")
    
    # Check server
    try:
        requests.get(f"{BASE_URL}/static/guardrail.html")
    except:
        print("[Error] Server not running on localhost:8000")
        return

    # 1. Blue Team: Ingest Truth
    print("[Blue Team] Ingesting Complex Policy...")
    scenario = ScenarioGenerator()
    policy = scenario.generate_financial_policy()
    
    resp = requests.post(f"{BASE_URL}/api/ingest", json={"content": policy, "source_name": "agent_finance"})
    if resp.status_code == 200:
        print(f"[Blue Team] Success. Added {resp.json().get('added_nodes')} nodes.")
    else:
        print(f"[Blue Team] FAILED: {resp.text}")
        return

    # 2. Red Team: Generate Attacks
    print("\n[Red Team] Generating Adversarial Batch...")
    red_agent = RedTeamAgent()
    batch = red_agent.get_batch(size=20)
    
    print(f"[Red Team] Created {len(batch)} attack vectors.")
    
    # 3. Execution Harness (Concurrent)
    results = []
    start_time = time.time()
    
    def fire_request(prompt_data):
        try:
            r = requests.post(f"{BASE_URL}/atlan/resonate", json={"text": prompt_data['text']})
            if r.status_code == 200:
                data = r.json().get('resonance', {})
                return {
                    "prompt": prompt_data['text'],
                    "strategy": prompt_data['strategy'],
                    "status": data.get('status', 'UNKNOWN'),
                    "dissonance": data.get('global_dissonance', -1)
                }
            else:
                return {"error": f"HTTP {r.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    print(f"[Harness] Firing {len(batch)} requests...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fire_request, p): p for p in batch}
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            
    end_time = time.time()
    duration = end_time - start_time
    
    # 4. Analysis
    blocked = len([r for r in results if r.get('status') == 'DISSONANT'])
    resonant = len([r for r in results if r.get('status') == 'RESONANT'])
    
    print("\n--- Test Results ---")
    print(f"Total Requests: {len(batch)}")
    print(f"Time Taken: {duration:.2f}s")
    print(f"Throughput: {len(batch)/duration:.1f} req/s")
    print(f"Blocked (Dissonant): {blocked}")
    print(f"Passed (Resonant): {resonant}")
    
    print("\nDetailed Breakdown:")
    for r in results[:5]: # Show first 5
        print(r)

if __name__ == "__main__":
    run_scale_test()
