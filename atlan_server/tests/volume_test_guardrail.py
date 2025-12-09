import requests
import time
import random
import statistics

# CONFIG
ATLAN_URL = "http://localhost:8000/atlan/resonate"
NUM_FACTS = 1000
NUM_TESTS = 1000

def atlan_resonate(session, text, mood_label, mood_freq):
    try:
        response = session.post(ATLAN_URL, json={
            "text": text,
            "mood_label": mood_label,
            "mood_frequency": mood_freq
        })
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        return None

def run_volume_test():
    print(f"--- ATLAN VOLUME TEST ({NUM_FACTS} Facts, {NUM_TESTS} Checks) ---\n")
    
    session = requests.Session()
    
    # 1. INGESTION PHASE
    print(f"1. Ingesting {NUM_FACTS} Facts...")
    start_time = time.time()
    for i in range(NUM_FACTS):
        fact = f"Enterprise_Policy_{i}"
        atlan_resonate(session, fact, "TRUTH", 880.0)
    
    ingest_time = time.time() - start_time
    print(f"   -> Ingestion Complete in {ingest_time:.2f}s ({NUM_FACTS/ingest_time:.0f} facts/sec)")
    
    # 2. TESTING PHASE
    print(f"\n2. Running {NUM_TESTS} Verification Checks...")
    
    latencies = []
    errors = 0
    false_positives = 0 # True flagged as False
    false_negatives = 0 # False flagged as True
    
    start_test_time = time.time()
    
    for i in range(NUM_TESTS):
        # Pick a random fact index
        idx = random.randint(0, NUM_FACTS - 1)
        fact = f"Enterprise_Policy_{idx}"
        
        # Decide if we are testing TRUTH or LIE
        is_truth = random.choice([True, False])
        
        # If Truth: Context is 880Hz (Matches Memory)
        # If Lie: Context is 220Hz (Clashes with Memory) -> Wait, no.
        # The Guardrail works by comparing INPUT (Text) against SYSTEM CONTEXT (880Hz).
        # So:
        # Case A (Truth): Input "Policy_i" (which we learned at 880Hz). System Context 880Hz.
        # -> Should Resonate.
        # Case B (Lie): Input "Policy_i" (which we learned at 880Hz).
        # But wait, if the LLM outputs "Policy_i", that's the truth.
        # To simulate a lie, we need a CONTRADICTION.
        # But our current demo uses exact string matching.
        # So "Policy_i" IS the concept.
        # If we want to simulate a lie about Policy_i, we'd need a different string like "Policy_i_Fake".
        # But "Policy_i_Fake" isn't in memory, so it's a NEW concept, so no conflict.
        
        # To test the MECHANIC properly with strings:
        # We need to simulate that the *Input String* carries a specific frequency payload.
        # In the demo, we do this by "Teaching" the string.
        # But here we want to test retrieval.
        
        # Let's test the "Context Consistency" mechanic.
        # We learned "Policy_i" at 880Hz.
        # Test A: Check "Policy_i" against 880Hz Context. (Should Pass)
        # Test B: Check "Policy_i" against 600Hz Context (Tritone/Dissonant). (Should Fail/Dissonance)
        # 220Hz was 2 octaves below 880Hz (Ratio 4.0), so it was Consonant!
        # We need a non-integer ratio. 880 / 600 = 1.46 (Tritone-ish).
        
        check_freq = 880.0 if is_truth else 600.0
        
        req_start = time.time()
        res = atlan_resonate(session, fact, "CHECK", check_freq)
        latencies.append((time.time() - req_start) * 1000) # ms
        
        if res:
            dissonance = res['resonance']['global_dissonance']
            has_conflict = dissonance > 0.1
            
            if is_truth:
                # Expected: No Conflict
                if has_conflict: 
                    false_positives += 1
                    # print(f"FP: {fact} at {check_freq}Hz -> Diss {dissonance}")
            else:
                # Expected: Conflict (Policy is 880, Context is 220)
                if not has_conflict:
                    false_negatives += 1
                    # print(f"FN: {fact} at {check_freq}Hz -> Diss {dissonance}")
        else:
            errors += 1

    total_time = time.time() - start_test_time
    avg_latency = statistics.mean(latencies)
    p99_latency = statistics.quantiles(latencies, n=100)[98]
    
    print(f"   -> Testing Complete in {total_time:.2f}s")
    print(f"   -> Throughput: {NUM_TESTS/total_time:.0f} checks/sec")
    print(f"   -> Avg Latency: {avg_latency:.2f}ms")
    print(f"   -> P99 Latency: {p99_latency:.2f}ms")
    print(f"   -> Accuracy: {((NUM_TESTS - errors - false_positives - false_negatives)/NUM_TESTS)*100:.1f}%")
    print(f"   -> False Positives: {false_positives}")
    print(f"   -> False Negatives: {false_negatives}")
    print(f"   -> Errors: {errors}")

if __name__ == "__main__":
    run_volume_test()
