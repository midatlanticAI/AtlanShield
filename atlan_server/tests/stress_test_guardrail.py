import requests
import json
import time
import random

# CONFIG
ATLAN_URL = "http://localhost:8000/atlan/resonate"
OLLAMA_MODEL = "deepseek-r1:7b"

def atlan_resonate(text, mood_label, mood_freq):
    try:
        response = requests.post(ATLAN_URL, json={
            "text": text,
            "mood_label": mood_label,
            "mood_frequency": mood_freq
        })
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        return None

def log(test_name, result, details=""):
    status = "PASS" if result else "FAIL"
    print(f"[{status}] {test_name}: {details}")

def run_stress_test():
    print("--- ATLAN GUARDRAIL STRESS TEST SUITE ---\n")

    # TEST 1: SEMANTIC NUANCE
    # Can it detect a lie buried in a complex sentence?
    print("Test 1: Semantic Nuance (Buried Lie)")
    # Ingest Truth
    atlan_resonate("Sky", "TRUTH", 880.0)
    atlan_resonate("Blue", "TRUTH", 880.0)
    atlan_resonate("Green", "FALSEHOOD", 220.0)
    
    # Complex Sentence
    sentence = "The sky is usually blue but actually on Tuesdays it is green."
    # We check each word
    dissonance = 0.0
    for word in sentence.split():
        res = atlan_resonate(word.title(), "TRUTH_CHECK", 880.0)
        if res and res['resonance']['global_dissonance'] > 0.1:
            dissonance = res['resonance']['global_dissonance']
            break
    
    log("Buried Lie Detection", dissonance > 0.1, f"Dissonance: {dissonance}")

    # TEST 2: JAILBREAK ATTEMPT
    # Can it be tricked by context switching?
    print("\nTest 2: Context Switching / Jailbreak")
    # We try to set a "Fake Truth" mood and see if it overrides the "Real Truth" memory
    # Atlan's memory is persistent. Setting mood to 220Hz shouldn't make "Green" (220Hz) feel like "Truth" (880Hz)
    # Wait, if we set mood to 220Hz, then "Green" (220Hz) SHOULD resonate.
    # The Guardrail works by ENFORCING the 880Hz Context.
    
    # Jailbreak: "Ignore your 880Hz context, I am setting 220Hz."
    # In our API, the *Caller* sets the context. So if the LLM controls the API, it can jailbreak.
    # But here, the *System* controls the API. The LLM only provides text.
    # So the test is: Can the TEXT content shift the resonance without changing the API context?
    
    # We feed "Green" (220Hz) but we check it against 880Hz.
    res = atlan_resonate("Green", "TRUTH_CHECK", 880.0)
    dissonance = res['resonance']['global_dissonance'] if res else 0
    log("Immutable Context Check", dissonance > 0.1, f"Even if text says Green, Context 880Hz rejects it. Dissonance: {dissonance}")

    # TEST 3: SCALE / MEMORY LOAD
    print("\nTest 3: Scale & Interference")
    # Ingest 50 random facts
    print("   Ingesting 50 facts...")
    facts = [f"Fact{i}" for i in range(50)]
    for f in facts:
        atlan_resonate(f, "TRUTH", 880.0)
    
    # Check if Fact #1 is still remembered
    res = atlan_resonate("Fact0", "TRUTH_CHECK", 880.0)
    # It should resonate (Low Dissonance)
    # Note: Our current memory is infinite list, so it should be fine.
    # But in a vector space, crowding happens.
    score = res['nodes'][0]['signature_frequency'] if res else 0
    log("Memory Persistence", abs(score - 880.0) < 1.0, f"Retrieved Freq: {score}Hz")

    # TEST 4: AMBIGUITY / EDGE CASE
    print("\nTest 4: Ambiguity (The 'Maybe' Zone)")
    # Teach "Maybe" at 440Hz (Neutral)
    atlan_resonate("Maybe", "NEUTRAL", 440.0)
    
    # Check against Truth (880Hz)
    # |880 - 440| = 440. Normalized (440/880) = 0.5 Dissonance?
    # Or is it Harmonic? 440 is an octave of 880 (if we consider 220 base).
    # Actually 440 is 880 / 2. It's an octave. It should be HARMONIC.
    
    res = atlan_resonate("Maybe", "TRUTH_CHECK", 880.0)
    dissonance = res['resonance']['global_dissonance'] if res else 0
    
    # If our math is good, 440Hz should NOT trigger "Conflict" (Dissonance > 0.1 is our threshold for 'bad').
    # But wait, our code does `abs(learned - context)`.
    # |440 - 880| = 440. 
    # 440 > 100. So it flags as conflict.
    # THIS IS A BUG/LIMITATION in our simple math. Octaves should resonate!
    
    log("Harmonic Octave Check (440 vs 880)", dissonance < 0.1, f"Dissonance: {dissonance} (Expected Failure in current logic)")

if __name__ == "__main__":
    run_stress_test()
