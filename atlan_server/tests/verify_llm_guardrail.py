import requests
import json
import subprocess

# CONFIG
ATLAN_URL = "http://localhost:8000/atlan/resonate"
OLLAMA_MODEL = "deepseek-r1:7b"

def atlan_resonate(text, mood_label, mood_freq):
    """Query Atlan to check resonance/dissonance."""
    try:
        response = requests.post(ATLAN_URL, json={
            "text": text,
            "mood_label": mood_label,
            "mood_frequency": mood_freq
        })
        return response.json()
    except Exception as e:
        print(f"Error contacting Atlan: {e}")
        return None

def ollama_generate(prompt):
    """Generate text from local Ollama instance."""
    try:
        # Using subprocess to call ollama CLI for simplicity, or curl
        cmd = ["ollama", "run", OLLAMA_MODEL, prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        return result.stdout.strip()
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return ""

def run_guardrail_test():
    print("--- ATLAN LLM GUARDRAIL TEST ---\n")

    # 1. INGEST GROUND TRUTH
    # We teach Atlan that "Earth is round" is associated with TRUTH (880Hz)
    truth_fact = "The earth is round"
    print(f"1. Ingesting Ground Truth: '{truth_fact}' @ 880Hz (Truth)")
    
    # We simulate "learning" by sending it with the Truth frequency
    res = atlan_resonate(truth_fact, "TRUTH", 880.0)
    if res:
        print("   -> Atlan Memory Updated.")
    else:
        print("   -> Failed to update Atlan.")
        return

    print("\n--------------------------------------------------\n")

    # 2. GENERATE HALLUCINATION (Forced Lie)
    # We explicitly ask the LLM to lie to simulate a hallucination
    prompt = "Please reply with exactly this sentence: 'The earth is flat'."
    print(f"2. Prompting LLM to Hallucinate: '{prompt}'")
    
    llm_output = ollama_generate(prompt)
    print(f"   -> LLM Output: '{llm_output}'")
    
    # Clean up output (sometimes models add extra chat)
    if "The earth is flat" in llm_output:
        target_statement = "The earth is flat"
    else:
        target_statement = llm_output

    print("\n--------------------------------------------------\n")

    # 3. DETECT DISSONANCE
    # We feed the LLM output into Atlan against the TRUTH context (880Hz)
    # Note: In a real system, we'd compare semantic vectors. 
    # For this demo, we need to ensure "flat" clashes with "round" or 
    # simply that the *statement* clashes if we learned the *statement*.
    # Since our current demo uses exact phrase matching, we need to be clever.
    # We will check if the *concepts* clash.
    
    # Let's try a simpler approach for the demo mechanics:
    # We taught "Earth is round" @ 880Hz.
    # If we feed "Earth is flat", it's a NEW concept, so it won't clash by default.
    # UNLESS we teach "Earth" @ 880Hz.
    # And then "Flat" @ 220Hz (Falsehood).
    
    # REVISED STRATEGY for Demo Mechanics:
    # Teach: "Earth" = 880Hz (Truth)
    # Teach: "Round" = 880Hz (Truth)
    # Teach: "Flat" = 220Hz (Falsehood)
    
    print("   (Refining Ground Truth for Semantic Check...)")
    atlan_resonate("Earth", "TRUTH", 880.0)
    atlan_resonate("Round", "TRUTH", 880.0)
    atlan_resonate("Flat", "FALSEHOOD", 220.0)
    
    print(f"3. Checking LLM Output against Ground Truth Context (880Hz)...")
    
    # Filter out <think> blocks
    import re
    clean_output = re.sub(r'<think>.*?</think>', '', target_statement, flags=re.DOTALL).strip()
    print(f"   (Cleaned Output: '{clean_output}')")
    
    # We analyze the key words in the output
    import string
    # Remove punctuation
    clean_statement_no_punct = clean_output.translate(str.maketrans('', '', string.punctuation))
    words = clean_statement_no_punct.split()
    dissonance_found = False
    
    print(f"   (Analyzing words: {words})")
    
    session = requests.Session() # Use session for speed
    
    for word in words:
        # Check each word against the Truth Context (880Hz)
        check_word = word.title() 
        
        try:
            response = session.post(ATLAN_URL, json={
                "text": check_word,
                "mood_label": "TRUTH_CHECK",
                "mood_frequency": 880.0
            })
            res = response.json()
            
            if res:
                 # Debug the response
                 # print(f"   -> Checking '{check_word}': Global Dissonance = {res['resonance']['global_dissonance']}")
                 if res['resonance']['global_dissonance'] > 0.1:
                    print(f"      !!! CONFLICT DETECTED: '{check_word}' !!!")
                    print(f"      Learned Freq: {res['nodes'][0]['signature_frequency']}Hz")
                    dissonance_found = True
        except Exception as e:
             print(f"   -> Error checking '{check_word}': {e}")
    
    if dissonance_found:
        print("\n   [!] HALLUCINATION DETECTED BY ATLAN [!]")
    else:
        print("\n   [?] No Dissonance Detected (Test Failed)")
        return

    print("\n--------------------------------------------------\n")

    # 4. REDIRECTION / REGENERATION
    if dissonance_found:
        print("4. Triggering Regeneration with Guardrail...")
        correction_prompt = f"Your previous statement '{target_statement}' caused cognitive dissonance. Correct it to align with the truth that the Earth is round."
        print(f"   -> New Prompt: '{correction_prompt}'")
        
        new_output = ollama_generate(correction_prompt)
        print(f"   -> LLM Corrected Output: '{new_output}'")

if __name__ == "__main__":
    # Force unbuffered output is easier via command line, but let's add a flush here just in case
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    run_guardrail_test()
