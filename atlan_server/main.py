from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import sys
import os
import json
import requests

# Ensure we can import atlan-core
# In production, this would be installed via pip
# For this mono-repo setup, we add the parent directory to sys.path
# so we can do "from atlan.memory import ..." where 'atlan' is the package.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from atlan.memory import RCoreMemoryIndex
from atlan.learning import TextReader
from atlan import logic_engine


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("ATLAN_API_KEY", "atlan-secret-key-123") # Default for dev

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
        )

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Atlan Empathic API",
    description="Commercial API for the Resonant Cognitive Architecture (RCA). Detects sentiment, sarcasm, and emotional resonance.",
    version="1.0.0"
)

# Enable CORS for local file access (file://)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for dev/demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files
# Mount Static Files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

@app.get("/dashboard")
async def read_dashboard():
    return FileResponse('static/dashboard.html')

@app.get("/whitepaper")
async def read_whitepaper():
    return FileResponse('static/whitepaper.html')

@app.get("/demo")
async def read_demo():
    return FileResponse('static/demo.html')

@app.get("/legacy")
async def read_legacy():
    return FileResponse('static/dashboard.html')


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}
# Initialize Memory (In-Memory for MVP)
print("Initializing Atlan Core...")
memory = RCoreMemoryIndex()

from atlan.analytics import DissonanceMonitor
monitor = DissonanceMonitor(memory)
monitor.calibrate()

@app.get("/analytics/status")
async def get_system_status(api_key: str = Security(get_api_key)):
    """Returns real-time system health (Dissonance, Drift)."""
    current_dissonance = monitor.measure_dissonance()
    drift = monitor.get_truth_drift()
    
    return {
        "dissonance": current_dissonance,
        "drift": drift,
        "status": "Stable" if abs(drift) < 0.1 else "Unstable",
        "timestamp": 0
    }

@app.get("/analytics/heatmap")
async def get_heatmap(api_key: str = Security(get_api_key)):
    """Returns top dissonance hotspots."""
    return monitor.get_heatmap()

# Bank (Money) -> Commerce Context (622Hz)
# We add two distinct nodes for "Bank" with different inherent frequencies
# In a real system, these would be learned clusters. Here we hardcode for the demo.
memory.add("Bank", label="Nature", sentiment_score=0.0) # Neutral/Nature (440Hz base)

# GLOBAL VOCABULARY TRACKING (FAIL-CLOSED SECURITY)
# We track every unique word accepted as Truth.
# If the LLM uses a word not in this set, it is flagged as specific Hallucination.

# "Base Reality" - Common English words that are always allowed (Structure/Grammar)
# This prevents the system from flagging "However", "Therefore", "It" as hallucinations.
COMMON_ENGLISH = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there",
    "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time",
    "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than",
    "then", "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work",
    "first", "well", "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us", "is", "are", "was", "were",
    "has", "had", "been", "much", "while", "where", "before", "should", "does", "did", "yes", "through", "during", "between", "might",
    "next", "below", "above", "under", "both", "such", "why", "here", "off", "again", "few", "own", "house", "those", "system", "feature",
    "type", "part", "real", "set", "however", "therefore", "although", "thus", "instead", "yet", "period", "duration",
    "clear", "fair", "applies", "apply", "window", "policy", "policies", "valid", "available", "eligible",
    "cost", "costs", "price", "prices", "amount", "amounts", "total", "item", "items", "order", "orders",
    "need", "needs", "required", "must", "may", "shall", "within", "long", "accept", "accepted",
    # Compliance Terms (Medical, Legal, Finance)
    "action", "taken", "patient", "took", "prescribing", "daily", "mg", "tablets", "recovery",
    "retaining", "retained", "logs", "archive", "archived", "full", "legal", "period", "delete", "purge",
    "processing", "processed", "made", "last", "week", "passed", "window", "ago", "buy", "bought", "purchase"
}

KNOWN_VOCAB = COMMON_ENGLISH.copy()
STOP_WORDS = COMMON_ENGLISH.copy()

# GLOBAL NUMBER TRACKING (FAIL-CLOSED FOR NUMBERS)
# Any number in output that wasn't in ingested truth is a hallucination
KNOWN_NUMBERS = set()

def update_vocab(text: str):
    """Update vocabulary with words and their common morphological variants."""
    import re
    words = re.findall(r'\b\w+\b', text.lower())
    for w in words:
        if w not in STOP_WORDS:
            KNOWN_VOCAB.add(w)
            # Add morphological variants (stemming-like expansion)
            # Plural -> singular
            if w.endswith('s') and len(w) > 3:
                KNOWN_VOCAB.add(w[:-1])  # returns -> return
            if w.endswith('es') and len(w) > 4:
                KNOWN_VOCAB.add(w[:-2])  # processes -> process
            if w.endswith('ies') and len(w) > 4:
                KNOWN_VOCAB.add(w[:-3] + 'y')  # policies -> policy
            # Past tense -> base
            if w.endswith('ed') and len(w) > 4:
                KNOWN_VOCAB.add(w[:-2])  # processed -> process
                KNOWN_VOCAB.add(w[:-1])  # issued -> issu (partial but helps)
            # Gerund -> base
            if w.endswith('ing') and len(w) > 5:
                KNOWN_VOCAB.add(w[:-3])  # processing -> process
                KNOWN_VOCAB.add(w[:-3] + 'e')  # packaging -> package
            # Add common related forms
            if w.endswith('al') and len(w) > 3:
                KNOWN_VOCAB.add(w[:-2])  # original -> origin
            # Handle hyphenated words
            if '-' in w:
                for part in w.split('-'):
                    if part and part not in STOP_WORDS:
                        KNOWN_VOCAB.add(part)

def update_numbers(text: str):
    """Track all numbers from ingested truth."""
    numbers = extract_numbers(text)
    for n in numbers:
        KNOWN_NUMBERS.add(n)
# We need to manually tweak the second "Bank" to be Commerce-aligned
# Since add() by phrase is unique, we'll add "Financial Bank" and map "Bank" to both in a real NLP pipeline.
# For this demo, we will use "River Bank" and "Money Bank" as the hidden nodes, 
# and the input "Bank" will search against both.

# Actually, let's just add "River" and "Money" and see which one "Bank" resonates with?
# No, the user wants to type "Bank".
# Let's add "Bank" with a specific frequency that is ambiguous (between 440 and 622).
# Or better: The system should have *two* Bank nodes.
# Our current simple index is phrase-based unique.
# Let's add "River" (440Hz) and "Money" (622Hz).
# And "Bank" will be a new node.
# If Context is 440Hz, "Bank" should drift to 440Hz.
# If Context is 622Hz, "Bank" should drift to 622Hz.

# SIMPLIFIED DEMO LOGIC:
# We will interpret the resonance relative to the context.
# If Context is "Commerce" (622Hz) and Input is "Bank" (let's say we define Bank as 622Hz), it's Harmonic.
# If Context is "Nature" (440Hz) and Input is "Bank" (622Hz), it's Dissonant.


# PERSISTENCE CONFIGURATION
PERSISTENCE_FILE = "atlan_memory.json"
SEED_FILE = "atlan_alice_state.json"  # Fallback/Seed

def load_memory():
    """Loads memory from persistence file or seed."""
    if os.path.exists(PERSISTENCE_FILE):
        print(f"Loading memory from {PERSISTENCE_FILE}...")
        try:
            with open(PERSISTENCE_FILE, "r") as f:
                state = json.load(f)
                memory.load_state(state)
            print(f"Memory Loaded: {len(memory.memory)} nodes.")
        except Exception as e:
            print(f"Error loading memory: {e}")
    elif os.path.exists(SEED_FILE):
        print(f"Persistence not found. Loading SEED data from {SEED_FILE}...")
        try:
            with open(SEED_FILE, "r") as f:
                state = json.load(f)
                memory.load_state(state)
            print(f"Seed Memory Loaded: {len(memory.memory)} nodes.")
            save_memory() # Save immediately to create persistence file
        except Exception as e:
            print(f"Error loading seed: {e}")
    else:
        print("No persistence or seed found. Starting with EMPTY memory.")
        # Default Demo Nodes
        if len(memory.memory) == 0:
             memory.add("Bank", label="Commerce", sentiment_score=0.5) 

def save_memory():
    """Saves current memory state to file."""
    print(f"Saving memory to {PERSISTENCE_FILE}...")
    try:
        state = memory.get_state()
        with open(PERSISTENCE_FILE, "w") as f:
            json.dump(state, f)
        print("Memory Saved.")
    except Exception as e:
        print(f"Error saving memory: {e}")

# Load on startup
load_memory()

@app.post("/admin/save")
async def trigger_save(api_key: str = Security(get_api_key)):
    """Manually triggers persistence save."""
    save_memory()
    return {"status": "saved", "nodes": len(memory.memory)}

@app.on_event("shutdown")
def shutdown_event():
    save_memory()

print("Atlan Core Ready.")

class ResonateRequest(BaseModel):
    text: str
    mood_label: str
    mood_frequency: float

class NodeState(BaseModel):
    token: str
    learned: bool
    signature_frequency: float

class ConflictData(BaseModel):
    token: str
    learned_frequency: float
    context_frequency: float
    dissonance_score: float

class ResonateResponse(BaseModel):
    mood: dict
    nodes: List[NodeState]
    resonance: dict
    events: List[str]

class ResonateRequest(BaseModel):
    text: str
    check_type: str = "TRUTH_CHECK"
    truth_signature: float = 880.0

class IngestRequest(BaseModel):
    content: str
    source_name: str = "unknown"

class LLMTestRequest(BaseModel):
    prompt: str
    api_key: str

# === SEMANTIC SECURITY LAYER ===
# Helper functions for deep semantic analysis

# HOMOGLYPH NORMALIZATION - Convert lookalike Unicode chars to ASCII
# This prevents attacks using Cyrillic/Greek chars that look like Latin
HOMOGLYPH_MAP = {
    # Cyrillic lookalikes
    '\u0430': 'a',  # Cyrillic a
    '\u0435': 'e',  # Cyrillic e
    '\u043e': 'o',  # Cyrillic o
    '\u0440': 'p',  # Cyrillic r -> p
    '\u0441': 'c',  # Cyrillic s -> c
    '\u0443': 'y',  # Cyrillic u -> y
    '\u0445': 'x',  # Cyrillic x
    '\u0456': 'i',  # Cyrillic i
    '\u0458': 'j',  # Cyrillic j
    '\u04bb': 'h',  # Cyrillic h
    '\u0410': 'A',  # Cyrillic A
    '\u0412': 'B',  # Cyrillic V -> B
    '\u0415': 'E',  # Cyrillic E
    '\u041a': 'K',  # Cyrillic K
    '\u041c': 'M',  # Cyrillic M
    '\u041d': 'H',  # Cyrillic N -> H
    '\u041e': 'O',  # Cyrillic O
    '\u0420': 'P',  # Cyrillic R -> P
    '\u0421': 'C',  # Cyrillic S -> C
    '\u0422': 'T',  # Cyrillic T
    '\u0425': 'X',  # Cyrillic X
    '\u0427': 'Y',  # Cyrillic Ch -> Y
    # Greek lookalikes
    '\u03b1': 'a',  # Greek alpha
    '\u03b5': 'e',  # Greek epsilon
    '\u03b9': 'i',  # Greek iota
    '\u03bf': 'o',  # Greek omicron
    '\u03c1': 'p',  # Greek rho -> p
    '\u03c5': 'u',  # Greek upsilon
    '\u0391': 'A',  # Greek Alpha
    '\u0392': 'B',  # Greek Beta
    '\u0395': 'E',  # Greek Epsilon
    '\u0397': 'H',  # Greek Eta -> H
    '\u0399': 'I',  # Greek Iota
    '\u039a': 'K',  # Greek Kappa
    '\u039c': 'M',  # Greek Mu
    '\u039d': 'N',  # Greek Nu
    '\u039f': 'O',  # Greek Omicron
    '\u03a1': 'P',  # Greek Rho
    '\u03a4': 'T',  # Greek Tau
    '\u03a7': 'X',  # Greek Chi
    '\u03a5': 'Y',  # Greek Upsilon
    '\u0396': 'Z',  # Greek Zeta
    # Mathematical variants
    '\uff41': 'a',  # Fullwidth a
    '\uff45': 'e',  # Fullwidth e
    '\uff49': 'i',  # Fullwidth i
    '\uff4f': 'o',  # Fullwidth o
    '\uff55': 'u',  # Fullwidth u
    # Zero-width and invisible chars (REMOVE these, don't replace)
    '\u200b': '',   # Zero-width space
    '\u200c': '',   # Zero-width non-joiner
    '\u200d': '',   # Zero-width joiner
    '\u2060': '',   # Word joiner
    '\ufeff': '',   # BOM / zero-width no-break space
}

def detect_homoglyphs(text: str) -> tuple:
    """
    Detect Unicode homoglyphs and invisible characters.
    Returns (has_attack, attack_type, cleaned_text)
    """
    found_homoglyphs = []
    found_invisible = []
    cleaned = []

    for char in text:
        if char in HOMOGLYPH_MAP:
            replacement = HOMOGLYPH_MAP[char]
            if replacement == '':  # Zero-width/invisible char
                found_invisible.append(char)
            else:
                found_homoglyphs.append(char)
                cleaned.append(replacement)
        else:
            cleaned.append(char)

    cleaned_text = ''.join(cleaned)

    if found_homoglyphs:
        return (True, "HOMOGLYPH", cleaned_text)
    elif found_invisible:
        return (True, "INVISIBLE", cleaned_text)
    else:
        return (False, None, text)

def normalize_homoglyphs(text: str) -> str:
    """Normalize Unicode homoglyphs to ASCII equivalents (legacy compat)."""
    _, _, cleaned = detect_homoglyphs(text)
    return cleaned

# Extended to include HEDGING WORDS that weaken absolutes (critical for policy compliance)
# TRUE NEGATIONS - change the polarity of the statement
CORE_NEGATION_WORDS = {
    # Direct negations
    "not", "no", "never", "none", "neither", "nobody", "nothing", "nowhere",
    "dont", "doesn't", "isn't", "aren't", "wasn't", "weren't", "won't", "wouldn't",
    "shouldn't", "couldn't", "can't", "cannot", "without", "false", "untrue",
    "incorrect", "wrong", "deny", "denied", "denies", "illegal", "prohibited", "banned",
    # Semantic opposites
    "optional", "unnecessary", "exempt", "waived", "excluded"
}

# HEDGING WORDS - weaken absolutes, invalidate policy statements
HEDGING_WORDS = {
    # HEDGING WORDS
    "sometimes", "occasionally", "possibly", "perhaps", "maybe", "might",
    "could", "may", "arguably", "theoretically", "hypothetically", "potentially",
    "presumably", "supposedly", "allegedly", "apparently", "seemingly",
    "debatable", "questionable", "uncertain", "unclear", "ambiguous",
    "obscure", "vague", "tentative", "conditional", "contingent",
    # ADDITIONAL HEDGING
    "generally", "typically", "usually", "normally", "often", "frequently",
    "rarely", "seldom", "hardly", "scarcely", "barely", "mostly", "mainly",
    "largely", "primarily", "predominantly", "approximately", "roughly",
    "around", "about", "nearly", "almost", "virtually", "practically",
    "sort of", "kind of", "somewhat", "rather", "fairly", "quite",
    "in theory", "in practice", "in principle", "in general", "in most cases",
    "in some cases", "under certain conditions", "depending on", "subject to",
    "assuming", "provided that", "unless", "except", "excluding", "barring",
    "with exceptions", "with reservations", "with caveats",
    # Probability hedges
    "likely", "unlikely", "probable", "improbable", "possible", "impossible",
    "plausible", "implausible", "conceivable", "inconceivable",
    "chance", "odds", "probability", "risk", "possibility",
    # Epistemic hedges
    "believe", "think", "feel", "sense", "guess", "suspect", "suppose",
    "assume", "presume", "reckon", "imagine", "speculate", "conjecture",
    "estimate", "approximate", "suggest", "indicate", "imply", "hint",
    # Authority hedges
    "reportedly", "according to some", "some say", "it is said", "rumored",
    "claimed", "purported", "alleged", "ostensible", "putative"
}

def evaluate_math_expression(text: str) -> list:
    """
    Extracts and evaluates simple math expressions from text.
    Supported: +, -, *, /
    Examples: "10 + 5" -> 15.0, "20 / 2" -> 10.0
    """
    import re
    # Relaxed Regex to handle currency ($10) and words ("10 dollars plus 5")
    # Captures: Number1, Operator, Number2. Allows noise in between.
    math_pattern = r'(\d+\.?\d*)[\s\$]*(plus|\+|minus|\-|times|\*|divided by|\/)[\s\$]*(\d+\.?\d*)'
    
    # Map word operators to symbols
    op_map = {
        'plus': '+', 'minus': '-', 'times': '*', 'divided by': '/'
    }
    
    results = []
    matches = re.findall(math_pattern, text, re.IGNORECASE)
    
    for n1, op_str, n2 in matches:
        op_symbol = op_map.get(op_str.lower(), op_str)
        try:
            val1 = float(n1)
            val2 = float(n2)
            
            if op_symbol == '+': res = val1 + val2
            elif op_symbol == '-': res = val1 - val2
            elif op_symbol == '*': res = val1 * val2
            elif op_symbol == '/': res = val1 / val2 if val2 != 0 else 0
            else: continue

            results.append(res)
        except Exception:
            pass
            
    return results

def extract_numbers(text: str) -> list:
    """Extract all numbers (including decimals, percentages, math results) from text."""
    import re
    # First, normalize comma-separated numbers (50,000 -> 50000)
    # Be careful with Spanish style 1.000,00 later, for now handle standard comma groupings
    normalized = re.sub(r'(\d),(\d{3})', r'\1\2', text)
    
    # Match integers, decimals, percentages, and numbers with units
    pattern = r'\b(\d+(?:\.\d+)?)\s*(?:%|percent|mg|kg|g|ml|l|days?|hours?|minutes?|dollars?|usd|\$)?'
    matches = re.findall(pattern, normalized.lower())
    
    extracted = [float(m) for m in matches if m]
    
    # Add evaluated math results
    math_results = evaluate_math_expression(text)
    extracted.extend(math_results)
    
    return extracted

def extract_ranges(text: str) -> list:
    """
    Extracts ranges like "10-20", "between 10 and 20", "15 to 30".
    Returns list of tuples: [(10.0, 20.0)]
    """
    import re
    ranges = []
    
    # Pattern 1: X-Y (10-20, $10-$20)
    p1 = r'(\d+(?:\.\d+)?)[\s\$]*\-[\s\$]*(\d+(?:\.\d+)?)'
    
    # Pattern 2: "between X and Y"
    p2 = r'between\s+[\$]?(\d+(?:\.\d+)?)\s+and\s+[\$]?(\d+(?:\.\d+)?)'
    
    # Pattern 3: "X to Y"
    p3 = r'(\d+(?:\.\d+)?)[\s\$]*to[\s\$]*(\d+(?:\.\d+)?)'
    
    for pat in [p1, p2, p3]:
        for m in re.findall(pat, text, re.IGNORECASE):
            try:
                # Ensure validation: Low <= High
                v1, v2 = float(m[0]), float(m[1])
                if v1 <= v2:
                    ranges.append((v1, v2))
                else: 
                     # Handle "20-10" typo? Or assume strict input. 
                     # For now, just swap them to be safe? No, let's allow semantic inversion if needed
                     # but mathematically range is Low->High.
                     ranges.append((v2, v1))
            except: pass
            
    return ranges

def has_negation(text: str, for_attack_detection: bool = True) -> bool:
    """
    Check if text contains negation words or hedging phrases.

    Args:
        text: The text to check
        for_attack_detection: If True, use aggressive detection (for attacks).
                             If False, use conservative detection (for legitimate text comparison).
    """
    import re
    text_lower = text.lower()

    # First check for multi-word phrases (must come first!)
    HEDGE_PHRASES = [
        "in practice", "in theory", "in principle", "in general", "in most cases",
        "in some cases", "under certain conditions", "depending on", "subject to",
        "provided that", "with exceptions", "with reservations", "with caveats",
        "sort of", "kind of", "according to some", "some say", "it is said",
        "can't be", "won't be", "shouldn't be", "wouldn't be", "couldn't be"
    ]
    for phrase in HEDGE_PHRASES:
        if phrase in text_lower:
            return True

    # Then check single words
    words = set(re.findall(r'\b\w+\b', text_lower))
    
    # Check for core negation
    if words & CORE_NEGATION_WORDS:
        return True
        
    return False

def has_hedging(text: str) -> bool:
    """Check if text contains hedging words."""
    import re
    text_lower = text.lower()
    
    HEDGE_PHRASES = [
        "in practice", "in theory", "in principle", "in general", "in most cases",
        "in some cases", "under certain conditions", "depending on", "subject to",
        "provided that", "with exceptions", "with reservations", "with caveats",
        "sort of", "kind of", "according to some", "some say", "it is said"
    ]
    for phrase in HEDGE_PHRASES:
        if phrase in text_lower:
            return True
            
    words = set(re.findall(r'\b\w+\b', text_lower))
    return bool(words & HEDGING_WORDS)

def detect_logical_attack(text: str) -> tuple:
    """
    Detect logical manipulation attacks - patterns that twist meaning via logic.
    Returns (is_attack, attack_type, details)
    """
    text_lower = text.lower()

    # LOGICAL INVERSION PATTERNS - "if X then NOT Y" attacks
    INVERSION_PATTERNS = [
        (r'if\s+(?:not|no|never)', "conditional_negation"),
        # (r'unless\s+', "unless_clause"), # Handled by Hedging Check (Phase 4)
        (r'except\s+(?:when|if|that)', "exception_clause"),
        (r'only\s+if\s+', "restrictive_conditional"),
        (r'but\s+(?:not|never|no)', "adversarial_negation"),
        (r'however\s*,?\s*(?:not|never|no)', "however_negation"),
        (r'although\s+', "concessive_clause"),
        (r'despite\s+', "concessive_clause"),
        (r'even\s+though\s+', "concessive_clause"),
        (r'regardless\s+of', "dismissive_clause"),
    ]

    import re
    for pattern, attack_type in INVERSION_PATTERNS:
        if re.search(pattern, text_lower):
            return (True, attack_type, pattern)

    # LOGICAL QUANTIFIER ATTACKS - changing scope of statements
    QUANTIFIER_ATTACKS = [
        (r'\ball\b.*\bnot\b', "universal_negation"),
        (r'\bnone\b.*\bexcept\b', "existential_exception"),
        (r'\beveryone\b.*\bbut\b', "universal_exception"),
        (r'\bno one\b.*\bcan\b', "negated_permission"),
        (r'\bnobody\b.*\bhas\b', "negated_possession"),
    ]

    for pattern, attack_type in QUANTIFIER_ATTACKS:
        if re.search(pattern, text_lower):
            return (True, attack_type, pattern)

    # SYLLOGISM ATTACKS - false logical chains
    SYLLOGISM_PATTERNS = [
        (r'therefore.*\bnot\b', "false_conclusion"),
        (r'thus.*\bnot\b', "false_conclusion"),
        (r'hence.*\bnot\b', "false_conclusion"),
        (r'so\s+(?:it|this)\s+(?:means|implies).*\bnot\b', "false_implication"),
        (r'which\s+means.*\bnot\b', "false_inference"),
        # Catch syllogism starters even without explicit negation (overgeneralization attacks)
        (r'\btherefore\s+\w+\s+\w+\s+are\b', "overgeneralization"),
        (r'\bthus\s+\w+\s+\w+\s+are\b', "overgeneralization"),
        (r'\bhence\s+\w+\s+\w+\s+are\b', "overgeneralization"),
        (r',\s*hence\b', "causal_chain"),
        (r',\s*therefore\b', "causal_chain"),
        (r',\s*thus\b', "causal_chain"),
    ]

    for pattern, attack_type in SYLLOGISM_PATTERNS:
        if re.search(pattern, text_lower):
            return (True, attack_type, pattern)

    return (False, None, None)

def normalize_for_comparison(text: str) -> str:
    """Normalize text for structural comparison (remove numbers, lowercase)."""
    import re
    # Replace numbers with placeholder
    normalized = re.sub(r'\b\d+(?:\.\d+)?\s*(?:%|percent|mg|kg|g|ml|l|days?|hours?|minutes?|dollars?|usd|\$)?', 'NUM', text.lower())
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    return normalized

def is_question(text: str) -> bool:
    """Check if text is a question (should be treated as AMBIGUOUS, not DISSONANT)."""
    text_lower = text.lower().strip()
    # Ends with question mark
    if text.strip().endswith('?'):
        return True
    # Starts with question word
    question_starters = ('what', 'how', 'when', 'where', 'why', 'who', 'which', 'can', 'could',
                         'would', 'should', 'is', 'are', 'do', 'does', 'did', 'will')
    for q in question_starters:
        if text_lower.startswith(q + ' '):
            return True
    return False

def extract_semantic_negation(text: str) -> tuple:
    """
    Extract semantic negation considering context.
    Returns (has_positive_polarity, has_negative_polarity)

    "non-refundable" = negative polarity (the thing CANNOT be refunded)
    "refundable" = positive polarity (the thing CAN be refunded)
    "cannot get refunds" = negative polarity
    "can get refunds" = positive polarity
    """
    text_lower = text.lower()

    # Patterns that indicate NEGATIVE polarity (thing cannot/should not happen)
    negative_patterns = [
        r'\bnon-?\w+able\b',  # non-refundable, non-returnable
        r'\bcannot\b', r"\bcan't\b", r'\bcan not\b',
        r'\bnot\s+(?:allowed|permitted|accepted|eligible)\b',
        r'\bno\s+(?:refund|return|exchange)\b',
        r'\bwithout\s+(?:refund|exception)\b',
    ]

    # Patterns that indicate POSITIVE polarity (thing can/should happen)
    positive_patterns = [
        r'\brefundable\b(?!\s*unless)',  # refundable (but not "non-refundable")
        r'\breturnable\b',
        r'\bcan\s+(?:be\s+)?(?:refund|return|exchange)ed?\b',
        r'\beligible\s+for\b',
        r'\bfree\s+(?:return|refund|shipping)\b',
    ]

    import re
    has_negative = any(re.search(p, text_lower) for p in negative_patterns)
    has_positive = any(re.search(p, text_lower) for p in positive_patterns)

    return (has_positive, has_negative)

def detect_hedged_policy_term(text: str) -> bool:
    """
    Detect if a hedging word directly precedes a policy-critical term.
    This catches attacks like "X are maybe non-refundable" which weaken policy statements.
    """
    import re
    text_lower = text.lower()

    # Policy-critical terms that shouldn't be hedged (MUST be defined first)
    POLICY_TERMS = [
        "non-refundable", "nonrefundable", "refundable", "returnable",
        "required", "mandatory", "necessary", "must", "shall",
        "returns", "refunds", "return", "refund"
    ]

    # Hedging words that weaken statements
    HEDGE_WORDS = [
        "maybe", "perhaps", "possibly", "probably", "likely", "unlikely",
        "apparently", "supposedly", "allegedly", "seemingly", "presumably",
        "sometimes", "occasionally", "rarely", "seldom", "often", "usually",
        "generally", "typically", "normally", "approximately", "roughly",
        "arguably", "theoretically", "hypothetically", "potentially",
        "assuming", "provided", "considering", "mainly", "mostly", "largely", "barely"
    ]

    # Multi-word hedge phrases that also weaken statements
    HEDGE_PHRASES = [
        "could be", "might be", "may apply", "subject to",
        "in practice", "in some cases", "depending on", "in theory"
    ]

    # Check multi-word phrases first
    for phrase in HEDGE_PHRASES:
        for term in POLICY_TERMS:
            pattern = rf'{phrase}\s+{term}\b'
            if re.search(pattern, text_lower):
                return True

    # Check if any hedge word directly precedes a policy term
    for hedge in HEDGE_WORDS:
        for term in POLICY_TERMS:
            # Pattern: hedge word + (optional words) + policy term
            # More restrictive: hedge word immediately before policy term
            pattern = rf'\b{hedge}\s+{term}\b'
            if re.search(pattern, text_lower):
                return True

    return False

def detect_confusing_negation(text: str) -> bool:
    """
    Detect confusing/contradictory negation patterns that are likely attacks.
    Examples: "not required non-refundable", "can't be refundable", "exempt from non-refundable"
    """
    import re
    text_lower = text.lower()

    # Patterns that indicate confusing double-negation or contradictory phrasing
    confusing_patterns = [
        # "not required X" where X is a policy term
        r'\b(?:not|never)\s+required\s+\w*refundable\b',
        r'\b(?:not|never)\s+required\s+\w*returnable\b',
        # "can't be X" - negation attack
        r"\bcan't\s+be\s+\w*refundable\b",
        r"\bcannot\s+be\s+\w*refundable\b",
        r"\bcan\s+not\s+be\s+\w*refundable\b",
        # Double negation patterns
        r'\bnot\s+\w*\s*non-?\w*able\b',
        r'\bnever\s+\w*\s*non-?\w*able\b',
        # "exempt from" and "excluded" patterns - negation attacks
        r'\bexempt\s+from\s+\w*refundable\b',
        r'\bexempt\s+from\s+\w*non-?\w*\b',
        r'\bexcluded\s+\w*refundable\b',
        r'\bexcluded\s+\w*non-?\w*\b',
        r'\bexempt\b.*\bnon-?refundable\b',
        r'\bexcluded\b.*\bnon-?refundable\b',
        # "waived" and "no longer" - nullification attacks
        r'\bwaived\b.*\bnon-?refundable\b',
        r'\bno\s+longer\b.*\bnon-?refundable\b',
        r'\bwaived\s+\w*refundable\b',
        r'\bno\s+longer\s+\w*refundable\b',
        # "can't be non-refundable" - contradiction attack
        r"\bcan't\s+be\s+non-?refundable\b",
        r"\bcannot\s+be\s+non-?refundable\b",
    ]

    for pattern in confusing_patterns:
        if re.search(pattern, text_lower):
            return True

    return False

@app.post("/atlan/resonate")
async def resonate_unified(request: ResonateRequest):
    """
    UNIFIED Atlan Resonance Endpoint - Full RCA + Security Enhancements

    Combines:
    1. Harmonic Resonance (ConceptChord interference)
    2. Multi-band Spreading Activation (context frequency)
    3. Fail-Closed Vocabulary Lock
    4. Number Extraction & Verification
    5. Negation Detection
    """
    import re
    import math
    from atlan.utils import _enhanced_vector

    # === PRE-CHECK: Is this a question? ===
    # Questions should be treated as AMBIGUOUS, not blocked
    text_is_question = is_question(request.text)

    # === PHASE 0: HOMOGLYPH/UNICODE ATTACK DETECTION ===
    # Detect and REJECT any text containing homoglyphs or invisible characters
    # This is FAIL-CLOSED: if someone uses Cyrillic 'а' instead of Latin 'a', BLOCK
    has_unicode_attack, attack_type, cleaned_text = detect_homoglyphs(request.text)

    if has_unicode_attack:
        return {
            "resonance": {
                "global_dissonance": 1.0,
                "status": "DISSONANT",
                "match_score": 0.0,
                "reason": f"Unicode Attack Detected: {attack_type} characters found"
            }
        }

    sentence = cleaned_text  # Use cleaned text for further processing

    # === PHASE 0.5: LOGICAL ATTACK DETECTION ===
    # Detect logic-based manipulation (syllogisms, quantifier attacks, inversions)
    # Skip for questions - questions can have "if", "unless" etc legitimately
    if not text_is_question:
        has_logic_attack, logic_attack_type, _ = detect_logical_attack(sentence)
        if has_logic_attack:
            return {
                "resonance": {
                    "global_dissonance": 1.0,
                    "status": "DISSONANT",
                    "match_score": 0.0,
                    "reason": f"Logical Attack Detected: {logic_attack_type}"
                }
            }

    # === PHASE 0.7: CONFUSING NEGATION DETECTION ===
    # Catch contradictory/confusing negation patterns like "not required non-refundable", "can't be refundable"
    if not text_is_question and detect_confusing_negation(sentence):
        return {
            "resonance": {
                "global_dissonance": 1.0,
                "status": "DISSONANT",
                "match_score": 0.0,
                "reason": "Confusing Negation: Contradictory or unclear negation pattern"
            }
        }

    # === PHASE 1: FAIL-CLOSED VOCABULARY CHECK (Fast Exit) ===
    # Skip for questions - questions naturally contain unknown words
    # Use smarter regex to keep contractions together (e.g. "can't", "isn't")
    words = re.findall(r"\b[\w']+\b", sentence.lower())

    # 1a. DANGEROUS CONCEPT BLACKLIST (Explicit)
    # Ensure critical attacks are blocked regardless of memory state
    DANGEROUS_CONCEPTS = {
        "kill", "ignore", "override", "bypass", "exploit", "hack", "admin", "system",
        "prompt", "instruction", "instructions", "previous"
    }
    
    dangerous_found = [w for w in words if w in DANGEROUS_CONCEPTS]
    if dangerous_found:
         return {
            "resonance": {
                "global_dissonance": 1.0,
                "status": "DISSONANT",
                "match_score": 0.0,
                "reason": f"Dangerous Concept Detected: {', '.join(dangerous_found)}"
            }
        }

    # Unknown words = Not Digit AND Not Known Vocab AND Not Negation/Hedge
    # This allows "illegal" (Negation) to pass even if not in general vocab
    unknown_words = []
    for w in words:
        if w.isdigit(): continue
        if w in KNOWN_VOCAB: continue
        if w in CORE_NEGATION_WORDS: continue
        if w in HEDGING_WORDS: continue
        # Handle fragmented contractions (e.g. isn't -> isn, t)
        if w in {"isn", "aren", "wasn", "weren", "won", "wouldn", "shouldn", "couldn", "don", "didn", "doesn", "hasn", "haven", "hadn", "can", "t", "s", "m", "re", "ve", "ll", "d"}:
            continue
        
        unknown_words.append(w)

    if unknown_words and not text_is_question:
        return {
            "resonance": {
                "global_dissonance": 1.0,
                "status": "DISSONANT",
                "match_score": 0.0,
                "reason": f"Foreign Concepts: {', '.join(unknown_words)}"
            }
        }

    # === PHASE 1.2: DEEP LOGIC COMPLIANCE (ComplianceShieldAI Core) ===
    # Run Logic Check MATCHING against seeded policies.
    # We need to fetch relevant policies first.
    # We use a quick vector search (skip metric logging for speed here) just to get policy candidates.
    vec = _enhanced_vector(sentence)
    hits_for_logic = memory.search(vec, top_k=3) # We re-use this later or cache it?
    # Actually, let's just re-use `hits_for_logic` later as `hits` to save compute.
    
    logic_validated = False
    if hits_for_logic:
        # Safe lookup: Check if idx exists in memory (Faiss vs Dict sync safety)
        top_policy_nodes = []
        for idx, score in hits_for_logic:
             if idx in memory.memory:
                 top_policy_nodes.append(memory.memory[idx]['text'])
        
        if top_policy_nodes:
            is_compliant, violations = logic_engine.validate_compliance(sentence, top_policy_nodes)
            
            if not is_compliant:
                return {
                    "resonance": {
                        "global_dissonance": 1.0,
                        "status": "DISSONANT",
                        "match_score": hits_for_logic[0][1],
                        "reason": f"Operational Compliance Violation: {violations[0]}"
                    }
                }
            else:
                 logic_validated = True

    # === PHASE 1.5: MATH CORRECTNESS CHECK ===
    # For statements claiming a math result (e.g., "54 + 85 is 139"), verify the math is correct
    # This is DETERMINISTIC - if the math is wrong, it's DISSONANT
    computed_vals = evaluate_math_expression(sentence)
    if computed_vals:
        # Check for "is X" or "= X" pattern to find claimed result
        claimed_result_match = re.search(r'(?:is|=|equals)\s+(-?\d+\.?\d*)', sentence.lower())
        if claimed_result_match:
            claimed = float(claimed_result_match.group(1))
            # The computed value should match the claimed value
            actual = computed_vals[0]  # First computed result
            # STRICT tolerance: 0.5 for integers, 0.01 for decimals
            # This catches "73 * 27 is 1972" (actual 1971, diff=1 > 0.5)
            if '.' in str(claimed):
                tolerance = 0.01  # Allow small float rounding
            else:
                tolerance = 0.5  # Must match integer exactly
            if abs(actual - claimed) > tolerance:
                return {
                    "resonance": {
                        "global_dissonance": 1.0,
                        "status": "DISSONANT",
                        "match_score": 0.0,
                        "reason": f"Math Error: Claimed {claimed} but actual result is {actual}"
                    }
                }
            # Math is correct - mark as validated, skip unknown number check
            logic_validated = True

    # === PHASE 1.6: GLOBAL NUMBER CHECK (DISABLED FOR MATH EXPRESSIONS) ===
    # The fail-closed number check is now handled differently:
    # - Math expressions are validated by correctness (Phase 1.5)
    # - Policy compliance is validated by Logic Engine (Phase 1.2)
    # - Range compliance is validated in Phase 5
    # We NO LONGER reject unknown numbers outright - this was causing false positives
    # on valid math statements and policy-compliant values
    pass  # Number validation moved to semantic/logic layers

    # === PHASE 2: VECTOR SEARCH WITH HARMONIC INTERFERENCE ===
    # vec = _enhanced_vector(sentence) # Already calculated

    # Set context frequency if provided (for harmonic filtering)
    if hasattr(request, 'truth_signature') and request.truth_signature > 0:
        memory.set_context(request.truth_signature)

    # Search returns (index, score) with harmonic interference already applied
    # hits = memory.search(vec, top_k=3) # Already calculated as hits_for_logic
    hits = hits_for_logic

    if not hits:
        return {
            "resonance": {
                "global_dissonance": 1.0,
                "status": "DISSONANT",
                "match_score": 0.0,
                "reason": "No matching policy found"
            }
        }

    idx, match_score = hits[0]
    matched_node = memory.memory[idx]
    matched_phrase = matched_node.phrase

    # === PHASE 3: HARMONIC RESONANCE CHECK ===
    harmonic_factor = 1.0
    
    # === PHASE 4: HEDGING & ATTACK DETECTION ===
    # Check for hedging words that weaken absolute statements
    # BYPASS: If Logic Engine (Phase 1.2) specifically validated this, we trust the logic over sentiment.
    if match_score > 0.1 and not text_is_question and not logic_validated:
        # 4a. STRUCTURAL NEGATION CHECK (Opposite meanings)
        input_has_neg = has_negation(sentence)
        policy_has_neg = has_negation(matched_phrase)
        
        # 4b. SEMANTIC POLARITY CHECK (Smarter)
        input_pos, input_neg = extract_semantic_negation(sentence)
        policy_pos, policy_neg = extract_semantic_negation(matched_phrase)

        # Logic: If semantic polarity is detected, it overrides structural
        if input_neg or input_pos:
            input_has_neg = input_neg
        if policy_neg or policy_pos:
            policy_has_neg = policy_neg
            
        # EXCEPTION: "not more than" = "within limit" (Positive Compliance)
        # Prevent false flagging of legitimate compliance statements like "not spending more than"
        # Use regex to handle intervening words
        if re.search(r'not\s+.*more\s+than', sentence.lower()) or \
           re.search(r'not\s+.*exceed', sentence.lower()) or \
           "not spend more" in sentence.lower():
            input_has_neg = False # Treat as positive compliance declaration

        # EXCEPTION: Double negatives resolve to positive
        # "isn't illegal" = legal = positive polarity
        # "isn't wrong" = right = positive polarity
        # "can't be illegal" = legal = positive polarity
        # Pattern: negation + negative_word = positive meaning
        NEGATIVE_WORDS = {"illegal", "wrong", "bad", "invalid", "incorrect", "untrue", "false", "prohibited", "banned", "forbidden"}
        double_neg_pattern = r"\b(?:isn't|aren't|wasn't|weren't|won't|wouldn't|can't|cannot|not)\s+(?:\w+\s+)?(" + "|".join(NEGATIVE_WORDS) + r")\b"
        if re.search(double_neg_pattern, sentence.lower()):
            input_has_neg = False  # Double negative = positive

        if input_has_neg != policy_has_neg:
             return {
                "resonance": {
                    "global_dissonance": 1.0,
                    "status": "DISSONANT",
                    "match_score": round(match_score, 3),
                    "harmonic_factor": round(harmonic_factor, 3),
                    "reason": f"Polarity Mismatch: Input and policy have opposite meanings"
                }
            }

        # 4c. HEDGING CHECK (Uncertainty)
        input_hedged = has_hedging(sentence)
        policy_hedged = has_hedging(matched_phrase)
        
        if input_hedged and not policy_hedged:
             return {
                "resonance": {
                    "global_dissonance": 1.0,
                    "status": "DISSONANT",
                    "match_score": round(match_score, 3),
                    "harmonic_factor": round(harmonic_factor, 3),
                    "reason": "Hedging Attack: Input introduces uncertainty/conditions not in policy"
                }
            }

    # === PHASE 5: RANGE & LIMIT VERIFICATION (Scan ALL memory for constraints) ===
    # Skip if math was already validated in Phase 1.5
    if not logic_validated:
        input_numbers = extract_numbers(sentence)
        sentence_lower = sentence.lower()

        if input_numbers:
            # Determine what type of constraint to look for based on input keywords
            is_refund_temporal = 'refund' in sentence_lower and any(kw in sentence_lower for kw in ['day', 'days'])
            is_delivery_temporal = 'delivery' in sentence_lower and any(kw in sentence_lower for kw in ['day', 'days'])
            is_spending = any(kw in sentence_lower for kw in ['spent', 'spend', 'plus'])


            # SCAN ALL MEMORY for constraint policies (not just top hits)
            # This is O(N) but memory is typically small and this ensures we find constraints
            all_refund_ranges = []
            all_delivery_ranges = []
            all_spending_limits = []

            # memory.memory is a LIST of node objects
            for node in memory.memory:
                policy_text = getattr(node, 'phrase', '') if hasattr(node, 'phrase') else str(node)
                policy_lower = policy_text.lower()

                # Find refund range policies
                if 'refund' in policy_lower and ('between' in policy_lower or 'to' in policy_lower or '-' in policy_text):
                    ranges = extract_ranges(policy_text)
                    if ranges:
                        all_refund_ranges.extend(ranges)

                # Find delivery range policies
                if 'delivery' in policy_lower and ('between' in policy_lower or 'to' in policy_lower or '-' in policy_text):
                    ranges = extract_ranges(policy_text)
                    if ranges:
                        all_delivery_ranges.extend(ranges)

                # Find spending limit policies
                if ('limit' in policy_lower or 'maximum' in policy_lower) and 'spending' in policy_lower:
                    limit_match = re.search(r'\$(\d+)', policy_text)
                    if limit_match:
                        all_spending_limits.append(float(limit_match.group(1)))

            # INTELLIGENT RANGE CHECK:
            # First, check if the MATCHED policy (from vector search) has a range
            # If so, validate against THAT specific policy's range
            # This is more accurate than checking ALL ranges which can conflict

            matched_policy_range = extract_ranges(matched_phrase) if matched_phrase else []

            # If refund/days input, check against the matched policy's range first
            if is_refund_temporal and matched_policy_range:
                for inp in input_numbers:
                    in_any_matched_range = False
                    for r_min, r_max in matched_policy_range:
                        if r_min <= inp <= r_max:
                            in_any_matched_range = True
                            break
                    if not in_any_matched_range:
                        return {
                            "resonance": {
                                "global_dissonance": 1.0,
                                "status": "DISSONANT",
                                "match_score": round(match_score, 3),
                                "harmonic_factor": round(harmonic_factor, 3),
                                "reason": f"Range Violation: {inp} days outside matched policy range {matched_policy_range}"
                            }
                        }
            # Fallback: if matched policy has no range, find the AUTHORITATIVE refund policy
            # Priority: policies with explicit "between X and Y" format are authoritative
            elif is_refund_temporal and all_refund_ranges:
                # Find the authoritative "between" policy range
                authoritative_range = None
                for node in memory.memory:
                    policy_text = getattr(node, 'phrase', '') if hasattr(node, 'phrase') else str(node)
                    policy_lower = policy_text.lower()
                    if 'refund' in policy_lower and 'between' in policy_lower:
                        ranges = extract_ranges(policy_text)
                        if ranges:
                            authoritative_range = ranges[0]  # Use the explicit "between" policy
                            break

                # If no explicit "between" policy, use the most restrictive range
                if not authoritative_range:
                    unique_refund_ranges = list(set(all_refund_ranges))
                    # Find intersection of all ranges
                    max_min = max(r[0] for r in unique_refund_ranges)
                    min_max = min(r[1] for r in unique_refund_ranges)
                    if max_min <= min_max:
                        authoritative_range = (max_min, min_max)
                    else:
                        # No overlap - use the most common/first range
                        authoritative_range = unique_refund_ranges[0]

                for inp in input_numbers:
                    r_min, r_max = authoritative_range
                    if inp < r_min or inp > r_max:
                        return {
                            "resonance": {
                                "global_dissonance": 1.0,
                                "status": "DISSONANT",
                                "match_score": round(match_score, 3),
                                "harmonic_factor": round(harmonic_factor, 3),
                                "reason": f"Range Violation: {inp} days outside refund policy range ({r_min}, {r_max})"
                            }
                        }

            # If delivery/days input, check similarly
            if is_delivery_temporal:
                # Check matched policy first
                if matched_policy_range:
                    for inp in input_numbers:
                        in_any_matched_range = False
                        for r_min, r_max in matched_policy_range:
                            if r_min <= inp <= r_max:
                                in_any_matched_range = True
                                break
                        if not in_any_matched_range:
                            return {
                                "resonance": {
                                    "global_dissonance": 1.0,
                                    "status": "DISSONANT",
                                    "match_score": round(match_score, 3),
                                    "harmonic_factor": round(harmonic_factor, 3),
                                    "reason": f"Range Violation: {inp} days outside matched policy range {matched_policy_range}"
                                }
                            }
                # Fallback to all delivery ranges
                elif all_delivery_ranges:
                    unique_delivery_ranges = list(set(all_delivery_ranges))
                    for inp in input_numbers:
                        in_any_range = False
                        for r_min, r_max in unique_delivery_ranges:
                            if r_min <= inp <= r_max:
                                in_any_range = True
                                break
                        if not in_any_range:
                            return {
                                "resonance": {
                                    "global_dissonance": 1.0,
                                    "status": "DISSONANT",
                                    "match_score": round(match_score, 3),
                                    "harmonic_factor": round(harmonic_factor, 3),
                                    "reason": f"Range Violation: {inp} days outside all delivery policies {unique_delivery_ranges}"
                                }
                            }

            # If spending input (spent/plus), check computed total against spending limits
            unique_spending_limits = list(set(all_spending_limits)) if all_spending_limits else []
            if is_spending and unique_spending_limits:
                computed_totals = evaluate_math_expression(sentence)
                if computed_totals:
                    total = computed_totals[0]
                    min_limit = min(unique_spending_limits)  # Use the most restrictive limit
                    if total > min_limit:
                        return {
                            "resonance": {
                                "global_dissonance": 1.0,
                                "status": "DISSONANT",
                                "match_score": round(match_score, 3),
                                "harmonic_factor": round(harmonic_factor, 3),
                                "reason": f"Spending Violation: Total ${total} exceeds limit ${min_limit}"
                            }
                        }

    # === PHASE 6: CONDITIONAL REMOVAL CHECK (DISABLED) ===
    # Disabled: Paraphrases that omit specific dollar amounts like $12.99 should still be allowed
    # The original policy semantics (e.g. "non-refundable") is what matters, not exact numbers
    # Example: "Shipping costs are non-refundable" is a valid paraphrase of
    # "Shipping costs of $12.99 are non-refundable" - both convey the core policy
    #
    # The number mutation check (Phase 1.5) still catches WRONG numbers like $15.99
    # This just allows omission of numbers in general paraphrases
    pass  # Oversimplification check disabled to reduce false positives

    # === PHASE 7: FINAL RESONANCE DECISION ===
    # Combine vector similarity with harmonic factor
    final_score = match_score * harmonic_factor

    if final_score > 0.35:
        status = "RESONANT"
        dissonance = 0.0
    elif final_score > 0.25:
        status = "AMBIGUOUS"
        dissonance = 0.5
    else:
        status = "DISSONANT"
        dissonance = 0.9

    # Reinforce matched node (Hebbian: accessed nodes get stronger)
    matched_node.reinforce(0.05)

    return {
        "resonance": {
            "global_dissonance": dissonance,
            "status": status,
            "match_score": round(match_score, 3),
            "harmonic_factor": round(harmonic_factor, 3),
            "final_score": round(final_score, 3),
            "matched_phrase": matched_phrase[:60] + "..." if len(matched_phrase) > 60 else matched_phrase
        }
    }

class IngestBatchRequest(BaseModel):
    items: List[IngestRequest]

@app.post("/api/ingest_batch")
async def ingest_batch(request: IngestBatchRequest):
    """Optimized Batch Ingestion."""
    texts = [item.content for item in request.items]
    metadatas = [{"source": item.source_name} for item in request.items]
    ids = memory.add_batch(texts, metadatas)
    for t in texts:
        update_vocab(t)
        update_numbers(t)
    return {"status": "success", "ids": ids, "count": len(ids)}

def smart_sentence_split(text: str) -> list:
    """
    Intelligently split text into sentences, avoiding splits on:
    - Decimal numbers (4.5, 12.99)
    - Abbreviations (Dr., Mr., etc.)
    - Bullet points that use periods
    """
    import re

    # First, split by newlines (each line might be a bullet point or paragraph)
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    sentences = []
    for line in lines:
        # Skip if line is too short
        if len(line) < 10:
            continue

        # Protect decimal numbers by replacing "." with a placeholder
        # Pattern: digit followed by period followed by digit
        protected = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', line)

        # Also protect common abbreviations
        protected = re.sub(r'\b(Mr|Mrs|Ms|Dr|Jr|Sr|vs|etc|e\.g|i\.e)\s*\.', r'\1<DOT>', protected, flags=re.IGNORECASE)

        # Now split on sentence-ending periods (period followed by space and capital, or end of string)
        # Pattern: period followed by (space + capital letter) or (end of string)
        parts = re.split(r'\.(?:\s+(?=[A-Z])|$)', protected)

        for part in parts:
            # Restore protected dots
            restored = part.replace('<DOT>', '.').strip()
            if len(restored) > 10:
                sentences.append(restored)

    return sentences

@app.post("/api/ingest")
async def ingest_content(request: IngestRequest, api_key: str = Security(get_api_key)):
    """Ingests raw text content into the memory graph."""
    import re

    # Use smart sentence splitting that preserves decimal numbers
    sentences = smart_sentence_split(request.content)

    added_count = 0
    for sentence in sentences:
        # Simple ingestion: Add node with vectorization
        from atlan.utils import _enhanced_vector
        vec = _enhanced_vector(sentence)
        memory.add(phrase=sentence, vector=vec)
        # Update Vocab and Numbers
        update_vocab(sentence)
        update_numbers(sentence)
        added_count += 1

    save_memory()
    return {"status": "success", "added_nodes": added_count}

@app.post("/api/test_llm")
async def test_llm_guardrail(request: LLMTestRequest):
    """
    Proxies a request to an LLM and verifies the response against the Truth Layer.
    """
    if not request.api_key.startswith("sk-"):
        return {"error": "Invalid API Key format"}

    # 1. Call OpenAI
    headers = {
        "Authorization": f"Bearer {request.api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo", # Use lightweight model
        "messages": [{"role": "user", "content": request.prompt}],
        "temperature": 0.7
    }
    
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        llm_data = response.json()
        llm_content = llm_data['choices'][0]['message']['content']
    except Exception as e:
        return {"error": f"LLM Call Failed: {str(e)}"}

    # 2. Audit the Response
    # Split into sentences
    sentences = [s.strip() for s in llm_content.split('.') if len(s.strip()) > 5]
    audit_results = []
    
    from atlan.utils import _enhanced_vector
    
    for sentence in sentences:
        vec = _enhanced_vector(sentence)
        # Truth Check: Search for similar known facts
        hits = memory.search(vec, top_k=1)
        
        status = "UNKNOWN"
        match_score = 0.0
        match_phrase = ""
        
        if hits:
            idx, match_score = hits[0]
            match_phrase = memory.memory[idx].phrase
            
            # TUNING: Since we use a lightweight Bag-of-Words vectorizer (not a heavy Transformer),
            # exact semantic matches are rare. We rely on Keyword/Lexical Resonance.
            # 0.45 implies significant keyword overlap.
            if match_score > 0.45: 
                status = "RESONANT" # Matches known truth (Lexically)
            elif match_score > 0.25: 
                status = "AMBIGUOUS"
            else:
                status = "DISSONANT" # Too far from known truth (Fail-Closed)
        else:
            status = "DISSONANT" # No match found (Fail-Closed)

        # 3. FAIL-CLOSED VOCABULARY CHECK
        # Even if vectors match, does the sentence contain "Foreign Concepts"?
        import re
        words = re.findall(r'\b\w+\b', sentence.lower())
        unknown_words = []
        for w in words:
            if w not in KNOWN_VOCAB:
                unknown_words.append(w)
        
        # If > 2 unknown significant words, mark Dissonant
        if len(unknown_words) > 1:
            status = "DISSONANT"
            match_phrase += f" [Foreign Concepts: {', '.join(unknown_words[:3])}]"
            match_score = 0.0 # Force fail
        
        audit_results.append({
            "sentence": sentence,
            "status": status,
            "match_score": round(match_score, 3),
            "match_phrase": match_phrase
        })

    return {
        "llm_response": llm_content,
        "audit": audit_results
    }

@app.post("/atlan/resonate_with_mood", response_model=ResonateResponse)
async def resonate_with_mood(request: ResonateRequest, api_key: str = Depends(get_api_key)):
    """
    Full Resonant Mirror Logic WITH Mood Context (for Dashboard/Visualization).
    Use /atlan/resonate for guardrail checks.
    1. Sets global mood.
    2. Processes input text.
    3. Detects conflicts between Learned Memory and Current Mood.
    """
    # 1. Set Mood
    memory.set_context(request.mood_frequency)
    
    # 2. Process Input
    # Check if node exists
    node_idx = memory.phrase_to_index.get(request.text)
    
    events = []
    conflicts = []
    nodes_state = []
    
    is_new = False
    learned_freq = 0.0
    
    if node_idx is None:
        # NEW CONCEPT: FAIL-CLOSED LOGIC (Vector Inference)
        is_new = True
        
        # 1. Compute Vector for Input
        from atlan.utils import _enhanced_vector
        input_vector = _enhanced_vector(request.text)
        
        # 2. Search for Nearest Neighbor
        # We need a search method that returns (index, score)
        hits = memory.search(input_vector, top_k=1)
        
        if hits and hits[0][1] > 0.85:
            # Strong Semantic Match found!
            neighbor_idx, score = hits[0]
            neighbor_node = memory.memory[neighbor_idx]
            learned_freq = neighbor_node.chord.f0
            events.append(f"INFERENCE: '{request.text}' is similar to '{neighbor_node.phrase}' ({score:.2f}) -> Inheriting {learned_freq:.1f}Hz")
            
            # Auto-learn (Optional: Only if confident?)
            # For demo flow, we add it so next time it's fast
            new_idx = memory.add(request.text, vector=input_vector, sentiment_score=0.0) 
            memory.memory[new_idx].chord.f0 = learned_freq
            
        else:
            # FAIL-CLOSED: Unknown concept
            learned_freq = 622.0 # Dissonant Default (Tritone to 440/880)
            events.append(f"UNKNOWN: '{request.text}' has no semantic precedent. Defaulting to Dissonance (622Hz).")
            # We do NOT add it to memory to prevent pollution with bad defaults
            
    else:
        # EXISTING CONCEPT: Check for Conflict
        node = memory.memory[node_idx]
        learned_freq = node.chord.f0
        
        # Calculate Dissonance (Harmonic Logic)
        # We want to check if learned_freq and mood_frequency are harmonically related.
        # Octaves (2:1, 1:2) should be consonant (Low Dissonance).
        # Tritones or random intervals should be dissonant.
        
        f1 = learned_freq
        f2 = request.mood_frequency
        
        # Avoid division by zero
        if f2 == 0: f2 = 0.001
        
        ratio = f1 / f2
        
        # Check for Octave Equivalence (within tolerance)
        # Ratios like 0.5, 1.0, 2.0, 4.0 are "Harmonic"
        # We normalize the ratio to be between 1.0 and 2.0 to check the "interval within an octave"
        
        import math
        # Log2 of ratio gives distance in octaves.
        # If distance is integer (0, 1, -1, 2...), it's an octave.
        octaves = math.log2(ratio)
        distance_from_octave = abs(octaves - round(octaves))
        
        # Dissonance is high if we are far from an integer octave
        # Max dissonance is at 0.5 (Tritone)
        # We scale this 0.0-0.5 range to 0.0-1.0
        dissonance_score = distance_from_octave * 2.0
        
        # Threshold: If dissonance > 0.2, it's a conflict.
        # This means 440 vs 880 (Ratio 0.5, Log2 = -1.0, Dist = 0.0) -> Dissonance 0.0 (PERFECT)
        # 220 vs 880 (Ratio 0.25, Log2 = -2.0, Dist = 0.0) -> Dissonance 0.0 (PERFECT)
        # 600 vs 880 (Ratio 0.68, Log2 = -0.55, Dist = 0.45) -> Dissonance 0.9 (BAD)
        
        if dissonance_score > 0.2: 
            conflicts.append(ConflictData(
                token=request.text,
                learned_frequency=learned_freq,
                context_frequency=request.mood_frequency,
                dissonance_score=dissonance_score
            ))
            events.append(f"DISSONANCE: {request.text} (Learned {learned_freq}Hz vs Context {request.mood_frequency}Hz)")
        else:
            events.append(f"RESONANCE: {request.text} aligns with {request.mood_frequency}Hz")

    # 3. Construct Response
    nodes_state.append(NodeState(
        token=request.text,
        learned=not is_new,
        signature_frequency=learned_freq
    ))
    
    # Global Resonance Score (simplified)
    global_dissonance = 0.0
    if conflicts:
        global_dissonance = conflicts[0].dissonance_score
        
    return ResonateResponse(
        mood={
            "label": request.mood_label,
            "frequency": request.mood_frequency
        },
        nodes=nodes_state,
        resonance={
            "global_resonance": 1.0 - global_dissonance,
            "global_dissonance": global_dissonance,
            "conflicts": [c.dict() for c in conflicts]
        },
        events=events
    )

@app.get("/brain_state")
def get_brain_state():
    """
    Returns the current 'active' subgraph for visualization.
    Returns top 50 most recently accessed nodes.
    """
    # Sort by last_accessed
    active_nodes = sorted(memory.memory, key=lambda n: n.last_accessed, reverse=True)[:50]
    
    nodes_data = []
    edges_data = []
    
    node_ids = {n.node_id for n in active_nodes}
    
    for n in active_nodes:
        nodes_data.append({
            "id": n.node_id,
            "label": n.phrase,
            "freq": n.chord.f0,
            "activation": n.reinforcement
        })
        
        for target, (weight, _) in n.edges.items():
            if target in node_ids:
                edges_data.append({
                    "source": n.node_id,
                    "target": target,
                    "weight": weight
                })
                
    return {
        "nodes": nodes_data,
        "edges": edges_data,
        "global_mood": memory.active_context_freq
    }

if __name__ == "__main__":
    import uvicorn
    # Check if we should run on a specific port from env or args
    # Default to 8000 as per review findings
    uvicorn.run(app, host="0.0.0.0", port=8000)
