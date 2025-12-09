from typing import List, Dict, Any

# Vocabulary Definition
# Format: "word": {"pos": "part_of_speech", "sentiment": float, "timbre": "instrument"}

# Musical Metaphor for Parts of Speech:
# - Nouns (Objects/Concepts) -> Bass/Cello (The Foundation)
# - Verbs (Action/Change) -> Drums/Percussion (The Rhythm/Drive)
# - Adjectives (Description) -> Woodwinds/Flute (The Color)
# - Adverbs (Modification) -> Brass (The Emphasis)
# - Determinerm/Conj (Connectors) -> Synth Pad (The Glue)

CORE_VOCABULARY = {
    # --- NOUNS (Bass) ---
    "world": {"pos": "noun", "sentiment": 0.1, "timbre": "bass"},
    "mind": {"pos": "noun", "sentiment": 0.0, "timbre": "bass"},
    "logic": {"pos": "noun", "sentiment": 0.0, "timbre": "bass"},
    "chaos": {"pos": "noun", "sentiment": -0.4, "timbre": "bass"},
    "nature": {"pos": "noun", "sentiment": 0.5, "timbre": "bass"},
    "harmony": {"pos": "noun", "sentiment": 0.6, "timbre": "bass"},
    "path": {"pos": "noun", "sentiment": 0.2, "timbre": "bass"},
    "void": {"pos": "noun", "sentiment": -0.3, "timbre": "bass"},
    "spark": {"pos": "noun", "sentiment": 0.5, "timbre": "bass"},
    "time": {"pos": "noun", "sentiment": 0.0, "timbre": "bass"},
    "truth": {"pos": "noun", "sentiment": 0.4, "timbre": "bass"},
    "lie": {"pos": "noun", "sentiment": -0.6, "timbre": "bass"},
    "pattern": {"pos": "noun", "sentiment": 0.2, "timbre": "bass"},
    "music": {"pos": "noun", "sentiment": 0.7, "timbre": "bass"},
    "silence": {"pos": "noun", "sentiment": -0.1, "timbre": "bass"},
    
    # --- VERBS (Percussion) ---
    "is": {"pos": "verb", "sentiment": 0.0, "timbre": "percussion"},
    "creates": {"pos": "verb", "sentiment": 0.3, "timbre": "percussion"},
    "destroys": {"pos": "verb", "sentiment": -0.5, "timbre": "percussion"},
    "flows": {"pos": "verb", "sentiment": 0.2, "timbre": "percussion"},
    "stops": {"pos": "verb", "sentiment": -0.1, "timbre": "percussion"},
    "seeks": {"pos": "verb", "sentiment": 0.1, "timbre": "percussion"},
    "finds": {"pos": "verb", "sentiment": 0.4, "timbre": "percussion"},
    "loses": {"pos": "verb", "sentiment": -0.4, "timbre": "percussion"},
    "resonates": {"pos": "verb", "sentiment": 0.5, "timbre": "percussion"},
    "clashes": {"pos": "verb", "sentiment": -0.3, "timbre": "percussion"},
    
    # --- ADJECTIVES (Woodwinds) ---
    "bright": {"pos": "adj", "sentiment": 0.6, "timbre": "woodwind"},
    "dark": {"pos": "adj", "sentiment": -0.6, "timbre": "woodwind"},
    "loud": {"pos": "adj", "sentiment": 0.1, "timbre": "woodwind"},
    "quiet": {"pos": "adj", "sentiment": 0.0, "timbre": "woodwind"},
    "fast": {"pos": "adj", "sentiment": 0.1, "timbre": "woodwind"},
    "slow": {"pos": "adj", "sentiment": -0.1, "timbre": "woodwind"},
    "happy": {"pos": "adj", "sentiment": 0.8, "timbre": "woodwind"},
    "joy": {"pos": "noun", "sentiment": 0.9, "timbre": "flute"},
    "sad": {"pos": "adj", "sentiment": -0.8, "timbre": "woodwind"},
    "sorrow": {"pos": "noun", "sentiment": -0.9, "timbre": "cello"},
    "infinite": {"pos": "adj", "sentiment": 0.3, "timbre": "woodwind"},
    "empty": {"pos": "adj", "sentiment": -0.4, "timbre": "woodwind"},
    "complex": {"pos": "adj", "sentiment": 0.0, "timbre": "woodwind"},
    "simple": {"pos": "adj", "sentiment": 0.2, "timbre": "woodwind"},
    
    # --- CONNECTORS (Pad) ---
    "the": {"pos": "det", "sentiment": 0.0, "timbre": "pad"},
    "a": {"pos": "det", "sentiment": 0.0, "timbre": "pad"},
    "and": {"pos": "conj", "sentiment": 0.0, "timbre": "pad"},
    "but": {"pos": "conj", "sentiment": -0.1, "timbre": "pad"},
    "or": {"pos": "conj", "sentiment": 0.0, "timbre": "pad"},
    "in": {"pos": "prep", "sentiment": 0.0, "timbre": "pad"},
    "of": {"pos": "prep", "sentiment": 0.0, "timbre": "pad"},
    "to": {"pos": "prep", "sentiment": 0.0, "timbre": "pad"},
}

# The Bush Topology (Connections)
# The Bush Topology (Connections)
# Format: (Source, Target, Weight, Optional[List[str]])
CONNECTIONS = [
    # World Cluster
    ("world", "nature", 0.8, None), 
    ("world", "chaos", 0.6, ["delta", "gamma"]), # Chaos resonates in Mood and Insight
    ("world", "logic", 0.6, ["beta"]), # Logic is purely intellectual
    ("nature", "flows", 0.9, ["theta"]), # Flow is sequential
    ("nature", "creates", 0.7, None),
    
    # Mind Cluster
    ("mind", "logic", 0.8, ["beta"]), 
    ("mind", "truth", 0.7, ["beta", "gamma"]), 
    ("mind", "lie", 0.6, ["beta"]),
    ("mind", "seeks", 0.9, ["theta"]), 
    ("mind", "finds", 0.8, ["gamma"]), # Finding is an insight
    
    # Emotional Cluster
    ("happy", "joy", 0.9, ["delta", "theta"]), 
    ("happy", "bright", 0.8, ["delta"]),
    ("sad", "sorrow", 0.9, ["delta", "theta"]), 
    ("sad", "dark", 0.8, ["delta"]), 
    ("sad", "void", 0.7, ["delta"]),
    
    # Action Flows
    ("creates", "harmony", 0.7, ["alpha", "gamma"]), 
    ("destroys", "chaos", 0.7, ["delta", "gamma"]),
    ("seeks", "truth", 0.8, ["beta"]), 
    ("finds", "path", 0.8, ["theta"]),
    
    # Grammatical/Structural Glue (Weak associations)
    ("the", "world", 0.3, ["theta"]), 
    ("the", "mind", 0.3, ["theta"]),
    ("is", "truth", 0.4, ["beta"]), 
    ("is", "lie", 0.4, ["beta"]),
    
    # INHIBITORY CONNECTIONS (The Balance)
    ("joy", "sorrow", -0.5, ["delta"]), # Emotional suppression
    ("sorrow", "joy", -0.5, ["delta"]),
    ("bright", "dark", -0.5, ["alpha"]), # Perceptual contrast
    ("dark", "bright", -0.5, ["alpha"]),
    ("truth", "lie", -0.8, ["beta"]), # Logical contradiction
    ("lie", "truth", -0.8, ["beta"]),
    ("creates", "destroys", -0.6, ["gamma"]), 
    ("destroys", "creates", -0.6, ["gamma"]),
    ("logic", "chaos", -0.5, ["beta", "delta"]), # Logic suppresses Chaos (Beta), Chaos disrupts Logic (Delta?)
    ("chaos", "logic", -0.5, ["beta", "delta"])
]

def seed_vocabulary(memory_index):
    """
    Populates the memory index with the core vocabulary and builds the graph.
    """
    print(f"Seeding memory with {len(CORE_VOCABULARY)} concepts...")
    
    # 1. Add Nodes
    phrase_to_index = {}
    for word, data in CORE_VOCABULARY.items():
        idx = memory_index.add(
            phrase=word, 
            label=data["pos"], 
            sentiment_score=data["sentiment"],
            pos=data["pos"],
            timbre=data["timbre"]
        )
        phrase_to_index[word] = idx
        
    # 2. Build Connections (The Bush)
    print("Weaving the graph connections...")
    count = 0
    for conn in CONNECTIONS:
        src = conn[0]
        tgt = conn[1]
        weight = conn[2]
        bands = conn[3] if len(conn) > 3 else None
        
        if src in phrase_to_index and tgt in phrase_to_index:
            # print(f"Connecting {src} -> {tgt}")
            memory_index.connect_nodes(phrase_to_index[src], phrase_to_index[tgt], weight, bands)
            count += 1
        else:
            print(f"Skipping connection {src} -> {tgt} (Node not found)")
            
    print(f"Seeding complete. Added {len(phrase_to_index)} nodes and {count} connections.")
