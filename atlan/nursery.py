from typing import List, Dict
from atlan.memory import RCoreMemoryIndex
from atlan.utils import _enhanced_vector

class PrimalManager:
    """
    Manages the 'Ground Truth' vectors for the agent.
    These are the immutable anchors of meaning.
    """
    def __init__(self):
        self.primals = self._define_primals()
        
    def _define_primals(self) -> Dict[str, str]:
        """
        Defines the set of ~200 Primal Concepts.
        Returns a dict of {Concept: Category}.
        """
        primals = {}
        
        # 1. Substantives (The Core)
        substantives = ["I", "YOU", "SOMEONE", "SOMETHING", "PEOPLE", "BODY", "WORLD", "THING"]
        for p in substantives: primals[p] = "SUBSTANTIVE"
            
        # 2. Relational
        relational = ["KIND", "PART", "SAME", "OTHER", "ELSE", "LIKE", "DIFFERENT"]
        for p in relational: primals[p] = "RELATIONAL"
            
        # 3. Spatial
        spatial = ["WHERE", "HERE", "THERE", "ABOVE", "BELOW", "FAR", "NEAR", "SIDE", "INSIDE", "OUTSIDE", "TOP", "BOTTOM", "FRONT", "BACK"]
        for p in spatial: primals[p] = "SPATIAL"
            
        # 4. Temporal
        temporal = ["WHEN", "NOW", "THEN", "BEFORE", "AFTER", "MOMENT", "WHILE", "TIME", "LONG", "SHORT"]
        for p in temporal: primals[p] = "TEMPORAL"
            
        # 5. Sensory (The "Fake" Senses - Grounding Anchors)
        # Visual
        visual = ["RED", "BLUE", "GREEN", "YELLOW", "BLACK", "WHITE", "DARK", "LIGHT", "BRIGHT", "DIM", "BIG", "SMALL", "ROUND", "FLAT", "SHAPE"]
        for p in visual: primals[p] = "SENSORY_VISUAL"
            
        # Auditory
        auditory = ["LOUD", "QUIET", "HIGH", "LOW", "NOISE", "SOUND", "SILENCE"]
        for p in auditory: primals[p] = "SENSORY_AUDITORY"
            
        # Tactile
        tactile = ["HOT", "COLD", "WARM", "COOL", "HARD", "SOFT", "ROUGH", "SMOOTH", "SHARP", "DULL", "WET", "DRY"]
        for p in tactile: primals[p] = "SENSORY_TACTILE"
            
        # 6. Emotional / Mental
        emotional = ["GOOD", "BAD", "WANT", "FEEL", "THINK", "KNOW", "BELIEVE", "SEE", "HEAR", "HAPPY", "SAD", "FEAR", "ANGER", "LOVE", "HATE", "JOY", "PAIN", "PLEASURE"]
        for p in emotional: primals[p] = "EMOTIONAL"
            
        # 7. Action / Event
        action = ["DO", "HAPPEN", "MOVE", "SAY", "LIVE", "DIE", "START", "STOP", "CONTINUE", "CHANGE", "GROW", "BREAK", "FIX", "MAKE", "GIVE", "TAKE"]
        for p in action: primals[p] = "ACTION"
            
        # 8. Logical / Quantifiers
        logical = ["NOT", "MAYBE", "CAN", "BECAUSE", "IF", "TRUE", "FALSE", "ONE", "TWO", "SOME", "ALL", "MANY", "FEW", "VERY", "MORE", "LESS"]
        for p in logical: primals[p] = "LOGICAL"
        
        return primals

    def get_vector(self, primal: str) -> List[float]:
        """
        Returns the immutable vector for a primal.
        Uses _enhanced_vector (MD5) but prefixes with 'PRIMAL_' to ensure separation from normal words.
        """
        # Prefix ensures "RED" (Primal) is distinct from "red" (Word) if we ever want them separate.
        # But actually, we want the word "red" to map to this vector.
        # So let's just use the word itself, but ensure it's uppercase for the seed?
        # No, let's use a special seed prefix to guarantee these are the "Ground Truth" versions.
        return _enhanced_vector(f"PRIMAL_{primal.upper()}")

class Nursery:
    """
    The Teacher. Responsible for initializing the agent's memory with Primals.
    """
    def __init__(self, memory: RCoreMemoryIndex):
        self.memory = memory
        self.manager = PrimalManager()
        
    def teach_primals(self):
        """
        Iterates through all Primals and adds them to memory as LOCKED nodes.
        """
        print(f"Welcome to The Nursery. Teaching {len(self.manager.primals)} Primal Concepts...")
        
        count = 0
        for primal, category in self.manager.primals.items():
            vector = self.manager.get_vector(primal)
            
            # Add to memory
            # Reinforcement is HIGH (100.0) to represent "Instinct" or "Core Knowledge"
            # Locked = True prevents merging/pruning
            self.memory.add(
                phrase=primal,
                vector=vector,
                label=category,
                sentiment_score=0.0, # Neutral by default, could customize
                pos="primal",
                timbre="sine", # Pure tone
                locked=True
            )
            
            # Also reinforce it immediately to ensure it's "hot"
            idx = self.memory.phrase_to_index[primal]
            self.memory.memory[idx].reinforcement = 100.0
            
            count += 1
            
        print(f"Nursery Session Complete. {count} Primals grounded.")

    def teach_flashcard(self, word: str, definition_primals: List[str]):
        """
        Teaches a new word by associating it with existing Primals.
        Example: teach_flashcard("APPLE", ["RED", "ROUND", "FRUIT"])
        """
        print(f"Flashcard: {word} = {definition_primals}")
        
        # 1. Add the new word (if not exists)
        # We don't lock it, it's a learned concept.
        word_idx = -1
        if word in self.memory.phrase_to_index:
            word_idx = self.memory.phrase_to_index[word]
        else:
            word_idx = self.memory.add(word, label="learned")
            
        word_node = self.memory.memory[word_idx]
        
        # 2. Link to Primals
        for primal in definition_primals:
            primal_upper = primal.upper()
            if primal_upper in self.memory.phrase_to_index:
                primal_idx = self.memory.phrase_to_index[primal_upper]
                primal_node = self.memory.memory[primal_idx]
                
                # Strong connection: Word -> Primal
                word_node.connect(primal_node.node_id, weight=0.9, bands=["beta"]) # Beta = Definition/Logic
                
                # Primal -> Word (weaker, association)
                primal_node.connect(word_node.node_id, weight=0.5, bands=["beta"])
            else:
                print(f"Warning: Primal '{primal}' not found in Nursery.")
