import time
import math
import random
from typing import List, Dict, Any, Optional, Tuple
from .memory import RCoreMemoryIndex
from .perception import DriftTracker
from .utils import _enhanced_vector

class SyntaxLearner:
    """
    Learns valid sentence structures (templates) from observed text.
    Replaces the hardcoded GrammarStateMachine.
    """
    def __init__(self):
        # Store learned templates: Set of POS tag tuples for O(1) lookup
        # e.g., {('determiner', 'adjective', 'noun', 'verb')}
        self.templates: Set[Tuple[str, ...]] = set()
        self.max_templates = 2000
        
        # Seed with some basic templates to start
        self.templates.add(("noun", "verb", "noun"))
        self.templates.add(("determiner", "noun", "verb"))
        self.templates.add(("determiner", "adjective", "noun", "verb"))
        
    def learn_template(self, pos_sequence: List[str]):
        """Records a new valid POS sequence."""
        if not pos_sequence: return
        
        # Normalize: convert to tuple for set storage
        template_tuple = tuple(pos_sequence)
        
        if len(template_tuple) > 10: return # Ignore too long
        
        if template_tuple not in self.templates:
            self.templates.add(template_tuple)
            # Pruning if too large (random removal is O(1) amortized but set doesn't support pop(0))
            # For now, let's just let it grow up to a limit and then stop learning?
            # Or convert to list to pop?
            # Actually, 2000 templates is fine for memory.
            if len(self.templates) > self.max_templates:
                # Simple strategy: Stop learning new ones, or clear half?
                # Let's just stop learning to prevent explosion.
                pass
                
    def get_template(self, length_hint: int = 5) -> List[str]:
        """Returns a learned template, preferably close to length_hint."""
        if not self.templates:
            return ["noun", "verb"] # Fallback
            
        # Pick a random template
        # Converting set to list is O(N), but we only do this when generating, not reading.
        # So it's fine.
        return list(random.choice(list(self.templates)))
        
    def get_valid_next_states(self, current_pos: str, position_in_template: int, template: List[str]) -> List[str]:
        """
        Returns the next expected POS based on the active template.
        """
        if position_in_template < len(template) - 1:
            return [template[position_in_template + 1]]
        else:
            return [] # End of template

class MelodicSequencer:
    """
    Generates thought sequences based on harmonic resonance and rhythmic (grammatical) constraints.
    """
    def __init__(self, memory_index: RCoreMemoryIndex, drift_tracker: DriftTracker):
        self.memory = memory_index
        self.drift_tracker = drift_tracker
        self.syntax = SyntaxLearner() # Replaces grammar
        self.theta_threshold = 0.5
        
    def generate_sequence(self, seed_phrase: str, length: int = 10, activation_map: Dict[int, float] = None) -> List[str]:
        """
        Generates a sequence of concepts starting from a seed.
        Uses learned Syntax Templates to structure the output.
        """
        sequence = []
        
        # 1. Find Seed Node
        seed_idx = -1
        for i, node in enumerate(self.memory.memory):
            if node.phrase == seed_phrase:
                seed_idx = i
                break
                
        if seed_idx == -1:
            return [seed_phrase] 
            
        current_node = self.memory.memory[seed_idx]
        sequence.append(current_node.phrase)
        
        # 2. Select a Template
        # We need a template that STARTS with the seed's POS
        # Or we just pick a template and try to fit.
        # Better: Pick a template, and if seed fits slot 0, good. If not, maybe seed is the subject?
        # For simplicity V1: Just try to follow *any* valid next step from a template?
        # No, we want structure.
        # Let's pick a template.
        template = self.syntax.get_template()
        
        # If seed POS doesn't match template[0], we might be in trouble.
        # But let's assume we are generating *from* the seed.
        # Actually, usually we want to generate a sentence *about* the seed.
        # So the seed might be the Subject (Noun).
        # If template is [Det, Adj, Noun, Verb], and seed is Noun, we might want to fill the Noun slot?
        # That's complex.
        # Simplified V1: Just generate a chain that *follows* the seed, using the SyntaxLearner's valid next states.
        # But SyntaxLearner stores whole templates.
        # Let's try to find a template that starts with seed.pos.
        
        matching_templates = [t for t in self.syntax.templates if t and t[0] == current_node.pos]
        if matching_templates:
            active_template = random.choice(matching_templates)
        else:
            # Fallback: Just use a generic one and ignore mismatch for first word?
            # Or just use the old logic but guided by *any* template segment?
            active_template = ["noun", "verb", "noun"] # Default
            
        # We are at index 0 of the template (the seed).
        # We need to generate index 1, 2, ...
        
        for i in range(1, len(active_template)):
            target_pos = active_template[i]
            
            # 3. Find Candidates (Spreading Activation + Bias)
            candidates = []
            for target_id, (weight, _) in current_node.edges.items():
                target_idx = -1
                target_node = None
                
                for k, n in enumerate(self.memory.memory):
                    if n.node_id == target_id:
                        target_node = n
                        target_idx = k
                        break
                
                # STRICT CONSTRAINT: Must match target_pos
                # (Relaxed for 'unknown')
                if target_node:
                    match = (target_node.pos == target_pos) or (target_node.pos == "unknown") or (target_pos == "unknown")
                    if match:
                        # Apply Bias
                        bias = 1.0
                        if activation_map and target_idx in activation_map:
                            bias = 1.0 + (activation_map[target_idx] * 20.0)
                        candidates.append((target_node, weight * bias))
            
            # 4. Select Next Node
            if candidates:
                total_weight = sum(w for _, w in candidates)
                r = random.uniform(0, total_weight)
                upto = 0
                next_node = candidates[0][0]
                for node, w in candidates:
                    if upto + w >= r:
                        next_node = node
                        break
                    upto += w
                    
                # Strip polysemous suffix for output (e.g., bank_2 -> bank)
                clean_phrase = next_node.phrase.split('_')[0]
                sequence.append(clean_phrase)
                current_node = next_node
            else:
                break
                
        return sequence

    def _simulate_wait_for_theta_peak(self) -> float:
        """
        Simulates the passage of time until the Theta wave hits a peak.
        Returns the simulated time elapsed.
        """
        elapsed = 0.0
        max_wait = 1.0 
        
        theta_band = self.drift_tracker.bands["theta"]
        
        while elapsed < max_wait:
            theta_val = theta_band.value()
            if theta_val > self.theta_threshold:
                return elapsed
            
            self.drift_tracker.update_time(dt=0.05)
            elapsed += 0.05
            
        return elapsed

class BeamSearchSequencer(MelodicSequencer):
    """
    Advanced sequencer using Beam Search to find the most coherent path.
    Optimizes for:
    1. Grammatical Correctness (Template Matching)
    2. Semantic Association (Edge Weights)
    3. Contextual Relevance (Activation Map)
    """
    def __init__(self, memory_index: RCoreMemoryIndex, drift_tracker: DriftTracker, beam_width: int = 5):
        super().__init__(memory_index, drift_tracker)
        self.beam_width = beam_width
        
    def generate_sequence(self, seed_phrase: str, length: int = 10, activation_map: Dict[int, float] = None) -> List[str]:
        """
        Generates a sequence using Beam Search.
        """
        # 1. Find Seed Node
        seed_idx = -1
        for i, node in enumerate(self.memory.memory):
            if node.phrase == seed_phrase:
                seed_idx = i
                break
                
        if seed_idx == -1:
            return [seed_phrase]
            
        current_node = self.memory.memory[seed_idx]
        
        # Beam State: (score, sequence_of_nodes, current_template)
        # We need to pick a template first.
        # Actually, beam search can explore *different templates* too?
        # For simplicity, let's pick one template for now, or try to fit multiple.
        # Let's stick to the "Active Template" idea but allow deviation if score is high.
        
        template = self.syntax.get_template()
        # Try to align template start with seed
        if template[0] != current_node.pos and len(template) > 1:
             # If mismatch, maybe just ignore template for first word?
             pass
             
        # Initial Beam: [(0.0, [current_node])]
        beam = [(0.0, [current_node])]
        
        for step in range(length - 1):
            candidates = []
            
            for score, path in beam:
                last_node = path[-1]
                path_len = len(path)
                
                # Expected POS from template
                expected_pos = template[path_len] if path_len < len(template) else None
                
                # Expand neighbors
                if not last_node.edges:
                    # Dead end, keep as is
                    candidates.append((score, path))
                    continue
                    
                for target_id, (weight, _) in last_node.edges.items():
                    # Resolve target node
                    target_node = None
                    for k, n in enumerate(self.memory.memory):
                        if n.node_id == target_id:
                            target_node = n
                            break
                    
                    if not target_node: continue

                    # Calculate Step Score
                    step_score = weight * 10.0 # Base edge weight
                    
                    # Template Bonus
                    if expected_pos and target_node.pos == expected_pos:
                        step_score += 2.0
                        
                    # Activation Bonus (Context)
                    if activation_map and target_id in activation_map:
                        step_score += activation_map[target_id] * 5.0
                        
                    # Repetition Penalty
                    if any(n.node_id == target_node.node_id for n in path):
                        step_score -= 10.0 # Strong penalty for immediate repetition
                        
                    new_path = path + [target_node]
                    new_score = score + step_score
                    candidates.append((new_score, new_path))
            
            # Prune to beam width
            # Sort by score descending
            candidates.sort(key=lambda x: x[0], reverse=True)
            beam = candidates[:self.beam_width]
            
            # Stop if all paths are stuck?
            if not beam: break
            
        # Return best path
        if not beam:
            return [seed_phrase]
            
        best_path = beam[0][1]
        return [n.phrase.split('_')[0] for n in best_path]
