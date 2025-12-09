import re
from typing import List, Dict, Optional, Tuple, Any
from atlan.memory import RCoreMemoryIndex
from atlan.utils import _enhanced_vector
from atlan.utils import _enhanced_vector
from atlan.similarity import cosine_similarity, clustering_similarity

class TextReader:
    """
    The 'Eye' of the agent. Reads raw text and builds the memory graph
    through unsupervised learning (co-occurrence mapping).
    Now includes Context Engine for Polysemy.
    """
    def __init__(self, memory_index: RCoreMemoryIndex, syntax_learner: Optional[Any] = None, window_size: int = 3):
        self.memory = memory_index
        self.syntax_learner = syntax_learner
        self.window_size = window_size
        self.stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "of"}
        self.context_window: List[List[float]] = [] # Stores vectors of recent words

    def read(self, text: str):
        """
        Ingests text, creating nodes and connections on the fly.
        """
        # 1. Tokenize (Simple regex for now)
        # Keep words, ignore punctuation
        tokens = re.findall(r'\b\w+\b', text)
        
        print(f"Reading {len(tokens)} tokens...")
        
        # Sliding Context Window for Association (Indices)
        recent_indices: List[int] = []
        
        # Sliding POS Window for Syntax Learning
        pos_window: List[str] = []
        
        count = 0
        for word in tokens:
            count += 1
            if count % 1000 == 0:
                print(f"Processed {count}/{len(tokens)} tokens...")
                
            if not word.strip(): continue
            
            # 1. Update Context Vector (Semantic Mood)
            word_vector = _enhanced_vector(word)
            self.context_window.append(word_vector)
            if len(self.context_window) > 5: # Keep context window slightly larger than association window
                self.context_window.pop(0)
                
            current_context = self._get_current_context_vector()
            
            # 2. Assimilate (Get or Create Node with Sense Disambiguation)
            current_idx = self._assimilate(word, current_context)
            
            # Get Node POS for Syntax Learning
            current_node = self.memory.memory[current_idx]
            pos_window.append(current_node.pos)
            
            # Learn Syntax Templates
            if self.syntax_learner and len(pos_window) >= 3:
                # Learn 3-grams and 4-grams
                self.syntax_learner.learn_template(pos_window[-3:])
                if len(pos_window) >= 4:
                    self.syntax_learner.learn_template(pos_window[-4:])
                    
            if len(pos_window) > 5:
                pos_window.pop(0)
            
            # 3. Associate (Wire to Context)
            self._associate(current_idx, recent_indices)
            
            # Update Association Window
            recent_indices.append(current_idx)
            if len(recent_indices) > self.window_size:
                recent_indices.pop(0)

    def _get_current_context_vector(self) -> List[float]:
        """Calculates the average vector of the current context window."""
        if not self.context_window:
            return [0.0] * 384
        
        # Simple average
        avg_vector = [0.0] * 384
        for vec in self.context_window:
            for i, val in enumerate(vec):
                avg_vector[i] += val
        
        count = len(self.context_window)
        return [v / count for v in avg_vector]

    def _assimilate(self, word: str, context_vector: List[float]) -> int:
        """
        Converts a word into a memory node index, handling Polysemy.
        If the word exists but in a different context, creates a new 'Sense Node'.
        """
        # 1. Find all candidates matching the word
        # OPTIMIZED: Use phrase_to_index for exact match first
        candidates = []
        
        # Check exact match
        if word in self.memory.phrase_to_index:
            candidates.append(self.memory.phrase_to_index[word])
            
        # Check for polysemous variants (word_1, word_2...)
        # This part is still potentially slow if we scan everything.
        # But usually we only have a few variants.
        # We can optimize this by storing a list of indices for a base word in memory?
        # For now, let's just check a reasonable number of suffixes: word_1 to word_10
        for i in range(1, 11):
            variant = f"{word}_{i}"
            if variant in self.memory.phrase_to_index:
                candidates.append(self.memory.phrase_to_index[variant])
        
        # 2. Sense Disambiguation
        
        # Optimization: Skip polysemy for stopwords
        if word in self.stop_words:
            if candidates:
                # Just return the first one (usually the base word)
                return candidates[0]
        
        best_idx = -1
        best_similarity = -1.0
        
        # Optimization: Limit candidate check to top 5
        for idx in candidates[:5]:
            node = self.memory.memory[idx]
            # Compare current context with node's stored context
            sim = cosine_similarity(context_vector, node.context_vector)
            if sim > best_similarity:
                best_similarity = sim
                best_idx = idx
                
        # Threshold for "Same Sense"
        # If similarity > 0.7, it's the same meaning.
        # If < 0.7, it's a new meaning (Polysemy).
        
        if best_idx != -1 and best_similarity > 0.7:
            # Reinforce existing sense
            self.memory.reinforce_by_index(best_idx)
            return best_idx
        else:
            # Create New Sense Node
            # If it's a new sense of an existing word, append suffix
            phrase = word
            if candidates:
                phrase = f"{word}_{len(candidates) + 1}" # e.g., bank_2
                
            # Simple POS Dictionary for Prototype
            pos_map = {
                "the": "determiner", "a": "determiner", "an": "determiner",
                "this": "determiner", "that": "determiner", "my": "determiner", "her": "determiner", "his": "determiner",
                "moon": "noun", "sea": "noun", "dark": "adjective", "shines": "verb",
                "fire": "noun", "burns": "verb", "dry": "adjective", "wood": "noun",
                "deep": "adjective", "is": "verb", "and": "conjunction", "on": "preposition",
                "bank": "noun", "river": "noun", "money": "noun", "deposit": "verb", "water": "noun",
                "fish": "noun", "swim": "verb", "safe": "adjective", "flows": "verb",
                "alice": "noun", "rabbit": "noun", "cat": "noun", "queen": "noun", "king": "noun",
                "white": "adjective", "pink": "adjective", "late": "adjective", "dear": "adjective",
                "run": "verb", "ran": "verb", "eat": "verb", "drink": "verb", "said": "verb", "thought": "verb",
                "was": "verb", "had": "verb", "went": "verb", "came": "verb",
                "in": "preposition", "at": "preposition", "to": "preposition", "of": "preposition", "with": "preposition", "by": "preposition",
                "she": "pronoun", "he": "pronoun", "it": "pronoun", "i": "pronoun", "you": "pronoun", "me": "pronoun"
            }
            
            pos = pos_map.get(word, "unknown")
            
            # Fallback Heuristics
            if pos == "unknown":
                if word.endswith("ing") or word.endswith("s"): pos = "verb" 
                elif word.endswith("ly"): pos = "adverb"
                elif word.endswith("ed"): pos = "verb"
            
            vector = _enhanced_vector(word)
            
            new_idx = self.memory.add(
                phrase=phrase,
                vector=vector,
                label=pos,
                pos=pos,
                timbre="sine",
                context_vector=context_vector # Store the context!
            )
            return new_idx

    def _associate(self, current_idx: int, context_indices: List[int]):
        """
        Connects the current node to nodes in the context buffer.
        """
        # Distance 1 (Immediate predecessor) is at end of list
        # Distance N is at start
        
        for i, past_idx in enumerate(reversed(context_indices)):
            distance = i + 1
            if distance > self.window_size: break
            
            # Weight decays with distance
            weight = 0.5 / distance
            
            # Bands: Theta (Sequence), Beta (Logic)
            bands = ["theta", "beta"]
            
            past_node = self.memory.memory[past_idx]
            current_node = self.memory.memory[current_idx]
            
            # Past -> Current (Sequence)
            if current_node.node_id in past_node.edges:
                past_node.strengthen_connection(current_node.node_id, amount=0.1, max_weight=1.0)
            else:
                past_node.connect(current_node.node_id, weight, bands)
                
            # Causal Hebbian Learning (Time-Asymmetric)
            # Strengthen P(Current | Past)
            # We treat the causal edge as a separate directed weight
            if current_node.node_id in past_node.causal_edges:
                past_node.causal_edges[current_node.node_id] = min(past_node.causal_edges[current_node.node_id] + 0.1, 1.0)
            else:
                past_node.causal_edges[current_node.node_id] = 0.1

    def optimize_memory(self, similarity_threshold: float = 0.85, prune_threshold: float = 0.1):
        """
        Consolidates memory by clustering similar nodes and pruning weak connections.
        This solves the "Graph Explosion" problem.
        """
        print("Optimizing Memory (Clustering & Pruning)...")
        initial_nodes = len([n for n in self.memory.memory if n.phrase != "DELETED"])
        
        # 1. Semantic Clustering
        # Find nodes with high vector similarity and merge them
        # Naive O(N^2) approach for prototype - needs optimization for production
        # We can limit the scan to recent nodes or use the index
        
        # Let's just scan the last 1000 nodes against the whole graph for now to save time
        scan_limit = 1000
        start_idx = max(0, len(self.memory.memory) - scan_limit)
        
        merged_count = 0
        
        for i in range(start_idx, len(self.memory.memory)):
            node_a = self.memory.memory[i]
            if node_a.phrase == "DELETED": continue
            
            # Check against potential duplicates
            # Optimization: Only check nodes with same phrase first (Polysemy collapse)
            # Then check vector similarity
            
            # 1. Collapse Polysemy (Same phrase, similar context)
            if node_a.phrase in self.memory.phrase_to_index:
                # This logic is tricky because phrase_to_index points to ONE node.
                # We need to find ALL nodes with this phrase.
                # For now, let's rely on vector similarity.
                pass
                
            # 2. Vector Similarity Scan
            # We look for nodes that are virtually identical
            # FIX: Use clustering_similarity (pure vector) instead of search (resonance/popularity)
            # We can't use self.memory.search() because it uses resonance.
            # We must manually scan or use a vector-only search if available.
            # For prototype, we'll manually scan the candidates returned by search but re-score them.
            
            candidates = self.memory.search(node_a.vector, top_k=10) # Get more candidates
            
            for idx_b, _ in candidates: # Ignore search score
                if idx_b == i: continue # Skip self
                
                node_b = self.memory.memory[idx_b]
                if node_b.phrase == "DELETED": continue
                
                # Re-calculate score using PURE vector similarity
                score = clustering_similarity(node_a.vector, node_b.vector)
                
                # If similarity is very high, merge A into B (or vice versa)
                # We prefer keeping the older/stronger node
                if score > similarity_threshold:
                    # Merge A into B
                    print(f"Merging '{node_a.phrase}' ({i}) into '{node_b.phrase}' ({idx_b}) (Score: {score:.2f})")
                    self.memory.merge_nodes(i, idx_b)
                    merged_count += 1
                    break # Node A is gone, move to next
                    
        # 2. Prune Graph
        removed_edges = self.memory.prune_graph(edge_threshold=prune_threshold)
        
        final_nodes = len([n for n in self.memory.memory if n.phrase != "DELETED"])
        print(f"Optimization Complete. Nodes: {initial_nodes} -> {final_nodes} (Merged {merged_count}). Edges Pruned: {removed_edges}")

