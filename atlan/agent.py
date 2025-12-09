import random
import time
import json
import re
from typing import Dict, List, Tuple, Any, Optional, Union

# Import components from other modules within the package
from .memory import RCoreMemoryIndex, SymbolicNode
from .perception import ToneEngine, DriftTracker
from .reasoning import BeliefSystem, DreamspaceSimulator
from .reflection import SelfAnchor
from .narrative import NarrativeNode
from .generator import MelodicSequencer, BeamSearchSequencer
from .utils import _enhanced_vector
from .learning import TextReader
from .persona import PersonaManager

class AtlanAgent:
    """Orchestrates the components of the Atlan symbolic cognitive engine."""
    def __init__(self, agent_id: str = "atlan_v1", config: Optional[Dict[str, Any]] = None):
        """Initializes the agent and its cognitive components."""
        self.agent_id = agent_id
        self._config = config if config else {}
        self._creation_time = time.time()
        
        # Memory configuration
        memory_config = self._config.get('memory', {})
        max_memory = memory_config.get('max_size', 1000)
        
        # Initialize components (allow passing custom configs/instances later)
        self.memory = RCoreMemoryIndex(
            max_memory_size=max_memory,
            config=memory_config
        )
        
        self.tone_engine = ToneEngine()
        self.drift_tracker = DriftTracker(max_window=self._config.get('drift_window', 30))
        
        # Narrative tracking
        self.narrator = NarrativeNode(
            entity_id=agent_id,
            drift_tracker=self.drift_tracker,
            tone_engine=self.tone_engine
        )
        self.last_narrative: Optional[Dict[str, Any]] = None
        self.projected_outcomes: Dict[str, float] = {} # Store projections for actions

        # Reasoning components
        self.belief_system = BeliefSystem(initial_beliefs=self._config.get('initial_beliefs'))
        self.dreamspace = DreamspaceSimulator(action_effects=self._config.get('action_effects'))
        self.anchor = SelfAnchor(self.belief_system, self.drift_tracker)
        
        # Generator (Atlan 6.0: Beam Search)
        self.sequencer = BeamSearchSequencer(self.memory, self.drift_tracker, beam_width=5)
        
        # Simulator (The Dreamspace)
        self.dream_simulator = DreamspaceSimulator()
        
        # Learning Engine (The Reader)
        # Pass the SyntaxLearner from the sequencer so it can learn templates
        self.reader = TextReader(self.memory, self.sequencer.syntax)
        
        # Persona Engine (Atlan 5.0)
        self.persona_manager = PersonaManager()
        
        # State tracking
        self.last_input_phrase: Optional[str] = None
        self.last_chosen_action: Optional[str] = None
        self.last_reflection: Optional[Dict[str, Any]] = None

    def read_text(self, text: str):
        """
        Ingests a text stream to build memory graph autonomously.
        """
        self.reader.read(text)

    def ingest_file(self, filepath: str):
        """
        Reads and ingests a text file.
        """
        print(f"Ingesting {filepath}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            self.read_text(text)
            print(f"Successfully ingested {len(text)} characters.")
        except Exception as e:
            print(f"Error ingesting file: {e}")
        
    def process_input(self, phrase: str, label: Optional[str] = None) -> Dict[str, Any]:
        """
        Processes a new input phrase: encodes, perceives mood, updates memory.
        
        Args:
            phrase: The input text to process
            label: Optional category label for the memory
            
        Returns:
            Dictionary with processing results
        """
        self.last_input_phrase = phrase
        
        # 1. Encode input
        vector = _enhanced_vector(phrase)
        
        # 2. Perceive Mood & Update Drift
        mood_score = self.tone_engine.score_phrase(phrase)
        self.drift_tracker.add_mood(mood_score)
        
        # 3. Memory Interaction (Search/Add/Reinforce)
        search_results = self.memory.search(vector, top_k=1)
        
        match_threshold = self._config.get('match_threshold', 0.95)
        
        match_info = {"status": "no_match", "reinforced_index": None, "score": None}
        
        if search_results and search_results[0][1] >= match_threshold:
            match_idx = search_results[0][0]
            self.memory.reinforce_by_index(match_idx)
            match_info = {"status": "reinforced", "reinforced_index": match_idx, "score": search_results[0][1]}
        else:
            # Pass mood_score to generate appropriate ConceptChord
            new_idx = self.memory.add(phrase=phrase, vector=vector, label=label, sentiment_score=mood_score)
            match_info = {"status": "added", "added_index": new_idx}
            
        # 4. Hebbian Learning (Plasticity)
        # Break input into words and wire them together
        # Simple tokenizer: split by space, strip punctuation
        words = [w.strip(".,!?").lower() for w in phrase.split()]
        if len(words) > 1:
            # High learning rate for direct conversation to ensure immediate recall
            self.memory.learn_sequence(words, learning_rate=0.8)
            
        # 5. Potentially trigger narrative update (not every time)
        if random.random() < 0.2: # 20% chance to update narrative
            self.last_narrative = self.narrator.build_narrative()
            
        return {
            "input_phrase": phrase,
            "mood_score": mood_score,
            "vector": vector,
            "memory_interaction": match_info,
            "drift": self.drift_tracker.drift(),
            "personality": self.narrator.evolve_personality() if self.last_narrative else []
        }

    def choose_action(self) -> str:
        """
        Uses Dreamspace to simulate and choose the best action based on current state.
        
        Returns:
            The chosen action string
        """
        current_drift = self.drift_tracker.drift()
        
        # Calculate average reinforcement from recent memories
        recent_memories = min(5, len(self.memory.memory))
        if recent_memories > 0:
            avg_reinforcement = sum(node.get("reinforcement", 1.0) for node in self.memory.memory[-recent_memories:]) / recent_memories
        else:
            avg_reinforcement = 1.0
            
        chosen_action, projected_resonance, projections = self.dreamspace.simulate_resonance_projection(
            current_drift, avg_reinforcement
        )
        
        # Store all projections for future reference
        self.projected_outcomes = {p["action"]: p["projected_resonance"] for p in projections}
        
        self.last_chosen_action = chosen_action
        return chosen_action

    def record_outcome(self, action: str, actual_outcome: float) -> Dict[str, Any]:
        """
        Records the actual outcome of an action and updates beliefs via SelfAnchor.
        
        Args:
            action: The action that was taken
            actual_outcome: The observed outcome value (-1.0 to 1.0)
            
        Returns:
            Dictionary with feedback results
        """
        # Find the projected outcome for the action taken
        projected_outcome = self.projected_outcomes.get(
            action,
            self.dreamspace.action_templates.get(action, 0.0) # Fallback if no projection stored
        )
        
        # Get feedback signal (reinforce/contradict/neutral)
        feedback_signal, alignment = self.dreamspace.learn_from_outcome(projected_outcome, actual_outcome)
        
        # Record feedback and trigger belief updates
        self.anchor.record_action_feedback(action, feedback_signal, alignment)
        
        return {
            "action": action,
            "feedback": feedback_signal,
            "alignment": alignment,
            "projected": projected_outcome,
            "actual": actual_outcome
        }

    def reflect(self) -> Dict[str, Any]:
        """
        Triggers a self-reflection cycle to revise beliefs and analyze state.
        
        Returns:
            Reflection results
        """
        # First, update narrative if it hasn't been updated recently
        if not self.last_narrative or (time.time() - self.last_narrative.get("timestamp", 0)) > 3600:
            self.last_narrative = self.narrator.build_narrative()
            
        # Then perform belief reflection
        self.last_reflection = self.anchor.reflect()
        
        # Add narrative context to reflection
        if self.last_reflection and self.last_narrative:
            self.last_reflection["narrative_context"] = {
                "personality": self.narrator.evolve_personality(),
                "avg_mood": self.last_narrative["avg_mood"],
                "mood_label": self.last_narrative["label"]
            }
        return self.last_reflection

    def generate_thought(self, seed_phrase: str, length: int = 5) -> str:
        """
        Generates a thought sequence using the Melodic Sequencer.
        
        Args:
            seed_phrase: Starting context
            length: Number of words to generate
            
        Returns:
            Generated string
        """
        sequence = self.sequencer.generate_sequence(seed_phrase, length)
        return " ".join(sequence)

    def get_state(self, include_memory_sample: int = 0) -> Dict[str, Any]:
        """
        Returns a snapshot of the agent's current internal state.
        
        Args:
            include_memory_sample: Number of recent memory nodes to include (0 for none)
            
        Returns:
            Dictionary with agent state
        """
        state = {
            "agent_id": self.agent_id,
            "uptime": time.time() - self._creation_time,
            "drift_state": self.drift_tracker.get_state(),
            "beliefs": self.belief_system.get_state(),
            "memory_stats": self.memory.get_state(), # Now returns full state
            "narrative_state": self.narrator.get_state(),
            "anchor_state": self.anchor.get_state(),
            "projected_outcomes": self.projected_outcomes,
            "config": self._config
        }
        
        if include_memory_sample > 0:
            sample_size = min(include_memory_sample, len(self.memory.memory))
            # Get a sample of the most recently added memory nodes
            state["memory_sample"] = self.memory.memory[-sample_size:]
            
        return state

    def get_response_template(self) -> str:
        """
        Gets a tone-appropriate response template based on current mood state.
        
        Returns:
            Response template string
        """
        avg_mood = self.drift_tracker.avg_mood()
        tone = self.tone_engine.infer_tone(avg_mood)
        return self.tone_engine.get_response_template(tone)

    def chat(self, query: str) -> str:
        """
        Answers a natural language query using the Graph-RAG mechanism.
        Supports Persona+ commands.
        """
        # 0. Check for Persona Commands
        system_response = self.persona_manager.handle_input(query)
        if system_response:
            return system_response
            
        # 1. Active Learning (Atlan 6.0)
        # Learn from the user's input before responding
        # But don't learn the query itself as a fact if it's a question?
        # E.g. "Is the sky green?" -> We shouldn't learn "sky is green".
        # Simple heuristic: If it ends with '?', treat as query only.
        # If it ends with '.', treat as fact.
        if not query.strip().endswith("?"):
             self.process_input(query)
            
        # Apply Persona Context
        effective_query = self.persona_manager.modify_context(query)
        print(f"Thinking about: '{effective_query}'...")
        
        # 2. Parse Query
        words = re.findall(r'\b\w+\b', effective_query.lower())
        # Filter out stop words AND question words for retrieval/seeding
        question_words = {"what", "who", "where", "when", "why", "how", "is", "are", "do", "does"}
        keywords = [w for w in words if w not in self.reader.stop_words and w not in question_words]
        
        if not keywords:
            # Fallback to all words if nothing left
            keywords = [w for w in words if w not in self.reader.stop_words]
            
        if not keywords:
            return "I have no thoughts on that."
            
        # 3. Activate Subgraph (Retrieval)
        # Find indices
        query_indices = []
        for w in keywords:
            if w in self.memory.phrase_to_index:
                query_indices.append(self.memory.phrase_to_index[w])
                
        # Add Working Memory (Last Input) to activation
        if self.last_input_phrase:
            last_words = re.findall(r'\b\w+\b', self.last_input_phrase.lower())
            last_keywords = [w for w in last_words if w not in self.reader.stop_words]
            for w in last_keywords:
                if w in self.memory.phrase_to_index:
                    query_indices.append(self.memory.phrase_to_index[w])
                
        if not query_indices:
            # Fallback: Try to find any word in the query
            for w in words:
                if w in self.memory.phrase_to_index:
                    query_indices.append(self.memory.phrase_to_index[w])
            
            if not query_indices:
                return f"I don't know about {' '.join(keywords)}."
            
        # Get Activation Map (Energy values)
        activated_nodes = self.memory.get_activated_subgraph(query_indices, depth=2)
        activation_map = {}
        for idx, spectrum in activated_nodes:
            # Sum energy across bands
            total_energy = sum(spectrum.values())
            activation_map[idx] = total_energy
            
        # 4. Generate Response (Beam Search)
        # Seed with the LAST keyword of the CURRENT query
        # (Working memory shouldn't dictate the seed, just the path)
        seed_phrase = keywords[-1] if keywords else words[-1]
        
        # Generate
        sequence = self.sequencer.generate_sequence(seed_phrase, length=10, activation_map=activation_map)
        
        # Format
        response = " ".join(sequence).capitalize() + "."
        return response

    def save_state(self, filepath: str) -> bool:
        """
        Saves the agent's current state to a JSON file.
        
        Args:
            filepath: Path to save the state file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            state = self.get_state()
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving state: {e}")
            return False

    def load_state(self, filepath: str) -> bool:
        """
        Loads the agent's state from a JSON file.
        
        Args:
            filepath: Path to load the state file from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
                
            # Restore components
            if "drift_state" in state:
                self.drift_tracker.load_state(state["drift_state"])
                
            if "beliefs" in state:
                self.belief_system.load_state(state["beliefs"])
                
            if "memory_stats" in state and "memory" in state["memory_stats"]:
                # This is tricky because RCoreMemoryIndex needs to reconstruct objects
                # For now, we rely on the memory component's load logic if implemented
                self.memory.load_state(state["memory_stats"])
                
            if "narrative_state" in state:
                self.narrator.load_state(state["narrative_state"])
                
            if "anchor_state" in state:
                self.anchor.load_state(state["anchor_state"])
                
            return True
        except Exception as e:
            print(f"Error loading state: {e}")
            return False
