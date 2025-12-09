import time
from typing import List, Tuple, Dict, Any

# 3. ToneEngine
class ToneEngine:
    """Scores phrase sentiment based on keywords and basic negation handling."""
    def __init__(self):
        # Simple keyword mapping for sentiment scoring
        self.affect_map = {
            # Positive terms
            "great": 0.6, "scheduled": 0.5, "ready": 0.5, "satisfaction": 0.7,
            "happy": 0.9, "hopeful": 0.7, "okay": 0.4, "neutral": 0.0,
            "grateful": 0.8, "energized": 0.6, "pleased": 0.7, "thanked": 0.7,
            "resolved": 0.6, "success": 0.8, "appreciate": 0.7, "satisfied": 0.6,
            "premium": 0.5, "interest": 0.4, "quick": 0.5, "renewal": 0.3,
            # Negative terms
            "missed": -0.4, "no reply": -0.6, "upset": -0.5, "unresolved": -0.7,
            "delayed": -0.6, "tired": -0.3, "anxious": -0.6, "angry": -0.8,
            "sad": -0.7, "frustrated": -0.6, "issue": -0.4, "complaint": -0.7,
            "urgent": -0.4, "problem": -0.5, "misunderstanding": -0.3, "billing": -0.2,
            # Neutral/context terms
            "help": 0.1, "followed up": 0.2, "no change": 0.0, "appointment": 0.1,
            "voicemail": -0.1, "meeting": 0.2, "call": 0.0, "service": 0.2
        }
        self.negation_words = {"not", "no", "didn't", "wasn't", "never", "none"}
        
        # Response templates for different emotional tones
        self.response_templates = {
            "uplifting": "You're doing great — keep the momentum going!",
            "encouraging": "I see progress — want to build on that?",
            "neutral": "I'm here if you want to explore more.",
            "gentle": "That sounds like a lot. Take your time.",
            "soothing": "Let's slow down and focus on one thing together."
        }

    def score_phrase(self, phrase: str) -> float:
        """Calculates a sentiment score for the input phrase."""
        score = 0.0
        words = phrase.lower().split()
        word_set = set(words)
        negation_present = bool(self.negation_words.intersection(word_set))
        
        for keyword, value in self.affect_map.items():
            # Check if keyword is present (can be improved with stemming/lemmatization)
            if keyword in phrase.lower(): # Simple substring check
                score += value
                
        # Apply basic negation: flip score if negation words are nearby
        # (This is very rudimentary sentiment analysis)
        if negation_present and score != 0:
            score *= -0.5 # Dampen and potentially flip score
            
        # Ensure score stays within a reasonable range, e.g., -1 to 1
        return round(max(-1.0, min(1.0, score)), 2)

    def infer_tone(self, mood_score: float) -> str:
        """
        Classifies tone based on average mood value.
        
        Args:
            mood_score: Mood value between -1.0 and 1.0
            
        Returns:
            Tone classification string
        """
        if mood_score > 0.5:
            return "uplifting"
        elif mood_score > 0.1:
            return "encouraging"
        elif mood_score < -0.5:
            return "soothing"
        elif mood_score < -0.1:
            return "gentle"
        else:
            return "neutral"

    def get_response_template(self, tone: str) -> str:
        """
        Returns a tone-calibrated response template.
        
        Args:
            tone: The tone classification
            
        Returns:
            Response template string
        """
        return self.response_templates.get(tone, self.response_templates["neutral"])

# 4. DriftTracker
import math

# 4. Multi-Band Oscillator (Replaces simple DriftTracker)
class Oscillator:
    """Represents a single frequency band oscillator."""
    def __init__(self, name: str, frequency: float, phase: float = 0.0):
        self.name = name
        self.frequency = frequency # Hz
        self.phase = phase
        self.amplitude = 0.0
        self.baseline_amplitude = 0.1 # Always some background noise
        self.decay_rate = 0.95
        
    def update(self, energy_input: float = 0.0, dt: float = 0.1):
        """
        Updates oscillator state.
        
        Args:
            energy_input: New energy injected into this band
            dt: Time step in seconds
        """
        # Decay current amplitude towards baseline
        self.amplitude = (self.amplitude * self.decay_rate) + (self.baseline_amplitude * (1 - self.decay_rate))
        
        # Add new energy (boost amplitude)
        self.amplitude += energy_input
        self.amplitude = min(1.0, self.amplitude) # Cap at 1.0
        
        # Advance phase
        self.phase += 2 * math.pi * self.frequency * dt
        self.phase %= (2 * math.pi)
        
    def value(self) -> float:
        """Returns current wave value (-amp to +amp)."""
        return self.amplitude * math.sin(self.phase)

class DriftTracker:
    """
    Tracks mood history using multi-band oscillators (Brainwave simulation).
    Maintains backward compatibility with simple drift/avg methods.
    """
    def __init__(self, max_window: int = 30):
        self.max_window = max_window
        self.mood_history: List[Tuple[float, float]] = []
        
        # Initialize Brainwave Bands
        # Frequencies are relative simulation values, not exact biological Hz
        self.bands = {
            "delta": Oscillator("Delta", 0.5),  # 0.5 Hz - Deep unconscious
            "theta": Oscillator("Theta", 2.0),  # 2 Hz - Emotion/Memory
            "alpha": Oscillator("Alpha", 5.0),  # 5 Hz - Idle/Reflection
            "beta":  Oscillator("Beta",  12.0), # 12 Hz - Active thought
            "gamma": Oscillator("Gamma", 30.0)  # 30 Hz - Insight/Peak
        }
        self.last_update_time = time.time()

    def add_mood(self, mood_score: float):
        """
        Adds a mood score and injects energy into relevant bands.
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # 1. Standard History Tracking (Legacy)
        if not isinstance(mood_score, (int, float)):
            return
        mood_score = float(mood_score)
        self.mood_history.append((current_time, mood_score))
        if len(self.mood_history) > self.max_window:
            self.mood_history.pop(0)
            
        # 2. Wave Dynamics Update
        # Distribute energy based on mood intensity and type
        abs_score = abs(mood_score)
        
        # Default energy inputs
        inputs = {b: 0.0 for b in self.bands}
        
        if abs_score > 0.8:
            # Intense mood -> Gamma (Insight/Shock) & Theta (Deep Emotion)
            inputs["gamma"] = 0.4
            inputs["theta"] = 0.3
        elif abs_score > 0.4:
            # Moderate mood -> Beta (Active processing)
            inputs["beta"] = 0.3
        else:
            # Low/Neutral -> Alpha (Idle)
            inputs["alpha"] = 0.2
            
        # Update all bands
        for name, band in self.bands.items():
            band.update(energy_input=inputs[name], dt=dt)

    def update_time(self, dt: float = 0.1):
        """Manually advance time for oscillators (for simulation loops)."""
        for band in self.bands.values():
            band.update(dt=dt)

    def get_wave_state(self) -> Dict[str, float]:
        """Returns current instantaneous value of all bands."""
        return {name: band.value() for name, band in self.bands.items()}

    def avg_mood(self) -> float:
        """Calculates the average mood score in the current window."""
        if not self.mood_history:
            return 0.0
        return round(sum(s for _, s in self.mood_history) / len(self.mood_history), 3)

    def drift(self) -> float:
        """Calculates the difference between the latest and earliest mood score."""
        if len(self.mood_history) < 2:
            return 0.0
        return round(self.mood_history[-1][1] - self.mood_history[0][1], 3)

    def classify_drift(self) -> str:
        """Classifies the current drift direction and magnitude."""
        drift = self.drift()
        if drift > 0.5: return "strong positive"
        elif drift > 0.1: return "mild positive"
        elif drift < -0.5: return "strong negative"
        elif drift < -0.1: return "mild negative"
        else: return "stable"

    def volatility(self) -> float:
        """Calculates the average absolute change between consecutive mood scores."""
        if len(self.mood_history) < 2:
            return 0.0
        diffs_sum = sum(abs(self.mood_history[i][1] - self.mood_history[i-1][1]) for i in range(1, len(self.mood_history)))
        return round(diffs_sum / (len(self.mood_history) - 1), 3)

    def get_state(self) -> Dict[str, Any]:
        """Returns the full state including wave data."""
        return {
            "average_mood": self.avg_mood(),
            "drift": self.drift(),
            "drift_classification": self.classify_drift(),
            "volatility": self.volatility(),
            "history_length": len(self.mood_history),
            "mood_history": self.mood_history,
            "waves": {
                name: {
                    "amp": band.amplitude,
                    "phase": band.phase,
                    "val": band.value()
                } for name, band in self.bands.items()
            }
        }

    def load_state(self, state: Dict[str, Any]):
        """Restores state."""
        if "mood_history" in state:
            self.mood_history = [tuple(x) for x in state["mood_history"]]
            
        # Restore wave state if present
        if "waves" in state:
            for name, data in state["waves"].items():
                if name in self.bands:
                    self.bands[name].amplitude = data["amp"]
                    self.bands[name].phase = data["phase"]

