from typing import List
import hashlib
import random
import math

def _get_word_vector(word: str, dim: int) -> List[float]:
    """Generates a deterministic random vector for a single word."""
    # Seed based on word content
    hash_object = hashlib.md5(word.encode())
    seed = int(hash_object.hexdigest(), 16)
    random.seed(seed)
    return [random.random() - 0.5 for _ in range(dim)]

STOP_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there",
    "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time",
    "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than",
    "then", "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work",
    "first", "well", "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us", "is", "are", "was", "were",
    "has", "had", "been", "much", "while", "where", "before", "should", "does", "did", "yes", "through", "during", "between", "might",
    "next", "below", "above", "under", "both", "such", "why", "here", "off", "again", "few", "own", "house", "those", "system", "feature",
    "type", "part", "real", "set", "however", "therefore", "although", "thus", "instead", "yet", "period", "duration"
}

def _enhanced_vector(phrase: str, dim: int = 384) -> List[float]:
    """
    Generates a sentence embedding using Character Trigrams (3-grams).
    Ignores STOP_WORDS to focus on Content Keywords.
    """
    phrase = phrase.lower()
    
    # Remove Stopwords to reduce noise
    import re
    tokens = re.findall(r'\b\w+\b', phrase)
    filtered_tokens = [t for t in tokens if t not in STOP_WORDS]
    
    if filtered_tokens:
        # Reconstruct phrase without stopwords
        phrase = " ".join(filtered_tokens)
        
    # Simple character trigrams
    trigrams = [phrase[i:i+3] for i in range(len(phrase)-2)]
    
    if not trigrams:
        # Fallback for very short words
        return _get_word_vector(phrase, dim)
        
    sentence_vector = [0.0] * dim
    
    for tri in trigrams:
        # We reuse _get_word_vector logic but pass the trigram
        # This gives us a deterministic vector for 'qua', 'uan', etc.
        vec = _get_word_vector(tri, dim)
        for i in range(dim):
            sentence_vector[i] += vec[i]
            
    # Normalize
    norm_sq = 0.0
    for i in range(dim):
        sentence_vector[i] **= 2 # Square it to emphasize peaks (Signal boosting)? 
        # No, just standard vector addition.
        pass
        
    # Re-loop to calculate norm sum (since we didn't just now)
    norm_sq = sum(x*x for x in sentence_vector)
    
    norm = math.sqrt(norm_sq)
    if norm > 0:
        return [x / norm for x in sentence_vector]
    else:
        return [0.0] * dim
