import math
from typing import List, Optional

# Try to use numpy for vectorized operations (10-100x faster)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Measures directional similarity between two vectors.
    Values closer to 1 mean high structural alignment.
    """
    if HAS_NUMPY:
        a = np.array(v1)
        b = np.array(v2)
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))
    else:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm_a = math.sqrt(sum(a * a for a in v1))
        norm_b = math.sqrt(sum(b * b for b in v2))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

def euclidean_distance(v1: List[float], v2: List[float]) -> float:
    """
    Measures magnitude closeness (proximity).
    Lower values mean more similarity.
    """
    if HAS_NUMPY:
        return float(np.linalg.norm(np.array(v1) - np.array(v2)))
    else:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

def resonance_score(v1: List[float], v2: List[float], reinforcement_weight: float = 0.05, reinforcement_value: float = 1.0) -> float:
    """
    Computes a composite symbolic resonance score using both cosine and Euclidean.
    Can be used to match input against symbolic memory nodes.
    """
    cos_sim = cosine_similarity(v1, v2)
    euc_dist = euclidean_distance(v1, v2)

    # DAMPEN REINFORCEMENT
    boost = 0.0
    if reinforcement_value > 0:
        boost = min(reinforcement_weight * math.log(1 + reinforcement_value), 0.2)

    return round(cos_sim - (0.1 * euc_dist) + boost, 4)

def clustering_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Computes pure vector similarity for clustering/merging.
    Ignores reinforcement to prevent 'Black Hole' nodes (popularity bias).
    """
    return round(cosine_similarity(v1, v2), 4)

# === BATCH OPERATIONS (10-100x faster for large memory) ===

def batch_cosine_similarity(query: List[float], vectors: List[List[float]]) -> List[float]:
    """
    Compute cosine similarity between a query and ALL vectors at once.
    This is the key optimization - vectorized matrix operations.
    """
    if not vectors:
        return []

    if HAS_NUMPY:
        q = np.array(query)
        V = np.array(vectors)

        # Normalize query
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return [0.0] * len(vectors)
        q_normalized = q / q_norm

        # Normalize all vectors at once
        v_norms = np.linalg.norm(V, axis=1, keepdims=True)
        v_norms[v_norms == 0] = 1  # Avoid division by zero
        V_normalized = V / v_norms

        # Batch dot product
        similarities = np.dot(V_normalized, q_normalized)
        return similarities.tolist()
    else:
        # Fallback to sequential
        return [cosine_similarity(query, v) for v in vectors]

def batch_resonance_scores(query: List[float], vectors: List[List[float]],
                          reinforcements: Optional[List[float]] = None,
                          reinforcement_weight: float = 0.05) -> List[float]:
    """
    Compute resonance scores for ALL vectors at once.
    Massive speedup for large memory graphs.
    """
    if not vectors:
        return []

    if HAS_NUMPY:
        q = np.array(query)
        V = np.array(vectors)

        # Batch cosine similarity
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return [0.0] * len(vectors)
        q_normalized = q / q_norm

        v_norms = np.linalg.norm(V, axis=1, keepdims=True)
        v_norms[v_norms == 0] = 1
        V_normalized = V / v_norms

        cos_sims = np.dot(V_normalized, q_normalized)

        # Batch euclidean distance
        euc_dists = np.linalg.norm(V - q, axis=1)

        # Reinforcement boost
        if reinforcements:
            r = np.array(reinforcements)
            boosts = np.minimum(reinforcement_weight * np.log(1 + r), 0.2)
        else:
            boosts = np.zeros(len(vectors))

        scores = cos_sims - (0.1 * euc_dists) + boosts
        return scores.tolist()
    else:
        # Fallback
        if reinforcements is None:
            reinforcements = [1.0] * len(vectors)
        return [
            resonance_score(query, v, reinforcement_weight, r)
            for v, r in zip(vectors, reinforcements)
        ]
