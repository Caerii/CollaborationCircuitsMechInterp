"""
Representation Geometry Analysis

Ported from experiment 02's representation geometry analysis.
Key techniques:
- Cosine similarity between class centroids
- Angular separation (degrees)
- Discriminability ratio (between/within class variance)
- The U-shaped separation curve across layers

Critical finding from exp 5: If probes don't transfer to naturalistic
data, the encoding is LEXICAL (token-based) not SEMANTIC (meaning-based).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression


@dataclass
class GeometryMetrics:
    """Geometry metrics for entity representations."""
    cosine_similarities: Dict[str, float]  # pair -> similarity
    angular_separations: Dict[str, float]  # pair -> degrees
    discriminability_ratio: float
    layer: int
    
    def to_dict(self) -> Dict:
        return {
            "layer": self.layer,
            "cosine_similarities": self.cosine_similarities,
            "angular_separations": self.angular_separations,
            "discriminability_ratio": self.discriminability_ratio,
        }


def compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def compute_angular_separation(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angular separation in degrees."""
    cos_sim = compute_cosine_similarity(v1, v2)
    # Clamp to valid range for arccos
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_sim)))


def compute_discriminability_ratio(
    activations_by_class: Dict[str, np.ndarray]
) -> float:
    """
    Compute discriminability ratio: between-class / within-class variance.
    
    Higher = more separable classes.
    """
    # Compute class centroids
    centroids = {k: np.mean(v, axis=0) for k, v in activations_by_class.items()}
    
    # Global centroid
    all_activations = np.vstack(list(activations_by_class.values()))
    global_centroid = np.mean(all_activations, axis=0)
    
    # Between-class variance
    between_var = 0.0
    for k, centroid in centroids.items():
        n_k = len(activations_by_class[k])
        between_var += n_k * np.sum((centroid - global_centroid) ** 2)
    between_var /= len(all_activations)
    
    # Within-class variance
    within_var = 0.0
    for k, acts in activations_by_class.items():
        centroid = centroids[k]
        within_var += np.sum((acts - centroid) ** 2)
    within_var /= len(all_activations)
    
    if within_var == 0:
        return float('inf')
    
    return float(between_var / within_var)


def analyze_layer_geometry(
    activations_by_class: Dict[str, np.ndarray],
    layer: int
) -> GeometryMetrics:
    """
    Analyze geometry of class representations at a single layer.
    
    Args:
        activations_by_class: Dict mapping class name to activations array
        layer: Layer index
        
    Returns:
        GeometryMetrics
    """
    # Compute centroids
    centroids = {k: np.mean(v, axis=0) for k, v in activations_by_class.items()}
    
    # Pairwise cosine similarities
    class_names = list(centroids.keys())
    cosine_sims = {}
    angular_seps = {}
    
    for i, c1 in enumerate(class_names):
        for c2 in class_names[i+1:]:
            pair = f"{c1}-{c2}"
            cosine_sims[pair] = compute_cosine_similarity(centroids[c1], centroids[c2])
            angular_seps[pair] = compute_angular_separation(centroids[c1], centroids[c2])
    
    # Discriminability
    disc_ratio = compute_discriminability_ratio(activations_by_class)
    
    return GeometryMetrics(
        cosine_similarities=cosine_sims,
        angular_separations=angular_seps,
        discriminability_ratio=disc_ratio,
        layer=layer,
    )


def analyze_geometry_across_layers(
    activations_by_layer: Dict[int, Dict[str, np.ndarray]]
) -> List[GeometryMetrics]:
    """
    Analyze geometry across multiple layers.
    
    Look for the U-shaped curve: classes similar at input,
    diverge in middle layers, partially converge at output.
    
    Args:
        activations_by_layer: Dict[layer -> Dict[class -> activations]]
        
    Returns:
        List of GeometryMetrics per layer
    """
    results = []
    for layer, acts_by_class in sorted(activations_by_layer.items()):
        metrics = analyze_layer_geometry(acts_by_class, layer)
        results.append(metrics)
    return results


def find_peak_separation_layer(geometry_results: List[GeometryMetrics]) -> int:
    """Find layer with maximum class separation (for intervention targeting)."""
    # Use discriminability ratio as the metric
    best_layer = max(geometry_results, key=lambda g: g.discriminability_ratio)
    return best_layer.layer


# ============================================================================
# TRANSFER TEST (Critical control from exp 5)
# ============================================================================

def test_transfer(
    train_activations: Dict[str, np.ndarray],
    test_activations: Dict[str, np.ndarray],
    train_labels: np.ndarray,
    test_labels: np.ndarray
) -> Dict:
    """
    Test if probes transfer from labeled to naturalistic data.
    
    CRITICAL CONTROL: If transfer fails, encoding is LEXICAL not SEMANTIC!
    
    Exp 5 finding: 100% within-domain accuracy but 32% transfer (chance!)
    means the model encodes TOKENS ("User:", "You:") not CONCEPTS.
    
    Args:
        train_activations: Activations from labeled data
        test_activations: Activations from naturalistic data
        train_labels: Labels for training
        test_labels: Labels for testing
        
    Returns:
        Dict with within-domain and transfer accuracies
    """
    # Combine train activations
    X_train = np.vstack(list(train_activations.values()))
    
    # Combine test activations
    X_test = np.vstack(list(test_activations.values()))
    
    # Train probe
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, train_labels)
    
    # Within-domain accuracy
    within_acc = clf.score(X_train, train_labels)
    
    # Transfer accuracy
    transfer_acc = clf.score(X_test, test_labels)
    
    # Gap
    gap = within_acc - transfer_acc
    
    # Interpretation
    is_semantic = transfer_acc > 0.5  # Better than chance
    
    return {
        "within_domain_accuracy": within_acc,
        "transfer_accuracy": transfer_acc,
        "gap": gap,
        "is_semantic": is_semantic,
        "interpretation": (
            f"Encoding is SEMANTIC (transfer={transfer_acc:.1%})"
            if is_semantic else
            f"Encoding is LEXICAL (transfer={transfer_acc:.1%} ~ chance)"
        ),
    }


def summarize_geometry(
    geometry_results: List[GeometryMetrics],
    class_pairs: Optional[List[str]] = None
) -> Dict:
    """
    Summarize geometry analysis results.
    
    Look for:
    - U-shaped curve (separation increases then decreases)
    - Peak separation layer
    - Which class pairs are most/least separable
    """
    if not geometry_results:
        return {}
    
    layers = [g.layer for g in geometry_results]
    
    # Find the pair to track (default to first)
    if class_pairs is None:
        class_pairs = list(geometry_results[0].angular_separations.keys())
    
    # Track separation across layers for each pair
    separation_curves = {}
    for pair in class_pairs:
        separation_curves[pair] = [
            g.angular_separations.get(pair, 0) for g in geometry_results
        ]
    
    # Find peak separation layer
    peak_layer = find_peak_separation_layer(geometry_results)
    
    # Discriminability curve
    disc_curve = [g.discriminability_ratio for g in geometry_results]
    
    return {
        "layers": layers,
        "separation_curves": separation_curves,
        "discriminability_curve": disc_curve,
        "peak_separation_layer": peak_layer,
        "peak_discriminability": max(disc_curve),
    }

