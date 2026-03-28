"""
Information-Theoretic Analysis Tools

Ported from experiment 12.
More principled than probe accuracy alone - directly measures
how much information about labels is encoded in activations.

Key techniques:
- Mutual Information I(activations; labels) 
- Kraskov-Stoegbauer-Grassberger (KSG) estimator for continuous MI
- Information flow between layers
- Redundancy/synergy analysis between layers
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from scipy.special import digamma


def kraskov_mi(x: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    """
    Kraskov-Stoegbauer-Grassberger (KSG) MI estimator.
    
    More accurate than binning for continuous variables.
    Uses k-nearest neighbors to estimate MI.
    
    Args:
        x: First variable (n_samples, n_features) or (n_samples,)
        y: Second variable (n_samples,) 
        k: Number of nearest neighbors
        
    Returns:
        Estimated mutual information in nats
    """
    from sklearn.neighbors import NearestNeighbors
    
    n = len(x)
    if n < k + 1:
        return 0.0
    
    # Reshape if needed
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    # Joint space
    xy = np.hstack([x, y])
    
    # Find k-th neighbor distances in joint space
    nn_xy = NearestNeighbors(n_neighbors=k+1, metric='chebyshev')
    nn_xy.fit(xy)
    distances, _ = nn_xy.kneighbors(xy)
    eps = distances[:, k]  # k-th neighbor distance
    
    # Count neighbors within eps in marginal spaces
    nn_x = NearestNeighbors(metric='chebyshev')
    nn_x.fit(x)
    
    nn_y = NearestNeighbors(metric='chebyshev')
    nn_y.fit(y)
    
    nx = np.array([
        len(nn_x.radius_neighbors([x[i]], eps[i], return_distance=False)[0]) - 1 
        for i in range(n)
    ])
    ny = np.array([
        len(nn_y.radius_neighbors([y[i]], eps[i], return_distance=False)[0]) - 1 
        for i in range(n)
    ])
    
    # KSG estimator
    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
    return max(0.0, float(mi))


def compute_layer_mi(
    activations: np.ndarray,
    labels: np.ndarray,
    n_pca_components: int = 10
) -> Dict:
    """
    Compute mutual information between layer activations and labels.
    
    Uses PCA for dimensionality reduction before MI estimation.
    
    Args:
        activations: (n_samples, hidden_size) activations
        labels: (n_samples,) labels
        n_pca_components: Number of PCA components
        
    Returns:
        Dict with MI estimates
    """
    n_samples = activations.shape[0]
    
    # PCA reduce for MI estimation
    n_components = min(n_pca_components, activations.shape[1], n_samples - 1)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(activations)
    
    # sklearn MI (fast, approximate)
    mi_sklearn = float(mutual_info_classif(X_pca, labels, random_state=42).mean())
    
    # KSG on top PC
    mi_ksg = kraskov_mi(X_pca[:, 0], labels.astype(float))
    
    return {
        "mi_sklearn": mi_sklearn,
        "mi_ksg": mi_ksg,
        "variance_explained": float(pca.explained_variance_ratio_.sum()),
    }


def compute_layer_mi_sweep(
    activations_by_layer: Dict[int, np.ndarray],
    labels: np.ndarray
) -> Dict:
    """
    Compute MI for multiple layers.
    
    Args:
        activations_by_layer: Dict[layer -> activations]
        labels: Labels
        
    Returns:
        Dict with per-layer MI and summary
    """
    results = {"layers": {}, "summary": {}}
    
    mi_values = []
    for layer, acts in sorted(activations_by_layer.items()):
        layer_mi = compute_layer_mi(acts, labels)
        results["layers"][layer] = layer_mi
        mi_values.append((layer, layer_mi["mi_sklearn"]))
    
    # Find peak MI layer
    if mi_values:
        peak_layer, peak_mi = max(mi_values, key=lambda x: x[1])
        results["summary"] = {
            "peak_mi_layer": peak_layer,
            "peak_mi_value": peak_mi,
            "mean_mi": np.mean([v for _, v in mi_values]),
        }
    
    return results


def compute_redundancy(
    layer1_acts: np.ndarray,
    layer2_acts: np.ndarray,
    labels: np.ndarray,
    n_pca: int = 5
) -> Dict:
    """
    Compute redundancy/synergy between two layers.
    
    - Redundancy > 0: Both layers encode same info (wasteful)
    - Synergy > 0: Combined layers encode more than either alone
    
    Args:
        layer1_acts: Activations from layer 1
        layer2_acts: Activations from layer 2  
        labels: Labels
        n_pca: Number of PCA components
        
    Returns:
        Dict with redundancy and synergy
    """
    n = min(layer1_acts.shape[0], layer2_acts.shape[0], len(labels))
    
    # PCA reduce each
    pca1 = PCA(n_components=min(n_pca, layer1_acts.shape[1], n - 1))
    pca2 = PCA(n_components=min(n_pca, layer2_acts.shape[1], n - 1))
    
    X1 = pca1.fit_transform(layer1_acts[:n])
    X2 = pca2.fit_transform(layer2_acts[:n])
    y = labels[:n]
    
    # MI of each with labels
    mi1 = mutual_info_classif(X1, y, random_state=42).mean()
    mi2 = mutual_info_classif(X2, y, random_state=42).mean()
    
    # Joint MI
    X_joint = np.hstack([X1, X2])
    mi_joint = mutual_info_classif(X_joint, y, random_state=42).mean()
    
    # Redundancy: R = I(X1;Y) + I(X2;Y) - I(X1,X2;Y)
    redundancy = mi1 + mi2 - mi_joint
    
    # Synergy: S = I(X1,X2;Y) - max(I(X1;Y), I(X2;Y))
    synergy = mi_joint - max(mi1, mi2)
    
    return {
        "mi_layer1": float(mi1),
        "mi_layer2": float(mi2),
        "mi_joint": float(mi_joint),
        "redundancy": float(redundancy),
        "synergy": float(synergy),
        "interpretation": (
            "Synergistic" if synergy > 0.01 else 
            "Redundant" if redundancy > 0.01 else
            "Independent"
        ),
    }

