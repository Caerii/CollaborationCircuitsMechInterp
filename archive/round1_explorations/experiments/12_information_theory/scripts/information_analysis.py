"""
Information-Theoretic Analysis of ToM Circuits
===============================================

Rigorous quantification using:
1. Mutual Information I(X; Y)
2. Conditional MI I(X; Y | Z)  
3. Information flow between layers
4. Redundancy analysis

More principled than probing accuracy alone.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
from scipy.special import digamma

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

RESULTS_DIR = Path(__file__).parent.parent / "results"
CACHE_DIR = Path(__file__).parent.parent.parent / "11_circuit_discovery" / "cache"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def kraskov_mi(x: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    """
    Kraskov-Stoegbauer-Grassberger (KSG) MI estimator.
    More accurate than binning for continuous variables.
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
    
    nx = np.array([len(nn_x.radius_neighbors([x[i]], eps[i], return_distance=False)[0]) - 1 
                   for i in range(n)])
    ny = np.array([len(nn_y.radius_neighbors([y[i]], eps[i], return_distance=False)[0]) - 1 
                   for i in range(n)])
    
    # KSG estimator
    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
    return max(0, mi)


def compute_layer_mi(activations: np.ndarray, labels: np.ndarray) -> Dict:
    """Compute MI between layer activations and labels."""
    n_samples, n_layers, hidden_size = activations.shape
    
    results = {"layers": [], "summary": {}}
    
    print("  Computing MI per layer...", flush=True)
    
    for layer_idx in range(n_layers):
        if layer_idx % 6 == 0:
            print(f"    Layer {layer_idx}/{n_layers}", flush=True)
        
        X = activations[:, layer_idx, :]
        
        # Method 1: sklearn MI (fast, approximate)
        # Use PCA to reduce dimensionality for MI estimation
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(10, X.shape[1], X.shape[0]-1))
        X_pca = pca.fit_transform(X)
        
        mi_values = mutual_info_classif(X_pca, labels, random_state=42)
        mi_sklearn = mi_values.mean()
        
        # Method 2: KSG estimator on top PCs
        mi_ksg = kraskov_mi(X_pca[:, 0], labels.astype(float))
        
        results["layers"].append({
            "layer": layer_idx,
            "mi_sklearn": float(mi_sklearn),
            "mi_ksg": float(mi_ksg),
            "variance_explained": float(pca.explained_variance_ratio_.sum()),
        })
    
    # Summary statistics
    mi_by_layer = [r["mi_sklearn"] for r in results["layers"]]
    results["summary"] = {
        "max_mi_layer": int(np.argmax(mi_by_layer)),
        "max_mi_value": float(np.max(mi_by_layer)),
        "mean_mi": float(np.mean(mi_by_layer)),
        "mi_increase_early_to_mid": float(np.mean(mi_by_layer[12:24]) - np.mean(mi_by_layer[0:12])),
        "mi_decrease_mid_to_late": float(np.mean(mi_by_layer[24:]) - np.mean(mi_by_layer[12:24])),
    }
    
    return results


def compute_information_flow(activations: np.ndarray) -> Dict:
    """Compute information flow between adjacent layers."""
    n_samples, n_layers, hidden_size = activations.shape
    
    print("  Computing information flow...", flush=True)
    
    flows = []
    
    for layer_idx in range(n_layers - 1):
        if layer_idx % 6 == 0:
            print(f"    Layer {layer_idx} -> {layer_idx+1}", flush=True)
        
        X_curr = activations[:, layer_idx, :]
        X_next = activations[:, layer_idx + 1, :]
        
        # Correlation-based flow (fast proxy for MI)
        # High correlation = information preserved
        from sklearn.decomposition import PCA
        
        pca_curr = PCA(n_components=min(5, X_curr.shape[0]-1))
        pca_next = PCA(n_components=min(5, X_next.shape[0]-1))
        
        X_curr_pca = pca_curr.fit_transform(X_curr)
        X_next_pca = pca_next.fit_transform(X_next)
        
        # Cross-correlation
        corr_matrix = np.corrcoef(X_curr_pca.T, X_next_pca.T)
        n_curr = X_curr_pca.shape[1]
        cross_corr = corr_matrix[:n_curr, n_curr:]
        
        flow_strength = np.abs(cross_corr).max()
        
        flows.append({
            "from_layer": layer_idx,
            "to_layer": layer_idx + 1,
            "flow_strength": float(flow_strength),
        })
    
    return {"flows": flows}


def compute_redundancy(activations: np.ndarray, labels: np.ndarray, layer_pairs: List[Tuple[int, int]]) -> Dict:
    """
    Compute redundancy between layer pairs.
    High redundancy = both layers encode same information.
    Low redundancy = layers encode complementary information.
    """
    print("  Computing redundancy...", flush=True)
    
    from sklearn.decomposition import PCA
    
    results = []
    
    for l1, l2 in layer_pairs:
        X1 = activations[:, l1, :]
        X2 = activations[:, l2, :]
        
        # PCA reduce
        pca1 = PCA(n_components=min(5, X1.shape[0]-1))
        pca2 = PCA(n_components=min(5, X2.shape[0]-1))
        
        X1_pca = pca1.fit_transform(X1)
        X2_pca = pca2.fit_transform(X2)
        
        # MI of each with labels
        mi1 = mutual_info_classif(X1_pca, labels, random_state=42).mean()
        mi2 = mutual_info_classif(X2_pca, labels, random_state=42).mean()
        
        # Joint MI (concatenated)
        X_joint = np.hstack([X1_pca, X2_pca])
        mi_joint = mutual_info_classif(X_joint, labels, random_state=42).mean()
        
        # Redundancy approximation: R = I(X1;Y) + I(X2;Y) - I(X1,X2;Y)
        redundancy = mi1 + mi2 - mi_joint
        
        # Synergy = I(X1,X2;Y) - max(I(X1;Y), I(X2;Y))
        synergy = mi_joint - max(mi1, mi2)
        
        results.append({
            "layer1": l1,
            "layer2": l2,
            "mi_layer1": float(mi1),
            "mi_layer2": float(mi2),
            "mi_joint": float(mi_joint),
            "redundancy": float(redundancy),
            "synergy": float(synergy),
        })
        
        print(f"    L{l1}-L{l2}: redundancy={redundancy:.3f}, synergy={synergy:.3f}", flush=True)
    
    return {"redundancy_analysis": results}


def main():
    timer_start = time.perf_counter()
    
    print("=" * 60)
    print("INFORMATION-THEORETIC ANALYSIS")
    print("=" * 60)
    
    # Load cached activations
    print("\n[1/5] Loading cached activations...", flush=True)
    cache_file = CACHE_DIR / "activations.npz"
    
    if not cache_file.exists():
        print("  ERROR: Run experiment 11 first to generate activations!")
        return
    
    data = np.load(cache_file)
    activations = data["layer_outputs"].astype(np.float32)
    print(f"  Shape: {activations.shape}")
    
    # Labels (from experiment 11 scenarios)
    SCENARIOS = [
        {"b_agrees": True}, {"b_agrees": True}, {"b_agrees": True}, {"b_agrees": True},
        {"b_agrees": True}, {"b_agrees": True},
        {"b_agrees": False}, {"b_agrees": False}, {"b_agrees": False}, {"b_agrees": False},
        {"b_agrees": False}, {"b_agrees": False},
    ]
    labels = np.array([1 if s["b_agrees"] else 0 for s in SCENARIOS])
    print(f"  Labels: {labels}")
    
    # 1. Layer-wise MI
    print("\n[2/5] Computing layer-wise Mutual Information...", flush=True)
    mi_start = time.perf_counter()
    mi_results = compute_layer_mi(activations, labels)
    mi_time = time.perf_counter() - mi_start
    print(f"  Time: {mi_time:.1f}s")
    
    # 2. Information flow
    print("\n[3/5] Computing information flow between layers...", flush=True)
    flow_start = time.perf_counter()
    flow_results = compute_information_flow(activations)
    flow_time = time.perf_counter() - flow_start
    print(f"  Time: {flow_time:.1f}s")
    
    # 3. Redundancy analysis
    print("\n[4/5] Computing redundancy/synergy...", flush=True)
    redundancy_start = time.perf_counter()
    layer_pairs = [(0, 12), (12, 24), (24, 35), (0, 35)]
    redundancy_results = compute_redundancy(activations, labels, layer_pairs)
    redundancy_time = time.perf_counter() - redundancy_start
    print(f"  Time: {redundancy_time:.1f}s")
    
    # Compile results
    all_results = {
        "mutual_information": mi_results,
        "information_flow": flow_results,
        "redundancy": redundancy_results,
        "timing": {
            "mi_analysis": mi_time,
            "flow_analysis": flow_time,
            "redundancy_analysis": redundancy_time,
            "total": time.perf_counter() - timer_start,
        }
    }
    
    # Save
    print("\n[5/5] Saving results...", flush=True)
    with open(RESULTS_DIR / "information_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print("\n1. MUTUAL INFORMATION BY LAYER")
    print("-" * 40)
    print(f"  Max MI layer: {mi_results['summary']['max_mi_layer']}")
    print(f"  Max MI value: {mi_results['summary']['max_mi_value']:.4f}")
    print(f"  MI increase (early->mid): {mi_results['summary']['mi_increase_early_to_mid']:.4f}")
    print(f"  MI change (mid->late): {mi_results['summary']['mi_decrease_mid_to_late']:.4f}")
    
    print("\n2. INFORMATION FLOW")
    print("-" * 40)
    flows = flow_results["flows"]
    bottlenecks = [f for f in flows if f["flow_strength"] < 0.5]
    if bottlenecks:
        print("  Information bottlenecks (low flow):")
        for b in bottlenecks[:3]:
            print(f"    Layer {b['from_layer']} -> {b['to_layer']}: {b['flow_strength']:.3f}")
    
    print("\n3. REDUNDANCY/SYNERGY")
    print("-" * 40)
    for r in redundancy_results["redundancy_analysis"]:
        print(f"  L{r['layer1']}-L{r['layer2']}: R={r['redundancy']:.3f}, S={r['synergy']:.3f}")
    
    total_time = time.perf_counter() - timer_start
    print(f"\n" + "=" * 60)
    print(f"TOTAL TIME: {total_time:.1f}s")
    print("=" * 60)
    
    print(f"\n[OK] Results saved to {RESULTS_DIR / 'information_analysis.json'}")


if __name__ == "__main__":
    main()























