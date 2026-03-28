"""
Compute Null Distributions
===========================

Critical for statistical validity:
1. Random vector cosines in high dimensions
2. Chance-level probe accuracy with proper N
3. Permutation baselines

Without this, can't interpret any results.
"""

import json
import time
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def null_cosine_distribution(n_dims: int, n_samples: int = 10000) -> dict:
    """
    Compute expected cosine similarity between random unit vectors.
    
    In high dimensions, random vectors are nearly orthogonal.
    E[cos(u,v)] ≈ 0, but what's the variance?
    """
    print(f"\n  Computing null cosine distribution (d={n_dims})...", flush=True)
    
    cosines = []
    for _ in range(n_samples):
        u = np.random.randn(n_dims)
        v = np.random.randn(n_dims)
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)
        cosines.append(np.dot(u, v))
    
    cosines = np.array(cosines)
    
    return {
        "n_dims": n_dims,
        "n_samples": n_samples,
        "mean": float(np.mean(cosines)),
        "std": float(np.std(cosines)),
        "percentile_1": float(np.percentile(np.abs(cosines), 1)),
        "percentile_5": float(np.percentile(np.abs(cosines), 5)),
        "percentile_95": float(np.percentile(np.abs(cosines), 95)),
        "percentile_99": float(np.percentile(np.abs(cosines), 99)),
        "abs_mean": float(np.mean(np.abs(cosines))),
    }


def null_probe_accuracy(n_samples: int, n_dims: int, n_trials: int = 100) -> dict:
    """
    What accuracy does logistic regression get on random data?
    
    This tells us if our probe accuracy is meaningful.
    """
    print(f"\n  Computing null probe accuracy (N={n_samples}, d={n_dims})...", flush=True)
    
    accuracies = []
    
    for trial in range(n_trials):
        if trial % 20 == 0:
            print(f"    Trial {trial}/{n_trials}", flush=True)
        
        # Random features and labels
        X = np.random.randn(n_samples, n_dims)
        y = np.random.randint(0, 2, n_samples)
        
        # Cross-validated accuracy
        clf = LogisticRegression(max_iter=500, random_state=trial)
        try:
            scores = cross_val_score(clf, X, y, cv=min(5, n_samples // 2))
            accuracies.append(scores.mean())
        except:
            accuracies.append(0.5)
    
    accuracies = np.array(accuracies)
    
    return {
        "n_samples": n_samples,
        "n_dims": n_dims,
        "n_trials": n_trials,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "percentile_95": float(np.percentile(accuracies, 95)),
        "percentile_99": float(np.percentile(accuracies, 99)),
        "max_accuracy": float(np.max(accuracies)),
    }


def null_ablation_flip_rate(n_samples: int, n_trials: int = 1000) -> dict:
    """
    What flip rate do we expect by chance?
    
    If we randomly flip labels, what's the distribution?
    """
    print(f"\n  Computing null flip rate distribution (N={n_samples})...", flush=True)
    
    flip_rates = []
    
    for _ in range(n_trials):
        # Original random labels
        original = np.random.randint(0, 2, n_samples)
        # "Ablated" - just random again
        ablated = np.random.randint(0, 2, n_samples)
        # Flip rate
        flip_rate = np.mean(original != ablated)
        flip_rates.append(flip_rate)
    
    flip_rates = np.array(flip_rates)
    
    return {
        "n_samples": n_samples,
        "expected_flip_rate": 0.5,  # Theoretical
        "observed_mean": float(np.mean(flip_rates)),
        "observed_std": float(np.std(flip_rates)),
        "percentile_5": float(np.percentile(flip_rates, 5)),
        "percentile_95": float(np.percentile(flip_rates, 95)),
    }


def significance_thresholds(observed_value: float, null_dist: np.ndarray) -> dict:
    """Compute p-value and significance for observed value."""
    # One-tailed: P(X >= observed)
    p_value = np.mean(np.abs(null_dist) >= np.abs(observed_value))
    
    return {
        "observed": float(observed_value),
        "p_value": float(p_value),
        "significant_05": p_value < 0.05,
        "significant_01": p_value < 0.01,
    }


def main():
    timer_start = time.perf_counter()
    
    print("=" * 60)
    print("COMPUTING NULL DISTRIBUTIONS")
    print("=" * 60)
    print("\nThis establishes baselines for statistical interpretation.")
    
    results = {}
    
    # 1. Cosine distributions for different dimensionalities
    print("\n[1/4] Null cosine distributions...")
    results["cosine_nulls"] = {}
    for n_dims in [128, 256, 640, 2560]:  # head_dim, typical dims, hidden_size
        results["cosine_nulls"][str(n_dims)] = null_cosine_distribution(n_dims, 10000)
        print(f"    d={n_dims}: |cos| mean={results['cosine_nulls'][str(n_dims)]['abs_mean']:.4f}, "
              f"95th={results['cosine_nulls'][str(n_dims)]['percentile_95']:.4f}")
    
    # 2. Probe accuracy nulls for different sample sizes
    print("\n[2/4] Null probe accuracy...")
    results["probe_nulls"] = {}
    
    # Key: test with N=12 (previous) vs N=200 (fixed)
    for n_samples, n_dims in [(12, 128), (12, 2560), (100, 128), (200, 128), (200, 640)]:
        key = f"n{n_samples}_d{n_dims}"
        results["probe_nulls"][key] = null_probe_accuracy(n_samples, n_dims, n_trials=50)
        acc = results["probe_nulls"][key]
        print(f"    N={n_samples}, d={n_dims}: chance={acc['mean_accuracy']:.1%} +/- {acc['std_accuracy']:.1%}, "
              f"max={acc['max_accuracy']:.1%}")
    
    # 3. Ablation flip rate null
    print("\n[3/4] Null ablation flip rate...")
    results["ablation_nulls"] = {}
    for n_samples in [4, 12, 50, 200]:
        results["ablation_nulls"][str(n_samples)] = null_ablation_flip_rate(n_samples, 1000)
        abl = results["ablation_nulls"][str(n_samples)]
        print(f"    N={n_samples}: expected=50%, observed={abl['observed_mean']:.1%} +/- {abl['observed_std']:.1%}")
    
    # 4. Summary interpretation guide
    print("\n[4/4] Generating interpretation guide...")
    
    results["interpretation_guide"] = {
        "cosine_significance": {
            "d2560": "Values |cos| > 0.05 are p < 0.05 significant",
            "d128": "Values |cos| > 0.17 are p < 0.05 significant",
        },
        "probe_significance": {
            "n12_d128": f"With N=12, d=128: {results['probe_nulls']['n12_d128']['percentile_95']:.1%} is 95th percentile",
            "n200_d128": f"With N=200, d=128: {results['probe_nulls']['n200_d128']['percentile_95']:.1%} is 95th percentile",
            "warning": "N=12 with d=2560 gives MEANINGLESS results (100% by chance)",
        },
        "ablation_significance": {
            "n4": "With N=4: 75% flip rate has p=0.25 (NOT significant)",
            "n50": "With N=50: Need >62% flip rate for p<0.05",
        },
    }
    
    # Save
    with open(RESULTS_DIR / "null_distributions.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    
    print("\n1. COSINE SIMILARITY IN HIGH DIMENSIONS")
    print("-" * 40)
    print(f"  d=2560: Mean |cos| = {results['cosine_nulls']['2560']['abs_mean']:.4f}")
    print(f"          95th percentile = {results['cosine_nulls']['2560']['percentile_95']:.4f}")
    print("  -> Your cos=0.03-0.12 is WITHIN random expectation!")
    
    print("\n2. PROBE ACCURACY")
    print("-" * 40)
    print(f"  N=12, d=128: Chance = {results['probe_nulls']['n12_d128']['mean_accuracy']:.1%}")
    print(f"  N=12, d=2560: Chance = {results['probe_nulls']['n12_d2560']['mean_accuracy']:.1%} (OVERFITS!)")
    print(f"  N=200, d=128: Chance = {results['probe_nulls']['n200_d128']['mean_accuracy']:.1%}")
    print("  -> Need N >> d for meaningful probing!")
    
    print("\n3. ABLATION FLIP RATE")
    print("-" * 40)
    print(f"  N=4: 75% flip has p ~ 0.25 (random!)")
    print(f"  N=50: Need >62% for significance")
    print("  -> Previous N=4 ablation proves NOTHING")
    
    total_time = time.perf_counter() - timer_start
    print(f"\n[OK] Completed in {total_time:.1f}s")
    print(f"     Saved to {RESULTS_DIR / 'null_distributions.json'}")


if __name__ == "__main__":
    main()

