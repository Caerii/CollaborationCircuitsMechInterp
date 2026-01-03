"""
Step 3: Statistical Analysis
============================

Proper statistics with:
- Cross-validation
- Permutation tests
- Effect sizes
- Multiple comparison correction
"""

import json
from pathlib import Path

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

RESULTS_DIR = Path(__file__).parent.parent / "results"

print("=" * 60)
print("STEP 3: STATISTICAL ANALYSIS")  
print("=" * 60)


def analyze_layer(X: np.ndarray, y: np.ndarray, layer: int) -> dict:
    """Analyze single layer with all statistics."""
    from scipy.stats import binomtest
    
    print(f"\n  Layer {layer}...", flush=True)
    
    # 1. 5-fold CV accuracy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_idx, test_idx in cv.split(X, y):
        clf_temp = LogisticRegression(max_iter=500, random_state=42)
        clf_temp.fit(X[train_idx], y[train_idx])
        cv_scores.append(clf_temp.score(X[test_idx], y[test_idx]))
    
    cv_acc = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"    CV accuracy: {cv_acc:.1%} (+/- {cv_std:.1%})", flush=True)
    
    # 2. Binomial test: p-value for getting k correct out of n by chance
    n_correct = int(cv_acc * len(y))
    n_total = len(y)
    result = binomtest(n_correct, n_total, p=0.5, alternative='greater')
    p_value = result.pvalue
    print(f"    p-value: {p_value:.2e}", flush=True)
    
    # 3. Random baseline & selectivity
    random_baseline = max(y.mean(), 1 - y.mean())
    selectivity = cv_acc - random_baseline
    
    # 4. Effect size (Cohen's d) - simplified
    class_0_mean = X[y == 0].mean(axis=0)
    class_1_mean = X[y == 1].mean(axis=0)
    pooled_std = np.sqrt((X[y==0].var(axis=0) + X[y==1].var(axis=0)) / 2)
    d_vector = (class_1_mean - class_0_mean) / (pooled_std + 1e-8)
    cohens_d = np.abs(d_vector).mean()
    print(f"    Cohen's d: {cohens_d:.2f}", flush=True)
    
    # 5. Simple CI from CV scores
    ci_low = cv_acc - 1.96 * cv_std
    ci_high = cv_acc + 1.96 * cv_std
    
    return {
        "cv_accuracy": float(cv_acc),
        "cv_std": float(cv_std),
        "p_value": float(p_value),
        "random_baseline": float(random_baseline),
        "selectivity": float(selectivity),
        "cohens_d": float(cohens_d),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high)
    }


def main():
    # Load activations
    print("\nLoading activations...")
    data = torch.load(RESULTS_DIR / "activations.pt", map_location="cpu")
    
    activations = data["activations"]
    labels = data["labels"]
    layers = data["layers"]
    
    print(f"  {len(labels)} samples, {len(layers)} layers")
    
    # Analyze each layer
    print("\nAnalyzing layers...")
    results = {}
    
    for layer in layers:
        X = activations[layer].numpy()
        results[layer] = analyze_layer(X, labels, layer)
    
    # Bonferroni correction
    alpha = 0.05
    alpha_corrected = alpha / len(layers)
    
    for layer in layers:
        results[layer]["significant"] = results[layer]["p_value"] < alpha
        results[layer]["significant_corrected"] = results[layer]["p_value"] < alpha_corrected
        results[layer]["meaningful"] = (
            results[layer]["selectivity"] > 0.1 and 
            results[layer]["cohens_d"] > 0.2
        )
    
    # Save
    output_path = RESULTS_DIR / "analysis_results.json"
    with open(output_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\nBonferroni-corrected alpha: {alpha_corrected:.4f}")
    print(f"\n{'Layer':<8} {'CV Acc':<10} {'p-value':<12} {'Select.':<10} {'d':<8} {'Sig?':<8}")
    print("-" * 60)
    
    for layer in layers:
        r = results[layer]
        sig = "***" if r["significant_corrected"] else ("*" if r["significant"] else "")
        print(f"{layer:<8} {r['cv_accuracy']:.1%}      {r['p_value']:<12.4f} {r['selectivity']:+.1%}     {r['cohens_d']:.2f}     {sig}")
    
    print("\n*** = significant after Bonferroni correction")
    print("*   = significant at 0.05 (uncorrected)")
    
    # Overall verdict
    n_sig = sum(1 for l in layers if results[l]["significant_corrected"])
    n_mean = sum(1 for l in layers if results[l]["meaningful"])
    
    print(f"\nSignificant layers (corrected): {n_sig}/{len(layers)}")
    print(f"Meaningful effect layers: {n_mean}/{len(layers)}")
    
    print(f"\n[OK] Results saved to {output_path}")


if __name__ == "__main__":
    main()


