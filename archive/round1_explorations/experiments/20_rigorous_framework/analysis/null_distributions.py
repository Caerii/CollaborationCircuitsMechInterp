"""
Null Distribution Computation

Ported from experiment 14. CRITICAL for statistical validity.

Without null distributions, you can't interpret:
- Whether a probe accuracy is meaningful
- Whether cosine similarities are random
- Whether ablation effects are chance

Key insight from exp 14:
- With N=12, d=2560: random data gives ~100% probe accuracy (MEANINGLESS!)
- With N=200, d=128: chance is ~51%, so 60% is significant
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


@dataclass
class NullDistribution:
    """A computed null distribution for comparison."""
    name: str
    mean: float
    std: float
    percentile_95: float
    percentile_99: float
    n_samples: int
    
    def is_significant(self, observed: float, alpha: float = 0.05) -> bool:
        """Check if observed value is significant."""
        threshold = self.percentile_95 if alpha == 0.05 else self.percentile_99
        return abs(observed) > threshold
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "mean": self.mean,
            "std": self.std,
            "percentile_95": self.percentile_95,
            "percentile_99": self.percentile_99,
            "n_samples": self.n_samples,
        }


def null_cosine_distribution(
    n_dims: int,
    n_samples: int = 10000
) -> NullDistribution:
    """
    Compute expected cosine similarity between random unit vectors.
    
    In high dimensions, random vectors are nearly orthogonal.
    This tells you what cosine values are "by chance".
    
    Args:
        n_dims: Dimensionality (e.g., 2560 for hidden_size)
        n_samples: Number of random pairs to sample
        
    Returns:
        NullDistribution for cosine similarities
    """
    cosines = []
    for _ in range(n_samples):
        u = np.random.randn(n_dims)
        v = np.random.randn(n_dims)
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)
        cosines.append(abs(np.dot(u, v)))  # Use absolute value
    
    cosines = np.array(cosines)
    
    return NullDistribution(
        name=f"cosine_d{n_dims}",
        mean=float(np.mean(cosines)),
        std=float(np.std(cosines)),
        percentile_95=float(np.percentile(cosines, 95)),
        percentile_99=float(np.percentile(cosines, 99)),
        n_samples=n_samples,
    )


def null_probe_accuracy(
    n_samples: int,
    n_dims: int,
    n_trials: int = 100,
    n_classes: int = 2
) -> NullDistribution:
    """
    What accuracy does logistic regression get on RANDOM data?
    
    CRITICAL: If your real accuracy is below this, your probe means nothing!
    
    Key findings from exp 14:
    - N=12, d=2560: probe gets ~100% on random data (overfits!)
    - N=200, d=128: probe gets ~51% (as expected)
    
    Args:
        n_samples: Number of samples in your dataset
        n_dims: Dimensionality of features
        n_trials: Number of random trials
        n_classes: Number of classes
        
    Returns:
        NullDistribution for probe accuracy
    """
    accuracies = []
    
    for trial in range(n_trials):
        # Random features and labels
        X = np.random.randn(n_samples, n_dims)
        y = np.random.randint(0, n_classes, n_samples)
        
        # Cross-validated accuracy
        clf = LogisticRegression(max_iter=500, random_state=trial)
        try:
            cv_folds = min(5, n_samples // (2 * n_classes))
            if cv_folds < 2:
                cv_folds = 2
            scores = cross_val_score(clf, X, y, cv=cv_folds)
            accuracies.append(scores.mean())
        except:
            accuracies.append(1.0 / n_classes)
    
    accuracies = np.array(accuracies)
    
    return NullDistribution(
        name=f"probe_n{n_samples}_d{n_dims}",
        mean=float(np.mean(accuracies)),
        std=float(np.std(accuracies)),
        percentile_95=float(np.percentile(accuracies, 95)),
        percentile_99=float(np.percentile(accuracies, 99)),
        n_samples=n_trials,
    )


def null_ablation_flip_rate(
    n_samples: int,
    n_trials: int = 1000
) -> NullDistribution:
    """
    What flip rate do we expect by chance from ablation?
    
    If original labels are random and "ablated" labels are random,
    expected flip rate is 50%.
    
    Args:
        n_samples: Number of samples per ablation test
        n_trials: Number of trials
        
    Returns:
        NullDistribution for flip rates
    """
    flip_rates = []
    
    for _ in range(n_trials):
        original = np.random.randint(0, 2, n_samples)
        ablated = np.random.randint(0, 2, n_samples)
        flip_rate = np.mean(original != ablated)
        flip_rates.append(flip_rate)
    
    flip_rates = np.array(flip_rates)
    
    return NullDistribution(
        name=f"ablation_n{n_samples}",
        mean=float(np.mean(flip_rates)),
        std=float(np.std(flip_rates)),
        percentile_95=float(np.percentile(flip_rates, 95)),
        percentile_99=float(np.percentile(flip_rates, 99)),
        n_samples=n_trials,
    )


def compute_standard_nulls() -> Dict[str, NullDistribution]:
    """
    Compute standard null distributions for common configurations.
    
    Call this once and cache the results.
    
    Returns:
        Dict mapping name to NullDistribution
    """
    nulls = {}
    
    # Cosine nulls for common dimensions
    for d in [128, 640, 2560, 4096]:
        key = f"cosine_d{d}"
        nulls[key] = null_cosine_distribution(d, 5000)
    
    # Probe nulls for common sample sizes
    for n, d in [(50, 128), (50, 640), (100, 640), (200, 640)]:
        key = f"probe_n{n}_d{d}"
        nulls[key] = null_probe_accuracy(n, d, n_trials=50)
    
    # Ablation nulls
    for n in [30, 50, 100]:
        key = f"ablation_n{n}"
        nulls[key] = null_ablation_flip_rate(n, 500)
    
    return nulls


def check_significance(
    observed: float,
    null: NullDistribution,
    alpha: float = 0.05
) -> Dict:
    """
    Check if an observed value is significant vs null distribution.
    
    Args:
        observed: Observed value
        null: Null distribution
        alpha: Significance level
        
    Returns:
        Dict with significance result
    """
    threshold = null.percentile_95 if alpha == 0.05 else null.percentile_99
    is_sig = abs(observed) > threshold
    
    # Approximate p-value (one-tailed)
    z_score = (observed - null.mean) / null.std if null.std > 0 else 0
    
    return {
        "observed": observed,
        "null_mean": null.mean,
        "null_std": null.std,
        "threshold": threshold,
        "is_significant": is_sig,
        "z_score": z_score,
        "interpretation": (
            f"SIGNIFICANT (p < {alpha}): observed {observed:.3f} > threshold {threshold:.3f}"
            if is_sig else
            f"NOT significant: observed {observed:.3f} <= threshold {threshold:.3f}"
        ),
    }


# Pre-computed common values for quick reference
QUICK_REFERENCE = {
    "cosine_d2560": "Values |cos| > 0.05 significant at p<0.05",
    "cosine_d128": "Values |cos| > 0.17 significant at p<0.05",
    "probe_n50_d640": "Accuracy > 55% significant at p<0.05",
    "probe_n100_d640": "Accuracy > 53% significant at p<0.05",
    "ablation_n50": "Flip rate > 60% significant at p<0.05",
}

