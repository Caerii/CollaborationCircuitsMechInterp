"""
Essential Statistical Functions

Just the essentials - no bloat.
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy import stats


def accuracy_with_ci(results: List[bool], confidence: float = 0.95) -> Dict:
    """
    Compute accuracy with Wilson score confidence interval.
    
    Args:
        results: List of True/False outcomes
        confidence: Confidence level (default 0.95)
        
    Returns:
        Dict with accuracy, ci_low, ci_high, n
    """
    n = len(results)
    if n == 0:
        return {"accuracy": 0, "ci_low": 0, "ci_high": 1, "n": 0}
    
    successes = sum(results)
    p = successes / n
    
    # Wilson score interval
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = (z / denominator) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    
    return {
        "accuracy": p,
        "ci_low": max(0, centre - margin),
        "ci_high": min(1, centre + margin),
        "n": n,
        "successes": successes,
    }


def compare_accuracies(acc1: float, n1: int, acc2: float, n2: int) -> Dict:
    """
    Compare two accuracies using Fisher's exact test.
    
    Returns p-value and Cohen's h effect size.
    """
    # Contingency table
    a = int(acc1 * n1)  # successes in group 1
    b = n1 - a          # failures in group 1
    c = int(acc2 * n2)  # successes in group 2
    d = n2 - c          # failures in group 2
    
    _, p_value = stats.fisher_exact([[a, b], [c, d]])
    
    # Cohen's h effect size
    phi1 = 2 * np.arcsin(np.sqrt(acc1))
    phi2 = 2 * np.arcsin(np.sqrt(acc2))
    cohens_h = abs(phi1 - phi2)
    
    return {
        "p_value": p_value,
        "cohens_h": cohens_h,
        "significant": p_value < 0.05,
        "effect_interpretation": (
            "large" if cohens_h >= 0.8 else
            "medium" if cohens_h >= 0.5 else
            "small" if cohens_h >= 0.2 else
            "negligible"
        ),
    }


def is_significant(p_value: float, alpha: float = 0.05) -> bool:
    """Simple significance check."""
    return p_value < alpha


def bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Bonferroni correction for multiple comparisons."""
    corrected_alpha = alpha / len(p_values)
    return [p < corrected_alpha for p in p_values]

