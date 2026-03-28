"""Statistical analysis with locked methodology.

All tests use permutation-based methods. No parametric assumptions.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class TestResult:
    """Result of a statistical test."""

    statistic: float
    p_value: float
    effect_size: float
    ci_lower: float
    ci_upper: float
    n: int
    method: str
    survives_bonferroni: bool = False

    def __str__(self) -> str:
        sig = "*" if self.p_value < 0.05 else "ns"
        bonf = " (Bonferroni)" if self.survives_bonferroni else ""
        return (
            f"stat={self.statistic:.3f}, p={self.p_value:.4f}{sig}{bonf}, "
            f"d={self.effect_size:.3f} [{self.ci_lower:.3f}, {self.ci_upper:.3f}], "
            f"n={self.n}"
        )


def permutation_test_accuracy(
    condition_a: np.ndarray,
    condition_b: np.ndarray,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> TestResult:
    """Two-sided permutation test on accuracy difference between conditions.

    Args:
        condition_a: Binary array (1=correct, 0=incorrect) for condition A
        condition_b: Binary array for condition B
        n_permutations: Number of permutations
        seed: Random seed

    Returns:
        TestResult with p-value, effect size (Cohen's h), and bootstrap CI
    """
    rng = np.random.RandomState(seed)

    observed_diff = np.mean(condition_a) - np.mean(condition_b)
    combined = np.concatenate([condition_a, condition_b])
    n_a = len(condition_a)

    null_diffs = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(combined)
        null_diffs[i] = np.mean(perm[:n_a]) - np.mean(perm[n_a:])

    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))

    # Cohen's h for proportions
    p1 = np.mean(condition_a)
    p2 = np.mean(condition_b)
    h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

    # Bootstrap CI on the difference
    ci_lower, ci_upper = bootstrap_ci(condition_a, condition_b, seed=seed)

    return TestResult(
        statistic=observed_diff,
        p_value=p_value,
        effect_size=h,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n=len(condition_a) + len(condition_b),
        method="permutation_test",
    )


def bootstrap_ci(
    condition_a: np.ndarray,
    condition_b: np.ndarray,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap 95% CI on accuracy difference."""
    rng = np.random.RandomState(seed)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        boot_a = rng.choice(condition_a, size=len(condition_a), replace=True)
        boot_b = rng.choice(condition_b, size=len(condition_b), replace=True)
        diffs[i] = np.mean(boot_a) - np.mean(boot_b)
    return float(np.percentile(diffs, 100 * alpha / 2)), float(np.percentile(diffs, 100 * (1 - alpha / 2)))


def null_probe_accuracy(
    n_samples: int,
    n_features: int,
    n_classes: int = 2,
    n_folds: int = 5,
    n_iterations: int = 100,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute null distribution for probe accuracy.

    Returns (mean, std) of accuracy on random data.
    Critical control: if your probe accuracy is within 2 SD of this, it's meaningless.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    rng = np.random.RandomState(seed)
    accuracies = []

    for i in range(n_iterations):
        X = rng.randn(n_samples, n_features)
        y = rng.randint(0, n_classes, n_samples)
        clf = LogisticRegression(max_iter=1000, random_state=seed + i)
        scores = cross_val_score(clf, X, y, cv=n_folds, scoring="accuracy")
        accuracies.append(np.mean(scores))

    return float(np.mean(accuracies)), float(np.std(accuracies))


def stability_analysis(
    run_experiment_fn,
    stimuli: list,
    n_subsets: int = 5,
    subset_fraction: float = 0.8,
    seed: int = 42,
) -> dict:
    """Run experiment on random subsets and measure agreement.

    Args:
        run_experiment_fn: Function that takes a list of stimuli and returns results dict
        stimuli: Full stimulus set
        n_subsets: Number of random subsets
        subset_fraction: Fraction of stimuli in each subset

    Returns:
        Dict with per-subset results and agreement metrics
    """
    rng = np.random.RandomState(seed)
    n = len(stimuli)
    k = int(n * subset_fraction)

    subset_results = []
    for i in range(n_subsets):
        indices = rng.choice(n, size=k, replace=False)
        subset = [stimuli[j] for j in indices]
        result = run_experiment_fn(subset)
        subset_results.append(result)

    return {
        "subset_results": subset_results,
        "n_subsets": n_subsets,
        "subset_size": k,
    }
