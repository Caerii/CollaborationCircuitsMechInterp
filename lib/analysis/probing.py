"""Linear probing with mandatory null baselines.

Every probe result is compared against a null distribution from random labels.
If probe accuracy is within 2 SD of null, it is reported as NOT meaningful.
This prevents the Round 1 mistake of interpreting overfitting as signal.
"""

import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from lib.analysis.statistics import null_probe_accuracy


@dataclass
class ProbeResult:
    """Result of a probing experiment with null baseline."""

    accuracy: float
    accuracy_std: float
    null_mean: float
    null_std: float
    above_null: bool  # True if accuracy > null_mean + 2*null_std
    margin: float  # How many SDs above null
    n_samples: int
    n_features: int
    n_classes: int
    per_class_accuracy: dict[int, float] | None = None

    def __str__(self) -> str:
        status = "MEANINGFUL" if self.above_null else "NOT MEANINGFUL (within null range)"
        return (
            f"Accuracy: {self.accuracy:.3f} +/- {self.accuracy_std:.3f} | "
            f"Null: {self.null_mean:.3f} +/- {self.null_std:.3f} | "
            f"Margin: {self.margin:.1f} SD | {status}"
        )


def run_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    compute_null: bool = True,
) -> ProbeResult:
    """Train and evaluate a linear probe with null baseline.

    Args:
        X: Activations array [n_samples, n_features]
        y: Labels array [n_samples]
        n_folds: Cross-validation folds
        seed: Random seed
        compute_null: Whether to compute null distribution (recommended: always True)

    Returns:
        ProbeResult with accuracy, null baseline, and significance
    """
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    # Fit probe
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=n_folds, scoring="accuracy")
    accuracy = float(np.mean(scores))
    accuracy_std = float(np.std(scores))

    # Null baseline
    if compute_null:
        null_mean, null_std = null_probe_accuracy(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_folds=n_folds,
            seed=seed,
        )
    else:
        null_mean = 1.0 / n_classes
        null_std = 0.0

    margin = (accuracy - null_mean) / null_std if null_std > 0 else float("inf")
    above_null = margin > 2.0

    # Per-class accuracy
    clf_full = LogisticRegression(max_iter=1000, random_state=seed)
    clf_full.fit(X, y)
    preds = clf_full.predict(X)
    per_class = {}
    for c in np.unique(y):
        mask = y == c
        per_class[int(c)] = float(np.mean(preds[mask] == y[mask]))

    return ProbeResult(
        accuracy=accuracy,
        accuracy_std=accuracy_std,
        null_mean=null_mean,
        null_std=null_std,
        above_null=above_null,
        margin=margin,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        per_class_accuracy=per_class,
    )


def transfer_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
) -> dict:
    """Train on one distribution, test on another.

    THE critical control that Round 1 failed to do until Experiment 5.
    If train accuracy >> test accuracy, the probe learned surface features, not concepts.
    """
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)

    train_acc = float(clf.score(X_train, y_train))
    test_acc = float(clf.score(X_test, y_test))
    chance = 1.0 / len(np.unique(y_test))

    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "chance": chance,
        "transfer_gap": train_acc - test_acc,
        "above_chance": test_acc > chance + 0.05,
        "warning": "POSSIBLE OVERFITTING" if (train_acc - test_acc) > 0.3 else "OK",
    }
