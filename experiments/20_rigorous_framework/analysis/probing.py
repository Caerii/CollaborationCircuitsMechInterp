"""
Linear Probing Pipeline

Consolidated from experiments 1, 9, 15, 17.
Provides standardized probing with cross-validation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder


@dataclass
class ProbeResult:
    """Result from training a linear probe."""
    layer: int
    accuracy: float
    std: float
    chance_level: float
    above_chance: bool
    n_samples: int
    n_classes: int
    cv_scores: List[float]
    coefficients: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        return {
            "layer": self.layer,
            "accuracy": float(self.accuracy),
            "std": float(self.std),
            "chance_level": float(self.chance_level),
            "above_chance": self.above_chance,
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
            "cv_scores": [float(s) for s in self.cv_scores],
        }


class ProbingPipeline:
    """
    Standardized linear probing with cross-validation.
    
    Consolidates probing patterns from:
    - Experiment 1: Entity classification (User/Self/Other)
    - Experiment 9: Belief location probing
    - Experiment 15: Multi-agent belief probing
    - Experiment 17: Presence tracking probing
    
    Example:
        pipeline = ProbingPipeline()
        
        # Probe activations for entity type
        results = pipeline.probe_multiple_layers(
            activations,  # Dict[int, np.ndarray]
            labels,       # np.ndarray of string labels
        )
        
        for layer, result in results.items():
            print(f"Layer {layer}: {result.accuracy:.1%}")
    """
    
    def __init__(
        self,
        cv_folds: int = 5,
        max_iter: int = 1000,
        random_state: int = 42
    ):
        """
        Initialize pipeline.
        
        Args:
            cv_folds: Number of cross-validation folds
            max_iter: Maximum iterations for LogisticRegression
            random_state: Random seed for reproducibility
        """
        self.cv_folds = cv_folds
        self.max_iter = max_iter
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
    
    def train_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        return_coefficients: bool = False
    ) -> ProbeResult:
        """
        Train a single linear probe with cross-validation.
        
        Args:
            X: Activation matrix [n_samples, hidden_size]
            y: Labels [n_samples]
            return_coefficients: Whether to return trained coefficients
            
        Returns:
            ProbeResult with accuracy and statistics
        """
        # Encode labels if needed
        if y.dtype.kind in ['U', 'S', 'O']:  # String types
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y
        
        n_classes = len(np.unique(y_encoded))
        n_samples = len(y_encoded)
        chance_level = 1.0 / n_classes
        
        # Determine CV splits
        n_splits = min(self.cv_folds, min(np.bincount(y_encoded)))
        if n_splits < 2:
            n_splits = 2
        
        # Create classifier
        clf = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver='lbfgs',
            multi_class='multinomial' if n_classes > 2 else 'auto',
        )
        
        # Cross-validation
        try:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(clf, X, y_encoded, cv=cv)
            accuracy = np.mean(cv_scores)
            std = np.std(cv_scores)
        except Exception as e:
            # Fallback if CV fails
            cv_scores = [chance_level]
            accuracy = chance_level
            std = 0.0
        
        # Check if above chance (2 std criterion)
        above_chance = accuracy > chance_level + 2 * std
        
        # Optionally get coefficients
        coefficients = None
        if return_coefficients:
            clf.fit(X, y_encoded)
            coefficients = clf.coef_
        
        return ProbeResult(
            layer=-1,  # Set by caller
            accuracy=accuracy,
            std=std,
            chance_level=chance_level,
            above_chance=above_chance,
            n_samples=n_samples,
            n_classes=n_classes,
            cv_scores=list(cv_scores),
            coefficients=coefficients,
        )
    
    def probe_multiple_layers(
        self,
        activations: Dict[int, np.ndarray],
        labels: np.ndarray,
        return_coefficients: bool = False,
        verbose: bool = False
    ) -> Dict[int, ProbeResult]:
        """
        Probe multiple layers for the same classification task.
        
        Args:
            activations: Dict mapping layer index to activation matrix
            labels: Labels for classification
            return_coefficients: Whether to return trained coefficients
            verbose: Print progress
            
        Returns:
            Dict mapping layer index to ProbeResult
        """
        results = {}
        
        for layer, X in sorted(activations.items()):
            if verbose:
                print(f"  Probing layer {layer}...", end=" ", flush=True)
            
            result = self.train_classifier(X, labels, return_coefficients)
            result.layer = layer
            results[layer] = result
            
            if verbose:
                print(f"{result.accuracy:.1%} (+/- {result.std:.1%})")
        
        return results
    
    def probe_multiple_targets(
        self,
        activations: Dict[int, np.ndarray],
        labels_dict: Dict[str, np.ndarray],
        verbose: bool = False
    ) -> Dict[str, Dict[int, ProbeResult]]:
        """
        Probe for multiple classification targets.
        
        Args:
            activations: Dict mapping layer to activations
            labels_dict: Dict mapping target name to labels
            verbose: Print progress
            
        Returns:
            Dict mapping target name to layer results
        """
        all_results = {}
        
        for target_name, labels in labels_dict.items():
            if verbose:
                print(f"\n=== Probing for: {target_name} ===")
            
            results = self.probe_multiple_layers(activations, labels, verbose=verbose)
            all_results[target_name] = results
        
        return all_results
    
    def extract_steering_direction(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Extract steering direction from binary classification.
        
        The steering direction is the normalized coefficient vector
        of a trained logistic regression classifier.
        
        Args:
            X: Activation matrix
            y: Binary labels
            
        Returns:
            Normalized steering direction vector
        """
        # Encode labels
        if y.dtype.kind in ['U', 'S', 'O']:
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y
        
        if len(np.unique(y_encoded)) != 2:
            raise ValueError("Steering direction requires binary classification")
        
        clf = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        clf.fit(X, y_encoded)
        
        direction = clf.coef_[0]
        direction = direction / np.linalg.norm(direction)
        
        return direction
    
    def find_best_layer(
        self,
        results: Dict[int, ProbeResult]
    ) -> Tuple[int, ProbeResult]:
        """
        Find the layer with highest probe accuracy.
        
        Args:
            results: Dict mapping layer to ProbeResult
            
        Returns:
            Tuple of (best_layer, best_result)
        """
        best_layer = max(results, key=lambda l: results[l].accuracy)
        return best_layer, results[best_layer]
    
    def summarize_results(
        self,
        results: Dict[int, ProbeResult]
    ) -> Dict:
        """
        Generate summary statistics for probe results.
        
        Args:
            results: Dict mapping layer to ProbeResult
            
        Returns:
            Summary dictionary
        """
        layers = sorted(results.keys())
        accuracies = [results[l].accuracy for l in layers]
        
        best_layer, best_result = self.find_best_layer(results)
        
        # Count layers above chance
        n_above_chance = sum(1 for r in results.values() if r.above_chance)
        
        return {
            "n_layers": len(results),
            "best_layer": best_layer,
            "best_accuracy": best_result.accuracy,
            "mean_accuracy": np.mean(accuracies),
            "n_above_chance": n_above_chance,
            "chance_level": best_result.chance_level,
            "n_samples": best_result.n_samples,
            "n_classes": best_result.n_classes,
        }


def quick_probe(
    activations: Dict[int, np.ndarray],
    labels: np.ndarray,
    verbose: bool = True
) -> Dict[int, float]:
    """
    Quick probing across layers with minimal setup.
    
    Args:
        activations: Dict mapping layer to activations
        labels: Classification labels
        verbose: Print results
        
    Returns:
        Dict mapping layer to accuracy
    """
    pipeline = ProbingPipeline()
    results = pipeline.probe_multiple_layers(activations, labels, verbose=verbose)
    
    if verbose:
        summary = pipeline.summarize_results(results)
        print(f"\nBest: Layer {summary['best_layer']} = {summary['best_accuracy']:.1%}")
    
    return {layer: r.accuracy for layer, r in results.items()}

