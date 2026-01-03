"""
Simple Result Validator

Just the ESSENTIAL checks. No bloat.
"""

from typing import Dict, List
from .statistics import accuracy_with_ci, compare_accuracies


class SimpleValidator:
    """
    Validate results with essential checks only.
    
    Essential checks:
    1. Sample size >= 50
    2. Model beats heuristic baselines
    3. p < 0.05 (if comparing conditions)
    4. Cohen's h >= 0.2 (meaningful effect)
    
    That's it. No over-engineering.
    """
    
    def __init__(
        self,
        min_samples: int = 50,
        alpha: float = 0.05,
        min_effect: float = 0.2
    ):
        self.min_samples = min_samples
        self.alpha = alpha
        self.min_effect = min_effect
    
    def validate(self, results: Dict) -> Dict:
        """
        Validate experiment results.
        
        Args:
            results: Dict with:
                - n: sample size
                - accuracy: model accuracy
                - heuristic_accuracy: best heuristic baseline
                - p_value: (optional) if comparing conditions
                - effect_size: (optional) Cohen's h
        
        Returns:
            Dict with passed/failed checks and overall valid status
        """
        checks = {}
        
        # 1. Sample size
        n = results.get("n", 0)
        passed = n >= self.min_samples
        checks["sample_size"] = {
            "passed": passed,
            "value": n,
            "required": self.min_samples,
            "message": f"n={n} {'>=' if passed else '<'} {self.min_samples}"
        }
        
        # 2. Beats heuristics
        model_acc = results.get("accuracy", 0)
        heuristic_acc = results.get("heuristic_accuracy", 0)
        
        if heuristic_acc > 0:
            beats = model_acc > heuristic_acc
            checks["beats_heuristics"] = {
                "passed": beats,
                "model": model_acc,
                "heuristic": heuristic_acc,
                "message": f"Model {model_acc:.1%} {'>' if beats else '<='} Heuristic {heuristic_acc:.1%}"
            }
        
        # 3. Statistical significance (if p_value provided)
        if "p_value" in results:
            p = results["p_value"]
            checks["significance"] = {
                "passed": p < self.alpha,
                "p_value": p,
                "alpha": self.alpha,
                "message": f"p={p:.4f} {'<' if p < self.alpha else '>='} {self.alpha}"
            }
        
        # 4. Effect size (if provided)
        if "effect_size" in results:
            h = results["effect_size"]
            checks["effect_size"] = {
                "passed": h >= self.min_effect,
                "value": h,
                "required": self.min_effect,
                "message": f"h={h:.2f} {'>=' if h >= self.min_effect else '<'} {self.min_effect}"
            }
        
        # Overall
        all_passed = all(c["passed"] for c in checks.values())
        
        return {
            "valid": all_passed,
            "checks": checks,
            "summary": "All checks passed" if all_passed else "Some checks failed"
        }
    
    def quick_check(
        self,
        correct: List[bool],
        heuristic_correct: List[bool] = None
    ) -> Dict:
        """
        Quick validation from raw results.
        
        Args:
            correct: List of True/False for model predictions
            heuristic_correct: List of True/False for best heuristic
        """
        n = len(correct)
        model_acc = sum(correct) / n if n > 0 else 0
        
        results = {"n": n, "accuracy": model_acc}
        
        if heuristic_correct:
            h_acc = sum(heuristic_correct) / len(heuristic_correct)
            results["heuristic_accuracy"] = h_acc
            
            # Compute significance
            stats = compare_accuracies(model_acc, n, h_acc, len(heuristic_correct))
            results["p_value"] = stats["p_value"]
            results["effect_size"] = stats["cohens_h"]
        
        return self.validate(results)
    
    def print_report(self, validation: Dict):
        """Print a simple validation report."""
        print("\n" + "=" * 50)
        print("VALIDATION REPORT")
        print("=" * 50)
        
        for name, check in validation["checks"].items():
            status = "[PASS]" if check["passed"] else "[FAIL]"
            print(f"{status} {name}: {check['message']}")
        
        print("-" * 50)
        status = "VALID" if validation["valid"] else "INVALID"
        print(f"RESULT: {status}")
        print("=" * 50)


# Convenience function
def validate(correct: List[bool], heuristic_correct: List[bool] = None) -> bool:
    """Quick validation - returns True if all checks pass."""
    v = SimpleValidator()
    result = v.quick_check(correct, heuristic_correct)
    return result["valid"]

