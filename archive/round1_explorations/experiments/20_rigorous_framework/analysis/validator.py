"""
Result Validator for Methodology Enforcement

Ensures all methodological requirements are met before allowing claims.
Based on lessons from PROPER_METHODOLOGY.md and steps 39-62.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np
from scipy import stats


@dataclass
class ValidationCheck:
    """A single validation check result."""
    name: str
    passed: bool
    message: str
    details: Dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report."""
    checks: List[ValidationCheck]
    all_passed: bool
    n_passed: int
    n_failed: int
    summary: str
    
    def to_dict(self) -> Dict:
        return {
            "all_passed": self.all_passed,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "summary": self.summary,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }
    
    def print_report(self):
        """Print formatted report."""
        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)
        
        for check in self.checks:
            status = "[PASS]" if check.passed else "[FAIL]"
            print(f"\n{status} {check.name}")
            print(f"       {check.message}")
        
        print("\n" + "-" * 60)
        print(f"RESULT: {self.n_passed}/{len(self.checks)} checks passed")
        
        if self.all_passed:
            print("STATUS: Claims can be made with confidence")
        else:
            print("STATUS: CANNOT make confident claims - address failures first")
        
        print("=" * 60)


class ResultValidator:
    """
    Validate experiment results against methodology requirements.
    
    Enforces requirements from PROPER_METHODOLOGY.md:
    - n >= 50 per condition
    - Model beats heuristic baselines
    - Statistical significance (p < 0.05)
    - Effect size (Cohen's h >= 0.2)
    - Proper counterbalancing
    - True belief controls
    
    Example:
        validator = ResultValidator(config)
        
        report = validator.validate_tom_results(results)
        
        if report.all_passed:
            print("Results are methodologically sound!")
        else:
            print("Address these issues:", report.summary)
    """
    
    def __init__(self, config=None):
        """
        Initialize validator.
        
        Args:
            config: ExperimentConfig with thresholds
        """
        if config:
            self.min_n = config.min_samples_per_condition
            self.significance_level = config.significance_level
            self.min_effect_size = config.min_effect_size
        else:
            self.min_n = 50
            self.significance_level = 0.05
            self.min_effect_size = 0.2
    
    def _check_sample_size(self, results: Dict) -> ValidationCheck:
        """Check if sample size meets minimum."""
        n = results.get("n_total", results.get("n", 0))
        passed = n >= self.min_n
        
        return ValidationCheck(
            name="Sample Size",
            passed=passed,
            message=f"n={n}, minimum required={self.min_n}",
            details={"n": n, "required": self.min_n}
        )
    
    def _check_beats_heuristics(self, results: Dict) -> ValidationCheck:
        """Check if model beats all heuristic baselines."""
        model_acc = results.get("model_accuracy", results.get("accuracy", 0))
        
        heuristic_accs = {
            "first_mention": results.get("first_mention_accuracy", 0),
            "recency": results.get("recency_accuracy", 0),
            "reality": results.get("reality_accuracy", 0),
        }
        
        best_heuristic = max(heuristic_accs.values()) if heuristic_accs else 0
        beats_all = model_acc > best_heuristic
        
        return ValidationCheck(
            name="Beats Heuristics",
            passed=beats_all,
            message=f"Model={model_acc:.1%} vs Best heuristic={best_heuristic:.1%}",
            details={
                "model_accuracy": model_acc,
                "heuristic_accuracies": heuristic_accs,
                "best_heuristic": best_heuristic,
                "margin": model_acc - best_heuristic,
            }
        )
    
    def _check_statistical_significance(self, results: Dict) -> ValidationCheck:
        """Check if result is statistically significant vs chance."""
        accuracy = results.get("accuracy", results.get("model_accuracy", 0))
        n = results.get("n_total", results.get("n", 0))
        n_correct = int(accuracy * n)
        
        # Options count for chance level
        n_options = results.get("n_options", 2)
        chance = 1.0 / n_options
        expected = int(chance * n)
        
        # Binomial test
        if n > 0:
            # One-sided test: is accuracy significantly above chance?
            p_value = stats.binom_test(n_correct, n, chance, alternative='greater')
        else:
            p_value = 1.0
        
        passed = p_value < self.significance_level
        
        return ValidationCheck(
            name="Statistical Significance",
            passed=passed,
            message=f"p={p_value:.4f}, threshold={self.significance_level}",
            details={
                "p_value": p_value,
                "threshold": self.significance_level,
                "observed": n_correct,
                "expected_by_chance": expected,
            }
        )
    
    def _check_effect_size(self, results: Dict) -> ValidationCheck:
        """Check if effect size is meaningful (Cohen's h)."""
        accuracy = results.get("accuracy", results.get("model_accuracy", 0))
        
        # Compare to chance or baseline
        n_options = results.get("n_options", 2)
        baseline = results.get("baseline_accuracy", 1.0 / n_options)
        
        # Cohen's h for proportions
        def arcsine_transform(p):
            return 2 * np.arcsin(np.sqrt(p))
        
        cohens_h = abs(arcsine_transform(accuracy) - arcsine_transform(baseline))
        passed = cohens_h >= self.min_effect_size
        
        # Interpret effect size
        if cohens_h < 0.2:
            interpretation = "negligible"
        elif cohens_h < 0.5:
            interpretation = "small"
        elif cohens_h < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return ValidationCheck(
            name="Effect Size",
            passed=passed,
            message=f"Cohen's h={cohens_h:.2f} ({interpretation}), minimum={self.min_effect_size}",
            details={
                "cohens_h": cohens_h,
                "interpretation": interpretation,
                "minimum": self.min_effect_size,
            }
        )
    
    def _check_counterbalancing(self, results: Dict) -> ValidationCheck:
        """Check if proper counterbalancing was used."""
        by_type = results.get("by_type", {})
        by_order = results.get("by_order", {})
        
        # Check for FB/TB balance
        has_fb = "FB" in by_type or "false_belief" in by_type
        has_tb = "TB" in by_type or "true_belief" in by_type
        has_belief_types = has_fb and has_tb
        
        # Check for order balance
        has_order_balance = "A-B" in by_order and "B-A" in by_order
        if has_order_balance:
            n_ab = by_order["A-B"].get("total", 0)
            n_ba = by_order["B-A"].get("total", 0)
            order_balanced = abs(n_ab - n_ba) <= max(n_ab, n_ba) * 0.1  # Within 10%
        else:
            order_balanced = False
        
        passed = has_belief_types and (not by_order or order_balanced)
        
        message_parts = []
        if has_belief_types:
            message_parts.append("FB/TB: present")
        else:
            message_parts.append("FB/TB: MISSING")
        
        if has_order_balance:
            message_parts.append(f"Order: A-B={n_ab}, B-A={n_ba}")
        else:
            message_parts.append("Order: not tested")
        
        return ValidationCheck(
            name="Counterbalancing",
            passed=passed,
            message=", ".join(message_parts),
            details={
                "has_fb_tb": has_belief_types,
                "has_order_balance": has_order_balance,
                "by_type": by_type,
                "by_order": by_order,
            }
        )
    
    def _check_confidence_interval(self, results: Dict) -> ValidationCheck:
        """Check if confidence interval doesn't include chance."""
        accuracy = results.get("accuracy", results.get("model_accuracy", 0))
        n = results.get("n_total", results.get("n", 0))
        
        # Wilson score interval
        if n > 0:
            z = 1.96  # 95% CI
            p = accuracy
            denominator = 1 + z**2 / n
            centre = (p + z**2 / (2*n)) / denominator
            margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator
            ci_low = max(0, centre - margin)
            ci_high = min(1, centre + margin)
        else:
            ci_low, ci_high = 0, 1
        
        n_options = results.get("n_options", 2)
        chance = 1.0 / n_options
        
        ci_excludes_chance = ci_low > chance
        
        return ValidationCheck(
            name="Confidence Interval",
            passed=ci_excludes_chance,
            message=f"95% CI: [{ci_low:.1%}, {ci_high:.1%}], chance={chance:.1%}",
            details={
                "ci_low": ci_low,
                "ci_high": ci_high,
                "chance": chance,
                "excludes_chance": ci_excludes_chance,
            }
        )
    
    def validate_tom_results(
        self,
        results: Dict,
        require_counterbalancing: bool = True,
        require_beats_heuristics: bool = True
    ) -> ValidationReport:
        """
        Validate ToM experiment results.
        
        Args:
            results: Results dictionary with accuracy, n, by_type, etc.
            require_counterbalancing: Whether to check for FB/TB and order balance
            require_beats_heuristics: Whether to require beating baselines
            
        Returns:
            ValidationReport
        """
        checks = []
        
        # Required checks
        checks.append(self._check_sample_size(results))
        checks.append(self._check_statistical_significance(results))
        checks.append(self._check_effect_size(results))
        checks.append(self._check_confidence_interval(results))
        
        # Optional checks
        if require_counterbalancing:
            checks.append(self._check_counterbalancing(results))
        
        if require_beats_heuristics and "first_mention_accuracy" in results:
            checks.append(self._check_beats_heuristics(results))
        
        # Aggregate
        n_passed = sum(1 for c in checks if c.passed)
        n_failed = len(checks) - n_passed
        all_passed = n_failed == 0
        
        # Summary
        if all_passed:
            summary = "All methodology requirements met. Results can be reported with confidence."
        else:
            failed_names = [c.name for c in checks if not c.passed]
            summary = f"Failed checks: {', '.join(failed_names)}. Address before making claims."
        
        return ValidationReport(
            checks=checks,
            all_passed=all_passed,
            n_passed=n_passed,
            n_failed=n_failed,
            summary=summary,
        )
    
    def validate_circuit_claim(
        self,
        results: Dict
    ) -> ValidationReport:
        """
        Validate circuit discovery claim.
        
        Requires:
        - n >= 30 for ablation tests
        - Significant behavioral change
        - Replication across scenarios
        """
        checks = []
        
        n = results.get("n_ablation_tests", results.get("n", 0))
        checks.append(ValidationCheck(
            name="Ablation Sample Size",
            passed=n >= 30,
            message=f"n={n}, minimum=30",
            details={"n": n}
        ))
        
        effect = results.get("ablation_effect", results.get("effect", 0))
        checks.append(ValidationCheck(
            name="Ablation Effect",
            passed=abs(effect) >= 0.1,
            message=f"Effect size={effect:.1%}, minimum=10%",
            details={"effect": effect}
        ))
        
        n_passed = sum(1 for c in checks if c.passed)
        
        return ValidationReport(
            checks=checks,
            all_passed=n_passed == len(checks),
            n_passed=n_passed,
            n_failed=len(checks) - n_passed,
            summary="Circuit claim validated" if n_passed == len(checks) else "Insufficient evidence"
        )


def quick_validate(results: Dict, verbose: bool = True) -> bool:
    """
    Quick validation check for results.
    
    Args:
        results: Results dictionary
        verbose: Print report
        
    Returns:
        True if all checks pass
    """
    validator = ResultValidator()
    report = validator.validate_tom_results(results)
    
    if verbose:
        report.print_report()
    
    return report.all_passed


class ComprehensiveValidator(ResultValidator):
    """
    Extended validator with additional scientific controls.
    
    Checks all requirements including:
    - Basic: sample size, significance, effect size
    - Methodology: counterbalancing, heuristic comparison
    - Robustness: attention checks, split-half reliability, prompt sensitivity
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.require_attention_checks = True
        self.require_ceiling_floor = True
        self.require_reliability = True
    
    def _check_attention_checks(self, results: Dict) -> ValidationCheck:
        """Check if attention checks passed."""
        attention_data = results.get("attention_checks", {})
        
        if not attention_data:
            return ValidationCheck(
                name="Attention Checks",
                passed=False,
                message="No attention check data provided",
                details={}
            )
        
        passed = attention_data.get("valid", False)
        accuracy = attention_data.get("accuracy", 0)
        
        return ValidationCheck(
            name="Attention Checks",
            passed=passed,
            message=f"Attention check accuracy: {accuracy:.1%} (need >=90%)",
            details=attention_data
        )
    
    def _check_ceiling_performance(self, results: Dict) -> ValidationCheck:
        """Check if ceiling scenarios show high performance."""
        ceiling_data = results.get("ceiling_test", {})
        
        if not ceiling_data:
            return ValidationCheck(
                name="Ceiling Test",
                passed=True,  # Optional
                message="No ceiling test data (optional)",
                details={}
            )
        
        accuracy = ceiling_data.get("accuracy", 0)
        passed = accuracy >= 0.9  # Should get 90%+ on easy scenarios
        
        return ValidationCheck(
            name="Ceiling Test",
            passed=passed,
            message=f"Ceiling accuracy: {accuracy:.1%} (need >=90%)",
            details=ceiling_data
        )
    
    def _check_split_half_reliability(self, results: Dict) -> ValidationCheck:
        """Check split-half reliability."""
        reliability_data = results.get("split_half_reliability", {})
        
        if not reliability_data:
            return ValidationCheck(
                name="Split-Half Reliability",
                passed=True,  # Optional
                message="No reliability data (optional)",
                details={}
            )
        
        reliability = reliability_data.get("spearman_brown_reliability", 0)
        passed = reliability >= 0.7  # Standard threshold
        
        return ValidationCheck(
            name="Split-Half Reliability",
            passed=passed,
            message=f"Reliability: {reliability:.2f} (need >=0.70)",
            details=reliability_data
        )
    
    def _check_prompt_sensitivity(self, results: Dict) -> ValidationCheck:
        """Check that results aren't overly prompt-sensitive."""
        sensitivity_data = results.get("prompt_sensitivity", {})
        
        if not sensitivity_data:
            return ValidationCheck(
                name="Prompt Sensitivity",
                passed=True,  # Optional
                message="No prompt sensitivity data (optional)",
                details={}
            )
        
        std = sensitivity_data.get("std_accuracy", 0)
        passed = std < 0.1  # <10% std across phrasings
        
        return ValidationCheck(
            name="Prompt Sensitivity",
            passed=passed,
            message=f"Accuracy std across prompts: {std:.1%} (need <10%)",
            details=sensitivity_data
        )
    
    def validate_comprehensive(
        self,
        results: Dict,
        require_all_controls: bool = False
    ) -> ValidationReport:
        """
        Run comprehensive validation including all controls.
        
        Args:
            results: Results dictionary with all control data
            require_all_controls: If True, fail on missing controls
            
        Returns:
            ValidationReport
        """
        checks = []
        
        # Basic checks (required)
        checks.append(self._check_sample_size(results))
        checks.append(self._check_statistical_significance(results))
        checks.append(self._check_effect_size(results))
        checks.append(self._check_confidence_interval(results))
        
        # Methodology checks
        if results.get("by_type") or results.get("by_order"):
            checks.append(self._check_counterbalancing(results))
        
        if "first_mention_accuracy" in results:
            checks.append(self._check_beats_heuristics(results))
        
        # Additional control checks
        if results.get("attention_checks") or require_all_controls:
            checks.append(self._check_attention_checks(results))
        
        if results.get("ceiling_test"):
            checks.append(self._check_ceiling_performance(results))
        
        if results.get("split_half_reliability"):
            checks.append(self._check_split_half_reliability(results))
        
        if results.get("prompt_sensitivity"):
            checks.append(self._check_prompt_sensitivity(results))
        
        # Aggregate
        n_passed = sum(1 for c in checks if c.passed)
        n_failed = len(checks) - n_passed
        all_passed = n_failed == 0
        
        # Summary
        if all_passed:
            summary = "All methodology requirements met including controls."
        else:
            failed_names = [c.name for c in checks if not c.passed]
            summary = f"Failed checks: {', '.join(failed_names)}"
        
        return ValidationReport(
            checks=checks,
            all_passed=all_passed,
            n_passed=n_passed,
            n_failed=n_failed,
            summary=summary,
        )

