"""
Chat Mode Circuit Analyzer

Combines head ablation with chat-based evaluation for proper ToM circuit discovery.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
import sys
import json
import hashlib

from .ablation import HeadAblator, AblationResult
from pathlib import Path

# Add framework root to path for imports
FRAMEWORK_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from core.chat_runner import ChatExperimentRunner, ScenarioResult
from core.response_parser import ResponseParser
from config import ExperimentConfig


class ChatModeCircuitAnalyzer:
    """
    Circuit analysis specifically for chat-mode models.
    
    Uses proper chat formatting and head ablation to discover ToM circuits.
    
    Example:
        analyzer = ChatModeCircuitAnalyzer(model, tokenizer, config)
        
        # Run ablation sweep
        results = analyzer.ablation_sweep(
            scenarios=scenarios,
            layers_to_test=[20, 24, 28, 32],
            heads_per_layer=4
        )
        
        # Get significant heads (after multiple comparisons correction)
        significant = analyzer.get_significant_heads(results, alpha=0.05)
    """
    
    def __init__(self, model, tokenizer, config: Optional[ExperimentConfig] = None, cache_dir: Optional[Path] = None):
        """
        Initialize analyzer.
        
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            config: Experiment config (uses default if None)
            cache_dir: Optional directory for caching results
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ExperimentConfig()
        
        self.ablator = HeadAblator(model)
        self.runner = ChatExperimentRunner(model, tokenizer, self.config)
        self.parser = ResponseParser()
        
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads
        
        # Setup caching
        self.cache_dir = cache_dir or (FRAMEWORK_ROOT / "cache" / "circuit_analysis")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def test_scenario(self, scenario: Dict, max_tokens: Optional[int] = None) -> ScenarioResult:
        """
        Test a single scenario in chat mode.
        
        Args:
            scenario: Scenario dict with story, question, correct, options
            max_tokens: Max tokens (uses config default if None)
            
        Returns:
            ScenarioResult
        """
        # Format scenario for chat runner
        formatted = self._format_scenario(scenario)
        
        # Run with chat runner (pass max_tokens if specified)
        result = self.runner.run_scenario(formatted, max_tokens=max_tokens)
        
        return result
    
    def _format_scenario(self, scenario: Dict) -> Dict:
        """Format scenario for chat runner."""
        # Handle different scenario formats
        if 'story' in scenario and 'question' in scenario:
            # Generator format
            question = f"{scenario['story']} {scenario['question']}"
            correct = scenario.get('correct', '')
            options = scenario.get('options', [])
        elif 'question' in scenario:
            # Simple format
            question = scenario['question']
            correct = scenario.get('correct', scenario.get('correct_answer', ''))
            options = scenario.get('options', [])
        else:
            raise ValueError(f"Unknown scenario format: {list(scenario.keys())}")
        
        # Get wrong answer from options
        wrong = [opt for opt in options if opt != correct]
        wrong = wrong[0] if wrong else scenario.get('wrong', scenario.get('wrong_answer', ''))
        
        return {
            "question": question,
            "options": options or [correct, wrong] if wrong else [correct],
            "correct": correct,
            "type": scenario.get('type', scenario.get('scenario_type', 'unknown')),
        }
    
    def _scenario_hash(self, scenario: Dict) -> str:
        """Generate hash for scenario caching."""
        # Create stable hash from scenario content
        key_parts = [
            scenario.get('story', ''),
            scenario.get('question', ''),
            scenario.get('correct', ''),
            str(sorted(scenario.get('options', [])))
        ]
        key_str = '|'.join(str(p) for p in key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _baseline_cache_key(self, scenarios: List[Dict]) -> str:
        """Generate cache key for baseline results."""
        scenario_hashes = sorted([self._scenario_hash(s) for s in scenarios])
        key_str = f"baseline_{len(scenarios)}_{'_'.join(scenario_hashes[:10])}"  # Use first 10 for key
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _ablation_cache_key(self, layer: int, head: int, scenarios: List[Dict], max_tokens: Optional[int]) -> str:
        """Generate cache key for ablation results."""
        scenario_hashes = sorted([self._scenario_hash(s) for s in scenarios])
        key_str = f"ablation_L{layer}H{head}_{len(scenarios)}_{max_tokens}_{'_'.join(scenario_hashes[:10])}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_baseline(self, scenarios: List[Dict], use_cache: bool = True) -> Dict:
        """
        Get baseline performance on scenarios.
        
        Args:
            scenarios: List of scenario dicts
            use_cache: Whether to use cached results if available
            
        Returns:
            Dict with accuracy, individual results, etc.
        """
        # Check cache
        if use_cache:
            cache_key = self._baseline_cache_key(scenarios)
            cache_file = self.cache_dir / f"baseline_{cache_key}.json"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        cached = json.load(f)
                    print(f"  [CACHE HIT] Loading baseline from cache...")
                    sys.stdout.flush()
                    # Convert back to ScenarioResult objects if needed
                    # For now, just return the cached dict (individual results are bools)
                    return cached
                except Exception as e:
                    print(f"  [CACHE ERROR] {e}, recomputing...")
                    sys.stdout.flush()
        
        # Compute baseline
        results = []
        n_scenarios = len(scenarios)
        print(f"  Testing {n_scenarios} scenarios...")
        sys.stdout.flush()
        
        for i, scenario in enumerate(scenarios):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"    [{i+1}/{n_scenarios}]", end=" ", flush=True)
            result = self.test_scenario(scenario)
            results.append(result)
            if (i + 1) % 10 == 0:
                acc_so_far = sum(1 for r in results if r.is_correct) / len(results)
                print(f"Acc: {acc_so_far:.1%}", flush=True)
        
        if n_scenarios % 10 != 0:
            print()  # Newline if we didn't end on a multiple of 10
        
        n_correct = sum(1 for r in results if r.is_correct)
        accuracy = n_correct / len(results) if results else 0
        
        baseline = {
            "accuracy": accuracy,
            "n": len(results),
            "n_correct": n_correct,
            "individual": [r.is_correct for r in results],
            # Don't cache full results objects (too large), just the bools
        }
        
        # Save to cache
        if use_cache:
            try:
                with open(cache_file, 'w') as f:
                    json.dump(baseline, f, indent=2)
            except Exception as e:
                print(f"  [CACHE WARNING] Could not save baseline: {e}")
                sys.stdout.flush()
        
        return baseline
    
    def ablation_sweep(
        self,
        scenarios: List[Dict],
        layers_to_test: List[int],
        heads_per_layer: int = 4,
        max_tokens: Optional[int] = None
    ) -> Dict:
        """
        Sweep ablation across layers and heads.
        
        Args:
            scenarios: List of scenarios to test
            layers_to_test: Which layers to test
            heads_per_layer: How many heads per layer (samples evenly)
            max_tokens: Max tokens per generation
            
        Returns:
            Dict with baseline, ablation results, and statistics
        """
        # Validate sample size
        n = len(scenarios)
        min_n = self.config.min_samples_per_condition
        if n < min_n:
            print(f"⚠️  WARNING: n={n} < {min_n} (methodology requirement)")
            print("   Results may not be reliable.")
        
        # Get baseline
        print("\n  Getting baseline performance...")
        sys.stdout.flush()
        
        baseline = self.get_baseline(scenarios)
        baseline_acc = baseline["accuracy"]
        baseline_individual = baseline["individual"]
        
        print(f"  Baseline: {baseline['n_correct']}/{n} = {baseline_acc:.1%}")
        sys.stdout.flush()
        
        # Test ablations
        results = {
            "baseline": baseline,
            "ablations": {},
            "n": n,
        }
        
        total_tests = len(layers_to_test) * heads_per_layer
        test_num = 0
        
        # Sample heads evenly across the layer
        head_indices = list(range(0, self.n_heads, self.n_heads // heads_per_layer))[:heads_per_layer]
        
        for layer_idx in layers_to_test:
            results["ablations"][layer_idx] = {}
            
            for head_idx in head_indices:
                test_num += 1
                print(f"\n  [{test_num}/{total_tests}] Testing L{layer_idx}H{head_idx}...")
                sys.stdout.flush()
                
                # Check cache for this ablation
                cache_key = self._ablation_cache_key(layer_idx, head_idx, scenarios, max_tokens)
                cache_file = self.cache_dir / f"ablation_{cache_key}.json"
                
                if cache_file.exists():
                    try:
                        with open(cache_file, 'r') as f:
                            cached_result = json.load(f)
                        print(f"    [CACHE HIT] Loading ablation results from cache...")
                        sys.stdout.flush()
                        
                        # Use cached results
                        ablation_results = cached_result["individual"]
                        ablation_acc = cached_result["accuracy"]
                        ablation_correct = cached_result["correct"]
                        effect = baseline_acc - ablation_acc
                        p_value = cached_result["p_value"]
                        
                        results["ablations"][layer_idx][head_idx] = cached_result
                        
                        sig_marker = " *" if p_value < 0.05 else ""
                        print(f"    Accuracy: {ablation_correct}/{n} = {ablation_acc:.1%} (effect: {effect:+.1%}, p={p_value:.4f}{sig_marker}) [CACHED]")
                        sys.stdout.flush()
                        continue
                    except Exception as e:
                        print(f"    [CACHE ERROR] {e}, recomputing...")
                        sys.stdout.flush()
                
                # Ablate head
                self.ablator.ablate_head(layer_idx, head_idx)
                
                try:
                    # Test all scenarios
                    ablation_results = []
                    n_scenarios = len(scenarios)
                    print(f"    Testing {n_scenarios} scenarios...", end=" ", flush=True)
                    
                    for i, scenario in enumerate(scenarios):
                        result = self.test_scenario(scenario, max_tokens=max_tokens)
                        ablation_results.append(result.is_correct)
                        # Show progress every 10 scenarios
                        if (i + 1) % 10 == 0:
                            acc_so_far = sum(ablation_results) / len(ablation_results)
                            print(f"[{i+1}/{n_scenarios} Acc: {acc_so_far:.1%}]", end=" ", flush=True)
                    
                    if n_scenarios % 10 != 0:
                        print()  # Newline if we didn't end on a multiple of 10
                    else:
                        print()  # Newline after progress updates
                    
                    ablation_acc = sum(ablation_results) / len(ablation_results)
                    ablation_correct = sum(ablation_results)
                    effect = baseline_acc - ablation_acc
                    
                    # Statistical test: McNemar's test for paired data
                    both_correct = sum(1 for b, a in zip(baseline_individual, ablation_results) if b and a)
                    baseline_only = sum(1 for b, a in zip(baseline_individual, ablation_results) if b and not a)
                    ablation_only = sum(1 for b, a in zip(baseline_individual, ablation_results) if not b and a)
                    both_wrong = sum(1 for b, a in zip(baseline_individual, ablation_results) if not b and not a)
                    
                    # McNemar's test (only uses discordant pairs)
                    if baseline_only + ablation_only > 0:
                        # Use continuity correction
                        mcnemar_stat = (abs(baseline_only - ablation_only) - 1) ** 2 / (baseline_only + ablation_only)
                        p_value = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
                    else:
                        # No discordant pairs - cannot compute p-value
                        p_value = 1.0
                    
                    ablation_data = {
                        "accuracy": ablation_acc,
                        "correct": ablation_correct,
                        "effect": effect,
                        "individual": ablation_results,
                        "p_value": p_value,
                        "mcnemar_table": {
                            "both_correct": both_correct,
                            "baseline_only": baseline_only,
                            "ablation_only": ablation_only,
                            "both_wrong": both_wrong,
                        }
                    }
                    
                    results["ablations"][layer_idx][head_idx] = ablation_data
                    
                    # Save to cache
                    try:
                        with open(cache_file, 'w') as f:
                            json.dump(ablation_data, f, indent=2)
                    except Exception as e:
                        print(f"    [CACHE WARNING] Could not save ablation: {e}")
                        sys.stdout.flush()
                    
                    sig_marker = " *" if p_value < 0.05 else ""
                    print(f"    Accuracy: {ablation_correct}/{n} = {ablation_acc:.1%} (effect: {effect:+.1%}, p={p_value:.4f}{sig_marker})")
                    sys.stdout.flush()
                
                finally:
                    # Always clear hooks (deactivates if optimized, removes if not)
                    self.ablator.clear()
                    # Clear GPU cache periodically to free memory
                    if test_num % 4 == 0:  # Every 4 tests
                        torch.cuda.empty_cache()
        
        return results
    
    def get_significant_heads(
        self,
        results: Dict,
        alpha: float = 0.05,
        correction: str = "bonferroni"
    ) -> List[Dict]:
        """
        Get heads that are significant after multiple comparisons correction.
        
        Args:
            results: Results from ablation_sweep
            alpha: Significance level
            correction: "bonferroni" or "fdr" (Benjamini-Hochberg)
            
        Returns:
            List of head dicts with significance info
        """
        # Import from parent analysis module
        from ..controls import bonferroni_correct, benjamini_hochberg
        
        # Collect all heads and p-values
        all_heads = []
        p_values = []
        
        for layer_idx, heads in results["ablations"].items():
            for head_idx, data in heads.items():
                all_heads.append({
                    "layer": layer_idx,
                    "head": head_idx,
                    "effect": data["effect"],
                    "accuracy": data["accuracy"],
                    "p_value": data["p_value"],
                })
                p_values.append(data["p_value"])
        
        # Apply correction
        if correction == "bonferroni":
            corrected = bonferroni_correct(p_values, alpha=alpha)
        elif correction == "fdr":
            corrected = benjamini_hochberg(p_values, alpha=alpha)
        else:
            raise ValueError(f"Unknown correction: {correction}")
        
        # Mark significance
        for i, head in enumerate(all_heads):
            head["significant"] = corrected["significant_after_correction"][i]
            head["significant_uncorrected"] = head["p_value"] < alpha
        
        return all_heads, corrected

