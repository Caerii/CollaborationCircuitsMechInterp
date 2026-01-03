"""
Evaluation utilities for Theory of Mind testing.

Provides standardized evaluation pipelines for:
- Single scenario evaluation
- Batch evaluation
- Intervention comparison
- Results aggregation
"""

import torch
from typing import List, Dict, Optional, Tuple, Union
from collections import defaultdict
import json
from pathlib import Path


class ToMEvaluator:
    """
    Evaluate Theory of Mind performance.
    
    Provides a clean interface for running ToM evaluations with
    optional interventions (ablation, amplification, etc.)
    
    Example:
        evaluator = ToMEvaluator(model, hook_manager)
        
        # Baseline
        baseline = evaluator.evaluate_batch(scenarios)
        
        # With ablation
        hook_manager.ablate_heads([(17, 4), (18, 11)])
        ablated = evaluator.evaluate_batch(scenarios)
        hook_manager.clear()
        
        # Compare
        comparison = evaluator.compare(baseline, ablated)
    """
    
    def __init__(self, model, hook_manager=None):
        """
        Initialize evaluator.
        
        Args:
            model: QwenModel instance
            hook_manager: Optional HookManager for interventions
        """
        self.model = model
        self.hooks = hook_manager
        
    def evaluate_scenario(self, scenario: Dict) -> Dict:
        """
        Evaluate a single ToM scenario.
        
        Args:
            scenario: Dict with 'prompt', 'correct', 'wrong' keys
            
        Returns:
            Dict with evaluation results
        """
        prompt = scenario['prompt']
        correct = scenario['correct']
        wrong = scenario['wrong']
        
        # Get probabilities
        result = self.model.compare_tokens(prompt, correct, wrong)
        
        return {
            'correct_prob': result['prob_a'],
            'wrong_prob': result['prob_b'],
            'predicts_correct': result['a_wins'],
            'margin': result['margin'],
            'scenario_type': scenario.get('type', 'unknown'),
        }
    
    def evaluate_batch(
        self,
        scenarios: List[Dict],
        verbose: bool = False
    ) -> Dict:
        """
        Evaluate a batch of scenarios.
        
        Args:
            scenarios: List of scenario dictionaries
            verbose: Print progress
            
        Returns:
            Dict with aggregated results
        """
        results = []
        correct_count = 0
        
        for i, scenario in enumerate(scenarios):
            result = self.evaluate_scenario(scenario)
            results.append(result)
            
            if result['predicts_correct']:
                correct_count += 1
                
            if verbose and (i + 1) % 10 == 0:
                print(f"  Evaluated {i+1}/{len(scenarios)}")
        
        accuracy = correct_count / len(scenarios) if scenarios else 0.0
        
        return {
            'n': len(scenarios),
            'correct': correct_count,
            'accuracy': accuracy,
            'results': results,
        }
    
    def evaluate_with_intervention(
        self,
        scenarios: List[Dict],
        intervention_fn: callable,
        verbose: bool = False
    ) -> Dict:
        """
        Evaluate with a custom intervention.
        
        Args:
            scenarios: List of scenarios
            intervention_fn: Function that installs hooks
            verbose: Print progress
            
        Returns:
            Dict with results
        """
        # Install intervention
        intervention_fn()
        
        # Evaluate
        results = self.evaluate_batch(scenarios, verbose=verbose)
        
        # Clear intervention
        if self.hooks:
            self.hooks.clear()
            
        return results
    
    def compare_conditions(
        self,
        scenarios: List[Dict],
        conditions: Dict[str, callable],
        verbose: bool = True
    ) -> Dict:
        """
        Compare multiple intervention conditions.
        
        Args:
            scenarios: List of scenarios
            conditions: Dict mapping condition name to intervention function
            verbose: Print results
            
        Returns:
            Dict with results for each condition
        """
        results = {}
        
        for name, intervention_fn in conditions.items():
            if verbose:
                print(f"\nEvaluating: {name}")
                
            # Run intervention (if any)
            if intervention_fn is not None:
                intervention_fn()
            
            # Evaluate
            result = self.evaluate_batch(scenarios, verbose=False)
            results[name] = result
            
            # Clear hooks
            if self.hooks:
                self.hooks.clear()
                
            if verbose:
                print(f"  Accuracy: {result['accuracy']*100:.1f}%")
        
        return results
    
    def detailed_analysis(
        self,
        scenarios: List[Dict],
        results: Dict
    ) -> Dict:
        """
        Perform detailed analysis of results.
        
        Args:
            scenarios: Original scenarios
            results: Results from evaluate_batch
            
        Returns:
            Dict with detailed analysis
        """
        # Group by scenario type
        by_type = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for scenario, result in zip(scenarios, results['results']):
            stype = scenario.get('type', 'unknown')
            by_type[stype]['total'] += 1
            if result['predicts_correct']:
                by_type[stype]['correct'] += 1
        
        # Calculate per-type accuracy
        type_accuracy = {}
        for stype, counts in by_type.items():
            type_accuracy[stype] = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
        
        # Find failure cases
        failures = []
        for i, (scenario, result) in enumerate(zip(scenarios, results['results'])):
            if not result['predicts_correct']:
                failures.append({
                    'index': i,
                    'scenario': scenario,
                    'result': result,
                })
        
        return {
            'overall_accuracy': results['accuracy'],
            'by_type': dict(by_type),
            'type_accuracy': type_accuracy,
            'failures': failures[:10],  # First 10 failures
            'n_failures': len(failures),
        }


class ResultsManager:
    """
    Manage experiment results: saving, loading, comparison.
    """
    
    def __init__(self, results_dir: Union[str, Path]):
        """
        Initialize results manager.
        
        Args:
            results_dir: Directory for storing results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def save(self, results: Dict, name: str) -> Path:
        """
        Save results to JSON.
        
        Args:
            results: Results dictionary
            name: Filename (without extension)
            
        Returns:
            Path to saved file
        """
        filepath = self.results_dir / f"{name}.json"
        
        # Convert any non-serializable types
        serializable = self._make_serializable(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
            
        return filepath
    
    def load(self, name: str) -> Dict:
        """
        Load results from JSON.
        
        Args:
            name: Filename (without extension)
            
        Returns:
            Results dictionary
        """
        filepath = self.results_dir / f"{name}.json"
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_results(self) -> List[str]:
        """List all saved result files."""
        return [p.stem for p in self.results_dir.glob("*.json")]
    
    def _make_serializable(self, obj):
        """Recursively convert to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (bool,)):
            return bool(obj)
        elif isinstance(obj, (int,)):
            return int(obj)
        elif isinstance(obj, (float,)):
            return float(obj)
        elif hasattr(obj, 'item'):  # Tensor
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        else:
            return obj


def quick_evaluate(
    model,
    scenarios: List[Dict],
    verbose: bool = True
) -> float:
    """
    Quick evaluation without hook manager.
    
    Args:
        model: QwenModel instance
        scenarios: List of scenarios
        verbose: Print accuracy
        
    Returns:
        Accuracy as float
    """
    evaluator = ToMEvaluator(model)
    results = evaluator.evaluate_batch(scenarios, verbose=False)
    
    if verbose:
        print(f"Accuracy: {results['accuracy']*100:.1f}% ({results['correct']}/{results['n']})")
        
    return results['accuracy']


