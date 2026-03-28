"""
Cross-Model Validation

Test findings across model sizes and families to ensure generalizability.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelResult:
    """Result from testing on a single model."""
    model_name: str
    accuracy: float
    n_samples: int
    details: Dict
    
    def to_dict(self) -> Dict:
        return {
            "model": self.model_name,
            "accuracy": float(self.accuracy),
            "n_samples": self.n_samples,
            "details": self.details,
        }


@dataclass
class CrossModelReport:
    """Report from cross-model validation."""
    finding_name: str
    holds_across_models: bool
    n_models_tested: int
    n_models_confirmed: int
    results: List[ModelResult]
    summary: str
    
    def to_dict(self) -> Dict:
        return {
            "finding": self.finding_name,
            "holds": self.holds_across_models,
            "n_tested": self.n_models_tested,
            "n_confirmed": self.n_models_confirmed,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
        }


class CrossModelValidator:
    """
    Test findings across model sizes and families.
    
    A finding is only considered robust if it holds across multiple models.
    
    Example:
        validator = CrossModelValidator()
        
        # Test if ToM finding holds
        report = validator.validate_finding(
            "ToM with proper prompting",
            scenarios,
            evaluator_fn,
            threshold=0.7  # Expect 70%+ accuracy
        )
        
        if report.holds_across_models:
            print("Finding is robust!")
    """
    
    # Models to test on (in order of preference)
    AVAILABLE_MODELS = [
        ("Qwen/Qwen3-4B", "primary"),
        ("Qwen/Qwen3-1.7B", "smaller"),
        ("Qwen/Qwen3-8B", "larger"),
    ]
    
    def __init__(
        self,
        models_to_use: Optional[List[str]] = None,
        device_map: str = "auto"
    ):
        """
        Initialize validator.
        
        Args:
            models_to_use: Optional list of model names to use
            device_map: Device mapping strategy
        """
        if models_to_use:
            self.models = [(m, "custom") for m in models_to_use]
        else:
            self.models = self.AVAILABLE_MODELS
        
        self.device_map = device_map
        self.loaded_model = None
        self.loaded_tokenizer = None
        self.loaded_name = None
    
    def load_model(self, model_name: str):
        """
        Load a model (caching to avoid repeated loads).
        
        Args:
            model_name: HuggingFace model name
        """
        if self.loaded_name == model_name:
            return self.loaded_model, self.loaded_tokenizer
        
        # Cleanup previous
        if self.loaded_model is not None:
            del self.loaded_model
            del self.loaded_tokenizer
            torch.cuda.empty_cache()
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.loaded_tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.loaded_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        self.loaded_model.eval()
        self.loaded_name = model_name
        
        return self.loaded_model, self.loaded_tokenizer
    
    def validate_finding(
        self,
        finding_name: str,
        scenarios: List[Dict],
        evaluator_fn: Callable,
        threshold: float = 0.6,
        min_models_to_confirm: int = 2,
        verbose: bool = True
    ) -> CrossModelReport:
        """
        Validate a finding across multiple models.
        
        Args:
            finding_name: Name of the finding
            scenarios: Test scenarios
            evaluator_fn: Function(model, tokenizer, scenarios) -> Dict with "accuracy"
            threshold: Minimum accuracy to confirm
            min_models_to_confirm: Minimum models that must confirm
            verbose: Print progress
            
        Returns:
            CrossModelReport
        """
        results = []
        
        for model_name, model_type in self.models:
            if verbose:
                print(f"\n=== Testing {model_name} ({model_type}) ===")
            
            try:
                model, tokenizer = self.load_model(model_name)
                
                result = evaluator_fn(model, tokenizer, scenarios)
                accuracy = result.get("accuracy", 0)
                
                results.append(ModelResult(
                    model_name=model_name,
                    accuracy=accuracy,
                    n_samples=len(scenarios),
                    details=result,
                ))
                
                if verbose:
                    print(f"  Accuracy: {accuracy:.1%}")
                    
            except Exception as e:
                if verbose:
                    print(f"  Error: {e}")
                results.append(ModelResult(
                    model_name=model_name,
                    accuracy=0.0,
                    n_samples=0,
                    details={"error": str(e)},
                ))
        
        # Check how many confirm
        n_confirmed = sum(1 for r in results if r.accuracy >= threshold)
        holds = n_confirmed >= min_models_to_confirm
        
        # Generate summary
        if holds:
            summary = f"Finding '{finding_name}' CONFIRMED on {n_confirmed}/{len(results)} models"
        else:
            summary = f"Finding '{finding_name}' NOT confirmed - only {n_confirmed}/{len(results)} models passed threshold"
        
        return CrossModelReport(
            finding_name=finding_name,
            holds_across_models=holds,
            n_models_tested=len(results),
            n_models_confirmed=n_confirmed,
            results=results,
            summary=summary,
        )
    
    def compare_scaling(
        self,
        scenarios: List[Dict],
        evaluator_fn: Callable,
        verbose: bool = True
    ) -> Dict:
        """
        Compare how finding scales with model size.
        
        Args:
            scenarios: Test scenarios
            evaluator_fn: Evaluation function
            verbose: Print progress
            
        Returns:
            Scaling analysis
        """
        results = []
        
        for model_name, model_type in self.models:
            if verbose:
                print(f"\nTesting {model_name}...")
            
            try:
                model, tokenizer = self.load_model(model_name)
                result = evaluator_fn(model, tokenizer, scenarios)
                
                # Estimate model size from name
                if "1.7B" in model_name or "1B" in model_name:
                    size = 1.7
                elif "4B" in model_name or "3B" in model_name:
                    size = 4.0
                elif "8B" in model_name or "7B" in model_name:
                    size = 8.0
                else:
                    size = 1.0
                
                results.append({
                    "model": model_name,
                    "size_b": size,
                    "accuracy": result.get("accuracy", 0),
                })
                
            except Exception as e:
                if verbose:
                    print(f"  Error: {e}")
        
        # Analyze scaling
        if len(results) >= 2:
            sizes = [r["size_b"] for r in results]
            accs = [r["accuracy"] for r in results]
            
            # Simple linear correlation
            if len(sizes) > 1:
                correlation = np.corrcoef(sizes, accs)[0, 1]
            else:
                correlation = 0.0
            
            scaling_trend = "positive" if correlation > 0.3 else "negative" if correlation < -0.3 else "neutral"
        else:
            correlation = 0.0
            scaling_trend = "insufficient_data"
        
        return {
            "results": results,
            "scaling_correlation": correlation,
            "scaling_trend": scaling_trend,
        }
    
    def cleanup(self):
        """Release model from memory."""
        if self.loaded_model is not None:
            del self.loaded_model
            del self.loaded_tokenizer
            self.loaded_model = None
            self.loaded_tokenizer = None
            self.loaded_name = None
            torch.cuda.empty_cache()


# Import numpy for scaling analysis
import numpy as np

