"""
Cross-Model Validation Utilities

Tools for testing findings across different model architectures
to ensure robustness and generalization.
"""

import torch
import gc
import numpy as np
from typing import List, Dict, Optional, Callable, Any
from scipy import stats


def wilson_ci(successes: int, total: int, confidence: float = 0.95) -> tuple:
    """
    Wilson score confidence interval for proportions.
    
    More accurate than normal approximation for small samples.
    
    Args:
        successes: Number of successes
        total: Total trials
        confidence: Confidence level (default 0.95)
        
    Returns:
        (proportion, ci_low, ci_high)
    """
    if total == 0:
        return 0.0, 0.0, 1.0
    
    p = successes / total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    
    return p, max(0, center - spread), min(1, center + spread)


def cohens_h(p1: float, p2: float) -> float:
    """
    Cohen's h effect size for comparing proportions.
    
    Interpretation:
    - 0.2: small effect
    - 0.5: medium effect
    - 0.8: large effect
    """
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return phi1 - phi2


class CrossModelTester:
    """
    Test findings across multiple models for robustness.
    
    Example:
        tester = CrossModelTester()
        
        results = tester.test_models(
            model_ids=["Qwen/Qwen3-4B", "Qwen/Qwen2.5-1.5B"],
            test_fn=my_test_function,
            scenarios=my_scenarios,
        )
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.current_model = None
        self.current_tokenizer = None
        
    def load_model(self, model_id: str, dtype=torch.float16):
        """Load a model, clearing previous model from memory."""
        # Clear previous model
        if self.current_model is not None:
            del self.current_model
            del self.current_tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading {model_id}...")
        self.current_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.current_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.current_model.eval()
        
        if self.current_tokenizer.pad_token is None:
            self.current_tokenizer.pad_token = self.current_tokenizer.eos_token
        
        return self.current_model, self.current_tokenizer
    
    def test_single_prompt(
        self,
        prompt: str,
        correct_token: str,
        wrong_token: str,
    ) -> Dict:
        """Test a single prompt on the current model."""
        if self.current_model is None:
            raise ValueError("No model loaded. Call load_model first.")
        
        inputs = self.current_tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Get token IDs
        correct_ids = self.current_tokenizer.encode(correct_token, add_special_tokens=False)
        wrong_ids = self.current_tokenizer.encode(wrong_token, add_special_tokens=False)
        
        if not correct_ids or not wrong_ids:
            return {"error": "Token encoding failed", "correct": False}
        
        with torch.no_grad():
            outputs = self.current_model(**inputs)
            logits = outputs.logits[0, -1, :]
        
        correct_logit = float(logits[correct_ids[0]])
        wrong_logit = float(logits[wrong_ids[0]])
        
        return {
            "correct": correct_logit > wrong_logit,
            "correct_logit": correct_logit,
            "wrong_logit": wrong_logit,
            "diff": correct_logit - wrong_logit,
        }
    
    def test_model(
        self,
        model_id: str,
        scenarios: List[Dict],
        prompt_key: str = "prompt",
        correct_key: str = "correct",
        wrong_key: str = "wrong",
    ) -> Dict:
        """
        Test a single model on a set of scenarios.
        
        Returns results with Wilson CI and statistics.
        """
        self.load_model(model_id)
        
        results = []
        for scenario in scenarios:
            prompt = scenario[prompt_key]
            correct = scenario[correct_key]
            wrong = scenario[wrong_key]
            
            result = self.test_single_prompt(prompt, correct, wrong)
            results.append(result)
        
        # Compute statistics
        successes = sum(1 for r in results if r.get("correct", False))
        total = len(results)
        
        accuracy, ci_low, ci_high = wilson_ci(successes, total)
        
        return {
            "model_id": model_id,
            "accuracy": accuracy,
            "successes": successes,
            "total": total,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "individual_results": results,
        }
    
    def test_models(
        self,
        model_ids: List[str],
        scenarios: List[Dict],
        prompt_key: str = "prompt",
        correct_key: str = "correct",
        wrong_key: str = "wrong",
    ) -> Dict[str, Dict]:
        """
        Test multiple models on the same scenarios.
        
        Returns comparison results with effect sizes.
        """
        all_results = {}
        
        for model_id in model_ids:
            model_results = self.test_model(
                model_id, scenarios, prompt_key, correct_key, wrong_key
            )
            all_results[model_id] = model_results
        
        # Compute pairwise comparisons
        comparisons = []
        model_list = list(all_results.keys())
        for i in range(len(model_list)):
            for j in range(i + 1, len(model_list)):
                m1, m2 = model_list[i], model_list[j]
                p1 = all_results[m1]["accuracy"]
                p2 = all_results[m2]["accuracy"]
                effect = cohens_h(p1, p2)
                
                comparisons.append({
                    "model_a": m1,
                    "model_b": m2,
                    "acc_a": p1,
                    "acc_b": p2,
                    "cohens_h": effect,
                })
        
        return {
            "models": all_results,
            "comparisons": comparisons,
        }
    
    def cleanup(self):
        """Free GPU memory."""
        if self.current_model is not None:
            del self.current_model
            del self.current_tokenizer
            self.current_model = None
            self.current_tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()


# Convenience functions
def compare_models_on_scenarios(
    model_ids: List[str],
    scenarios: List[Dict],
    **kwargs
) -> Dict:
    """
    One-liner for cross-model comparison.
    
    Example:
        results = compare_models_on_scenarios(
            ["Qwen/Qwen3-4B", "Qwen/Qwen2.5-1.5B"],
            scenarios
        )
    """
    tester = CrossModelTester()
    try:
        return tester.test_models(model_ids, scenarios, **kwargs)
    finally:
        tester.cleanup()

