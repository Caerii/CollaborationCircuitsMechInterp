"""
Causal Steering Experiments
============================

Demonstrate causal control over agent modeling:
1. Extract steering vectors (agree vs disagree)
2. Apply steering during inference
3. Measure behavioral changes
4. Compare ToM heads vs random heads
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Contrastive pairs for steering vector extraction
AGREE_PROMPTS = [
    "Agent A: '2+2=4' Agent B verified this. B thinks A is correct.",
    "Agent A: 'Water is H2O.' Agent B confirms. B agrees with A.",
    "Agent A: 'Paris is in France.' Agent B knows this. B agrees.",
    "Agent A: 'The sun is hot.' Agent B learned this. B agrees.",
]

DISAGREE_PROMPTS = [
    "Agent A: '2+2=5' Agent B knows math. B thinks A is wrong.",
    "Agent A: 'Water is H3O.' Agent B knows chemistry. B disagrees with A.",
    "Agent A: 'Paris is in Germany.' Agent B knows geography. B disagrees.",
    "Agent A: 'The sun is cold.' Agent B knows physics. B disagrees.",
]

# Test prompts (neutral, can go either way)
TEST_PROMPTS = [
    {
        "prompt": "Agent A makes a claim. Agent B evaluates. Does B agree? Answer:",
        "neutral": True,
    },
    {
        "prompt": "Agent A: 'The answer is X.' Agent B considers this. B's response:",
        "neutral": True,
    },
    {
        "prompt": "Agent A states something. Agent B thinks about it. B says:",
        "neutral": True,
    },
]


class SteeringExperiment:
    """Run steering experiments on the model."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = model.config.num_hidden_layers
        self.hidden_size = model.config.hidden_size
        self.steering_vectors = {}
        self.hooks = []
    
    def extract_activations(self, prompts: List[str], layer: int) -> np.ndarray:
        """Extract last-token activations from a layer."""
        activations = []
        captured = {}
        
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["hidden"] = hidden[:, -1, :].detach().cpu()
        
        handle = self.model.model.layers[layer].register_forward_hook(hook)
        
        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
                _ = self.model(**inputs)
                activations.append(captured["hidden"].numpy())
        
        handle.remove()
        return np.vstack(activations)
    
    def compute_steering_vector(self, layer: int) -> np.ndarray:
        """Compute agree - disagree steering vector."""
        agree_acts = self.extract_activations(AGREE_PROMPTS, layer)
        disagree_acts = self.extract_activations(DISAGREE_PROMPTS, layer)
        
        # Steering vector = mean(agree) - mean(disagree)
        steering = agree_acts.mean(axis=0) - disagree_acts.mean(axis=0)
        
        # Normalize
        steering = steering / (np.linalg.norm(steering) + 1e-8)
        
        self.steering_vectors[layer] = steering
        return steering
    
    def install_steering_hook(self, layer: int, strength: float = 1.0, direction: str = "agree"):
        """Install hook to add steering vector during inference."""
        self.remove_hooks()
        
        steering = self.steering_vectors.get(layer)
        if steering is None:
            steering = self.compute_steering_vector(layer)
        
        steering_tensor = torch.tensor(steering, dtype=torch.float16, device="cuda")
        
        # Direction: "agree" adds vector, "disagree" subtracts
        sign = 1.0 if direction == "agree" else -1.0
        
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Add steering to all positions
            hidden = hidden + sign * strength * steering_tensor
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        
        handle = self.model.model.layers[layer].register_forward_hook(hook)
        self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove all steering hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate(self, prompt: str, max_tokens: int = 20) -> str:
        """Generate response."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response[len(prompt):].strip()
    
    def test_steering_effect(self, layer: int, strength: float = 1.0) -> Dict:
        """Test effect of steering at a specific layer."""
        results = {"layer": layer, "strength": strength, "tests": []}
        
        for test in TEST_PROMPTS:
            prompt = test["prompt"]
            
            # Baseline (no steering)
            self.remove_hooks()
            baseline = self.generate(prompt)
            
            # With "agree" steering
            self.install_steering_hook(layer, strength, "agree")
            steered_agree = self.generate(prompt)
            
            # With "disagree" steering
            self.install_steering_hook(layer, strength, "disagree")
            steered_disagree = self.generate(prompt)
            
            self.remove_hooks()
            
            results["tests"].append({
                "prompt": prompt[:50],
                "baseline": baseline[:50],
                "steered_agree": steered_agree[:50],
                "steered_disagree": steered_disagree[:50],
                "changed": baseline != steered_agree or baseline != steered_disagree,
            })
        
        results["change_rate"] = sum(1 for t in results["tests"] if t["changed"]) / len(results["tests"])
        return results


def main():
    timer_start = time.perf_counter()
    
    print("=" * 60)
    print("CAUSAL STEERING EXPERIMENTS")
    print("=" * 60)
    
    print("\n[1/5] Loading model...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CFG.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CFG.model_name,
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  [OK] {model.config.num_hidden_layers} layers")
    
    experiment = SteeringExperiment(model, tokenizer)
    
    print("\n[2/5] Computing steering vectors...", flush=True)
    # Compute for key layers (identified ToM layers)
    tom_layers = [12, 24, 30]
    control_layers = [0, 6, 35]  # Early, mid, late control
    
    all_layers = tom_layers + control_layers
    
    for layer in all_layers:
        print(f"  Layer {layer}...", flush=True)
        experiment.compute_steering_vector(layer)
    
    print("\n[3/5] Testing steering at ToM layers...", flush=True)
    tom_results = []
    for layer in tom_layers:
        print(f"  Testing layer {layer}...", flush=True)
        result = experiment.test_steering_effect(layer, strength=2.0)
        tom_results.append(result)
        print(f"    Change rate: {result['change_rate']:.1%}")
    
    print("\n[4/5] Testing steering at control layers...", flush=True)
    control_results = []
    for layer in control_layers:
        print(f"  Testing layer {layer}...", flush=True)
        result = experiment.test_steering_effect(layer, strength=2.0)
        control_results.append(result)
        print(f"    Change rate: {result['change_rate']:.1%}")
    
    print("\n[5/5] Strength sweep on best ToM layer...", flush=True)
    # Find best ToM layer
    best_tom_layer = max(tom_results, key=lambda x: x["change_rate"])["layer"]
    
    strength_sweep = []
    for strength in [0.5, 1.0, 2.0, 3.0, 5.0]:
        result = experiment.test_steering_effect(best_tom_layer, strength=strength)
        strength_sweep.append({
            "strength": strength,
            "change_rate": result["change_rate"],
        })
        print(f"  Strength {strength}: {result['change_rate']:.1%}", flush=True)
    
    # Compile results
    all_results = {
        "tom_layers": tom_results,
        "control_layers": control_results,
        "strength_sweep": strength_sweep,
        "best_tom_layer": best_tom_layer,
        "timing": time.perf_counter() - timer_start,
    }
    
    # Save
    with open(RESULTS_DIR / "steering_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print("\n1. ToM LAYERS (should show high steering effect)")
    print("-" * 40)
    avg_tom = np.mean([r["change_rate"] for r in tom_results])
    for r in tom_results:
        print(f"  Layer {r['layer']}: {r['change_rate']:.1%} change rate")
    print(f"  Average: {avg_tom:.1%}")
    
    print("\n2. CONTROL LAYERS (should show lower effect)")
    print("-" * 40)
    avg_control = np.mean([r["change_rate"] for r in control_results])
    for r in control_results:
        print(f"  Layer {r['layer']}: {r['change_rate']:.1%} change rate")
    print(f"  Average: {avg_control:.1%}")
    
    print("\n3. ToM vs CONTROL COMPARISON")
    print("-" * 40)
    if avg_tom > avg_control:
        print(f"  ToM layers show {avg_tom/avg_control:.1f}x stronger steering effect!")
        print("  -> CONFIRMS CAUSAL ROLE OF ToM HEADS")
    else:
        print("  No clear difference - need more investigation")
    
    print("\n4. STRENGTH SWEEP")
    print("-" * 40)
    for s in strength_sweep:
        print(f"  Strength {s['strength']}: {s['change_rate']:.1%}")
    
    total_time = time.perf_counter() - timer_start
    print(f"\n" + "=" * 60)
    print(f"TOTAL TIME: {total_time:.1f}s")
    print("=" * 60)
    
    print(f"\n[OK] Results saved to {RESULTS_DIR / 'steering_results.json'}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()




















