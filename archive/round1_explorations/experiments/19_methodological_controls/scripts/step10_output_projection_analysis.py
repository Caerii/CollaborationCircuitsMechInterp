"""
OUTPUT PROJECTION ANALYSIS: Why Do Inhibitors Veto?
====================================================

Key mystery: Inhibitors ATTEND to the new location but still SUPPRESS belief update.
The suppression must happen in the OUTPUT (value projection), not attention.

This script analyzes:
1. What direction do inhibitors ADD to the residual stream?
2. What direction do enablers ADD?
3. Is there a "suppression direction" vs "update direction"?
4. Path patching: trace information flow

Method:
- Hook into the OUTPUT of each head (after attention, before residual add)
- Compare activations between correct vs incorrect predictions
- Find the "belief update direction" in activation space
"""

import json
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

RESULTS_DIR = Path(__file__).parent.parent / "results"


class OutputAnalyzer:
    """Analyze what attention heads output to residual stream."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = []
        self.captured_outputs = {}
        self.n_heads = model.config.num_attention_heads
        
    def _create_capture_hook(self, layer_idx: int):
        """Capture the input to o_proj (attention output before combination)."""
        def hook(module, args, output):
            # args[0] is the input to o_proj - this is the attention output
            # Shape: (batch, seq, hidden_dim)
            self.captured_outputs[layer_idx] = args[0].detach().cpu()
        return hook
    
    def install_capture_hooks(self, layer_indices: list):
        """Install hooks to capture attention outputs."""
        self.clear_hooks()
        self.captured_outputs = {}
        
        for layer_idx in layer_indices:
            o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
            hook = o_proj.register_forward_hook(self._create_capture_hook(layer_idx))
            self.hooks.append(hook)
    
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.captured_outputs = {}
    
    def get_head_output(self, layer_idx: int, head_idx: int, seq_position: int = -1):
        """Extract specific head's output from captured activations."""
        if layer_idx not in self.captured_outputs:
            return None
        
        full_output = self.captured_outputs[layer_idx]  # (batch, seq, hidden)
        batch, seq_len, hidden = full_output.shape
        head_dim = hidden // self.n_heads
        
        # Reshape to (batch, seq, n_heads, head_dim)
        reshaped = full_output.view(batch, seq_len, self.n_heads, head_dim)
        
        # Get specific head at specific position
        head_output = reshaped[0, seq_position, head_idx, :]  # (head_dim,)
        
        return head_output.numpy()
    
    def run_and_capture(self, prompt: str, layer_indices: list):
        """Run model and capture head outputs."""
        self.install_capture_hooks(layer_indices)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]
        
        return logits.cpu(), self.captured_outputs.copy()


def main():
    print("=" * 70)
    print("OUTPUT PROJECTION ANALYSIS")
    print("=" * 70)
    print("Why do inhibitors VETO despite SEEING the update?")
    print()
    
    # Load fixed scenarios
    scenarios_file = RESULTS_DIR / "fixed_scenarios.json"
    with open(scenarios_file) as f:
        scenarios = json.load(f)
    print(f"Loaded {len(scenarios)} fixed scenarios")
    
    # Load model
    print("\nLoading model...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CFG.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CFG.model_name,
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    print("  [OK]", flush=True)
    
    analyzer = OutputAnalyzer(model, tokenizer)
    
    # Heads to analyze
    TOP_INHIBITORS = [(18, 11), (17, 4), (18, 14)]
    CRITICAL_ENABLERS = [(15, 9), (19, 2), (19, 15)]
    
    ALL_LAYERS = sorted(set([l for l, h in TOP_INHIBITORS + CRITICAL_ENABLERS]))
    ALL_HEADS = TOP_INHIBITORS + CRITICAL_ENABLERS
    
    print(f"\nAnalyzing layers: {ALL_LAYERS}")
    print(f"Heads: {[f'L{l}H{h}' for l, h in ALL_HEADS]}")
    
    # Collect head outputs for correct vs incorrect scenarios
    print("\n[1/4] Collecting head outputs...", flush=True)
    
    correct_outputs = {f"L{l}H{h}": [] for l, h in ALL_HEADS}
    incorrect_outputs = {f"L{l}H{h}": [] for l, h in ALL_HEADS}
    
    n_analyze = 50
    
    for i, scenario in enumerate(scenarios[:n_analyze]):
        logits, captured = analyzer.run_and_capture(scenario["prompt"], ALL_LAYERS)
        
        # Check if prediction is correct
        correct_id = tokenizer.encode(scenario["correct"], add_special_tokens=False)[0]
        wrong_id = tokenizer.encode(scenario["wrong"], add_special_tokens=False)[0]
        is_correct = logits[correct_id] > logits[wrong_id]
        
        # Extract head outputs
        for layer, head in ALL_HEADS:
            key = f"L{layer}H{head}"
            head_out = analyzer.get_head_output(layer, head, seq_position=-1)
            
            if head_out is not None:
                if is_correct:
                    correct_outputs[key].append(head_out)
                else:
                    incorrect_outputs[key].append(head_out)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{n_analyze} scenarios...", flush=True)
    
    analyzer.clear_hooks()
    
    # Analyze differences
    print("\n[2/4] Analyzing output differences...", flush=True)
    print("-" * 60)
    
    results = {}
    
    print("\n  HEAD      | CORRECT | INCORRECT | DIFF NORM | COSINE")
    print("  " + "-" * 55)
    
    for key in correct_outputs.keys():
        if len(correct_outputs[key]) > 5 and len(incorrect_outputs[key]) > 5:
            correct_mean = np.mean(correct_outputs[key], axis=0)
            incorrect_mean = np.mean(incorrect_outputs[key], axis=0)
            
            # Compute statistics
            correct_norm = np.linalg.norm(correct_mean)
            incorrect_norm = np.linalg.norm(incorrect_mean)
            diff = correct_mean - incorrect_mean
            diff_norm = np.linalg.norm(diff)
            
            # Cosine similarity
            cosine = np.dot(correct_mean, incorrect_mean) / (correct_norm * incorrect_norm + 1e-8)
            
            results[key] = {
                "correct_norm": float(correct_norm),
                "incorrect_norm": float(incorrect_norm),
                "diff_norm": float(diff_norm),
                "cosine_similarity": float(cosine),
                "n_correct": len(correct_outputs[key]),
                "n_incorrect": len(incorrect_outputs[key]),
            }
            
            print(f"  {key:9s} | {correct_norm:7.3f} | {incorrect_norm:9.3f} | {diff_norm:9.3f} | {cosine:6.3f}")
    
    # Train probes to predict correct/incorrect from head outputs
    print("\n[3/4] Training probes on head outputs...", flush=True)
    print("-" * 60)
    print("\n  Can we predict correct/incorrect from each head's output?")
    print()
    
    probe_results = {}
    
    for key in correct_outputs.keys():
        if len(correct_outputs[key]) > 10 and len(incorrect_outputs[key]) > 10:
            # Prepare data
            X = np.vstack(correct_outputs[key] + incorrect_outputs[key])
            y = [1] * len(correct_outputs[key]) + [0] * len(incorrect_outputs[key])
            
            # Train probe
            probe = LogisticRegression(max_iter=1000)
            probe.fit(X, y)
            accuracy = probe.score(X, y)
            
            probe_results[key] = {
                "probe_accuracy": float(accuracy),
                "coef_norm": float(np.linalg.norm(probe.coef_)),
            }
            
            marker = ""
            if accuracy > 0.75:
                marker = " ** PREDICTIVE **"
            elif accuracy > 0.65:
                marker = " * moderate *"
            
            print(f"  {key}: {accuracy:.1%} accuracy{marker}")
    
    results["probe_accuracy"] = probe_results
    
    # Analyze inhibitors vs enablers
    print("\n[4/4] Comparing inhibitors vs enablers...", flush=True)
    print("-" * 60)
    
    inhibitor_keys = [f"L{l}H{h}" for l, h in TOP_INHIBITORS]
    enabler_keys = [f"L{l}H{h}" for l, h in CRITICAL_ENABLERS]
    
    inhibitor_norms = [results[k]["diff_norm"] for k in inhibitor_keys if k in results]
    enabler_norms = [results[k]["diff_norm"] for k in enabler_keys if k in results]
    
    inhibitor_probes = [probe_results[k]["probe_accuracy"] for k in inhibitor_keys if k in probe_results]
    enabler_probes = [probe_results[k]["probe_accuracy"] for k in enabler_keys if k in probe_results]
    
    print(f"\n  INHIBITORS:")
    print(f"    Avg diff norm: {np.mean(inhibitor_norms):.3f}")
    print(f"    Avg probe accuracy: {np.mean(inhibitor_probes):.1%}")
    
    print(f"\n  ENABLERS:")
    print(f"    Avg diff norm: {np.mean(enabler_norms):.3f}")
    print(f"    Avg probe accuracy: {np.mean(enabler_probes):.1%}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if np.mean(inhibitor_probes) > np.mean(enabler_probes):
        print("\n  INHIBITORS are MORE predictive of correct/incorrect")
        print("  --> They actively compute a 'veto signal'")
    else:
        print("\n  ENABLERS are MORE predictive of correct/incorrect")
        print("  --> They actively compute the 'update signal'")
    
    if np.mean(inhibitor_norms) > np.mean(enabler_norms):
        print("\n  INHIBITORS have LARGER output difference between correct/incorrect")
        print("  --> They contribute more to the final decision")
    else:
        print("\n  ENABLERS have LARGER output difference")
        print("  --> They contribute more to the final decision")
    
    # Save
    output_file = RESULTS_DIR / "output_projection_analysis.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Saved to {output_file}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

