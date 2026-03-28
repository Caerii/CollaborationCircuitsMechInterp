"""
DIRECTION ANALYSIS: What Direction Do Inhibitors Push?
=======================================================

Key hypothesis: Inhibitors add a "wrong answer" direction to residual stream.

This script:
1. Extracts the "correct - incorrect" direction from each head
2. Projects final logits onto these directions
3. Tests if inhibitors push AGAINST the correct direction
"""

import json
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

RESULTS_DIR = Path(__file__).parent.parent / "results"


class DirectionAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = []
        self.captured_outputs = {}
        self.n_heads = model.config.num_attention_heads
        
    def _create_capture_hook(self, layer_idx: int):
        def hook(module, args, output):
            self.captured_outputs[layer_idx] = args[0].detach().cpu()
        return hook
    
    def install_hooks(self, layer_indices: list):
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
    
    def get_head_output(self, layer_idx: int, head_idx: int):
        if layer_idx not in self.captured_outputs:
            return None
        full_output = self.captured_outputs[layer_idx]
        batch, seq_len, hidden = full_output.shape
        head_dim = hidden // self.n_heads
        reshaped = full_output.view(batch, seq_len, self.n_heads, head_dim)
        return reshaped[0, -1, head_idx, :].numpy()
    
    def run(self, prompt: str, layer_indices: list):
        self.install_hooks(layer_indices)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]
        return logits.cpu(), self.captured_outputs.copy()


def main():
    print("=" * 70)
    print("DIRECTION ANALYSIS: What Direction Do Inhibitors Push?")
    print("=" * 70)
    
    # Load scenarios
    with open(RESULTS_DIR / "fixed_scenarios.json") as f:
        scenarios = json.load(f)
    print(f"Loaded {len(scenarios)} scenarios")
    
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
    
    analyzer = DirectionAnalyzer(model, tokenizer)
    
    TOP_INHIBITORS = [(18, 11), (17, 4), (18, 14)]
    CRITICAL_ENABLERS = [(15, 9), (19, 2), (19, 15)]
    ALL_HEADS = TOP_INHIBITORS + CRITICAL_ENABLERS
    ALL_LAYERS = sorted(set([l for l, h in ALL_HEADS]))
    
    # Collect data
    print("\n[1/3] Collecting head outputs and logit differences...", flush=True)
    
    head_outputs_correct = {f"L{l}H{h}": [] for l, h in ALL_HEADS}
    head_outputs_incorrect = {f"L{l}H{h}": [] for l, h in ALL_HEADS}
    logit_diffs = []  # correct_logit - wrong_logit
    
    n_analyze = 50
    
    for scenario in scenarios[:n_analyze]:
        logits, captured = analyzer.run(scenario["prompt"], ALL_LAYERS)
        
        correct_id = tokenizer.encode(scenario["correct"], add_special_tokens=False)[0]
        wrong_id = tokenizer.encode(scenario["wrong"], add_special_tokens=False)[0]
        
        logit_diff = logits[correct_id].item() - logits[wrong_id].item()
        logit_diffs.append(logit_diff)
        is_correct = logit_diff > 0
        
        for layer, head in ALL_HEADS:
            key = f"L{layer}H{head}"
            out = analyzer.get_head_output(layer, head)
            if out is not None:
                if is_correct:
                    head_outputs_correct[key].append(out)
                else:
                    head_outputs_incorrect[key].append(out)
    
    analyzer.clear_hooks()
    
    # Compute "correct direction" for each head
    print("\n[2/3] Computing 'correct direction' for each head...", flush=True)
    
    directions = {}
    
    for key in head_outputs_correct.keys():
        if len(head_outputs_correct[key]) > 5 and len(head_outputs_incorrect[key]) > 5:
            correct_mean = np.mean(head_outputs_correct[key], axis=0)
            incorrect_mean = np.mean(head_outputs_incorrect[key], axis=0)
            
            # Direction pointing toward "correct"
            direction = correct_mean - incorrect_mean
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            
            directions[key] = direction
    
    # Analyze how each head's output relates to correct direction
    print("\n[3/3] Analyzing head contributions...", flush=True)
    print("-" * 60)
    
    print("\n  For each scenario, we compute:")
    print("  - Projection of head output onto 'correct direction'")
    print("  - Positive = pushing toward correct, Negative = pushing toward wrong")
    print()
    
    results = {}
    
    for key in directions.keys():
        all_outputs = head_outputs_correct[key] + head_outputs_incorrect[key]
        all_labels = [1] * len(head_outputs_correct[key]) + [0] * len(head_outputs_incorrect[key])
        
        projections = [np.dot(out, directions[key]) for out in all_outputs]
        
        # Correlation between projection and correct label
        correct_projs = [p for p, l in zip(projections, all_labels) if l == 1]
        incorrect_projs = [p for p, l in zip(projections, all_labels) if l == 0]
        
        avg_correct_proj = np.mean(correct_projs)
        avg_incorrect_proj = np.mean(incorrect_projs)
        
        results[key] = {
            "avg_projection_when_correct": float(avg_correct_proj),
            "avg_projection_when_incorrect": float(avg_incorrect_proj),
            "difference": float(avg_correct_proj - avg_incorrect_proj),
        }
    
    # Summary
    print("\n  HEAD      | When Correct | When Incorrect | Difference")
    print("  " + "-" * 55)
    
    inhibitor_keys = [f"L{l}H{h}" for l, h in TOP_INHIBITORS]
    enabler_keys = [f"L{l}H{h}" for l, h in CRITICAL_ENABLERS]
    
    for key in inhibitor_keys + enabler_keys:
        if key in results:
            r = results[key]
            head_type = "INH" if key in inhibitor_keys else "ENA"
            print(f"  {key:9s} [{head_type}] | {r['avg_projection_when_correct']:+8.3f} | {r['avg_projection_when_incorrect']:+11.3f} | {r['difference']:+8.3f}")
    
    # Key analysis
    print("\n" + "=" * 70)
    print("KEY ANALYSIS: How Do Heads Differ?")
    print("=" * 70)
    
    inh_diffs = [results[k]["difference"] for k in inhibitor_keys if k in results]
    ena_diffs = [results[k]["difference"] for k in enabler_keys if k in results]
    
    print(f"\n  INHIBITORS avg difference: {np.mean(inh_diffs):.3f}")
    print(f"  ENABLERS avg difference:   {np.mean(ena_diffs):.3f}")
    
    # The key insight: when model is INCORRECT, what are heads doing?
    print("\n  When model predicts INCORRECTLY:")
    
    for key in inhibitor_keys + enabler_keys:
        if key in results:
            proj = results[key]["avg_projection_when_incorrect"]
            head_type = "INHIBITOR" if key in inhibitor_keys else "ENABLER"
            if proj > 0:
                print(f"    {key} [{head_type}]: Still pushing toward correct! ({proj:+.3f})")
            else:
                print(f"    {key} [{head_type}]: Pushing toward WRONG ({proj:+.3f})")
    
    print("\n" + "-" * 60)
    print("INTERPRETATION:")
    print("-" * 60)
    
    # Count how many heads push wrong when model is incorrect
    inh_wrong_push = sum(1 for k in inhibitor_keys if k in results and results[k]["avg_projection_when_incorrect"] < 0)
    ena_wrong_push = sum(1 for k in enabler_keys if k in results and results[k]["avg_projection_when_incorrect"] < 0)
    
    print(f"\n  Inhibitors pushing wrong when incorrect: {inh_wrong_push}/{len(inhibitor_keys)}")
    print(f"  Enablers pushing wrong when incorrect:   {ena_wrong_push}/{len(enabler_keys)}")
    
    if inh_wrong_push > ena_wrong_push:
        print("\n  --> INHIBITORS are more likely to push toward wrong answer")
        print("      when the model makes a mistake!")
    elif ena_wrong_push > inh_wrong_push:
        print("\n  --> ENABLERS are pushing wrong when model errs")
        print("      (surprising!)")
    else:
        print("\n  --> Both types behave similarly")
    
    # Save
    with open(RESULTS_DIR / "direction_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Saved to {RESULTS_DIR / 'direction_analysis.json'}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

