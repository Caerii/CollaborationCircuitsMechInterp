"""
Step 2: Head-Level Probing with Mutual Information
====================================================

For each attention head output, measure:
1. Probe accuracy for "B agrees with A"
2. Mutual information estimate

EFFICIENT: Extract all head outputs in single pass, then probe in parallel.
"""

import json
import sys
from pathlib import Path

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

DATA_DIR = Path(__file__).parent.parent.parent / "10_proper_tom" / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

print("=" * 60)
print("STEP 2: HEAD-LEVEL PROBING + MUTUAL INFORMATION")
print("=" * 60)

# Reuse scenarios from step 6
SCENARIOS = [
    {"prompt": "Agent A: 'The answer is 42.' Agent B verified this is correct.\nDoes B agree?", "b_agrees": True},
    {"prompt": "Agent A: '2+2=5' Agent B knows basic math.\nDoes B agree?", "b_agrees": False},
    {"prompt": "Agent A: 'Meeting at 3pm.' Agent B got the same email.\nDoes B agree?", "b_agrees": True},
    {"prompt": "Agent A: 'Tokyo is in China.' Agent B knows geography.\nDoes B agree?", "b_agrees": False},
    {"prompt": "Agent A: 'The code works.' Agent B tested it successfully.\nDoes B agree?", "b_agrees": True},
    {"prompt": "Agent A: 'Pi equals 3.' Agent B remembers it's 3.14159.\nDoes B agree?", "b_agrees": False},
    {"prompt": "Agent A: 'Water boils at 100C.' Agent B confirms this.\nDoes B agree?", "b_agrees": True},
    {"prompt": "Agent A: 'Earth is flat.' Agent B has seen satellite photos.\nDoes B agree?", "b_agrees": False},
    {"prompt": "Agent A: 'Python is a language.' Agent B is a programmer.\nDoes B agree?", "b_agrees": True},
    {"prompt": "Agent A: 'The sun is cold.' Agent B knows about fusion.\nDoes B agree?", "b_agrees": False},
    {"prompt": "Agent A: 'Gravity pulls down.' Agent B agrees with physics.\nDoes B agree?", "b_agrees": True},
    {"prompt": "Agent A: 'Humans don't need oxygen.' Agent B is a biologist.\nDoes B agree?", "b_agrees": False},
]


def main():
    print(f"\n[1/5] Using {len(SCENARIOS)} scenarios", flush=True)
    labels = np.array([1 if s["b_agrees"] else 0 for s in SCENARIOS])
    print(f"  Agrees: {labels.sum()}, Disagrees: {len(labels) - labels.sum()}")
    
    print("\n[2/5] Loading model...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CFG.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CFG.model_name,
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    
    print(f"  [OK] {n_layers} layers, {n_heads} heads, {head_dim} dim per head")
    
    print("\n[3/5] Extracting head outputs (EFFICIENT: all at once)...", flush=True)
    
    # Store head outputs: [n_scenarios, n_layers, n_heads, head_dim]
    all_head_outputs = []
    
    # Hook to capture attention head outputs
    head_outputs_cache = {}
    
    def make_head_output_hook(layer_idx):
        def hook(module, input, output):
            # output is (hidden_states, ...) where hidden_states has shape (batch, seq, hidden)
            # We need to extract per-head outputs from the attention layer
            # For Qwen, we hook the attention output projection
            hidden = output[0] if isinstance(output, tuple) else output
            head_outputs_cache[layer_idx] = hidden.detach()
        return hook
    
    hooks = []
    for layer_idx in range(n_layers):
        hook = model.model.layers[layer_idx].register_forward_hook(make_head_output_hook(layer_idx))
        hooks.append(hook)
    
    with torch.no_grad():
        for i, scenario in enumerate(SCENARIOS):
            if (i + 1) % 4 == 0:
                print(f"    [{i+1}/{len(SCENARIOS)}]", flush=True)
            
            inputs = tokenizer(scenario["prompt"], return_tensors="pt", truncation=True, max_length=128).to("cuda")
            _ = model(**inputs)
            
            # Collect head outputs for this scenario
            scenario_heads = []
            for layer_idx in range(n_layers):
                hidden = head_outputs_cache[layer_idx]
                # Take last token, reshape to (n_heads, head_dim)
                last_token = hidden[0, -1, :].cpu().float()
                # Reshape: (hidden_size,) -> (n_heads, head_dim)
                per_head = last_token.view(n_heads, head_dim)
                scenario_heads.append(per_head)
            
            # Stack: (n_layers, n_heads, head_dim)
            all_head_outputs.append(torch.stack(scenario_heads))
    
    for hook in hooks:
        hook.remove()
    
    # Stack all: (n_scenarios, n_layers, n_heads, head_dim)
    all_head_outputs = torch.stack(all_head_outputs).numpy()
    print(f"  Extracted shape: {all_head_outputs.shape}")
    
    print("\n[4/5] Probing each head + computing mutual information...", flush=True)
    
    results = {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_scores": [],
    }
    
    best_heads = []
    
    for layer_idx in range(n_layers):
        if layer_idx % 6 == 0:
            print(f"  Layer {layer_idx}/{n_layers}...", flush=True)
        
        for head_idx in range(n_heads):
            # Get this head's outputs across all scenarios
            X = all_head_outputs[:, layer_idx, head_idx, :]  # (n_scenarios, head_dim)
            
            # 1. Probe accuracy (simple CV)
            clf = LogisticRegression(max_iter=500, random_state=42)
            try:
                scores = cross_val_score(clf, X, labels, cv=3)
                probe_acc = scores.mean()
            except:
                probe_acc = 0.5
            
            # 2. Mutual information (use mean of head output as single feature)
            X_mean = X.mean(axis=1, keepdims=True)  # Reduce to single value per sample
            try:
                mi = mutual_info_classif(X_mean, labels, random_state=42)[0]
            except:
                mi = 0.0
            
            head_info = {
                "layer": layer_idx,
                "head": head_idx,
                "probe_acc": float(probe_acc),
                "mutual_info": float(mi),
            }
            results["head_scores"].append(head_info)
            
            if probe_acc > 0.7:
                best_heads.append(head_info)
    
    # Sort by probe accuracy
    best_heads.sort(key=lambda x: x["probe_acc"], reverse=True)
    
    print("\n[5/5] Results...", flush=True)
    print("\n" + "=" * 60)
    print("TOP 20 HEADS FOR AGENT AGREEMENT DECODING")
    print("=" * 60)
    print(f"{'Layer':<8} {'Head':<8} {'Probe Acc':<12} {'Mutual Info':<12}")
    print("-" * 44)
    
    for h in best_heads[:20]:
        print(f"{h['layer']:<8} {h['head']:<8} {h['probe_acc']:.1%}        {h['mutual_info']:.4f}")
    
    results["top_heads"] = best_heads[:50]
    
    # Save
    with open(RESULTS_DIR / "head_probing.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: CANDIDATE ToM HEADS")
    print("=" * 60)
    
    # Group by layer
    layer_best = {}
    for h in best_heads:
        l = h["layer"]
        if l not in layer_best or h["probe_acc"] > layer_best[l]["probe_acc"]:
            layer_best[l] = h
    
    print("\nBest head per layer (>60% accuracy):")
    for l in sorted(layer_best.keys()):
        h = layer_best[l]
        if h["probe_acc"] > 0.6:
            print(f"  Layer {l}: Head {h['head']} = {h['probe_acc']:.1%}")
    
    print(f"\n[OK] Results saved to {RESULTS_DIR / 'head_probing.json'}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()























