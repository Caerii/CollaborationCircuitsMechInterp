"""
Per-Agent Belief Probing
========================

Test: Can we decode Agent A's belief SEPARATELY from Agent B's belief?

This is the key test for genuine multi-agent ToM:
- If model has separate representations: A's and B's probes will be different
- If model conflates agents: A's and B's beliefs will be encoded similarly
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_activations(model, tokenizer, prompts: List[str], layers: List[int]) -> Dict:
    """Extract activations for a list of prompts."""
    all_activations = {layer: [] for layer in layers}
    captured = {}
    hooks = []
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden.detach()
        return hook
    
    for layer_idx in layers:
        hook = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)
    
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
            _ = model(**inputs)
            
            for layer_idx in layers:
                hidden = captured[layer_idx]
                # Mean pool over sequence
                mean_activation = hidden[0].mean(dim=0).cpu().float()
                all_activations[layer_idx].append(mean_activation)
    
    for hook in hooks:
        hook.remove()
    
    for layer in layers:
        all_activations[layer] = torch.stack(all_activations[layer])
    
    return all_activations


def probe_agent_beliefs(activations: Dict, scenarios: List[dict], layers: List[int]) -> Dict:
    """
    Probe for each agent's belief separately.
    
    Key test: Can we decode A's belief independently of B's belief?
    """
    results = {"layers": {}, "summary": {}}
    
    # Get unique locations for label encoding
    all_locations = set()
    for s in scenarios:
        all_locations.add(s.get("target_belief", s.get("a_belief", "")))
    all_locations = sorted([l for l in all_locations if l])
    loc_to_idx = {loc: i for i, loc in enumerate(all_locations)}
    
    if len(loc_to_idx) < 2:
        print("  [!] Not enough location variety for probing")
        return results
    
    # Create labels based on target agent's belief
    labels = np.array([
        loc_to_idx.get(s.get("target_belief", s.get("a_belief", "")), 0) 
        for s in scenarios
    ])
    
    # Ensure we have enough samples per class
    unique, counts = np.unique(labels, return_counts=True)
    min_count = min(counts)
    
    print(f"  Label distribution: {dict(zip(unique, counts))}")
    print(f"  Min samples per class: {min_count}")
    
    if min_count < 5:
        print("  [!] Not enough samples per class for reliable CV")
    
    for layer in layers:
        print(f"\n  Layer {layer}:", flush=True)
        X = activations[layer].numpy()
        
        # Determine CV splits
        n_splits = min(5, min_count)
        if n_splits < 2:
            n_splits = 2
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Probe for target agent's belief
        clf = LogisticRegression(max_iter=1000, random_state=42)
        try:
            scores = cross_val_score(clf, X, labels, cv=cv)
            accuracy = scores.mean()
            std = scores.std()
        except Exception as e:
            print(f"    [!] CV failed: {e}")
            accuracy = 1.0 / len(unique)
            std = 0.0
        
        # Chance baseline
        chance = 1.0 / len(unique)
        
        # Fit full model for direction analysis
        clf_full = LogisticRegression(max_iter=1000, random_state=42)
        clf_full.fit(X, labels)
        
        results["layers"][str(layer)] = {
            "accuracy": float(accuracy),
            "std": float(std),
            "chance": float(chance),
            "n_classes": len(unique),
            "above_chance": accuracy > chance + 2 * std,
        }
        
        print(f"    Accuracy: {accuracy:.1%} (+/- {std:.1%})")
        print(f"    Chance: {chance:.1%}")
        print(f"    Above chance: {accuracy > chance + 2*std}")
    
    return results


def test_belief_separation(activations: Dict, scenarios: List[dict], layers: List[int]) -> Dict:
    """
    Test if A's belief direction is different from B's belief direction.
    
    For scenarios where both A and B have known beliefs, check if their
    encoding directions are orthogonal.
    """
    results = {"layers": {}}
    
    # Filter to scenarios that have both a_belief and b_belief
    dual_scenarios = [s for s in scenarios if "a_belief" in s and "b_belief" in s]
    
    if len(dual_scenarios) < 20:
        print(f"  [!] Only {len(dual_scenarios)} scenarios with dual beliefs")
        return results
    
    # Create labels for A and B separately
    all_locs = set()
    for s in dual_scenarios:
        all_locs.add(s["a_belief"])
        all_locs.add(s["b_belief"])
    all_locs = sorted(all_locs)
    loc_to_idx = {loc: i for i, loc in enumerate(all_locs)}
    
    # Get indices of dual scenarios in original list
    dual_indices = [i for i, s in enumerate(scenarios) if "a_belief" in s and "b_belief" in s]
    
    a_labels = np.array([loc_to_idx[scenarios[i]["a_belief"]] for i in dual_indices])
    b_labels = np.array([loc_to_idx[scenarios[i]["b_belief"]] for i in dual_indices])
    
    for layer in layers:
        X_dual = activations[layer][dual_indices].numpy()
        
        # Fit probes for A and B
        clf_a = LogisticRegression(max_iter=1000, random_state=42)
        clf_b = LogisticRegression(max_iter=1000, random_state=42)
        
        try:
            clf_a.fit(X_dual, a_labels)
            clf_b.fit(X_dual, b_labels)
            
            # Get mean directions
            dir_a = clf_a.coef_.mean(axis=0)
            dir_a = dir_a / (np.linalg.norm(dir_a) + 1e-8)
            
            dir_b = clf_b.coef_.mean(axis=0)
            dir_b = dir_b / (np.linalg.norm(dir_b) + 1e-8)
            
            # Compute cosine similarity
            cosine = np.abs(np.dot(dir_a, dir_b))
            
            results["layers"][str(layer)] = {
                "cosine_a_b": float(cosine),
                "interpretation": (
                    "ORTHOGONAL (different encoding)" if cosine < 0.3 else
                    "PARTIALLY ALIGNED" if cosine < 0.7 else
                    "ALIGNED (same encoding)"
                )
            }
            
            print(f"  Layer {layer}: A-B cosine = {cosine:.3f} [{results['layers'][str(layer)]['interpretation']}]")
            
        except Exception as e:
            print(f"  Layer {layer}: [!] Failed: {e}")
    
    return results


def main():
    timer_start = time.perf_counter()
    
    print("=" * 60)
    print("PER-AGENT BELIEF PROBING")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading scenarios...", flush=True)
    with open(DATA_DIR / "multi_agent_scenarios.json") as f:
        data = json.load(f)
    
    # Combine divergent scenarios (they have clear per-agent beliefs)
    all_scenarios = data["divergent"] + data["comparison"]
    print(f"  Total scenarios for probing: {len(all_scenarios)}")
    
    # Load model
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
    print("  [OK]", flush=True)
    
    # Extract activations
    print("\n[3/5] Extracting activations...", flush=True)
    prompts = [s["story"] for s in all_scenarios]
    layers = [0, 12, 18, 24, 30, 35]
    activations = extract_activations(model, tokenizer, prompts, layers)
    print(f"  Shape: {activations[layers[0]].shape}")
    
    all_results = {
        "n_scenarios": len(all_scenarios),
        "layers_tested": layers,
    }
    
    # Test 1: Probe for target agent's belief
    print("\n[4/5] Probing for target agent's belief...", flush=True)
    probe_results = probe_agent_beliefs(activations, all_scenarios, layers)
    all_results["target_belief_probing"] = probe_results
    
    # Test 2: Check if A and B beliefs are encoded differently
    print("\n[5/5] Testing A vs B belief separation...", flush=True)
    separation_results = test_belief_separation(activations, all_scenarios, layers)
    all_results["a_b_separation"] = separation_results
    
    # Compute null baseline for cosine
    print("\n  Computing random cosine baseline (d=2560)...", flush=True)
    n_random = 1000
    random_cosines = []
    d = activations[layers[0]].shape[1]
    for _ in range(n_random):
        v1 = np.random.randn(d)
        v2 = np.random.randn(d)
        v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
        random_cosines.append(np.abs(np.dot(v1, v2)))
    
    all_results["random_cosine_baseline"] = {
        "mean": float(np.mean(random_cosines)),
        "percentile_95": float(np.percentile(random_cosines, 95)),
        "d": int(d),
    }
    print(f"  Random baseline: mean={np.mean(random_cosines):.3f}, 95th={np.percentile(random_cosines, 95):.3f}")
    
    # Save
    with open(RESULTS_DIR / "per_agent_probing.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\n1. TARGET BELIEF PROBING:")
    print(f"{'Layer':<8} {'Accuracy':<12} {'Chance':<10} {'Above Chance':<12}")
    print("-" * 42)
    for layer in layers:
        r = probe_results["layers"].get(str(layer), {})
        if r:
            print(f"{layer:<8} {r['accuracy']:.1%}        {r['chance']:.1%}       {'YES' if r['above_chance'] else 'no'}")
    
    print("\n2. A vs B BELIEF SEPARATION:")
    print(f"Random baseline (d={d}): cos < {np.percentile(random_cosines, 95):.3f} expected")
    print(f"{'Layer':<8} {'Cosine':<10} {'Interpretation':<30}")
    print("-" * 48)
    for layer in layers:
        r = separation_results.get("layers", {}).get(str(layer), {})
        if r:
            print(f"{layer:<8} {r['cosine_a_b']:.3f}      {r['interpretation']}")
    
    print("\n3. INTERPRETATION:")
    # Check if A-B separation is real
    sep_cosines = [
        separation_results.get("layers", {}).get(str(l), {}).get("cosine_a_b", 1.0)
        for l in layers
    ]
    random_95 = np.percentile(random_cosines, 95)
    
    n_separated = sum(1 for c in sep_cosines if c < random_95)
    
    if n_separated >= len(layers) // 2:
        print("  [+] A and B beliefs appear to be encoded DIFFERENTLY")
        print("      This suggests genuine multi-agent belief tracking!")
    else:
        print("  [-] A and B beliefs may be encoded SIMILARLY")
        print("      Model may not distinguish per-agent beliefs")
    
    total_time = time.perf_counter() - timer_start
    print(f"\nCompleted in {total_time:.1f}s")
    print(f"Saved to {RESULTS_DIR / 'per_agent_probing.json'}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()



