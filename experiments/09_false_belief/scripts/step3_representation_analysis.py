"""
Step 3: Representation Analysis
================================

Extract activations and test:
1. Can we decode the BELIEVED location from activations?
2. Can we decode the ACTUAL location from activations?
3. Are these represented in DIFFERENT directions?
"""

import json
import sys
from pathlib import Path

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

print("=" * 60)
print("STEP 3: REPRESENTATION ANALYSIS")
print("=" * 60)


def extract_activations(model, tokenizer, texts, layers):
    """Extract activations using hooks."""
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
        for i, text in enumerate(texts):
            if (i + 1) % 20 == 0:
                print(f"    [{i+1}/{len(texts)}]", flush=True)
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
            _ = model(**inputs)
            
            for layer_idx in layers:
                hidden = captured[layer_idx]
                last_token = hidden[0, -1, :].cpu().float()
                all_activations[layer_idx].append(last_token)
    
    for hook in hooks:
        hook.remove()
    
    for layer in layers:
        all_activations[layer] = torch.stack(all_activations[layer])
    
    return all_activations


def main():
    print("\n[1/4] Loading data...", flush=True)
    with open(DATA_DIR / "prompts.json") as f:
        prompts = json.load(f)
    
    # Filter to false belief scenarios only (where belief != reality)
    false_belief_prompts = [p for p in prompts if p["is_false_belief"]][:50]
    print(f"  Using {len(false_belief_prompts)} false belief scenarios")
    
    print("\n[2/4] Loading model...", flush=True)
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
    
    print("\n[3/4] Extracting activations...", flush=True)
    
    layers = [0, 8, 16, 24, 35]
    
    # Extract from belief prompts (asking where agent THINKS it is)
    print("  Extracting belief prompt activations...")
    belief_texts = [p["belief_prompt"] for p in false_belief_prompts]
    belief_activations = extract_activations(model, tokenizer, belief_texts, layers)
    
    # Extract from reality prompts (asking where it ACTUALLY is)
    print("  Extracting reality prompt activations...")
    reality_texts = [p["reality_prompt"] for p in false_belief_prompts]
    reality_activations = extract_activations(model, tokenizer, reality_texts, layers)
    
    # Create labels
    # For belief prompts: the BELIEVED location
    # For reality prompts: the ACTUAL location
    # In false belief, these are DIFFERENT
    
    # Get unique locations
    all_believed = [p["believed_location"] for p in false_belief_prompts]
    all_actual = [p["actual_location"] for p in false_belief_prompts]
    unique_locations = sorted(set(all_believed + all_actual))
    loc_to_idx = {loc: i for i, loc in enumerate(unique_locations)}
    
    belief_labels = np.array([loc_to_idx[p["believed_location"]] for p in false_belief_prompts])
    reality_labels = np.array([loc_to_idx[p["actual_location"]] for p in false_belief_prompts])
    
    print(f"  Unique locations: {unique_locations}")
    print(f"  Belief labels: {np.bincount(belief_labels)}")
    print(f"  Reality labels: {np.bincount(reality_labels)}")
    
    print("\n[4/4] Analyzing representations...", flush=True)
    
    results = {"layers": layers, "analysis": {}}
    
    for layer in layers:
        print(f"\n  === Layer {layer} ===", flush=True)
        
        X_belief = belief_activations[layer].numpy()
        X_reality = reality_activations[layer].numpy()
        
        # 1. Can we decode BELIEVED location from belief prompts?
        clf = LogisticRegression(max_iter=500, random_state=42)
        belief_scores = cross_val_score(clf, X_belief, belief_labels, cv=5)
        belief_acc = belief_scores.mean()
        print(f"    Decode BELIEVED location: {belief_acc:.1%}")
        
        # 2. Can we decode ACTUAL location from reality prompts?
        reality_scores = cross_val_score(clf, X_reality, reality_labels, cv=5)
        reality_acc = reality_scores.mean()
        print(f"    Decode ACTUAL location: {reality_acc:.1%}")
        
        # 3. KEY TEST: From BELIEF prompts, can we still decode the ACTUAL location?
        # If the model truly separates belief from reality, this should be LOWER
        cross_scores = cross_val_score(clf, X_belief, reality_labels, cv=5)
        cross_acc = cross_scores.mean()
        print(f"    Cross-decode (belief prompt -> actual loc): {cross_acc:.1%}")
        
        # 4. Compute directions and check orthogonality
        clf_belief = LogisticRegression(max_iter=500, random_state=42)
        clf_belief.fit(X_belief, belief_labels)
        
        clf_reality = LogisticRegression(max_iter=500, random_state=42)
        clf_reality.fit(X_reality, reality_labels)
        
        # Get mean directions (average of OvR coefficients)
        belief_dir = clf_belief.coef_.mean(axis=0)
        belief_dir = belief_dir / np.linalg.norm(belief_dir)
        
        reality_dir = clf_reality.coef_.mean(axis=0)
        reality_dir = reality_dir / np.linalg.norm(reality_dir)
        
        cosine = np.abs(np.dot(belief_dir, reality_dir))
        print(f"    Belief-Reality cosine: {cosine:.3f}")
        
        # Chance level for location classification
        n_classes = len(unique_locations)
        chance = 1.0 / n_classes
        
        results["analysis"][str(layer)] = {
            "belief_decode_acc": float(belief_acc),
            "reality_decode_acc": float(reality_acc),
            "cross_decode_acc": float(cross_acc),
            "belief_reality_cosine": float(cosine),
            "chance_level": float(chance),
            "n_classes": n_classes,
        }
    
    # Save results
    with open(RESULTS_DIR / "representation_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("REPRESENTATION ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\nChance level: {1/len(unique_locations):.1%} ({len(unique_locations)} locations)")
    
    print("\nLayer-by-layer results:")
    print("-" * 60)
    print(f"{'Layer':<8} {'Belief':<10} {'Reality':<10} {'Cross':<10} {'Cosine':<10}")
    print("-" * 60)
    
    for layer in layers:
        r = results["analysis"][str(layer)]
        print(f"{layer:<8} {r['belief_decode_acc']:.1%}      {r['reality_decode_acc']:.1%}      {r['cross_decode_acc']:.1%}      {r['belief_reality_cosine']:.3f}")
    
    print("\n" + "-" * 60)
    print("INTERPRETATION:")
    
    # Average metrics
    avg_belief = np.mean([results["analysis"][str(l)]["belief_decode_acc"] for l in layers])
    avg_reality = np.mean([results["analysis"][str(l)]["reality_decode_acc"] for l in layers])
    avg_cross = np.mean([results["analysis"][str(l)]["cross_decode_acc"] for l in layers])
    avg_cosine = np.mean([results["analysis"][str(l)]["belief_reality_cosine"] for l in layers])
    
    print(f"\n  Avg belief decode: {avg_belief:.1%}")
    print(f"  Avg reality decode: {avg_reality:.1%}")
    print(f"  Avg cross decode: {avg_cross:.1%}")
    print(f"  Avg cosine: {avg_cosine:.3f}")
    
    if avg_belief > 0.4 and avg_reality > 0.4 and avg_cross < avg_belief * 0.8:
        print("\n>>> EVIDENCE FOR BELIEF-REALITY SEPARATION <<<")
        print("    Model encodes believed and actual locations differently!")
    elif avg_belief > 0.3 or avg_reality > 0.3:
        print("\n>>> PARTIAL EVIDENCE <<<")
        print("    Some location encoding, but separation unclear")
    else:
        print("\n>>> WEAK EVIDENCE <<<")
        print("    Location encoding is weak or not separable")
    
    print(f"\n[OK] Results saved to {RESULTS_DIR / 'representation_analysis.json'}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
























