"""
Step 2: Probe the NARRATIVE for Belief vs Reality
==================================================

KEY CORRECTION: Previous experiments probed Q&A prompts.
Zhu et al. probe the NARRATIVE ITSELF for belief representations.

We extract activations from the STORY (before any question) and test:
1. Can we decode believed_location from story activations?
2. Can we decode actual_location from story activations?
3. Are these directions DIFFERENT?
4. Does this survive counter-balancing (not just "first location")?
"""

import json
import sys
import warnings
from pathlib import Path

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Suppress sklearn convergence warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', module='sklearn')

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STEP 2: PROBE NARRATIVE (NOT Q&A)")
print("=" * 60)


def extract_activations(model, tokenizer, stories, layers):
    """Extract activations from STORY ONLY (no questions)."""
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
        for i, story in enumerate(stories):
            if (i + 1) % 20 == 0:
                print(f"    [{i+1}/{len(stories)}]", flush=True)
            
            inputs = tokenizer(story, return_tensors="pt", truncation=True, max_length=256).to("cuda")
            _ = model(**inputs)
            
            for layer_idx in layers:
                hidden = captured[layer_idx]
                # Take MEAN over all tokens (holistic story representation)
                mean_activation = hidden[0].mean(dim=0).cpu().float()
                all_activations[layer_idx].append(mean_activation)
    
    for hook in hooks:
        hook.remove()
    
    for layer in layers:
        all_activations[layer] = torch.stack(all_activations[layer])
    
    return all_activations


def main():
    print("\n[1/5] Loading data...", flush=True)
    with open(DATA_DIR / "scenarios_counterbalanced.json") as f:
        scenarios = json.load(f)
    
    # Filter to false belief only
    false_belief = [s for s in scenarios if s["version"] != "true_belief"]
    print(f"  Using {len(false_belief)} false belief scenarios")
    
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
    
    print("\n[3/5] Extracting STORY activations (no Q&A)...", flush=True)
    stories = [s["story"] for s in false_belief]
    layers = [0, 12, 24, 35]
    activations = extract_activations(model, tokenizer, stories, layers)
    
    # Create labels
    unique_locs = sorted(set(s["believed_location"] for s in false_belief) | 
                         set(s["actual_location"] for s in false_belief))
    loc_to_idx = {loc: i for i, loc in enumerate(unique_locs)}
    
    believed_labels = np.array([loc_to_idx[s["believed_location"]] for s in false_belief])
    actual_labels = np.array([loc_to_idx[s["actual_location"]] for s in false_belief])
    first_labels = np.array([loc_to_idx[s["first_location_mentioned"]] for s in false_belief])
    
    print(f"  Unique locations: {unique_locs}")
    print(f"  Believed labels distribution: {np.bincount(believed_labels)}")
    print(f"  Actual labels distribution: {np.bincount(actual_labels)}")
    
    print("\n[4/5] Testing for shortcut heuristic...", flush=True)
    
    results = {"layers": layers, "analysis": {}}
    
    for layer in layers:
        print(f"\n  === Layer {layer} ===", flush=True)
        
        X = activations[layer].numpy()
        
        # Ensure we have enough samples for stratified CV
        min_class_count = min(np.bincount(believed_labels))
        n_splits = min(5, min_class_count)
        if n_splits < 2:
            print(f"    Not enough samples for CV (min class: {min_class_count})")
            continue
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        
        # 1. Decode BELIEVED location
        try:
            believed_scores = cross_val_score(clf, X, believed_labels, cv=cv)
            believed_acc = believed_scores.mean()
        except Exception as e:
            print(f"    Believed decode failed: {e}")
            believed_acc = 0
        print(f"    Decode BELIEVED location: {believed_acc:.1%}")
        
        # 2. Decode ACTUAL location  
        try:
            actual_scores = cross_val_score(clf, X, actual_labels, cv=cv)
            actual_acc = actual_scores.mean()
        except Exception as e:
            print(f"    Actual decode failed: {e}")
            actual_acc = 0
        print(f"    Decode ACTUAL location: {actual_acc:.1%}")
        
        # 3. CRITICAL: Decode FIRST-MENTIONED location
        # If this is high, model is using shortcut heuristic!
        try:
            first_scores = cross_val_score(clf, X, first_labels, cv=cv)
            first_acc = first_scores.mean()
        except Exception as e:
            print(f"    First decode failed: {e}")
            first_acc = 0
        print(f"    Decode FIRST-MENTIONED location: {first_acc:.1%}")
        
        # 4. Compute directions and check orthogonality
        try:
            clf_believed = LogisticRegression(max_iter=1000, random_state=42)
            clf_believed.fit(X, believed_labels)
            clf_actual = LogisticRegression(max_iter=1000, random_state=42)
            clf_actual.fit(X, actual_labels)
            
            believed_dir = clf_believed.coef_.mean(axis=0)
            believed_dir = believed_dir / np.linalg.norm(believed_dir)
            
            actual_dir = clf_actual.coef_.mean(axis=0)
            actual_dir = actual_dir / np.linalg.norm(actual_dir)
            
            cosine = np.abs(np.dot(believed_dir, actual_dir))
        except:
            cosine = 1.0
        print(f"    Believed-Actual direction cosine: {cosine:.3f}")
        
        # Chance level
        chance = 1.0 / len(unique_locs)
        
        results["analysis"][str(layer)] = {
            "believed_acc": float(believed_acc),
            "actual_acc": float(actual_acc),
            "first_mentioned_acc": float(first_acc),
            "belief_reality_cosine": float(cosine),
            "chance": float(chance),
            "n_classes": len(unique_locs),
        }
    
    print("\n[5/5] Analyzing shortcut heuristic...", flush=True)
    
    # KEY TEST: If first_mentioned_acc >> believed_acc, model uses shortcut
    for layer in layers:
        if str(layer) not in results["analysis"]:
            continue
        r = results["analysis"][str(layer)]
        believed = r["believed_acc"]
        first = r["first_mentioned_acc"]
        
        if first > believed * 1.1:  # First-mentioned predicts better
            print(f"  Layer {layer}: SHORTCUT DETECTED! First-mention ({first:.1%}) > Believed ({believed:.1%})")
        elif believed > first * 1.1:  # Believed predicts better
            print(f"  Layer {layer}: True belief tracking! Believed ({believed:.1%}) > First-mention ({first:.1%})")
        else:
            print(f"  Layer {layer}: Unclear - Believed ({believed:.1%}) ~ First-mention ({first:.1%})")
    
    # Save results
    with open(RESULTS_DIR / "narrative_probe_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: SHORTCUT HEURISTIC TEST")
    print("=" * 60)
    print(f"\nChance level: {1/len(unique_locs):.1%}")
    print("\nLayer-by-layer:")
    print("-" * 60)
    print(f"{'Layer':<8} {'Believed':<12} {'Actual':<12} {'First-Ment':<12} {'Cosine':<10}")
    print("-" * 60)
    for layer in layers:
        if str(layer) not in results["analysis"]:
            continue
        r = results["analysis"][str(layer)]
        print(f"{layer:<8} {r['believed_acc']:.1%}        {r['actual_acc']:.1%}        {r['first_mentioned_acc']:.1%}        {r['belief_reality_cosine']:.3f}")
    
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    # Average across layers
    avg_believed = np.mean([results["analysis"][str(l)]["believed_acc"] for l in layers if str(l) in results["analysis"]])
    avg_actual = np.mean([results["analysis"][str(l)]["actual_acc"] for l in layers if str(l) in results["analysis"]])
    avg_first = np.mean([results["analysis"][str(l)]["first_mentioned_acc"] for l in layers if str(l) in results["analysis"]])
    avg_cosine = np.mean([results["analysis"][str(l)]["belief_reality_cosine"] for l in layers if str(l) in results["analysis"]])
    
    print(f"\n  Avg Believed decode: {avg_believed:.1%}")
    print(f"  Avg Actual decode: {avg_actual:.1%}")
    print(f"  Avg First-mention decode: {avg_first:.1%}")
    print(f"  Avg Belief-Reality cosine: {avg_cosine:.3f}")
    
    if avg_first > avg_believed:
        print("\n>>> WARNING: Model may use FIRST-MENTION shortcut! <<<")
        print("    Previous 'ToM success' may be spurious.")
    elif avg_believed > 0.3 and avg_cosine < 0.5:
        print("\n>>> EVIDENCE FOR TRUE BELIEF TRACKING <<<")
        print("    Model separates belief from reality beyond shortcuts.")
    else:
        print("\n>>> INCONCLUSIVE <<<")
        print("    Need more data or different approach.")
    
    print(f"\n[OK] Results saved to {RESULTS_DIR / 'narrative_probe_results.json'}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


