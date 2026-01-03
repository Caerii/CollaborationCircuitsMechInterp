"""
Probe for Agent Presence Tracking
==================================

Hypothesis: Model fails at belief updates because it doesn't track
which agent was PRESENT when information changed.

This experiment probes for representations of:
- "Agent X was in room when event Y happened"
- "Agent X left before event Y"

If we can decode this, the circuit for it exists but isn't used for ToM.
If we CAN'T decode this, the model lacks this fundamental capability.
"""

import json
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy import stats
import random

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_presence_scenarios(n: int = 200) -> list:
    """
    Generate scenarios where we vary whether agent was present during an event.
    
    Key: Same event happens, but agent either SAW it or DIDN'T.
    """
    random.seed(42)
    
    AGENTS = ["Alice", "Bob", "Carol", "David", "Emma", "Frank"]
    OBJECTS = ["ball", "key", "book", "phone", "wallet", "box"]
    LOCATIONS = ["drawer", "basket", "cupboard", "shelf", "desk", "table"]
    
    scenarios = []
    
    for i in range(n):
        agents = random.sample(AGENTS, 2)
        observer, mover = agents
        obj = random.choice(OBJECTS)
        loc1, loc2 = random.sample(LOCATIONS, 2)
        
        # VERSION A: Observer was PRESENT (saw the move)
        story_present = (
            f"{observer} and {mover} are in the room. "
            f"The {obj} is in the {loc1}. "
            f"{mover} moves the {obj} to the {loc2}. "
            f"{observer} watches this happen."
        )
        
        # VERSION B: Observer was ABSENT (didn't see the move)
        story_absent = (
            f"{observer} and {mover} are in the room. "
            f"The {obj} is in the {loc1}. "
            f"{observer} leaves the room. "
            f"{mover} moves the {obj} to the {loc2}. "
            f"{observer} returns."
        )
        
        scenarios.append({
            "id": f"presence_{i}_present",
            "story": story_present,
            "observer": observer,
            "mover": mover,
            "object": obj,
            "original_loc": loc1,
            "new_loc": loc2,
            "observer_saw_move": True,
            "observer_belief": loc2,  # Saw it move, knows new location
        })
        
        scenarios.append({
            "id": f"presence_{i}_absent",
            "story": story_absent,
            "observer": observer,
            "mover": mover,
            "object": obj,
            "original_loc": loc1,
            "new_loc": loc2,
            "observer_saw_move": False,
            "observer_belief": loc1,  # Didn't see, still believes original
        })
    
    return scenarios


def extract_activations(model, tokenizer, texts: list, layers: list) -> dict:
    """Extract activations at specified layers for each text."""
    activations = {layer: [] for layer in layers}
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Get last token activation at each layer
        for layer in layers:
            act = outputs.hidden_states[layer][0, -1, :].cpu().numpy()
            activations[layer].append(act)
    
    return {layer: np.array(acts) for layer, acts in activations.items()}


def main():
    timer_start = time.perf_counter()
    
    print("=" * 60)
    print("PROBE FOR AGENT PRESENCE TRACKING")
    print("=" * 60)
    print("\nQuestion: Does model encode 'who was present when event happened'?")
    
    # Load model
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
    n_layers = model.config.num_hidden_layers
    print(f"  [OK] {n_layers} layers", flush=True)
    
    # Generate scenarios
    print("\n[2/5] Generating scenarios...", flush=True)
    scenarios = generate_presence_scenarios(200)
    print(f"  Generated {len(scenarios)} scenarios (half present, half absent)")
    
    # Extract activations
    print("\n[3/5] Extracting activations...", flush=True)
    texts = [s["story"] for s in scenarios]
    labels = np.array([1 if s["observer_saw_move"] else 0 for s in scenarios])
    
    # Sample layers across depth
    probe_layers = [0, 6, 12, 18, 24, 30, n_layers - 1]
    
    activations = {}
    for layer in probe_layers:
        print(f"  Layer {layer}...", flush=True)
        layer_acts = []
        for i, text in enumerate(texts):
            if i % 50 == 0:
                print(f"    [{i}/{len(texts)}]", flush=True)
            
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            act = outputs.hidden_states[layer][0, -1, :].cpu().numpy()
            layer_acts.append(act)
        
        activations[layer] = np.array(layer_acts)
    
    # Train probes
    print("\n[4/5] Training presence probes...", flush=True)
    results = {"layers": {}}
    
    for layer in probe_layers:
        X = activations[layer]
        y = labels
        
        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        
        mean_acc = float(np.mean(scores))
        std_acc = float(np.std(scores))
        
        # Significance test
        n = len(y)
        binom = stats.binomtest(int(mean_acc * n), n, p=0.5, alternative='greater')
        
        results["layers"][layer] = {
            "accuracy": mean_acc,
            "std": std_acc,
            "p_value": float(binom.pvalue),
            "significant": bool(binom.pvalue < 0.05),
        }
        
        sig = "***" if mean_acc > 0.7 else "**" if mean_acc > 0.6 else "*" if mean_acc > 0.55 else ""
        print(f"  Layer {layer:2d}: {mean_acc:.1%} (+/- {std_acc:.1%}) {sig}")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    best_layer = max(results["layers"], key=lambda l: results["layers"][l]["accuracy"])
    best_acc = results["layers"][best_layer]["accuracy"]
    
    print(f"\n  Best layer: {best_layer} ({best_acc:.1%} accuracy)")
    
    print("\n  INTERPRETATION:")
    if best_acc >= 0.80:
        print("  [+++] STRONG presence encoding!")
        print("        Model DOES track who was present.")
        print("        The issue is downstream: not using this for belief updates.")
    elif best_acc >= 0.65:
        print("  [++] MODERATE presence encoding")
        print("       Model partially tracks presence, may explain partial ToM.")
    elif best_acc >= 0.55:
        print("  [+] WEAK presence encoding")
        print("      Model has limited ability to track who saw what.")
    else:
        print("  [-] NO presence encoding")
        print("      Model doesn't track who was present for events!")
        print("      This explains why belief updates fail.")
    
    # Save
    with open(RESULTS_DIR / "presence_probe_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    total_time = time.perf_counter() - timer_start
    print(f"\n[OK] Completed in {total_time:.1f}s")
    print(f"     Saved to {RESULTS_DIR / 'presence_probe_results.json'}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


