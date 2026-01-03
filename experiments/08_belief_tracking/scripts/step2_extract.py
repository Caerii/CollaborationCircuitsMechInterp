"""
Step 2: Extract Activations for Belief Tracking
================================================

Fast extraction using PyTorch hooks.
"""

import json
import sys
from pathlib import Path

import torch
import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STEP 2: EXTRACT ACTIVATIONS")
print("=" * 60)


def extract_activations(model, tokenizer, texts, layers_to_hook):
    """Extract last-token activations using hooks."""
    
    # Storage
    all_activations = {layer: [] for layer in layers_to_hook}
    captured = {}
    hooks = []
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden.detach()
        return hook
    
    # Register hooks
    for layer_idx in layers_to_hook:
        hook = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)
    
    print(f"  Extracting from {len(texts)} texts...", flush=True)
    
    with torch.no_grad():
        for i, text in enumerate(texts):
            if (i + 1) % 20 == 0 or i == 0:
                print(f"    [{i+1}/{len(texts)}]", flush=True)
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to("cuda")
            _ = model(**inputs)
            
            for layer_idx in layers_to_hook:
                hidden = captured[layer_idx]
                last_token = hidden[0, -1, :].cpu().float()
                all_activations[layer_idx].append(last_token)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Stack
    for layer in layers_to_hook:
        all_activations[layer] = torch.stack(all_activations[layer])
    
    return all_activations


def main():
    # Load data
    print("\n[1/4] Loading data...", flush=True)
    
    with open(DATA_DIR / "minimal_pairs.json") as f:
        pairs = json.load(f)
    with open(DATA_DIR / "belief_scenarios.json") as f:
        scenarios = json.load(f)
    
    print(f"  Minimal pairs: {len(pairs)}")
    print(f"  Belief scenarios: {len(scenarios)}")
    
    # Load model
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
    print("  [OK] Model loaded!", flush=True)
    
    layers = [0, 8, 16, 24, 35]  # Sample of layers
    
    # Extract for minimal pairs
    print("\n[3/4] Extracting minimal pair activations...", flush=True)
    pair_texts = [p["text"] for p in pairs]
    pair_activations = extract_activations(model, tokenizer, pair_texts, layers)
    
    # Extract labels
    pair_labels = {
        "agent": np.array([0 if p["agent"] == "Alice" else 1 for p in pairs]),
        "category": np.array([["password", "location", "plan", "fact"].index(p["content_category"]) for p in pairs]),
    }
    
    # Save minimal pairs
    torch.save({
        "activations": pair_activations,
        "labels": pair_labels,
        "layers": layers,
        "metadata": pairs,
    }, RESULTS_DIR / "minimal_pairs_activations.pt")
    print(f"  Saved minimal pairs to {RESULTS_DIR / 'minimal_pairs_activations.pt'}")
    
    # Extract for belief scenarios
    print("\n[4/4] Extracting belief scenario activations...", flush=True)
    scenario_texts = [s["text"] for s in scenarios]
    scenario_activations = extract_activations(model, tokenizer, scenario_texts, layers)
    
    # Extract labels
    scenario_labels = {
        "alice_knows": np.array([1 if s["alice_knows"] else 0 for s in scenarios]),
        "bob_knows": np.array([1 if s["bob_knows"] else 0 for s in scenarios]),
        "state": np.array([["neither", "alice_only", "bob_only", "both_know"].index(s["state"]) for s in scenarios]),
    }
    
    # Save scenarios
    torch.save({
        "activations": scenario_activations,
        "labels": scenario_labels,
        "layers": layers,
        "metadata": scenarios,
    }, RESULTS_DIR / "belief_scenarios_activations.pt")
    print(f"  Saved scenarios to {RESULTS_DIR / 'belief_scenarios_activations.pt'}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    print("\n[OK] Extraction complete!")
    print(f"  Layers: {layers}")
    print(f"  Minimal pairs shape: {pair_activations[layers[0]].shape}")
    print(f"  Scenarios shape: {scenario_activations[layers[0]].shape}")


if __name__ == "__main__":
    main()





















