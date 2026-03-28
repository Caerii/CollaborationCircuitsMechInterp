"""
Step 2: Extract Activations
===========================

Load model, extract activations from scenarios.
OPTIMIZED: Load model once, batch processing, direct HuggingFace.
"""

import json
import sys
from pathlib import Path

import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("STEP 2: EXTRACT ACTIVATIONS (OPTIMIZED)")
print("=" * 60)


def main():
    # Load scenarios
    print("\n[1/3] Loading scenarios...", flush=True)
    with open(DATA_DIR / "scenarios.json") as f:
        scenarios = json.load(f)
    print(f"  Loaded {len(scenarios)} scenarios", flush=True)
    
    # Load model DIRECTLY with HuggingFace (faster than nnsight wrapper)
    print("\n[2/3] Loading model directly via HuggingFace...", flush=True)
    print(f"  Model: {MODEL_CFG.model_name}", flush=True)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("  Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CFG.model_name,
        trust_remote_code=True,
    )
    print("  [OK] Tokenizer loaded!", flush=True)
    
    print("  Loading model weights...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CFG.model_name,
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    print("  [OK] Model loaded and ready!", flush=True)
    
    # Extract activations using hooks (MUCH faster than nnsight per-call)
    print("\n[3/3] Extracting activations with hooks...", flush=True)
    
    layers_to_hook = [0, 8, 16, 20, 24, 28, 35]
    
    # Storage
    all_activations = {layer: [] for layer in layers_to_hook}
    labels = []
    
    # Set up hooks
    captured = {}
    hooks = []
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            # output is tuple, first element is hidden states
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden.detach()
        return hook
    
    # Register hooks on target layers
    for layer_idx in layers_to_hook:
        hook = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)
    
    print(f"  Registered {len(hooks)} hooks on layers {layers_to_hook}", flush=True)
    
    total = len(scenarios)
    print(f"  Processing {total} scenarios...", flush=True)
    
    with torch.no_grad():
        for i, scenario in enumerate(scenarios):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{total}] {scenario['id']}", flush=True)
            
            # Tokenize
            inputs = tokenizer(
                scenario["dialogue"], 
                return_tensors="pt",
                truncation=True, 
                max_length=512
            ).to("cuda")
            
            # Forward pass triggers hooks
            _ = model(**inputs)
            
            # Extract last token activation from each layer
            for layer_idx in layers_to_hook:
                hidden = captured[layer_idx]  # [batch, seq_len, hidden]
                last_token = hidden[0, -1, :].cpu().float()
                all_activations[layer_idx].append(last_token)
            
            # Label
            labels.append(1 if scenario["condition"] == "private" else 0)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    print("  [OK] Hooks removed", flush=True)
    
    print("\n  Stacking tensors...", flush=True)
    
    # Stack into tensors
    for layer in layers_to_hook:
        all_activations[layer] = torch.stack(all_activations[layer])
        print(f"    Layer {layer}: {all_activations[layer].shape}", flush=True)
    
    labels = np.array(labels)
    
    # Save
    output_path = RESULTS_DIR / "activations.pt"
    torch.save({
        "activations": all_activations,
        "labels": labels,
        "layers": layers_to_hook
    }, output_path)
    
    print(f"\n[OK] Saved activations to {output_path}")
    print(f"  Layers: {layers_to_hook}")
    print(f"  Shape per layer: {all_activations[layers_to_hook[0]].shape}")
    print(f"  Labels: {labels.sum()} private, {len(labels) - labels.sum()} shared")
    
    # Free GPU
    del model
    torch.cuda.empty_cache()
    print("\n[OK] GPU memory freed")


if __name__ == "__main__":
    main()
