"""
Step 15: Cross-Layer Feature Tracking

HOW DO BELIEF FEATURES EVOLVE THROUGH THE NETWORK?

METHOD:
1. Train SAEs on multiple layers (8, 12, 20, 28, 32)
2. Find discriminative features at each layer
3. Track how the representation evolves

OUTPUT: results/step15_crosslayer.json, figures/step15_*.png
"""

import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

# Add framework to path
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig
from analysis.sae_analysis import SimpleSAE, SAEConfig, SAETrainer

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def collect_mlp_activations(model, tokenizer, prompts, layer):
    """Collect MLP output activations for a list of prompts."""
    activations = []
    
    def hook(module, input, output):
        activations.append(output[0, -1, :].detach().cpu())
    
    mlp = model.model.layers[layer].mlp
    handle = mlp.register_forward_hook(hook)
    
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            model(**inputs)
    
    handle.remove()
    
    return torch.stack(activations)


def main():
    print("=" * 70)
    print("STEP 15: CROSS-LAYER FEATURE TRACKING")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    
    # Layers to analyze
    layers_to_analyze = [4, 8, 12, 16, 20, 24, 28, 32]
    
    # Scenarios
    scenarios = [
        ("Alice put the ball in the drawer. Alice left. Bob moved it to the basket. Where does Alice think the ball is?", "false_belief"),
        ("Tom hid the key in the box. Tom left. Jerry moved the key to the drawer. Where does Tom think the key is?", "false_belief"),
        ("Chef put ingredients in cabinet A. Chef left. Waiter moved them to cabinet B. Where does Chef think they are?", "false_belief"),
        ("Sally put the toy in the basket. Sally went outside. Anne moved it to the box. Where does Sally think it is?", "false_belief"),
        ("Alice put the ball in the drawer. Alice stayed and watched. Bob moved it to the basket. Where does Alice think the ball is?", "true_belief"),
        ("Tom hid the key in the box. Tom watched Jerry move the key to the drawer. Where does Tom think the key is?", "true_belief"),
        ("Chef put ingredients in cabinet A. Chef saw Waiter move them to cabinet B. Where does Chef think they are?", "true_belief"),
        ("Sally put the toy in the basket. Sally watched Anne move it to the box. Where does Sally think it is?", "true_belief"),
    ]
    
    prompts = [s[0] for s in scenarios]
    labels = [s[1] for s in scenarios]
    
    # Load model
    print("\nLoading model...")
    sys.stdout.flush()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded!")
    sys.stdout.flush()
    
    d_model = model.config.hidden_size
    
    # ========================================
    # TRAIN SAEs FOR EACH LAYER
    # ========================================
    print(f"\n{'='*60}")
    print("TRAINING SAEs FOR EACH LAYER")
    print(f"{'='*60}")
    
    layer_results = {}
    
    for layer in layers_to_analyze:
        print(f"\n--- Layer {layer} ---")
        sys.stdout.flush()
        
        # Collect activations
        activations = collect_mlp_activations(model, tokenizer, prompts, layer)
        activations = activations.float()
        
        # Train SAE
        sae_config = SAEConfig(d_model=d_model, d_sae=d_model * 4, l1_coeff=1e-3, lr=1e-3)
        sae = SimpleSAE(sae_config)
        trainer = SAETrainer(sae, lr=sae_config.lr)
        
        for epoch in range(300):  # Faster training
            loss = trainer.step(activations)
        
        print(f"  Final loss: {trainer.losses[-1]:.4f}")
        
        # Analyze features
        sae.eval()
        with torch.no_grad():
            features = sae.get_feature_activations(activations)
        
        fb_mask = torch.tensor([l == "false_belief" for l in labels])
        tb_mask = torch.tensor([l == "true_belief" for l in labels])
        
        fb_features = features[fb_mask].mean(dim=0)
        tb_features = features[tb_mask].mean(dim=0)
        diff = fb_features - tb_features
        
        # Metrics
        sparsity = (features > 0).float().mean()
        discriminability = diff.abs().max()
        n_discriminative = (diff.abs() > 0.5).sum()
        
        # Top discriminative features
        disc_vals, disc_idx = diff.abs().topk(5)
        
        layer_results[layer] = {
            "sparsity": float(sparsity),
            "max_discriminability": float(discriminability),
            "n_discriminative_features": int(n_discriminative),
            "top_features": [(int(idx), float(diff[idx])) for idx, val in zip(disc_idx, disc_vals)],
            "final_loss": trainer.losses[-1],
        }
        
        print(f"  Sparsity: {sparsity:.1%}")
        print(f"  Max discriminability: {discriminability:.3f}")
        print(f"  # discriminative features (diff>0.5): {n_discriminative}")
        sys.stdout.flush()
    
    # ========================================
    # ANALYZE EVOLUTION
    # ========================================
    print(f"\n{'='*60}")
    print("EVOLUTION OF BELIEF ENCODING")
    print(f"{'='*60}")
    
    print("\nDiscriminability by layer:")
    for layer in layers_to_analyze:
        disc = layer_results[layer]["max_discriminability"]
        n_disc = layer_results[layer]["n_discriminative_features"]
        bar = "#" * int(disc * 20)
        print(f"  Layer {layer:2d}: {disc:.3f} {bar} ({n_disc} features)")
    
    # Find peak discriminability
    peak_layer = max(layers_to_analyze, key=lambda l: layer_results[l]["max_discriminability"])
    print(f"\nPeak discriminability at Layer {peak_layer}")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": config.model_name,
            "layers_analyzed": layers_to_analyze,
            "n_scenarios": len(scenarios),
        },
        "layer_results": layer_results,
        "summary": {
            "peak_discriminability_layer": peak_layer,
            "evolution_pattern": "See figure",
        },
    }
    
    output_path = RESULTS_DIR / "step15_crosslayer.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # ========================================
    # GENERATE FIGURES
    # ========================================
    print("\nGenerating figures...")
    
    import matplotlib.pyplot as plt
    
    # Figure 1: Discriminability evolution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Discriminability over layers
    ax1 = axes[0]
    layers = list(layer_results.keys())
    disc_values = [layer_results[l]["max_discriminability"] for l in layers]
    n_disc_values = [layer_results[l]["n_discriminative_features"] for l in layers]
    
    ax1.plot(layers, disc_values, marker='o', linewidth=2, color='coral', markersize=10)
    ax1.fill_between(layers, disc_values, alpha=0.3, color='coral')
    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("Max Discriminability", fontsize=12)
    ax1.set_title("When Does Belief Become Discriminable?", fontsize=14, fontweight='bold')
    ax1.axvline(x=peak_layer, color='red', linestyle='--', alpha=0.5, label=f'Peak: L{peak_layer}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Number of discriminative features
    ax2 = axes[1]
    ax2.bar(layers, n_disc_values, color='steelblue', edgecolor='black')
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("# Discriminative Features (diff>0.5)", fontsize=12)
    ax2.set_title("Feature Count by Layer", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step15_evolution.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    # Figure 2: Sparsity evolution
    fig, ax = plt.subplots(figsize=(10, 5))
    
    sparsity_values = [layer_results[l]["sparsity"] * 100 for l in layers]
    ax.plot(layers, sparsity_values, marker='s', linewidth=2, color='purple', markersize=10)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Sparsity (%)", fontsize=12)
    ax.set_title("SAE Sparsity Across Layers", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step15_sparsity.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("STEP 15 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

