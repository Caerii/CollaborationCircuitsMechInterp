"""
Step 13: SAE Feature Analysis

Train a Sparse Autoencoder on MLP activations to find
interpretable features for ToM/belief tracking.

GOAL: Go from "Layer 12 encodes belief" to 
      "Feature #X = 'agent has outdated belief'"

METHOD:
1. Collect MLP activations from ToM scenarios
2. Train simple SAE on these activations
3. Analyze which features correlate with belief type
4. Visualize feature activations

OUTPUT: results/step13_sae.json, figures/step13_*.png
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


def create_belief_scenarios():
    """Create scenarios for SAE training."""
    scenarios = []
    
    # FALSE BELIEF
    false_belief = [
        ("Alice put the ball in the drawer. Alice left. Bob moved it to the basket. Where does Alice think the ball is?", "false_belief"),
        ("Tom hid the key in the box. Tom left. Jerry moved the key to the drawer. Where does Tom think the key is?", "false_belief"),
        ("Chef put ingredients in cabinet A. Chef left. Waiter moved them to cabinet B. Where does Chef think they are?", "false_belief"),
        ("Sally put the toy in the basket. Sally went outside. Anne moved it to the box. Where does Sally think it is?", "false_belief"),
        ("Dad put cookies in the jar. Dad left for work. Mom moved them to the cupboard. Where does Dad think they are?", "false_belief"),
        ("Teacher put chalk in drawer 1. Teacher left. Student moved it to drawer 2. Where does Teacher think it is?", "false_belief"),
        ("Mark put his phone on the table. Mark went to shower. Lisa moved it to the shelf. Where does Mark think it is?", "false_belief"),
        ("Emma put her keys in her purse. Emma left. Jack moved them to the hook. Where does Emma think they are?", "false_belief"),
    ]
    
    # TRUE BELIEF
    true_belief = [
        ("Alice put the ball in the drawer. Alice stayed and watched. Bob moved it to the basket. Where does Alice think the ball is?", "true_belief"),
        ("Tom hid the key in the box. Tom watched Jerry move the key to the drawer. Where does Tom think the key is?", "true_belief"),
        ("Chef put ingredients in cabinet A. Chef saw Waiter move them to cabinet B. Where does Chef think they are?", "true_belief"),
        ("Sally put the toy in the basket. Sally watched Anne move it to the box. Where does Sally think it is?", "true_belief"),
        ("Dad put cookies in the jar. Dad watched Mom move them to the cupboard. Where does Dad think they are?", "true_belief"),
        ("Teacher put chalk in drawer 1. Teacher saw Student move it to drawer 2. Where does Teacher think it is?", "true_belief"),
        ("Mark put his phone on the table. Mark watched Lisa move it to the shelf. Where does Mark think it is?", "true_belief"),
        ("Emma put her keys in her purse. Emma watched Jack move them to the hook. Where does Emma think they are?", "true_belief"),
    ]
    
    scenarios.extend(false_belief)
    scenarios.extend(true_belief)
    
    return scenarios


def collect_mlp_activations(model, tokenizer, prompts, layer):
    """Collect MLP output activations."""
    activations = []
    
    def hook(module, input, output):
        # Get last token activation
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
    print("STEP 13: SAE FEATURE ANALYSIS")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    
    # Create scenarios
    scenarios = create_belief_scenarios()
    prompts = [s[0] for s in scenarios]
    labels = [s[1] for s in scenarios]
    print(f"\nCreated {len(scenarios)} scenarios")
    print(f"  False belief: {sum(1 for l in labels if l == 'false_belief')}")
    print(f"  True belief: {sum(1 for l in labels if l == 'true_belief')}")
    sys.stdout.flush()
    
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
    
    # Get hidden size
    d_model = model.config.hidden_size
    print(f"Hidden size: {d_model}")
    
    # Layer to analyze (from our probing results)
    target_layer = 12  # Best belief encoding
    
    # ========================================
    # COLLECT ACTIVATIONS
    # ========================================
    print(f"\n{'='*60}")
    print(f"COLLECTING MLP ACTIVATIONS (Layer {target_layer})")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    activations = collect_mlp_activations(model, tokenizer, prompts, target_layer)
    print(f"Collected activations: {activations.shape}")
    
    # ========================================
    # TRAIN SAE
    # ========================================
    print(f"\n{'='*60}")
    print("TRAINING SPARSE AUTOENCODER")
    print(f"{'='*60}")
    
    # SAE config
    expansion_factor = 4
    sae_config = SAEConfig(
        d_model=d_model,
        d_sae=d_model * expansion_factor,
        l1_coeff=1e-3,
        lr=1e-3,
    )
    
    print(f"SAE dimensions: {d_model} -> {sae_config.d_sae} -> {d_model}")
    print(f"L1 coefficient: {sae_config.l1_coeff}")
    
    # Create and train SAE
    sae = SimpleSAE(sae_config)
    trainer = SAETrainer(sae, lr=sae_config.lr)
    
    # Convert to float32 for training
    activations_train = activations.float()
    
    n_epochs = 500
    batch_size = len(prompts)
    
    print(f"\nTraining for {n_epochs} epochs...")
    sys.stdout.flush()
    
    for epoch in range(n_epochs):
        loss = trainer.step(activations_train)
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}: loss = {loss:.4f}")
            sys.stdout.flush()
    
    print(f"Final loss: {trainer.losses[-1]:.4f}")
    
    # ========================================
    # ANALYZE FEATURES
    # ========================================
    print(f"\n{'='*60}")
    print("ANALYZING SAE FEATURES")
    print(f"{'='*60}")
    
    sae.eval()
    with torch.no_grad():
        features = sae.get_feature_activations(activations_train)
    
    # Sparsity analysis
    n_active = (features > 0).float().sum(dim=1).mean()
    sparsity = n_active / sae_config.d_sae
    print(f"\nMean active features per input: {n_active:.1f} / {sae_config.d_sae} ({sparsity:.1%})")
    
    # Features by belief type
    fb_mask = torch.tensor([l == "false_belief" for l in labels])
    tb_mask = torch.tensor([l == "true_belief" for l in labels])
    
    fb_features = features[fb_mask].mean(dim=0)
    tb_features = features[tb_mask].mean(dim=0)
    
    # Differential features
    diff = fb_features - tb_features
    
    # Top features for false belief
    fb_top_vals, fb_top_idx = fb_features.topk(10)
    print("\nTop features for FALSE BELIEF:")
    for i, (idx, val) in enumerate(zip(fb_top_idx, fb_top_vals)):
        tb_val = tb_features[idx]
        print(f"  {i+1}. Feature #{idx.item():4d}: FB={val:.3f}, TB={tb_val:.3f}, diff={val-tb_val:+.3f}")
    
    # Top features for true belief
    tb_top_vals, tb_top_idx = tb_features.topk(10)
    print("\nTop features for TRUE BELIEF:")
    for i, (idx, val) in enumerate(zip(tb_top_idx, tb_top_vals)):
        fb_val = fb_features[idx]
        print(f"  {i+1}. Feature #{idx.item():4d}: TB={val:.3f}, FB={fb_val:.3f}, diff={fb_val-val:+.3f}")
    
    # Most discriminative features
    abs_diff = diff.abs()
    disc_vals, disc_idx = abs_diff.topk(10)
    print("\nMost DISCRIMINATIVE features (|FB - TB|):")
    for i, (idx, d) in enumerate(zip(disc_idx, disc_vals)):
        fb_val = fb_features[idx]
        tb_val = tb_features[idx]
        direction = "FB>TB" if fb_val > tb_val else "TB>FB"
        print(f"  {i+1}. Feature #{idx.item():4d}: diff={d:.3f} ({direction})")
    
    # ========================================
    # RECONSTRUCTION QUALITY
    # ========================================
    print(f"\n{'='*60}")
    print("RECONSTRUCTION QUALITY")
    print(f"{'='*60}")
    
    with torch.no_grad():
        x_hat, _, _ = sae(activations_train)
        mse = F.mse_loss(x_hat, activations_train)
        cos_sim = F.cosine_similarity(x_hat, activations_train, dim=1).mean()
    
    print(f"MSE: {mse:.4f}")
    print(f"Cosine similarity: {cos_sim:.4f}")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": config.model_name,
            "layer": target_layer,
            "d_model": d_model,
            "d_sae": sae_config.d_sae,
            "l1_coeff": sae_config.l1_coeff,
            "n_epochs": n_epochs,
        },
        "training": {
            "final_loss": trainer.losses[-1],
            "loss_history": trainer.losses[::50],  # Every 50th
        },
        "analysis": {
            "mean_active_features": float(n_active),
            "sparsity": float(sparsity),
            "reconstruction_mse": float(mse),
            "reconstruction_cos_sim": float(cos_sim),
        },
        "top_features": {
            "false_belief": [(int(idx), float(val)) for idx, val in zip(fb_top_idx[:5], fb_top_vals[:5])],
            "true_belief": [(int(idx), float(val)) for idx, val in zip(tb_top_idx[:5], tb_top_vals[:5])],
            "discriminative": [(int(idx), float(d)) for idx, d in zip(disc_idx[:5], disc_vals[:5])],
        },
    }
    
    output_path = RESULTS_DIR / "step13_sae.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # ========================================
    # GENERATE FIGURES
    # ========================================
    print("\nGenerating figures...")
    
    import matplotlib.pyplot as plt
    
    # Figure 1: Training loss
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(trainer.losses, color='steelblue', linewidth=0.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("SAE Training Loss", fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step13_sae_training.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    # Figure 2: Feature activations comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # FB vs TB feature comparison
    ax1 = axes[0]
    x = np.arange(10)
    width = 0.35
    ax1.bar(x - width/2, [float(fb_features[i]) for i in disc_idx[:10]], width, label='False Belief', color='coral')
    ax1.bar(x + width/2, [float(tb_features[i]) for i in disc_idx[:10]], width, label='True Belief', color='seagreen')
    ax1.set_xlabel("Feature Index", fontsize=12)
    ax1.set_ylabel("Activation", fontsize=12)
    ax1.set_title("Top Discriminative Features", fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"#{i}" for i in disc_idx[:10].tolist()], rotation=45)
    ax1.legend()
    
    # Sparsity histogram
    ax2 = axes[1]
    active_per_sample = (features > 0).float().sum(dim=1).numpy()
    ax2.hist(active_per_sample, bins=20, color='purple', edgecolor='black', alpha=0.7)
    ax2.axvline(x=n_active, color='red', linestyle='--', label=f'Mean: {n_active:.0f}')
    ax2.set_xlabel("Number of Active Features", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Feature Sparsity Distribution", fontsize=14, fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step13_sae_features.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    # Figure 3: Feature heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get top 10 discriminative features
    top_disc_features = disc_idx[:10]
    feature_matrix = features[:, top_disc_features].numpy()
    
    im = ax.imshow(feature_matrix.T, aspect='auto', cmap='viridis')
    ax.set_xlabel("Scenario", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title("SAE Feature Activations (Top 10 Discriminative)", fontsize=14, fontweight='bold')
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"#{i}" for i in top_disc_features.tolist()])
    
    # Mark FB vs TB boundary
    ax.axvline(x=7.5, color='white', linestyle='--', linewidth=2, label='FB|TB boundary')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Activation", fontsize=10)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step13_sae_heatmap.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("STEP 13 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

