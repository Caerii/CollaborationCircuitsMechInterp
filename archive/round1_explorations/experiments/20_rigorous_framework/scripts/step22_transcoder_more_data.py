"""
Step 22: Transcoder with More Data

Step 17 transcoder had 0 active features due to insufficient data.
Let's try:
1. More training scenarios (50+)
2. Lower L1 penalty
3. Longer training

OUTPUT: results/step22_transcoder.json, figures/step22_*.png
"""

import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

# Add framework to path
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


class ImprovedTranscoder(nn.Module):
    """Transcoder with better initialization and training."""
    
    def __init__(self, d_in, d_out, d_sparse, l1_coeff=1e-5):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_sparse = d_sparse
        self.l1_coeff = l1_coeff
        
        # Use kaiming initialization
        self.W_enc = nn.Parameter(torch.empty(d_sparse, d_in))
        nn.init.kaiming_uniform_(self.W_enc, a=np.sqrt(5))
        self.b_enc = nn.Parameter(torch.zeros(d_sparse) + 0.5)  # Positive bias
        
        self.W_dec = nn.Parameter(torch.empty(d_out, d_sparse))
        nn.init.kaiming_uniform_(self.W_dec, a=np.sqrt(5))
        self.b_dec = nn.Parameter(torch.zeros(d_out))
    
    def forward(self, x_in, x_out_target):
        # LeakyReLU instead of ReLU
        pre_act = x_in @ self.W_enc.T + self.b_enc
        f = F.leaky_relu(pre_act, negative_slope=0.01)
        
        x_out_pred = f @ self.W_dec.T + self.b_dec
        
        mse_loss = F.mse_loss(x_out_pred, x_out_target)
        l1_loss = self.l1_coeff * f.abs().mean()
        
        return x_out_pred, f, mse_loss + l1_loss
    
    def get_features(self, x_in):
        pre_act = x_in @ self.W_enc.T + self.b_enc
        return F.leaky_relu(pre_act, negative_slope=0.01)


def collect_mlp_io(model, tokenizer, prompts, layer):
    """Collect MLP input and output activations."""
    inputs_list = []
    outputs_list = []
    
    def pre_hook(module, inp):
        inputs_list.append(inp[0][0, -1, :].detach().cpu())
    
    def hook(module, inp, out):
        outputs_list.append(out[0, -1, :].detach().cpu())
    
    mlp = model.model.layers[layer].mlp
    h1 = mlp.register_forward_pre_hook(pre_hook)
    h2 = mlp.register_forward_hook(hook)
    
    with torch.no_grad():
        for prompt in prompts:
            tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
            model(**tokens)
    
    h1.remove()
    h2.remove()
    
    return torch.stack(inputs_list), torch.stack(outputs_list)


def generate_many_scenarios(n=50):
    """Generate many ToM scenarios for training."""
    agents = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Ivy", "Jack"]
    objects = ["ball", "key", "book", "phone", "wallet", "cup", "pen", "toy", "box", "bag"]
    loc1s = ["drawer", "basket", "shelf", "table", "bed", "chair", "desk", "floor", "couch", "counter"]
    loc2s = ["cupboard", "box", "cabinet", "closet", "bag", "pocket", "bin", "case", "trunk", "tray"]
    
    scenarios = []
    
    for i in range(n // 2):
        a1 = agents[i % len(agents)]
        a2 = agents[(i + 1) % len(agents)]
        obj = objects[i % len(objects)]
        l1 = loc1s[i % len(loc1s)]
        l2 = loc2s[i % len(loc2s)]
        
        # False belief
        fb_prompt = f"{a1} put the {obj} in the {l1}. {a1} left. {a2} moved the {obj} to the {l2}. Where does {a1} think the {obj} is?"
        scenarios.append((fb_prompt, "false_belief", l1, l2))
        
        # True belief  
        tb_prompt = f"{a1} put the {obj} in the {l1}. {a1} watched {a2} move the {obj} to the {l2}. Where does {a1} think the {obj} is?"
        scenarios.append((tb_prompt, "true_belief", l2, l1))
    
    return scenarios


def main():
    print("=" * 70)
    print("STEP 22: TRANSCODER WITH MORE DATA")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\nStep 17 had 0 active features. Trying:")
    print("  - 50+ training scenarios (was 10)")
    print("  - LeakyReLU instead of ReLU")
    print("  - Lower L1 coefficient")
    print("  - Kaiming initialization")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    target_layer = 28  # Peak discriminability
    
    # Generate many scenarios
    scenarios = generate_many_scenarios(50)
    prompts = [s[0] for s in scenarios]
    labels = [s[1] for s in scenarios]
    print(f"\nGenerated {len(scenarios)} scenarios")
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
    
    d_model = model.config.hidden_size
    
    # ========================================
    # COLLECT MLP I/O
    # ========================================
    print(f"\n{'='*60}")
    print(f"COLLECTING MLP I/O (Layer {target_layer})")
    print(f"{'='*60}")
    
    print("Collecting...", end="")
    sys.stdout.flush()
    mlp_inputs, mlp_outputs = collect_mlp_io(model, tokenizer, prompts, target_layer)
    mlp_inputs = mlp_inputs.float()
    mlp_outputs = mlp_outputs.float()
    print(f" done! Shape: {mlp_inputs.shape}")
    sys.stdout.flush()
    
    # ========================================
    # TRAIN TRANSCODER
    # ========================================
    print(f"\n{'='*60}")
    print("TRAINING IMPROVED TRANSCODER")
    print(f"{'='*60}")
    
    d_sparse = d_model * 2  # Smaller expansion
    transcoder = ImprovedTranscoder(d_model, d_model, d_sparse, l1_coeff=1e-6)
    optimizer = torch.optim.Adam(transcoder.parameters(), lr=1e-3)
    
    print(f"Transcoder: {d_model} -> {d_sparse} -> {d_model}")
    print(f"L1 coefficient: 1e-6")
    print(f"Training for 1000 epochs...")
    sys.stdout.flush()
    
    losses = []
    for epoch in range(1000):
        optimizer.zero_grad()
        _, features, loss = transcoder(mlp_inputs, mlp_outputs)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
        
        if (epoch + 1) % 200 == 0:
            n_active = (features > 0.01).float().sum(dim=1).mean()
            print(f"  Epoch {epoch+1}: loss={loss:.4f}, active_features={n_active:.1f}")
            sys.stdout.flush()
    
    # ========================================
    # ANALYZE FEATURES
    # ========================================
    print(f"\n{'='*60}")
    print("ANALYZING TRANSCODER FEATURES")
    print(f"{'='*60}")
    
    transcoder.eval()
    with torch.no_grad():
        features = transcoder.get_features(mlp_inputs)
    
    # Count active features
    threshold = 0.01
    n_active = (features > threshold).float().sum(dim=1).mean()
    sparsity = (features > threshold).float().mean()
    
    print(f"\nMean active features: {n_active:.1f} / {d_sparse} ({sparsity:.1%})")
    
    # FB vs TB analysis
    fb_mask = torch.tensor([l == "false_belief" for l in labels])
    tb_mask = torch.tensor([l == "true_belief" for l in labels])
    
    fb_features = features[fb_mask].mean(dim=0)
    tb_features = features[tb_mask].mean(dim=0)
    diff = fb_features - tb_features
    
    # Top discriminative
    disc_vals, disc_idx = diff.abs().topk(10)
    
    print(f"\nTop DISCRIMINATIVE features:")
    for i, (idx, d) in enumerate(zip(disc_idx, disc_vals)):
        direction = "FB>TB" if diff[idx] > 0 else "TB>FB"
        fb_val = float(fb_features[idx])
        tb_val = float(tb_features[idx])
        print(f"  {i+1}. Feature #{idx.item():4d}: FB={fb_val:.3f}, TB={tb_val:.3f}, diff={d:.3f} ({direction})")
    
    # Reconstruction quality
    with torch.no_grad():
        out_pred, _, _ = transcoder(mlp_inputs, mlp_outputs)
        mse = F.mse_loss(out_pred, mlp_outputs)
        cos_sim = F.cosine_similarity(out_pred, mlp_outputs, dim=1).mean()
    
    print(f"\nReconstruction:")
    print(f"  MSE: {mse:.4f}")
    print(f"  Cosine similarity: {cos_sim:.4f}")
    
    # ========================================
    # INTERPRET TOP FEATURES
    # ========================================
    print(f"\n{'='*60}")
    print("INTERPRETING TOP FEATURES")
    print(f"{'='*60}")
    
    for i in range(3):
        idx = disc_idx[i].item()
        
        # What input dimensions activate this feature?
        enc_weights = transcoder.W_enc[idx, :].detach().cpu().numpy()
        top_input_dims = np.argsort(np.abs(enc_weights))[-5:]
        
        # What output dimensions does it write to?
        dec_weights = transcoder.W_dec[:, idx].detach().cpu().numpy()
        top_output_dims = np.argsort(np.abs(dec_weights))[-5:]
        
        print(f"\nFeature #{idx} ({'FB>TB' if diff[idx] > 0 else 'TB>FB'}):")
        print(f"  Reads from input dims: {top_input_dims}")
        print(f"  Writes to output dims: {top_output_dims}")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": config.model_name,
            "layer": target_layer,
            "n_scenarios": len(scenarios),
            "d_sparse": d_sparse,
        },
        "training": {
            "final_loss": losses[-1],
            "n_epochs": 1000,
        },
        "analysis": {
            "mean_active_features": float(n_active),
            "sparsity": float(sparsity),
            "reconstruction_mse": float(mse),
            "reconstruction_cos_sim": float(cos_sim),
        },
        "top_features": [
            {
                "idx": int(disc_idx[i]),
                "diff": float(disc_vals[i]),
                "direction": "FB>TB" if diff[disc_idx[i]] > 0 else "TB>FB",
            }
            for i in range(min(5, len(disc_idx)))
        ],
    }
    
    output_path = RESULTS_DIR / "step22_transcoder.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # Generate figure
    print("\nGenerating figure...")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Training loss
    ax1 = axes[0]
    ax1.plot(losses, color='steelblue', linewidth=0.5)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Transcoder Training", fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    
    # Feature comparison
    ax2 = axes[1]
    x = np.arange(min(10, len(disc_idx)))
    fb_vals = [float(fb_features[i]) for i in disc_idx[:10]]
    tb_vals = [float(tb_features[i]) for i in disc_idx[:10]]
    width = 0.35
    ax2.bar(x - width/2, fb_vals, width, label='False Belief', color='coral')
    ax2.bar(x + width/2, tb_vals, width, label='True Belief', color='seagreen')
    ax2.set_xlabel("Feature Index", fontsize=12)
    ax2.set_ylabel("Activation", fontsize=12)
    ax2.set_title("Top Discriminative Features", fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"#{i}" for i in disc_idx[:10].tolist()])
    ax2.legend()
    
    # Sparsity distribution
    ax3 = axes[2]
    active_per_sample = (features > threshold).float().sum(dim=1).numpy()
    ax3.hist(active_per_sample, bins=20, color='purple', edgecolor='black', alpha=0.7)
    ax3.axvline(x=float(n_active), color='red', linestyle='--', label=f'Mean: {n_active:.0f}')
    ax3.set_xlabel("# Active Features", fontsize=12)
    ax3.set_ylabel("Count", fontsize=12)
    ax3.set_title("Feature Sparsity", fontsize=14, fontweight='bold')
    ax3.legend()
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step22_transcoder.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("STEP 22 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

