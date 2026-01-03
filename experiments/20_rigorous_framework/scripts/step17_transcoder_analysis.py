"""
Step 17: Transcoder Analysis - What Computation Does the MLP Perform?

TRANSCODERS vs SAEs:
- SAE: Encode/decode SAME activation → "What features exist?"
- Transcoder: Map MLP_INPUT → MLP_OUTPUT → "What computation happens?"

WHY TRANSCODERS MATTER:
Step 15 found L28 has PEAK discriminability (10.94!).
Transcoder tells us WHAT COMPUTATION L28 performs to achieve this.

We learn: "Feature X in input → Feature Y in output" transformations.

OUTPUT: results/step17_transcoder.json, figures/step17_*.png
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
from analysis.sae_analysis import Transcoder

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def collect_mlp_io(model, tokenizer, prompts, layer):
    """Collect MLP input and output activations."""
    inputs_list = []
    outputs_list = []
    
    def input_hook(module, inp, out):
        # MLP input is from the residual stream after attention
        inputs_list.append(inp[0][0, -1, :].detach().cpu())
    
    def output_hook(module, inp, out):
        outputs_list.append(out[0, -1, :].detach().cpu())
    
    mlp = model.model.layers[layer].mlp
    
    # Hook before and after MLP
    h1 = mlp.register_forward_pre_hook(lambda m, i: input_hook(m, i, None))
    h2 = mlp.register_forward_hook(output_hook)
    
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
            model(**tokens)
            print(f".", end="")
            sys.stdout.flush()
    
    print(" done!")
    sys.stdout.flush()
    
    h1.remove()
    h2.remove()
    
    return torch.stack(inputs_list), torch.stack(outputs_list)


class TranscoderTrainer:
    """Trainer for transcoders."""
    
    def __init__(self, transcoder, lr=1e-4):
        self.transcoder = transcoder
        self.optimizer = torch.optim.Adam(transcoder.parameters(), lr=lr)
        self.losses = []
    
    def step(self, x_in, x_out):
        """Single training step."""
        self.optimizer.zero_grad()
        _, _, loss = self.transcoder(x_in, x_out)
        loss.backward()
        self.optimizer.step()
        
        # Normalize decoder weights
        with torch.no_grad():
            self.transcoder.W_dec.data = F.normalize(self.transcoder.W_dec.data, dim=0)
        
        self.losses.append(float(loss))
        return float(loss)


def main():
    print("=" * 70)
    print("STEP 17: TRANSCODER ANALYSIS")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\nTRANSCODER vs SAE:")
    print("  SAE: 'What features exist in this activation?'")
    print("  Transcoder: 'What computation transforms input to output?'")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    
    # Target layer (L28 = peak discriminability from Step 15!)
    target_layer = 28
    comparison_layer = 12  # For comparison
    
    print(f"\nAnalyzing layers: L{target_layer} (peak) and L{comparison_layer} (comparison)")
    
    # Scenarios
    scenarios = [
        ("Alice put the ball in the drawer. Alice left. Bob moved it to the basket. Where does Alice think the ball is?", "false_belief"),
        ("Tom hid the key in the box. Tom left. Jerry moved the key to the drawer. Where does Tom think the key is?", "false_belief"),
        ("Chef put ingredients in cabinet A. Chef left. Waiter moved them to cabinet B. Where does Chef think they are?", "false_belief"),
        ("Sally put the toy in the basket. Sally went outside. Anne moved it to the box. Where does Sally think it is?", "false_belief"),
        ("Dad put cookies in the jar. Dad left for work. Mom moved them to the cupboard. Where does Dad think they are?", "false_belief"),
        ("Alice put the ball in the drawer. Alice stayed and watched. Bob moved it to the basket. Where does Alice think the ball is?", "true_belief"),
        ("Tom hid the key in the box. Tom watched Jerry move the key to the drawer. Where does Tom think the key is?", "true_belief"),
        ("Chef put ingredients in cabinet A. Chef saw Waiter move them to cabinet B. Where does Chef think they are?", "true_belief"),
        ("Sally put the toy in the basket. Sally watched Anne move it to the box. Where does Sally think it is?", "true_belief"),
        ("Dad put cookies in the jar. Dad watched Mom move them to the cupboard. Where does Dad think they are?", "true_belief"),
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
    d_sparse = d_model * 4  # 4x expansion
    
    results = {}
    
    for layer in [target_layer, comparison_layer]:
        print(f"\n{'='*60}")
        print(f"ANALYZING LAYER {layer}")
        print(f"{'='*60}")
        sys.stdout.flush()
        
        # ========================================
        # COLLECT MLP INPUT/OUTPUT
        # ========================================
        print(f"\nCollecting MLP input/output for L{layer}...")
        sys.stdout.flush()
        mlp_inputs, mlp_outputs = collect_mlp_io(model, tokenizer, prompts, layer)
        mlp_inputs = mlp_inputs.float()
        mlp_outputs = mlp_outputs.float()
        print(f"  Input shape: {mlp_inputs.shape}")
        print(f"  Output shape: {mlp_outputs.shape}")
        sys.stdout.flush()
        
        # ========================================
        # TRAIN TRANSCODER
        # ========================================
        print(f"\nTraining transcoder...")
        sys.stdout.flush()
        
        # Lower L1 to get non-zero sparse features
        transcoder = Transcoder(
            d_in=d_model,
            d_out=d_model,
            d_sparse=d_sparse,
            l1_coeff=1e-5,  # Much lower L1 to avoid all-zero features
        )
        
        trainer = TranscoderTrainer(transcoder, lr=1e-3)
        
        for epoch in range(500):
            loss = trainer.step(mlp_inputs, mlp_outputs)
            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}: loss = {loss:.4f}")
                sys.stdout.flush()
        
        print(f"  Final loss: {trainer.losses[-1]:.4f}")
        sys.stdout.flush()
        
        # ========================================
        # ANALYZE TRANSCODER FEATURES
        # ========================================
        print(f"\nAnalyzing transcoder features...")
        
        transcoder.eval()
        with torch.no_grad():
            features = transcoder.get_features(mlp_inputs)
        
        # Sparsity
        sparsity = (features > 0).float().mean()
        n_active = (features > 0).float().sum(dim=1).mean()
        
        print(f"  Mean active features: {n_active:.1f} / {d_sparse} ({sparsity:.1%})")
        
        # FB vs TB difference
        fb_mask = torch.tensor([l == "false_belief" for l in labels])
        tb_mask = torch.tensor([l == "true_belief" for l in labels])
        
        fb_features = features[fb_mask].mean(dim=0)
        tb_features = features[tb_mask].mean(dim=0)
        diff = fb_features - tb_features
        
        # Most discriminative features
        disc_vals, disc_idx = diff.abs().topk(10)
        
        print(f"\n  Top DISCRIMINATIVE computation features:")
        for i, (idx, d) in enumerate(zip(disc_idx, disc_vals)):
            direction = "FB>TB" if diff[idx] > 0 else "TB>FB"
            print(f"    {i+1}. Feature #{idx.item():4d}: diff={d:.3f} ({direction})")
        
        # ========================================
        # RECONSTRUCTION QUALITY
        # ========================================
        with torch.no_grad():
            out_pred, _, _ = transcoder(mlp_inputs, mlp_outputs)
            mse = F.mse_loss(out_pred, mlp_outputs)
            cos_sim = F.cosine_similarity(out_pred, mlp_outputs, dim=1).mean()
        
        print(f"\n  Reconstruction quality:")
        print(f"    MSE: {mse:.4f}")
        print(f"    Cosine similarity: {cos_sim:.4f}")
        
        # ========================================
        # WHAT DOES THE TRANSCODER LEARN?
        # ========================================
        print(f"\n  Analyzing learned computation...")
        
        # The transcoder's W_enc tells us what input patterns it detects
        # The transcoder's W_dec tells us what output it produces
        
        # For the most discriminative feature, analyze:
        top_feature_idx = disc_idx[0].item()
        
        # Input pattern for this feature (what triggers it)
        input_pattern = transcoder.W_enc[top_feature_idx, :].detach().cpu().numpy()
        top_input_dims = np.argsort(np.abs(input_pattern))[-5:]
        
        # Output pattern for this feature (what it produces)
        output_pattern = transcoder.W_dec[:, top_feature_idx].detach().cpu().numpy()
        top_output_dims = np.argsort(np.abs(output_pattern))[-5:]
        
        print(f"    Feature #{top_feature_idx}:")
        print(f"      Input: activates most for dims {top_input_dims}")
        print(f"      Output: writes most to dims {top_output_dims}")
        
        results[layer] = {
            "sparsity": float(sparsity),
            "n_active_features": float(n_active),
            "reconstruction_mse": float(mse),
            "reconstruction_cos_sim": float(cos_sim),
            "top_discriminative_features": [
                {
                    "idx": int(idx),
                    "diff": float(diff[idx]),
                    "direction": "FB>TB" if diff[idx] > 0 else "TB>FB",
                }
                for idx in disc_idx[:5]
            ],
            "top_feature_analysis": {
                "feature_idx": top_feature_idx,
                "top_input_dims": top_input_dims.tolist(),
                "top_output_dims": top_output_dims.tolist(),
            },
            "final_loss": trainer.losses[-1],
        }
    
    # ========================================
    # COMPARE LAYERS
    # ========================================
    print(f"\n{'='*60}")
    print("LAYER COMPARISON")
    print(f"{'='*60}")
    
    print(f"\n| Layer | Sparsity | Active Features | Top Discriminability |")
    print(f"|-------|----------|-----------------|---------------------|")
    for layer in [target_layer, comparison_layer]:
        r = results[layer]
        top_disc = abs(r["top_discriminative_features"][0]["diff"])
        print(f"| L{layer:2d}   | {r['sparsity']:.1%}     | {r['n_active_features']:.0f}              | {top_disc:.3f}               |")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": config.model_name,
            "target_layer": target_layer,
            "comparison_layer": comparison_layer,
            "d_sparse": d_sparse,
        },
        "layer_results": results,
        "key_insight": "Transcoder reveals what COMPUTATION each layer performs, not just what features exist",
    }
    
    output_path = RESULTS_DIR / "step17_transcoder.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # ========================================
    # GENERATE FIGURES
    # ========================================
    print("\nGenerating figures...")
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Figure 1: Compare discriminability across layers
    ax1 = axes[0]
    layers = [comparison_layer, target_layer]
    disc_values = [
        abs(results[l]["top_discriminative_features"][0]["diff"]) 
        for l in layers
    ]
    colors = ['steelblue', 'coral']
    bars = ax1.bar([f"L{l}" for l in layers], disc_values, color=colors, edgecolor='black')
    ax1.set_ylabel("Max Discriminability", fontsize=12)
    ax1.set_title("Transcoder: Computation Discriminability by Layer", fontsize=14, fontweight='bold')
    
    for bar, val in zip(bars, disc_values):
        ax1.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Figure 2: Feature activation comparison (FB vs TB)
    ax2 = axes[1]
    
    # Use target layer for detailed analysis
    r = results[target_layer]
    feature_indices = [f["idx"] for f in r["top_discriminative_features"][:5]]
    fb_values = []
    tb_values = []
    
    transcoder.eval()
    with torch.no_grad():
        features = transcoder.get_features(mlp_inputs)
    
    for idx in feature_indices:
        fb_values.append(float(features[fb_mask, idx].mean()))
        tb_values.append(float(features[tb_mask, idx].mean()))
    
    x = np.arange(5)
    width = 0.35
    ax2.bar(x - width/2, fb_values, width, label='False Belief', color='coral')
    ax2.bar(x + width/2, tb_values, width, label='True Belief', color='seagreen')
    ax2.set_xlabel("Feature Index", fontsize=12)
    ax2.set_ylabel("Activation", fontsize=12)
    ax2.set_title(f"L{target_layer} Transcoder: Computation Features", fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"#{i}" for i in feature_indices])
    ax2.legend()
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step17_transcoder.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("KEY INSIGHTS")
    print(f"{'='*60}")
    print("""
TRANSCODER vs SAE:
==================
- SAE asks: "What features are PRESENT in this activation?"
- Transcoder asks: "What COMPUTATION transforms input to output?"

For belief/ToM:
- SAE: "The activation contains a 'belief update' feature"
- Transcoder: "The MLP COMPUTES belief update from input pattern X to output pattern Y"

TRANSCODER ADVANTAGE:
- More causal: directly models the transformation
- More interpretable: see input->output mapping
- Better for understanding WHAT MLPs DO, not just what they contain
""")
    
    print(f"\n{'='*60}")
    print("STEP 17 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

