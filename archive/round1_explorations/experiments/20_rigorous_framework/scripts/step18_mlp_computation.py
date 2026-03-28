"""
Step 18: Direct MLP Computation Analysis

Since transcoder training is tricky, let's analyze MLP computation directly:
1. Look at the MLP weight matrices
2. Find what input directions map to what output directions
3. Test with belief-related activation differences

This is more direct than learning a transcoder.

OUTPUT: results/step18_mlp_computation.json, figures/step18_*.png
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

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def collect_activations(model, tokenizer, prompts, layer, position="last"):
    """Collect residual stream activations."""
    activations = []
    
    def hook(module, input, output):
        # Output can be tuple or tensor
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        
        # Handle different tensor shapes
        if hidden.dim() == 3:
            if position == "last":
                activations.append(hidden[0, -1, :].detach().cpu())
            else:
                activations.append(hidden[0, :, :].detach().cpu())
        elif hidden.dim() == 2:
            activations.append(hidden[-1, :].detach().cpu())
        else:
            activations.append(hidden.detach().cpu())
    
    # Hook on layer output (residual after attention + mlp)
    layer_module = model.model.layers[layer]
    handle = layer_module.register_forward_hook(hook)
    
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            model(**inputs)
    
    handle.remove()
    return torch.stack(activations)


def main():
    print("=" * 70)
    print("STEP 18: DIRECT MLP COMPUTATION ANALYSIS")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\nAnalyzing MLP weights directly (no training needed!)")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    
    # Target layers
    layers_to_analyze = [12, 28, 32]  # Key layers from previous findings
    
    # Scenarios
    fb_prompts = [
        "Alice put the ball in the drawer. Alice left. Bob moved it to the basket. Where does Alice think the ball is?",
        "Tom hid the key in the box. Tom left. Jerry moved the key to the drawer. Where does Tom think the key is?",
        "Chef put ingredients in cabinet A. Chef left. Waiter moved them to cabinet B. Where does Chef think they are?",
        "Sally put the toy in the basket. Sally went outside. Anne moved it to the box. Where does Sally think it is?",
    ]
    
    tb_prompts = [
        "Alice put the ball in the drawer. Alice stayed and watched. Bob moved it to the basket. Where does Alice think the ball is?",
        "Tom hid the key in the box. Tom watched Jerry move the key to the drawer. Where does Tom think the key is?",
        "Chef put ingredients in cabinet A. Chef saw Waiter move them to cabinet B. Where does Chef think they are?",
        "Sally put the toy in the basket. Sally watched Anne move it to the box. Where does Sally think it is?",
    ]
    
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
    
    results = {}
    
    for layer in layers_to_analyze:
        print(f"\n{'='*60}")
        print(f"ANALYZING LAYER {layer}")
        print(f"{'='*60}")
        sys.stdout.flush()
        
        # Get MLP weights
        mlp = model.model.layers[layer].mlp
        
        # Qwen uses gate_proj, up_proj, down_proj
        W_gate = mlp.gate_proj.weight.data.cpu().float()  # (intermediate, hidden)
        W_up = mlp.up_proj.weight.data.cpu().float()      # (intermediate, hidden)
        W_down = mlp.down_proj.weight.data.cpu().float()  # (hidden, intermediate)
        
        intermediate_size = W_gate.shape[0]
        print(f"  MLP structure: {d_model} -> {intermediate_size} -> {d_model}")
        print(f"  W_gate shape: {W_gate.shape}")
        print(f"  W_down shape: {W_down.shape}")
        sys.stdout.flush()
        
        # ========================================
        # COLLECT ACTIVATIONS
        # ========================================
        print(f"\n  Collecting activations...")
        sys.stdout.flush()
        
        fb_acts = collect_activations(model, tokenizer, fb_prompts, layer)
        tb_acts = collect_activations(model, tokenizer, tb_prompts, layer)
        
        # Compute mean difference direction (convert to float)
        fb_mean = fb_acts.mean(dim=0).float()
        tb_mean = tb_acts.mean(dim=0).float()
        diff_direction = fb_mean - tb_mean
        diff_norm = diff_direction / (diff_direction.norm() + 1e-8)
        
        print(f"    FB mean norm: {fb_mean.norm():.2f}")
        print(f"    TB mean norm: {tb_mean.norm():.2f}")
        print(f"    Diff direction norm: {diff_direction.norm():.2f}")
        sys.stdout.flush()
        
        # ========================================
        # ANALYZE MLP RESPONSE TO DIFF DIRECTION
        # ========================================
        print(f"\n  Analyzing MLP response to belief diff direction...")
        sys.stdout.flush()
        
        # Pass diff direction through MLP
        # SwiGLU: out = down @ (silu(gate @ x) * up @ x)
        gate_response = W_gate @ diff_norm
        up_response = W_up @ diff_norm
        
        # SiLU activation
        silu_gate = F.silu(gate_response)
        hidden_activation = silu_gate * up_response
        
        # Down projection
        output_direction = W_down @ hidden_activation
        output_norm = output_direction / (output_direction.norm() + 1e-8)
        
        # How much is the output aligned with input diff?
        alignment = float(torch.dot(output_norm, diff_norm))
        amplification = float(output_direction.norm() / diff_direction.norm())
        
        print(f"    Input-output alignment: {alignment:.3f}")
        print(f"    Amplification factor: {amplification:.3f}")
        
        # Find top activated neurons
        top_neurons_gate = gate_response.abs().topk(5)
        top_neurons_up = up_response.abs().topk(5)
        top_hidden = hidden_activation.abs().topk(5)
        
        print(f"    Top gate neurons: {top_neurons_gate.indices.tolist()}")
        print(f"    Top hidden neurons: {top_hidden.indices.tolist()}")
        sys.stdout.flush()
        
        # ========================================
        # ANALYZE WHAT OUTPUT DIMS ARE AFFECTED
        # ========================================
        print(f"\n  Analyzing output dimensions affected...")
        
        top_output_dims = output_direction.abs().topk(10)
        print(f"    Top output dims: {top_output_dims.indices.tolist()}")
        print(f"    Top output magnitudes: {[f'{v:.3f}' for v in top_output_dims.values.tolist()]}")
        sys.stdout.flush()
        
        results[layer] = {
            "mlp_structure": {
                "input_dim": d_model,
                "intermediate_dim": intermediate_size,
            },
            "diff_direction_norm": float(diff_direction.norm()),
            "alignment": alignment,
            "amplification": amplification,
            "top_gate_neurons": top_neurons_gate.indices.tolist(),
            "top_hidden_neurons": top_hidden.indices.tolist(),
            "top_output_dims": top_output_dims.indices.tolist(),
        }
    
    # ========================================
    # COMPARE LAYERS
    # ========================================
    print(f"\n{'='*60}")
    print("LAYER COMPARISON")
    print(f"{'='*60}")
    
    print(f"\n| Layer | Alignment | Amplification | Interpretation |")
    print(f"|-------|-----------|---------------|----------------|")
    for layer in layers_to_analyze:
        r = results[layer]
        interp = ""
        if r["alignment"] > 0.5:
            interp = "Preserves belief diff"
        elif r["alignment"] < -0.5:
            interp = "Inverts belief diff!"
        else:
            interp = "Transforms direction"
        print(f"| L{layer:2d}   | {r['alignment']:+.3f}     | {r['amplification']:.2f}x          | {interp} |")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": config.model_name,
            "layers_analyzed": layers_to_analyze,
        },
        "layer_results": results,
        "key_insight": "Direct MLP weight analysis shows how belief difference is transformed layer by layer",
    }
    
    output_path = RESULTS_DIR / "step18_mlp_computation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # ========================================
    # GENERATE FIGURES
    # ========================================
    print("\nGenerating figures...")
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Figure 1: Alignment across layers
    ax1 = axes[0]
    alignments = [results[l]["alignment"] for l in layers_to_analyze]
    colors = ['coral' if a > 0 else 'steelblue' for a in alignments]
    ax1.bar([f"L{l}" for l in layers_to_analyze], alignments, color=colors, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel("Alignment (input vs output direction)", fontsize=12)
    ax1.set_title("MLP Alignment with Belief Diff", fontsize=14, fontweight='bold')
    ax1.set_ylim(-1, 1)
    
    # Figure 2: Amplification across layers
    ax2 = axes[1]
    amplifications = [results[l]["amplification"] for l in layers_to_analyze]
    ax2.bar([f"L{l}" for l in layers_to_analyze], amplifications, color='purple', edgecolor='black')
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No change')
    ax2.set_ylabel("Amplification Factor", fontsize=12)
    ax2.set_title("MLP Amplification of Belief Diff", fontsize=14, fontweight='bold')
    ax2.legend()
    
    # Figure 3: Diff direction norm across layers
    ax3 = axes[2]
    diff_norms = [results[l]["diff_direction_norm"] for l in layers_to_analyze]
    ax3.plot([f"L{l}" for l in layers_to_analyze], diff_norms, marker='o', linewidth=2, color='seagreen', markersize=10)
    ax3.set_ylabel("FB-TB Difference Norm", fontsize=12)
    ax3.set_title("Belief Separability by Layer", fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step18_mlp_computation.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("KEY INSIGHTS")
    print(f"{'='*60}")
    print("""
WHAT EACH MLP DOES TO BELIEF DIFFERENCE:
========================================
- ALIGNMENT > 0: MLP preserves the FB vs TB distinction
- ALIGNMENT < 0: MLP inverts the distinction!
- AMPLIFICATION > 1: MLP strengthens the signal
- AMPLIFICATION < 1: MLP weakens the signal

This tells us HOW each layer transforms belief information
without needing to train a sparse model!
""")
    
    print(f"\n{'='*60}")
    print("STEP 18 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

