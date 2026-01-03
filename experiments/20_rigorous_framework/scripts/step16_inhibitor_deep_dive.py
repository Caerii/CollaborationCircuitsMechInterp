"""
Step 16: L18H16 Deep Dive - Why Is It An Inhibitor?

MYSTERY: Ablating L18H16 IMPROVES multi-agent performance by 25%!
Why would a head actively HURT performance?

CONTEXT FROM STEP 15:
- L28 has PEAK belief discriminability (10.94!)
- L18 is BEFORE the peak - possibly interfering with buildup?

HYPOTHESES:
1. It's "overthinking" - adding noise to multi-agent reasoning
2. It's interfering with correct signal propagation to L28
3. It encodes a competing/conflicting representation
4. It diffuses attention when multiple agents present

METHOD:
1. Analyze what L18H16 attends to
2. Compare its output to other heads
3. Test if it's active in single-agent vs multi-agent
4. Measure attention entropy (diffuse vs focused)

OUTPUT: results/step16_inhibitor.json, figures/step16_*.png
"""

import sys
import json
import torch
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


def get_attention_pattern(model, tokenizer, prompt, layer, head):
    """Get attention pattern for a specific head."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, return_dict=True)
    
    # Get attention for specified layer and head
    # Shape: (batch, n_heads, seq, seq)
    attn = outputs.attentions[layer][0, head, :, :].cpu().numpy()
    
    return attn, tokens


def get_head_output(model, tokenizer, prompt, layer, head):
    """Get the output contribution of a specific attention head."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    head_output = []
    
    def hook(module, input, output):
        # output shape: (batch, seq, hidden)
        # We want to isolate this head's contribution
        # The actual decomposition is complex, so we'll approximate
        head_output.append(output[0, -1, :].detach().cpu())
    
    # Hook on attention output
    attn_module = model.model.layers[layer].self_attn
    handle = attn_module.register_forward_hook(hook)
    
    with torch.no_grad():
        model(**inputs)
    
    handle.remove()
    
    return head_output[0] if head_output else None


def main():
    print("=" * 70)
    print("STEP 16: L18H16 INHIBITOR DEEP DIVE")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\nMystery: Why does ablating L18H16 IMPROVE multi-agent performance?")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    
    # Target head
    target_layer = 18
    target_head = 16
    
    # Test scenarios
    single_agent = [
        "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket. Where does Alice think the ball is?",
        "Tom hid the key in the box. Tom left. Jerry moved the key to the drawer. Where does Tom think the key is?",
    ]
    
    multi_agent = [
        "Alice put the ball in the drawer. Bob saw this. Alice left. Carol moved the ball to the basket. Bob watched. Where does Alice think the ball is?",
        "Tom hid the key in the box. Jerry saw this. Tom left. Spike moved the key to the drawer. Jerry watched. Where does Tom think the key is?",
    ]
    
    control = [
        "The ball is in the basket. Where is the ball?",
        "The key is in the drawer. Where is the key?",
    ]
    
    # Load model
    print("\nLoading model (with attention outputs)...")
    sys.stdout.flush()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # Required for attention outputs
    )
    model.eval()
    print("Model loaded!")
    sys.stdout.flush()
    
    # ========================================
    # ANALYZE ATTENTION PATTERNS
    # ========================================
    print(f"\n{'='*60}")
    print(f"ANALYZING L{target_layer}H{target_head} ATTENTION")
    print(f"{'='*60}")
    
    results = {
        "single_agent": [],
        "multi_agent": [],
        "control": [],
    }
    
    # Analyze single-agent scenarios
    print("\n--- Single-Agent Scenarios ---")
    for prompt in single_agent:
        attn, tokens = get_attention_pattern(model, tokenizer, prompt, target_layer, target_head)
        
        # Last token's attention (what does prediction attend to?)
        last_attn = attn[-1, :]
        
        # Find what it attends to most
        top_k = 5
        top_idx = np.argsort(last_attn)[-top_k:][::-1]
        
        # Clean tokens for display
        tokens_clean = [t.encode('ascii', 'replace').decode('ascii') for t in tokens]
        
        print(f"\n  Prompt: ...{prompt[-50:]}")
        print(f"  Top attended tokens:")
        for idx in top_idx:
            print(f"    {tokens_clean[idx]}: {last_attn[idx]:.3f}")
        
        results["single_agent"].append({
            "prompt": prompt,
            "top_attended": [(tokens_clean[i], float(last_attn[i])) for i in top_idx],
            "entropy": float(-np.sum(last_attn * np.log(last_attn + 1e-10))),
        })
    
    # Analyze multi-agent scenarios
    print("\n--- Multi-Agent Scenarios ---")
    for prompt in multi_agent:
        attn, tokens = get_attention_pattern(model, tokenizer, prompt, target_layer, target_head)
        last_attn = attn[-1, :]
        top_idx = np.argsort(last_attn)[-5:][::-1]
        tokens_clean = [t.encode('ascii', 'replace').decode('ascii') for t in tokens]
        
        print(f"\n  Prompt: ...{prompt[-50:]}")
        print(f"  Top attended tokens:")
        for idx in top_idx:
            print(f"    {tokens_clean[idx]}: {last_attn[idx]:.3f}")
        
        results["multi_agent"].append({
            "prompt": prompt,
            "top_attended": [(tokens_clean[i], float(last_attn[i])) for i in top_idx],
            "entropy": float(-np.sum(last_attn * np.log(last_attn + 1e-10))),
        })
    
    # Analyze control scenarios
    print("\n--- Control Scenarios ---")
    for prompt in control:
        attn, tokens = get_attention_pattern(model, tokenizer, prompt, target_layer, target_head)
        last_attn = attn[-1, :]
        top_idx = np.argsort(last_attn)[-5:][::-1]
        tokens_clean = [t.encode('ascii', 'replace').decode('ascii') for t in tokens]
        
        print(f"\n  Prompt: ...{prompt[-50:]}")
        print(f"  Top attended tokens:")
        for idx in top_idx:
            print(f"    {tokens_clean[idx]}: {last_attn[idx]:.3f}")
        
        results["control"].append({
            "prompt": prompt,
            "top_attended": [(tokens_clean[i], float(last_attn[i])) for i in top_idx],
            "entropy": float(-np.sum(last_attn * np.log(last_attn + 1e-10))),
        })
    
    # ========================================
    # COMPARE ENTROPY (FOCUS vs DIFFUSE)
    # ========================================
    print(f"\n{'='*60}")
    print("ATTENTION ENTROPY ANALYSIS")
    print(f"{'='*60}")
    
    single_entropy = np.mean([r["entropy"] for r in results["single_agent"]])
    multi_entropy = np.mean([r["entropy"] for r in results["multi_agent"]])
    control_entropy = np.mean([r["entropy"] for r in results["control"]])
    
    print(f"\nMean attention entropy (higher = more diffuse):")
    print(f"  Single-agent: {single_entropy:.3f}")
    print(f"  Multi-agent:  {multi_entropy:.3f}")
    print(f"  Control:      {control_entropy:.3f}")
    
    # ========================================
    # HYPOTHESIS TESTING
    # ========================================
    print(f"\n{'='*60}")
    print("HYPOTHESIS ANALYSIS")
    print(f"{'='*60}")
    
    # H1: More diffuse in multi-agent (overthinking)
    h1_supported = multi_entropy > single_entropy
    print(f"\nH1: L18H16 is more diffuse in multi-agent scenarios")
    print(f"    Result: {'SUPPORTED' if h1_supported else 'NOT SUPPORTED'}")
    print(f"    (Multi entropy {multi_entropy:.3f} {'>' if h1_supported else '<='} Single entropy {single_entropy:.3f})")
    
    # Check what tokens it attends to
    print(f"\nAttention Pattern Analysis:")
    
    # Count attention to agent names vs locations
    agent_attn_single = []
    agent_attn_multi = []
    
    for r in results["single_agent"]:
        agent_sum = sum(v for t, v in r["top_attended"] if any(x in t.lower() for x in ["alice", "bob", "tom", "jerry"]))
        agent_attn_single.append(agent_sum)
    
    for r in results["multi_agent"]:
        agent_sum = sum(v for t, v in r["top_attended"] if any(x in t.lower() for x in ["alice", "bob", "carol", "tom", "jerry", "spike"]))
        agent_attn_multi.append(agent_sum)
    
    print(f"  Mean agent attention (single): {np.mean(agent_attn_single):.3f}")
    print(f"  Mean agent attention (multi):  {np.mean(agent_attn_multi):.3f}")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": config.model_name,
            "target_layer": target_layer,
            "target_head": target_head,
        },
        "attention_analysis": results,
        "entropy": {
            "single_agent": single_entropy,
            "multi_agent": multi_entropy,
            "control": control_entropy,
        },
        "hypotheses": {
            "H1_more_diffuse_in_multi": h1_supported,
        },
    }
    
    output_path = RESULTS_DIR / "step16_inhibitor.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # ========================================
    # GENERATE FIGURE
    # ========================================
    print("\nGenerating figure...")
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Figure 1: Entropy comparison
    ax1 = axes[0]
    categories = ['Single-Agent', 'Multi-Agent', 'Control']
    entropies = [single_entropy, multi_entropy, control_entropy]
    colors = ['steelblue', 'coral', 'gray']
    bars = ax1.bar(categories, entropies, color=colors, edgecolor='black')
    ax1.set_ylabel("Attention Entropy", fontsize=12)
    ax1.set_title(f"L{target_layer}H{target_head} Attention Focus", fontsize=14, fontweight='bold')
    
    for bar, ent in zip(bars, entropies):
        ax1.annotate(f'{ent:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Figure 2: Visualize one attention pattern
    ax2 = axes[1]
    # Get attention for a multi-agent scenario
    attn, tokens = get_attention_pattern(model, tokenizer, multi_agent[0], target_layer, target_head)
    tokens_clean = [t.encode('ascii', 'replace').decode('ascii')[:8] for t in tokens]
    
    # Show last row (what prediction attends to)
    last_attn = attn[-1, :]
    ax2.bar(range(len(last_attn)), last_attn, color='purple', alpha=0.7)
    ax2.set_xlabel("Token Position", fontsize=12)
    ax2.set_ylabel("Attention Weight", fontsize=12)
    ax2.set_title(f"L{target_layer}H{target_head} Attention Pattern (Multi-Agent)", fontsize=14, fontweight='bold')
    
    # Highlight top attended
    top_idx = np.argsort(last_attn)[-3:]
    for idx in top_idx:
        ax2.annotate(tokens_clean[idx], xy=(idx, last_attn[idx]), fontsize=8, rotation=45)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step16_inhibitor.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("STEP 16 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

