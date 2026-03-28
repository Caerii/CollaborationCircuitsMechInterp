"""
Step 24: Find the First-Mention Heuristic Circuit

Step 20 revealed model always predicts first-mentioned location.
Let's find WHERE this heuristic is implemented:

1. Test if swapping location order in prompt flips prediction
2. Find which heads attend to first-mentioned location
3. Ablate those heads to break the heuristic

OUTPUT: results/step24_heuristic_circuit.json, figures/step24_*.png
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add framework to path
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig
from analysis.circuit_analysis import CircuitAnalysis

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def get_prediction(model, tokenizer, prompt):
    """Get next token prediction."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    
    # Get top prediction
    top_id = logits.argmax().item()
    return tokenizer.decode(top_id)


def get_logit_diff(model, tokenizer, prompt, token1, token2):
    """Get logit difference between two tokens."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    id1 = tokenizer.encode(token1, add_special_tokens=False)[0]
    id2 = tokenizer.encode(token2, add_special_tokens=False)[0]
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    
    return float(logits[id1] - logits[id2])


def main():
    print("=" * 70)
    print("STEP 24: FIRST-MENTION HEURISTIC CIRCUIT HUNT")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\nFinding where the first-mention heuristic is implemented")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    
    # Test scenarios: same story, different first-mention
    test_pairs = [
        {
            "name": "drawer_first",
            "prompt": "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket. Alice looks in the",
            "first": " drawer",
            "second": " basket",
        },
        {
            "name": "basket_first",
            "prompt": "Alice saw the basket. Alice put the ball in the basket. Alice left. Bob moved the ball to the drawer. Alice looks in the",
            "first": " basket",
            "second": " drawer",
        },
        {
            "name": "box_first",
            "prompt": "Tom put the key in the box. Tom went out. Jerry moved the key to the shelf. Tom looks in the",
            "first": " box",
            "second": " shelf",
        },
        {
            "name": "shelf_first",
            "prompt": "Tom noticed the shelf. Tom put the key on the shelf. Tom went out. Jerry moved the key to the box. Tom looks in the",
            "first": " shelf",
            "second": " box",
        },
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
        output_attentions=True,
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded!")
    sys.stdout.flush()
    
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    
    # ========================================
    # TEST FIRST-MENTION EFFECT
    # ========================================
    print(f"\n{'='*60}")
    print("TESTING FIRST-MENTION EFFECT")
    print(f"{'='*60}")
    
    first_mention_results = []
    for pair in test_pairs:
        diff = get_logit_diff(model, tokenizer, pair["prompt"], pair["first"], pair["second"])
        predicts_first = diff > 0
        first_mention_results.append({
            "name": pair["name"],
            "predicts_first": predicts_first,
            "logit_diff": diff,
        })
        
        print(f"\n{pair['name']}:")
        print(f"  First-mentioned: {pair['first']}")
        print(f"  Predicts first: {'YES' if predicts_first else 'NO'} (diff={diff:.2f})")
    
    first_rate = sum(1 for r in first_mention_results if r["predicts_first"]) / len(first_mention_results)
    print(f"\nFirst-mention prediction rate: {first_rate:.0%}")
    
    # ========================================
    # FIND ATTENTION TO FIRST-MENTION
    # ========================================
    print(f"\n{'='*60}")
    print("FINDING HEADS THAT ATTEND TO FIRST LOCATION")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    # Use a single prompt to analyze attention
    analysis_prompt = "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket. Alice looks in the"
    inputs = tokenizer(analysis_prompt, return_tensors="pt").to(model.device)
    
    # Get token positions
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Find position of "drawer" and "basket"
    drawer_pos = None
    basket_pos = None
    for i, tok in enumerate(tokens):
        if "drawer" in tok.lower():
            drawer_pos = i
        if "basket" in tok.lower():
            basket_pos = i
    
    print(f"Token positions: drawer={drawer_pos}, basket={basket_pos}")
    print(f"Total tokens: {len(tokens)}")
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions  # (n_layers, batch, n_heads, seq, seq)
    
    # Find heads that attend strongly to first-mentioned (drawer)
    head_attention_to_first = {}
    head_attention_to_second = {}
    
    last_pos = len(tokens) - 1  # Position we're predicting from
    
    for layer_idx, layer_attn in enumerate(attentions):
        attn = layer_attn[0]  # Remove batch dim: (n_heads, seq, seq)
        for head_idx in range(n_heads):
            if drawer_pos is not None:
                first_attn = float(attn[head_idx, last_pos, drawer_pos])
            else:
                first_attn = 0
            if basket_pos is not None:
                second_attn = float(attn[head_idx, last_pos, basket_pos])
            else:
                second_attn = 0
            
            head_attention_to_first[(layer_idx, head_idx)] = first_attn
            head_attention_to_second[(layer_idx, head_idx)] = second_attn
    
    # Find heads with highest attention to first location
    sorted_by_first = sorted(head_attention_to_first.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 heads attending to FIRST-MENTIONED (drawer):")
    for (layer, head), attn in sorted_by_first[:10]:
        second_attn = head_attention_to_second[(layer, head)]
        ratio = attn / (second_attn + 1e-8)
        print(f"  L{layer}H{head}: {attn:.3f} (vs {second_attn:.3f} to basket, ratio={ratio:.1f}x)")
    
    # ========================================
    # ABLATE TOP FIRST-MENTION HEADS
    # ========================================
    print(f"\n{'='*60}")
    print("ABLATING TOP FIRST-MENTION HEADS")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    circuit = CircuitAnalysis(model, tokenizer, n_layers, n_heads)
    
    # Get top 5 first-mention heads
    top_first_heads = [(l, h) for (l, h), _ in sorted_by_first[:5]]
    print(f"Ablating heads: {top_first_heads}")
    
    # Baseline
    baseline_diff = get_logit_diff(model, tokenizer, analysis_prompt, " drawer", " basket")
    print(f"\nBaseline: drawer-basket diff = {baseline_diff:.2f}")
    
    # Ablate and test
    circuit.ablate_heads(top_first_heads)
    ablated_diff = get_logit_diff(model, tokenizer, analysis_prompt, " drawer", " basket")
    circuit._clear_hooks()
    
    print(f"After ablation: drawer-basket diff = {ablated_diff:.2f}")
    print(f"Change: {ablated_diff - baseline_diff:+.2f}")
    
    if ablated_diff < baseline_diff:
        print("\n*** Ablating first-mention heads REDUCES preference for first location! ***")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {"model": config.model_name},
        "first_mention_results": first_mention_results,
        "first_mention_rate": first_rate,
        "top_first_mention_heads": [
            {"layer": l, "head": h, "attention": float(a)} 
            for (l, h), a in sorted_by_first[:10]
        ],
        "ablation": {
            "heads": [{"layer": l, "head": h} for l, h in top_first_heads],
            "baseline_diff": baseline_diff,
            "ablated_diff": ablated_diff,
            "change": ablated_diff - baseline_diff,
        },
    }
    
    output_path = RESULTS_DIR / "step24_heuristic_circuit.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # Generate figure
    print("\nGenerating figure...")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Figure 1: Attention to first vs second
    ax1 = axes[0]
    top_heads = sorted_by_first[:15]
    labels = [f"L{l}H{h}" for (l, h), _ in top_heads]
    first_attns = [a for _, a in top_heads]
    second_attns = [head_attention_to_second[(l, h)] for (l, h), _ in top_heads]
    
    x = np.arange(len(labels))
    width = 0.35
    ax1.bar(x - width/2, first_attns, width, label='First (drawer)', color='coral')
    ax1.bar(x + width/2, second_attns, width, label='Second (basket)', color='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylabel("Attention Weight", fontsize=12)
    ax1.set_title("Head Attention to Locations", fontsize=14, fontweight='bold')
    ax1.legend()
    
    # Figure 2: Ablation effect
    ax2 = axes[1]
    conditions = ["Baseline", "After Ablation"]
    diffs = [baseline_diff, ablated_diff]
    colors = ['coral' if d > 0 else 'steelblue' for d in diffs]
    bars = ax2.bar(conditions, diffs, color=colors, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel("Logit Diff (drawer - basket)", fontsize=12)
    ax2.set_title("Effect of Ablating First-Mention Heads", fontsize=14, fontweight='bold')
    
    for bar, d in zip(bars, diffs):
        ax2.annotate(f'{d:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom' if d > 0 else 'top', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step24_heuristic_circuit.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("STEP 24 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

