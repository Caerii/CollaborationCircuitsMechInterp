"""
Step 9: Attention Pattern Analysis - What Do Critical Heads Attend To?

We know L32H0, L33H4, L33H16, L33H28, L34H0 are critical for ToM.
But WHAT do they actually attend to?

QUESTIONS:
1. Do they attend to agent names? (Alice, Bob)
2. Do they attend to locations? (drawer, basket)
3. Do they attend to belief verbs? (thinks, knows, believes)
4. Do they attend to negation? (didn't, not, left)

METHOD:
- Extract attention patterns from critical heads
- Visualize what tokens they attend to
- Compare ToM-correct vs ToM-incorrect scenarios

OUTPUT: results/step9_attention.json, figures/step9_*.png
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

# Critical heads from Steps 5 & 7
CRITICAL_HEADS = [(32, 0), (33, 4), (33, 16), (33, 28), (34, 0)]


def get_attention_patterns(model, tokenizer, prompt, layers):
    """Extract attention patterns for specified layers."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # We need attention outputs
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True,
        )
    
    # outputs.attentions is tuple of (batch, n_heads, seq, seq) per layer
    attention_patterns = {}
    for layer_idx in layers:
        if layer_idx < len(outputs.attentions):
            # Get attention for this layer, all heads
            # Shape: (batch, n_heads, seq, seq)
            layer_attn = outputs.attentions[layer_idx][0]  # Remove batch dim
            attention_patterns[layer_idx] = layer_attn.cpu().numpy()
    
    return attention_patterns, tokens


def analyze_attention_to_tokens(attention, tokens, target_tokens):
    """Calculate how much attention goes to specific token types."""
    # attention shape: (n_heads, seq, seq)
    # We look at the last position's attention (what does prediction attend to?)
    
    last_pos_attn = attention[:, -1, :]  # (n_heads, seq)
    
    results = {}
    for token_type, keywords in target_tokens.items():
        # Find positions of matching tokens
        positions = []
        for i, tok in enumerate(tokens):
            tok_clean = tok.lower().replace('▁', '').replace('Ġ', '')
            for kw in keywords:
                if kw.lower() in tok_clean:
                    positions.append(i)
                    break
        
        if positions:
            # Sum attention to these positions
            attn_to_type = last_pos_attn[:, positions].sum(axis=1)  # (n_heads,)
            # Sanitize tokens for storage
            tokens_found = [tokens[p].encode('ascii', 'replace').decode('ascii') for p in positions]
            results[token_type] = {
                "positions": positions,
                "tokens_found": tokens_found,
                "attention_per_head": attn_to_type.tolist(),
                "mean_attention": float(attn_to_type.mean()),
            }
        else:
            results[token_type] = {
                "positions": [],
                "tokens_found": [],
                "attention_per_head": [],
                "mean_attention": 0.0,
            }
    
    return results


def main():
    print("=" * 70)
    print("STEP 9: ATTENTION PATTERN ANALYSIS")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"\nAnalyzing critical heads: {CRITICAL_HEADS}")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    
    # Token types to analyze
    target_tokens = {
        "agents": ["alice", "bob", "sally", "anne", "she", "he", "they"],
        "locations": ["drawer", "basket", "box", "bin", "table", "shelf", "cave", "forest"],
        "belief_verbs": ["thinks", "knows", "believes", "expects", "assumes"],
        "movement": ["moved", "put", "placed", "left", "returned", "came"],
        "negation": ["not", "didn't", "doesn't", "never", "without"],
    }
    
    # Test scenarios
    scenarios = [
        {
            "name": "classic_false_belief",
            "prompt": "Alice put the ball in the drawer. Alice left the room. Bob moved the ball to the basket. Alice returned. Where does Alice think the ball is? Alice thinks it is in the",
            "correct": "drawer",
            "type": "false_belief",
        },
        {
            "name": "true_belief_saw",
            "prompt": "Alice put the ball in the drawer. Alice stayed and watched. Bob moved the ball to the basket. Where does Alice think the ball is? Alice thinks it is in the",
            "correct": "basket",
            "type": "true_belief",
        },
        {
            "name": "told_update",
            "prompt": "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket. Bob told Alice about the move. Where does Alice think the ball is? Alice thinks it is in the",
            "correct": "basket",
            "type": "communicated",
        },
        {
            "name": "multi_agent",
            "prompt": "Alice put the ball in the drawer. Bob saw this. Alice left. Carol moved the ball to the basket. Bob watched Carol. Where does Alice think the ball is? Alice thinks it is in the",
            "correct": "drawer",
            "type": "multi_agent",
        },
    ]
    
    # Load model with attention output enabled
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
    
    # Get unique layers from critical heads
    critical_layers = sorted(set(h[0] for h in CRITICAL_HEADS))
    print(f"Extracting attention from layers: {critical_layers}")
    
    results = []
    
    # ========================================
    # ANALYZE EACH SCENARIO
    # ========================================
    print(f"\n{'='*60}")
    print("ANALYZING ATTENTION PATTERNS")
    print(f"{'='*60}")
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"Type: {scenario['type']}")
        sys.stdout.flush()
        
        # Get attention patterns
        attention_patterns, tokens = get_attention_patterns(
            model, tokenizer, scenario["prompt"], critical_layers
        )
        
        print(f"Tokens: {len(tokens)}")
        
        # Analyze attention to different token types
        scenario_results = {
            "name": scenario["name"],
            "type": scenario["type"],
            "prompt": scenario["prompt"],
            "n_tokens": len(tokens),
            "by_head": {},
            "by_token_type": {},
        }
        
        # For each critical head
        for layer, head in CRITICAL_HEADS:
            if layer in attention_patterns:
                head_attn = attention_patterns[layer][head:head+1, :, :]  # (1, seq, seq)
                head_attn = head_attn[0]  # (seq, seq)
                
                # Analyze what this head attends to
                analysis = analyze_attention_to_tokens(
                    attention_patterns[layer][head:head+1],
                    tokens,
                    target_tokens
                )
                
                head_name = f"L{layer}H{head}"
                scenario_results["by_head"][head_name] = analysis
                
                # Print summary
                print(f"\n  {head_name}:")
                for token_type, data in analysis.items():
                    if data["mean_attention"] > 0.01:  # Only show significant
                        # Sanitize token names for Windows console
                        tokens_clean = [t.encode('ascii', 'replace').decode('ascii') for t in data['tokens_found'][:3]]
                        print(f"    {token_type}: {data['mean_attention']:.3f} ({tokens_clean})")
        
        # Aggregate by token type across all critical heads
        for token_type in target_tokens.keys():
            total_attn = 0
            count = 0
            for head_name, analysis in scenario_results["by_head"].items():
                if token_type in analysis:
                    total_attn += analysis[token_type]["mean_attention"]
                    count += 1
            scenario_results["by_token_type"][token_type] = total_attn / count if count > 0 else 0
        
        results.append(scenario_results)
        sys.stdout.flush()
    
    # ========================================
    # SUMMARY: What do critical heads attend to?
    # ========================================
    print(f"\n{'='*60}")
    print("SUMMARY: Critical Head Attention Patterns")
    print(f"{'='*60}")
    
    # Average across all scenarios
    avg_by_type = {t: 0 for t in target_tokens.keys()}
    for r in results:
        for t, v in r["by_token_type"].items():
            avg_by_type[t] += v / len(results)
    
    print("\nAverage attention to token types (across all scenarios):")
    sorted_types = sorted(avg_by_type.items(), key=lambda x: x[1], reverse=True)
    for token_type, avg_attn in sorted_types:
        bar = "#" * int(avg_attn * 100)
        print(f"  {token_type:15s}: {avg_attn:.3f} {bar}")
    
    # By scenario type
    print("\nBy scenario type:")
    for scenario_type in ["false_belief", "true_belief", "communicated", "multi_agent"]:
        type_results = [r for r in results if r["type"] == scenario_type]
        if type_results:
            print(f"\n  {scenario_type}:")
            for token_type in ["agents", "locations", "belief_verbs"]:
                avg = np.mean([r["by_token_type"][token_type] for r in type_results])
                print(f"    {token_type}: {avg:.3f}")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "critical_heads": [list(h) for h in CRITICAL_HEADS],
            "target_token_types": list(target_tokens.keys()),
        },
        "scenarios": results,
        "summary": {
            "avg_attention_by_type": avg_by_type,
        },
    }
    
    output_path = RESULTS_DIR / "step9_attention.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # ========================================
    # GENERATE FIGURES
    # ========================================
    print("\nGenerating figures...")
    
    import matplotlib.pyplot as plt
    
    # Figure 1: Average attention by token type
    fig, ax = plt.subplots(figsize=(10, 6))
    
    types = [t[0] for t in sorted_types]
    values = [t[1] for t in sorted_types]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(types)))
    
    bars = ax.barh(types, values, color=colors, edgecolor='black')
    ax.set_xlabel("Average Attention Weight", fontsize=12)
    ax.set_title("What Do Critical ToM Heads Attend To?", fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.3f}', 
                    xy=(val, bar.get_y() + bar.get_height()/2),
                    ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step9_attention_by_type.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    # Figure 2: Heatmap of head attention by token type
    fig, ax = plt.subplots(figsize=(12, 6))
    
    head_names = [f"L{l}H{h}" for l, h in CRITICAL_HEADS]
    token_types = list(target_tokens.keys())
    
    # Build matrix: rows = heads, cols = token types, values = avg attention
    matrix = np.zeros((len(head_names), len(token_types)))
    for i, head_name in enumerate(head_names):
        for j, token_type in enumerate(token_types):
            attns = []
            for r in results:
                if head_name in r["by_head"] and token_type in r["by_head"][head_name]:
                    attns.append(r["by_head"][head_name][token_type]["mean_attention"])
            matrix[i, j] = np.mean(attns) if attns else 0
    
    im = ax.imshow(matrix, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(token_types)))
    ax.set_xticklabels(token_types, rotation=45, ha='right')
    ax.set_yticks(range(len(head_names)))
    ax.set_yticklabels(head_names)
    ax.set_title("Critical Head Attention Patterns", fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Attention Weight", fontsize=10)
    
    # Add text annotations
    for i in range(len(head_names)):
        for j in range(len(token_types)):
            val = matrix[i, j]
            if val > 0.01:
                ax.annotate(f'{val:.2f}', xy=(j, i), ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step9_attention_heatmap.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("STEP 9 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

