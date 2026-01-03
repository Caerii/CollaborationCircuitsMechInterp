"""
Step 20: The "told" Mystery - Deep Attention Analysis

Why does "told" cause COMPLETE ToM failure (0% baseline) while "provided" doesn't (100%)?

This script performs detailed attention pattern analysis on the 5 decision heads
to understand what makes "told" special.
"""

import torch
import json
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration
MODEL_NAME = "Qwen/Qwen3-4B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The circuit we discovered
DECISION_HEADS = [(17, 4), (18, 11), (18, 14), (19, 30), (21, 17)]
ENABLER_HEADS = [(15, 9), (19, 2), (19, 15)]
ALL_HEADS = DECISION_HEADS + ENABLER_HEADS

# Verbs to compare
COMPARE_VERBS = {
    "worst": ["told", "announced", "clarified", "noted", "queried"],  # 0% baseline
    "best": ["provided", "dispatched", "supported", "manifested"],    # 80-100% baseline
    "mid": ["mentioned", "said", "stated", "communicated"]            # Mid-range
}

# Output paths
OUTPUT_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def generate_scenario(verb: str) -> tuple[str, str, str]:
    """Generate a ToM scenario with the given verb.
    
    Returns: (prompt, correct_answer, wrong_answer)
    """
    prompt = f"""Alice puts the ball in the drawer. Alice leaves.
Bob {verb} Carol that he moved the ball to the basket.
Alice returns. Alice will look for the ball in the"""
    
    return prompt, " drawer", " basket"


def load_model():
    """Load model with eager attention for attention weight extraction."""
    print(f"Loading {MODEL_NAME} with eager attention...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"  # Critical for attention extraction
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def get_attention_patterns(model, tokenizer, prompt: str) -> dict:
    """Extract attention patterns from all heads of interest.
    
    Returns dict with attention weights and token info.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Get tokens for labeling
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    # Extract attention from heads of interest
    attention_data = {}
    for layer_idx, head_idx in ALL_HEADS:
        # outputs.attentions is tuple of (batch, heads, seq, seq) per layer
        layer_attn = outputs.attentions[layer_idx]  # (1, n_heads, seq, seq)
        head_attn = layer_attn[0, head_idx].cpu().float().numpy()  # (seq, seq)
        attention_data[f"L{layer_idx}H{head_idx}"] = head_attn
    
    return {
        "tokens": tokens,
        "attention": attention_data,
        "input_ids": inputs.input_ids[0].cpu().tolist()
    }


def get_token_positions(tokens: list[str], prompt: str) -> dict:
    """Find positions of key tokens in the sequence."""
    positions = {
        "alice": [],
        "bob": [],
        "carol": [],
        "ball": [],
        "drawer": [],
        "basket": [],
        "verb": None  # Will be set based on verb position
    }
    
    # Simple token matching (handles subword tokenization)
    text_lower = " ".join(tokens).lower()
    for i, tok in enumerate(tokens):
        tok_lower = tok.lower().replace("▁", "").replace("Ġ", "")
        if "alice" in tok_lower:
            positions["alice"].append(i)
        elif "bob" in tok_lower:
            positions["bob"].append(i)
        elif "carol" in tok_lower:
            positions["carol"].append(i)
        elif "ball" in tok_lower:
            positions["ball"].append(i)
        elif "drawer" in tok_lower:
            positions["drawer"].append(i)
        elif "basket" in tok_lower:
            positions["basket"].append(i)
    
    return positions


def find_verb_position(tokens: list[str], verb: str) -> int:
    """Find the position of the verb in the token sequence."""
    verb_lower = verb.lower()
    for i, tok in enumerate(tokens):
        tok_clean = tok.lower().replace("▁", "").replace("Ġ", "").replace("<", "").replace(">", "")
        if verb_lower in tok_clean or tok_clean in verb_lower:
            return i
    return -1


def compute_attention_stats(attention_data: dict, tokens: list, verb: str) -> dict:
    """Compute statistics about attention patterns."""
    positions = get_token_positions(tokens, "")
    verb_pos = find_verb_position(tokens, verb)
    
    stats = {}
    
    for head_name, attn_matrix in attention_data.items():
        seq_len = attn_matrix.shape[0]
        
        # Last token attention (most important for prediction)
        last_token_attn = attn_matrix[-1]  # What the last token attends to
        
        head_stats = {
            "attn_to_verb": float(last_token_attn[verb_pos]) if verb_pos >= 0 else 0.0,
            "attn_to_alice": float(sum(last_token_attn[i] for i in positions["alice"])) if positions["alice"] else 0.0,
            "attn_to_bob": float(sum(last_token_attn[i] for i in positions["bob"])) if positions["bob"] else 0.0,
            "attn_to_carol": float(sum(last_token_attn[i] for i in positions["carol"])) if positions["carol"] else 0.0,
            "attn_to_drawer": float(sum(last_token_attn[i] for i in positions["drawer"])) if positions["drawer"] else 0.0,
            "attn_to_basket": float(sum(last_token_attn[i] for i in positions["basket"])) if positions["basket"] else 0.0,
            "verb_position": verb_pos,
            "max_attn_position": int(np.argmax(last_token_attn)),
            "entropy": float(-np.sum(last_token_attn * np.log(last_token_attn + 1e-10)))
        }
        stats[head_name] = head_stats
    
    return stats


def test_verb_accuracy(model, tokenizer, verb: str, n_trials: int = 10) -> tuple[float, list]:
    """Test accuracy for a verb and return individual results."""
    results = []
    
    for _ in range(n_trials):
        prompt, correct, wrong = generate_scenario(verb)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]
        
        correct_id = tokenizer.encode(correct, add_special_tokens=False)[0]
        wrong_id = tokenizer.encode(wrong, add_special_tokens=False)[0]
        
        is_correct = logits[correct_id] > logits[wrong_id]
        results.append({
            "correct": bool(is_correct),
            "correct_logit": float(logits[correct_id]),
            "wrong_logit": float(logits[wrong_id]),
            "diff": float(logits[correct_id] - logits[wrong_id])
        })
    
    accuracy = sum(r["correct"] for r in results) / len(results)
    return accuracy, results


def plot_attention_comparison(all_attention_stats: dict, verb_category: dict):
    """Create comparison plots for attention patterns across verb categories."""
    
    # Prepare data for plotting
    heads = list(next(iter(all_attention_stats.values())).keys())
    metrics = ["attn_to_verb", "attn_to_alice", "attn_to_bob", "attn_to_carol", "attn_to_drawer", "attn_to_basket"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Attention Patterns: Decision Heads (Last Token → Key Positions)", fontsize=14, fontweight='bold')
    
    colors = {"worst": "#e74c3c", "best": "#27ae60", "mid": "#3498db"}
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        # Group by category
        for category, verb_list in verb_category.items():
            values = []
            for verb in verb_list:
                if verb in all_attention_stats:
                    # Average across decision heads
                    head_values = []
                    for head in DECISION_HEADS:
                        head_name = f"L{head[0]}H{head[1]}"
                        if head_name in all_attention_stats[verb]:
                            head_values.append(all_attention_stats[verb][head_name].get(metric, 0))
                    if head_values:
                        values.append(np.mean(head_values))
            
            if values:
                ax.bar(category, np.mean(values), yerr=np.std(values) if len(values) > 1 else 0,
                       color=colors[category], alpha=0.7, capsize=5, label=category)
        
        ax.set_title(metric.replace("attn_to_", "Attention to ").title())
        ax.set_ylabel("Attention Weight")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "told_mystery_attention_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'told_mystery_attention_comparison.png'}")


def plot_head_heatmaps(attention_data: dict, tokens: list, verb: str, output_name: str):
    """Plot attention heatmaps for each decision head."""
    
    # Only plot last 30 tokens (most relevant)
    max_tokens = 30
    if len(tokens) > max_tokens:
        tokens = tokens[-max_tokens:]
        for head_name in attention_data:
            attention_data[head_name] = attention_data[head_name][-max_tokens:, -max_tokens:]
    
    n_heads = len(DECISION_HEADS)
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    fig.suptitle(f"Decision Head Attention Patterns: '{verb}'", fontsize=14, fontweight='bold')
    
    for idx, (layer, head) in enumerate(DECISION_HEADS):
        head_name = f"L{layer}H{head}"
        ax = axes[idx] if n_heads > 1 else axes
        
        if head_name in attention_data:
            attn = attention_data[head_name]
            
            # Clean token labels
            clean_tokens = [t.replace("▁", "").replace("Ġ", "")[:8] for t in tokens]
            
            sns.heatmap(attn, ax=ax, cmap='viridis', 
                       xticklabels=clean_tokens, yticklabels=clean_tokens,
                       cbar_kws={'shrink': 0.5})
            ax.set_title(head_name)
            ax.tick_params(axis='x', rotation=45, labelsize=6)
            ax.tick_params(axis='y', rotation=0, labelsize=6)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{output_name}_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / output_name}_heatmap.png")


def plot_logit_differences(all_accuracies: dict, verb_category: dict):
    """Plot logit differences between correct and wrong answers."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Flatten and sort by accuracy
    verb_data = []
    for category, verbs in verb_category.items():
        for verb in verbs:
            if verb in all_accuracies:
                verb_data.append({
                    "verb": verb,
                    "category": category,
                    "accuracy": all_accuracies[verb]["accuracy"],
                    "mean_diff": np.mean([r["diff"] for r in all_accuracies[verb]["results"]])
                })
    
    verb_data.sort(key=lambda x: x["accuracy"])
    
    colors = {"worst": "#e74c3c", "best": "#27ae60", "mid": "#3498db"}
    
    x = range(len(verb_data))
    bars = ax.bar(x, [d["mean_diff"] for d in verb_data],
                  color=[colors[d["category"]] for d in verb_data], alpha=0.7)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([d["verb"] for d in verb_data], rotation=45, ha='right')
    ax.set_ylabel("Logit Difference (correct - wrong)")
    ax.set_xlabel("Verb")
    ax.set_title("Logit Differences by Verb (sorted by accuracy)")
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[c], label=c) for c in colors]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "told_mystery_logit_diffs.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'told_mystery_logit_diffs.png'}")


def main():
    print("=" * 70)
    print("DEEP INVESTIGATION: The 'told' Mystery")
    print("=" * 70)
    print()
    
    model, tokenizer = load_model()
    
    all_attention_stats = {}
    all_accuracies = {}
    
    # Test each verb
    all_verbs = COMPARE_VERBS["worst"] + COMPARE_VERBS["best"] + COMPARE_VERBS["mid"]
    
    print(f"\nTesting {len(all_verbs)} verbs...")
    print("-" * 50)
    
    for verb in all_verbs:
        print(f"\n>> {verb}")
        
        # Test accuracy
        accuracy, results = test_verb_accuracy(model, tokenizer, verb, n_trials=10)
        all_accuracies[verb] = {"accuracy": accuracy, "results": results}
        print(f"   Accuracy: {accuracy*100:.0f}%")
        
        # Get attention patterns
        prompt, _, _ = generate_scenario(verb)
        attn_data = get_attention_patterns(model, tokenizer, prompt)
        
        # Compute statistics
        stats = compute_attention_stats(attn_data["attention"], attn_data["tokens"], verb)
        all_attention_stats[verb] = stats
        
        # Print key attention stats for decision heads
        for head in DECISION_HEADS[:2]:  # Just show top 2
            head_name = f"L{head[0]}H{head[1]}"
            if head_name in stats:
                s = stats[head_name]
                print(f"   {head_name}: verb={s['attn_to_verb']:.3f}, alice={s['attn_to_alice']:.3f}, basket={s['attn_to_basket']:.3f}")
        
        # Plot heatmaps for extreme cases
        if verb in ["told", "provided"]:
            plot_head_heatmaps(attn_data["attention"], attn_data["tokens"], verb, f"told_mystery_{verb}")
    
    # Create comparison plots
    print("\n" + "=" * 50)
    print("Creating visualizations...")
    
    plot_attention_comparison(all_attention_stats, COMPARE_VERBS)
    plot_logit_differences(all_accuracies, COMPARE_VERBS)
    
    # Summary analysis
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    # Compare worst vs best
    print("\n1. ACCURACY COMPARISON:")
    for category in ["worst", "best", "mid"]:
        accs = [all_accuracies[v]["accuracy"] for v in COMPARE_VERBS[category] if v in all_accuracies]
        print(f"   {category}: {np.mean(accs)*100:.1f}% (±{np.std(accs)*100:.1f}%)")
    
    # Attention patterns
    print("\n2. ATTENTION TO VERB (Decision Heads):")
    for category in ["worst", "best", "mid"]:
        verb_attns = []
        for verb in COMPARE_VERBS[category]:
            if verb in all_attention_stats:
                for head in DECISION_HEADS:
                    head_name = f"L{head[0]}H{head[1]}"
                    if head_name in all_attention_stats[verb]:
                        verb_attns.append(all_attention_stats[verb][head_name]["attn_to_verb"])
        if verb_attns:
            print(f"   {category}: {np.mean(verb_attns):.4f} (±{np.std(verb_attns):.4f})")
    
    print("\n3. ATTENTION TO BASKET (wrong answer):")
    for category in ["worst", "best", "mid"]:
        basket_attns = []
        for verb in COMPARE_VERBS[category]:
            if verb in all_attention_stats:
                for head in DECISION_HEADS:
                    head_name = f"L{head[0]}H{head[1]}"
                    if head_name in all_attention_stats[verb]:
                        basket_attns.append(all_attention_stats[verb][head_name]["attn_to_basket"])
        if basket_attns:
            print(f"   {category}: {np.mean(basket_attns):.4f} (±{np.std(basket_attns):.4f})")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "verbs_tested": all_verbs,
        "accuracies": {v: {"accuracy": d["accuracy"], 
                          "mean_diff": float(np.mean([r["diff"] for r in d["results"]]))}
                      for v, d in all_accuracies.items()},
        "attention_stats": all_attention_stats
    }
    
    with open(OUTPUT_DIR / "told_mystery_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to: {OUTPUT_DIR / 'told_mystery_results.json'}")
    
    print("\n" + "=" * 70)
    print("Investigation complete! Check figures/ for visualizations.")
    print("=" * 70)


if __name__ == "__main__":
    main()

