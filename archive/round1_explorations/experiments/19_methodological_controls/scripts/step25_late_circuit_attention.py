"""
Step 25: Late Circuit Attention Analysis

Goal: Understand HOW the 10 late-layer heads (L32-L35) override correct ToM.

Questions to answer:
1. What tokens do these heads attend to?
2. How does attention differ for "told" vs "announced"?
3. What is the mechanism of the override?
"""

import torch
import json
import numpy as np
import sys
import io
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# The 10 critical late-layer heads
LATE_CIRCUIT_HEADS = [
    (32, 6), (32, 31),
    (33, 6), (33, 13), (33, 17), (33, 31),
    (34, 17),
    (35, 0), (35, 1), (35, 17)
]

# Verbs to compare
BAD_VERBS = ["told", "said", "mentioned", "informed", "stated"]
GOOD_VERBS = ["announced", "asked", "hinted", "explained", "shouted"]

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"


def load_model():
    """Load model with eager attention for pattern extraction."""
    print("Loading Qwen3-4B with eager attention...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager"  # Required for attention weights
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model.eval()
    return model, tokenizer


def create_prompt(verb):
    """Create Sally-Anne style ToM prompt with given verb."""
    return f"""Alice put the ball in the drawer. Alice then left the room.
While Alice was away, Bob {verb} Carol that he moved the ball to the basket.
When Alice returned, Alice looked for the ball. Alice searched in the"""


def get_attention_patterns(model, tokenizer, prompt):
    """Extract attention patterns from all late-layer heads."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # outputs.attentions is tuple of (batch, n_heads, seq, seq) per layer
    attention_patterns = {}
    
    for layer_idx, head_idx in LATE_CIRCUIT_HEADS:
        layer_attn = outputs.attentions[layer_idx]  # (batch, n_heads, seq, seq)
        head_attn = layer_attn[0, head_idx].cpu().float().numpy()  # (seq, seq)
        attention_patterns[(layer_idx, head_idx)] = head_attn
    
    # Get token labels for visualization
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    return attention_patterns, tokens


def find_key_token_positions(tokens, verb):
    """Find positions of key tokens in the sequence."""
    positions = {
        'verb': None,
        'alice': [],
        'bob': None,
        'carol': None,
        'drawer': None,
        'basket': None,
        'ball': [],
        'last': len(tokens) - 1
    }
    
    for i, token in enumerate(tokens):
        token_lower = token.lower().replace('▁', '').replace('Ġ', '')
        if verb.lower() in token_lower and positions['verb'] is None:
            positions['verb'] = i
        if 'alice' in token_lower:
            positions['alice'].append(i)
        if 'bob' in token_lower:
            positions['bob'] = i
        if 'carol' in token_lower:
            positions['carol'] = i
        if 'drawer' in token_lower:
            positions['drawer'] = i
        if 'basket' in token_lower:
            positions['basket'] = i
        if 'ball' in token_lower:
            positions['ball'].append(i)
    
    return positions


def analyze_attention_to_key_tokens(attention_patterns, tokens, positions):
    """Analyze what the last token attends to via late circuit heads."""
    last_pos = positions['last']
    
    analysis = {}
    for (layer, head), attn in attention_patterns.items():
        head_key = f"L{layer}H{head}"
        # Attention FROM last token TO all other tokens
        last_token_attn = attn[last_pos, :]
        
        analysis[head_key] = {
            'to_verb': float(last_token_attn[positions['verb']]) if positions['verb'] else 0,
            'to_alice_first': float(last_token_attn[positions['alice'][0]]) if positions['alice'] else 0,
            'to_alice_last': float(last_token_attn[positions['alice'][-1]]) if positions['alice'] else 0,
            'to_bob': float(last_token_attn[positions['bob']]) if positions['bob'] else 0,
            'to_carol': float(last_token_attn[positions['carol']]) if positions['carol'] else 0,
            'to_drawer': float(last_token_attn[positions['drawer']]) if positions['drawer'] else 0,
            'to_basket': float(last_token_attn[positions['basket']]) if positions['basket'] else 0,
            'to_ball_first': float(last_token_attn[positions['ball'][0]]) if positions['ball'] else 0,
            'max_attn': float(np.max(last_token_attn)),
            'max_attn_pos': int(np.argmax(last_token_attn)),
            'max_attn_token': tokens[int(np.argmax(last_token_attn))].replace('\u0120', '').replace('\u2581', '')
        }
    
    return analysis


def run_verb_comparison():
    """Compare attention patterns for bad vs good verbs."""
    model, tokenizer = load_model()
    
    results = {
        'bad_verbs': {},
        'good_verbs': {},
        'comparison': {}
    }
    
    print("\n" + "="*60)
    print("ANALYZING BAD VERBS (0% baseline)")
    print("="*60)
    
    for verb in BAD_VERBS:
        print(f"\n--- {verb.upper()} ---")
        prompt = create_prompt(verb)
        attention_patterns, tokens = get_attention_patterns(model, tokenizer, prompt)
        positions = find_key_token_positions(tokens, verb)
        analysis = analyze_attention_to_key_tokens(attention_patterns, tokens, positions)
        results['bad_verbs'][verb] = {
            'analysis': analysis,
            'positions': {k: v if not isinstance(v, list) else v for k, v in positions.items()},
            'n_tokens': len(tokens)
        }
        
        # Print summary
        print(f"  Tokens: {len(tokens)}")
        print(f"  Verb position: {positions['verb']}")
        for head_key, head_analysis in analysis.items():
            print(f"  {head_key}: verb={head_analysis['to_verb']:.3f}, "
                  f"drawer={head_analysis['to_drawer']:.3f}, "
                  f"basket={head_analysis['to_basket']:.3f}, "
                  f"max={head_analysis['max_attn']:.3f} @ '{head_analysis['max_attn_token']}'")
    
    print("\n" + "="*60)
    print("ANALYZING GOOD VERBS (100% baseline)")
    print("="*60)
    
    for verb in GOOD_VERBS:
        print(f"\n--- {verb.upper()} ---")
        prompt = create_prompt(verb)
        attention_patterns, tokens = get_attention_patterns(model, tokenizer, prompt)
        positions = find_key_token_positions(tokens, verb)
        analysis = analyze_attention_to_key_tokens(attention_patterns, tokens, positions)
        results['good_verbs'][verb] = {
            'analysis': analysis,
            'positions': {k: v if not isinstance(v, list) else v for k, v in positions.items()},
            'n_tokens': len(tokens)
        }
        
        # Print summary
        print(f"  Tokens: {len(tokens)}")
        print(f"  Verb position: {positions['verb']}")
        for head_key, head_analysis in analysis.items():
            print(f"  {head_key}: verb={head_analysis['to_verb']:.3f}, "
                  f"drawer={head_analysis['to_drawer']:.3f}, "
                  f"basket={head_analysis['to_basket']:.3f}, "
                  f"max={head_analysis['max_attn']:.3f} @ '{head_analysis['max_attn_token']}'")
    
    # Compute average attention differences
    print("\n" + "="*60)
    print("COMPUTING DIFFERENCES: BAD - GOOD")
    print("="*60)
    
    for head in [f"L{l}H{h}" for l, h in LATE_CIRCUIT_HEADS]:
        bad_verb_attn = np.mean([results['bad_verbs'][v]['analysis'][head]['to_verb'] for v in BAD_VERBS])
        good_verb_attn = np.mean([results['good_verbs'][v]['analysis'][head]['to_verb'] for v in GOOD_VERBS])
        
        bad_basket_attn = np.mean([results['bad_verbs'][v]['analysis'][head]['to_basket'] for v in BAD_VERBS])
        good_basket_attn = np.mean([results['good_verbs'][v]['analysis'][head]['to_basket'] for v in GOOD_VERBS])
        
        bad_drawer_attn = np.mean([results['bad_verbs'][v]['analysis'][head]['to_drawer'] for v in BAD_VERBS])
        good_drawer_attn = np.mean([results['good_verbs'][v]['analysis'][head]['to_drawer'] for v in GOOD_VERBS])
        
        results['comparison'][head] = {
            'verb_diff': float(bad_verb_attn - good_verb_attn),
            'basket_diff': float(bad_basket_attn - good_basket_attn),
            'drawer_diff': float(bad_drawer_attn - good_drawer_attn),
            'bad_verb_mean': float(bad_verb_attn),
            'good_verb_mean': float(good_verb_attn),
            'bad_basket_mean': float(bad_basket_attn),
            'good_basket_mean': float(good_basket_attn)
        }
        
        print(f"{head}:")
        print(f"  Verb attention:   Bad={bad_verb_attn:.4f}, Good={good_verb_attn:.4f}, Diff={bad_verb_attn - good_verb_attn:+.4f}")
        print(f"  Basket attention: Bad={bad_basket_attn:.4f}, Good={good_basket_attn:.4f}, Diff={bad_basket_attn - good_basket_attn:+.4f}")
        print(f"  Drawer attention: Bad={bad_drawer_attn:.4f}, Good={good_drawer_attn:.4f}, Diff={bad_drawer_attn - good_drawer_attn:+.4f}")
    
    return results, model, tokenizer


def plot_attention_heatmaps(model, tokenizer, verb, save_prefix):
    """Plot attention heatmaps for all late circuit heads."""
    prompt = create_prompt(verb)
    attention_patterns, tokens = get_attention_patterns(model, tokenizer, prompt)
    
    # Truncate token labels for readability
    tokens_short = [t[:10] for t in tokens]
    
    # Create subplot grid for 10 heads
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle(f'Late Circuit Attention Patterns: "{verb}"', fontsize=16)
    
    for idx, ((layer, head), attn) in enumerate(attention_patterns.items()):
        ax = axes[idx // 5, idx % 5]
        
        # Only show last 30 tokens attending to all tokens for clarity
        n_show = min(30, attn.shape[0])
        attn_subset = attn[-n_show:, :]
        
        sns.heatmap(attn_subset, ax=ax, cmap='viridis', 
                   xticklabels=False, yticklabels=False,
                   cbar_kws={'shrink': 0.5})
        ax.set_title(f'L{layer}H{head}')
        ax.set_xlabel('Key position')
        ax.set_ylabel('Query position (last 30)')
    
    plt.tight_layout()
    save_path = FIGURES_DIR / f"{save_prefix}_heatmaps.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_comparison_bar_chart(results):
    """Plot bar chart comparing bad vs good verbs attention to key tokens."""
    heads = list(results['comparison'].keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Attention to verb
    bad_verb = [results['comparison'][h]['bad_verb_mean'] for h in heads]
    good_verb = [results['comparison'][h]['good_verb_mean'] for h in heads]
    
    x = np.arange(len(heads))
    width = 0.35
    
    axes[0].bar(x - width/2, bad_verb, width, label='Bad verbs (told, said...)', color='red', alpha=0.7)
    axes[0].bar(x + width/2, good_verb, width, label='Good verbs (announced...)', color='green', alpha=0.7)
    axes[0].set_ylabel('Attention weight')
    axes[0].set_title('Attention to Communication Verb')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(heads, rotation=45, ha='right')
    axes[0].legend()
    
    # Plot 2: Attention to basket (wrong answer location)
    bad_basket = [results['comparison'][h]['bad_basket_mean'] for h in heads]
    good_basket = [results['comparison'][h]['good_basket_mean'] for h in heads]
    
    axes[1].bar(x - width/2, bad_basket, width, label='Bad verbs', color='red', alpha=0.7)
    axes[1].bar(x + width/2, good_basket, width, label='Good verbs', color='green', alpha=0.7)
    axes[1].set_ylabel('Attention weight')
    axes[1].set_title('Attention to "basket" (wrong answer)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(heads, rotation=45, ha='right')
    axes[1].legend()
    
    # Plot 3: Difference (bad - good)
    verb_diff = [results['comparison'][h]['verb_diff'] for h in heads]
    basket_diff = [results['comparison'][h]['basket_diff'] for h in heads]
    drawer_diff = [results['comparison'][h]['drawer_diff'] for h in heads]
    
    axes[2].bar(x - width, verb_diff, width, label='Verb diff', color='blue', alpha=0.7)
    axes[2].bar(x, basket_diff, width, label='Basket diff', color='orange', alpha=0.7)
    axes[2].bar(x + width, drawer_diff, width, label='Drawer diff', color='purple', alpha=0.7)
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2].set_ylabel('Difference (Bad - Good)')
    axes[2].set_title('Attention Difference by Token Type')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(heads, rotation=45, ha='right')
    axes[2].legend()
    
    plt.tight_layout()
    save_path = FIGURES_DIR / "late_circuit_attention_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def analyze_what_heads_attend_to(model, tokenizer):
    """Deep analysis: What does each head maximally attend to?"""
    print("\n" + "="*60)
    print("DEEP ANALYSIS: What does each late head attend to?")
    print("="*60)
    
    analysis = {}
    
    for verb_type, verbs in [("BAD", BAD_VERBS), ("GOOD", GOOD_VERBS)]:
        print(f"\n{verb_type} VERBS:")
        
        for verb in verbs:
            prompt = create_prompt(verb)
            attention_patterns, tokens = get_attention_patterns(model, tokenizer, prompt)
            
            print(f"\n  '{verb}':")
            for (layer, head), attn in attention_patterns.items():
                # Last token's attention distribution
                last_attn = attn[-1, :]
                top_5_idx = np.argsort(last_attn)[-5:][::-1]
                top_5_tokens = [(tokens[i], float(last_attn[i])) for i in top_5_idx]
                
                print(f"    L{layer}H{head}: ", end="")
                for tok, weight in top_5_tokens:
                    tok_clean = tok.replace('\u0120', '').replace('\u2581', '')
                    print(f"'{tok_clean}':{weight:.3f} ", end="")
                print()
    
    return analysis


def main():
    print("="*60)
    print("STEP 25: Late Circuit Attention Analysis")
    print("="*60)
    print("\nGoal: Understand HOW the 10 late heads override correct ToM")
    print(f"Heads to analyze: {len(LATE_CIRCUIT_HEADS)}")
    print(f"Bad verbs: {BAD_VERBS}")
    print(f"Good verbs: {GOOD_VERBS}")
    
    # Run main comparison
    results, model, tokenizer = run_verb_comparison()
    
    # Plot comparison
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    plot_comparison_bar_chart(results)
    
    # Plot heatmaps for "told" and "announced"
    plot_attention_heatmaps(model, tokenizer, "told", "late_circuit_told")
    plot_attention_heatmaps(model, tokenizer, "announced", "late_circuit_announced")
    
    # Deep analysis
    analyze_what_heads_attend_to(model, tokenizer)
    
    # Save results
    save_path = RESULTS_DIR / "late_circuit_attention_results.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results: {save_path}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nKey findings from attention analysis:")
    
    # Find heads with largest differences
    verb_diffs = [(h, results['comparison'][h]['verb_diff']) for h in results['comparison']]
    verb_diffs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\nHeads with largest VERB attention difference (Bad - Good):")
    for head, diff in verb_diffs[:5]:
        print(f"  {head}: {diff:+.4f}")
    
    basket_diffs = [(h, results['comparison'][h]['basket_diff']) for h in results['comparison']]
    basket_diffs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\nHeads with largest BASKET attention difference:")
    for head, diff in basket_diffs[:5]:
        print(f"  {head}: {diff:+.4f}")


if __name__ == "__main__":
    main()

