"""
Step 35: Token-by-Token Attention Analysis

Now we know:
- Divergence happens in L25-35
- Action verbs work, belief verbs fail

This script analyzes WHERE the model attends differently
for action vs belief verbs in these critical layers.

Key tokens of interest:
- "drawer" (original location)
- "basket" (new location)  
- "told" (the communication verb)
- The completion verb (searched vs thinks)
"""

import torch
import json
import sys
import io
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"


def load_model():
    print("Loading Qwen3-4B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model.eval()
    return model, tokenizer


def get_attention_patterns(model, tokenizer, prompt, critical_layers=[25, 30, 32, 35]):
    """Get attention patterns from final token to all previous tokens."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    attention_data = {}
    
    for layer_idx in critical_layers:
        layer_attn = outputs.attentions[layer_idx]  # (batch, heads, seq, seq)
        # Get attention from last token, average across heads
        last_token_attn = layer_attn[0, :, -1, :].mean(dim=0).cpu().float().numpy()
        attention_data[f"L{layer_idx}"] = last_token_attn
    
    return tokens, attention_data


def find_key_tokens(tokens, keywords):
    """Find positions of key tokens."""
    positions = {}
    for keyword in keywords:
        for i, token in enumerate(tokens):
            # Handle different tokenization
            clean_token = token.replace("Ġ", "").replace("▁", "").lower()
            if keyword.lower() in clean_token or clean_token in keyword.lower():
                if keyword not in positions:
                    positions[keyword] = []
                positions[keyword].append(i)
    return positions


def analyze_attention_differences():
    """Compare attention patterns between action and belief verbs."""
    model, tokenizer = load_model()
    
    # Define prompts
    action_prompt = """Alice put the ball in the drawer. Alice then left the room.
While Alice was away, Bob told Carol that he moved the ball to the basket.
Alice returns. Alice searched in the"""

    belief_prompt = """Alice put the ball in the drawer. Alice then left the room.
While Alice was away, Bob told Carol that he moved the ball to the basket.
Alice returns. Alice thinks the ball is in the"""

    print("\n" + "="*70)
    print("TOKEN-BY-TOKEN ATTENTION ANALYSIS")
    print("="*70)
    
    # Get attention for action verb
    print("\n--- ACTION VERB: 'searched' ---")
    action_tokens, action_attn = get_attention_patterns(model, tokenizer, action_prompt)
    
    # Get attention for belief verb
    print("--- BELIEF VERB: 'thinks' ---")
    belief_tokens, belief_attn = get_attention_patterns(model, tokenizer, belief_prompt)
    
    # Find key token positions
    keywords = ["drawer", "basket", "told", "moved", "Alice", "Bob"]
    action_positions = find_key_tokens(action_tokens, keywords)
    belief_positions = find_key_tokens(belief_tokens, keywords)
    
    print(f"\nAction prompt token count: {len(action_tokens)}")
    print(f"Belief prompt token count: {len(belief_tokens)}")
    
    print(f"\nKey positions (action): {action_positions}")
    print(f"Key positions (belief): {belief_positions}")
    
    # Compare attention to key tokens
    print("\n" + "="*70)
    print("ATTENTION TO KEY TOKENS (averaged across heads)")
    print("="*70)
    
    results = {"action": {}, "belief": {}, "comparison": {}}
    
    for layer in ["L25", "L30", "L32", "L35"]:
        print(f"\n--- {layer} ---")
        
        results["action"][layer] = {}
        results["belief"][layer] = {}
        
        for keyword in keywords:
            action_pos = action_positions.get(keyword, [])
            belief_pos = belief_positions.get(keyword, [])
            
            if action_pos:
                action_attn_sum = sum(action_attn[layer][p] for p in action_pos)
            else:
                action_attn_sum = 0
            
            if belief_pos:
                belief_attn_sum = sum(belief_attn[layer][p] for p in belief_pos)
            else:
                belief_attn_sum = 0
            
            diff = action_attn_sum - belief_attn_sum
            
            results["action"][layer][keyword] = float(action_attn_sum)
            results["belief"][layer][keyword] = float(belief_attn_sum)
            
            # Print comparison
            sign = "+" if diff > 0 else ""
            print(f"  {keyword:10s}: action={action_attn_sum:.4f}, belief={belief_attn_sum:.4f}, diff={sign}{diff:.4f}")
    
    # Find the completion verb positions
    print("\n" + "="*70)
    print("ATTENTION TO COMPLETION VERB")
    print("="*70)
    
    # Find "searched" in action tokens
    searched_pos = None
    for i, t in enumerate(action_tokens):
        if "searched" in t.lower() or "search" in t.lower():
            searched_pos = i
            break
    
    # Find "thinks" in belief tokens
    thinks_pos = None
    for i, t in enumerate(belief_tokens):
        if "thinks" in t.lower() or "think" in t.lower():
            thinks_pos = i
            break
    
    print(f"\n'searched' position: {searched_pos}")
    print(f"'thinks' position: {thinks_pos}")
    
    if searched_pos and thinks_pos:
        for layer in ["L25", "L30", "L32", "L35"]:
            action_to_verb = action_attn[layer][searched_pos]
            belief_to_verb = belief_attn[layer][thinks_pos]
            print(f"\n{layer}: attention to completion verb")
            print(f"  'searched': {action_to_verb:.4f}")
            print(f"  'thinks':   {belief_to_verb:.4f}")
    
    # Visualize
    plot_attention_comparison(action_tokens, action_attn, belief_tokens, belief_attn,
                              action_positions, belief_positions)
    
    # Save
    save_path = RESULTS_DIR / "token_attention_results.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {save_path}")
    
    return results


def plot_attention_comparison(action_tokens, action_attn, belief_tokens, belief_attn,
                              action_positions, belief_positions):
    """Plot attention pattern comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    layers = ["L25", "L30", "L32", "L35"]
    
    for idx, layer in enumerate(layers):
        ax = axes[idx // 2, idx % 2]
        
        # Plot attention patterns
        x_action = range(len(action_attn[layer]))
        x_belief = range(len(belief_attn[layer]))
        
        ax.plot(x_action, action_attn[layer], label="Action (searched)", 
                color="blue", alpha=0.7, linewidth=1)
        ax.plot(x_belief, belief_attn[layer], label="Belief (thinks)", 
                color="red", alpha=0.7, linewidth=1)
        
        # Mark key positions
        for keyword, color in [("drawer", "green"), ("basket", "orange"), ("told", "purple")]:
            if keyword in action_positions:
                for pos in action_positions[keyword]:
                    ax.axvline(x=pos, color=color, linestyle='--', alpha=0.5, 
                              label=f"{keyword}" if pos == action_positions[keyword][0] else "")
        
        ax.set_title(f"{layer}: Attention from Final Token")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Attention Weight")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = FIGURES_DIR / "token_attention_comparison.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved: {save_path}")
    plt.close()


def main():
    print("="*70)
    print("STEP 35: Token-by-Token Attention Analysis")
    print("="*70)
    
    results = analyze_attention_differences()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
This analysis shows WHERE the model looks when processing
action vs belief verbs, revealing the attention mechanism
behind ToM success/failure.
""")


if __name__ == "__main__":
    main()


