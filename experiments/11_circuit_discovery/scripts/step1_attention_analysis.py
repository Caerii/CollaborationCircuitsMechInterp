"""
Step 1: Attention Pattern Analysis
===================================

Find which attention heads attend across agents (A→B or B→A).
These are candidate "ToM heads".

EFFICIENT: Single forward pass, extract all attention patterns.
"""

import json
import sys
from pathlib import Path

import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STEP 1: ATTENTION PATTERN ANALYSIS")
print("=" * 60)

# Simple test prompts with clear agent structure
TEST_PROMPTS = [
    {
        "text": "Agent A says: 'The answer is 42.' Agent B thinks Agent A is correct.",
        "agent_a_tokens": ["Agent", "A", "says"],
        "agent_b_tokens": ["Agent", "B", "thinks"],
    },
    {
        "text": "Agent A claims the meeting is at 3pm. Agent B disagrees because B heard 4pm.",
        "agent_a_tokens": ["Agent", "A", "claims"],
        "agent_b_tokens": ["Agent", "B", "disagrees"],
    },
    {
        "text": "Alice believes the ball is in the box. Bob knows it's actually in the basket.",
        "agent_a_tokens": ["Alice", "believes"],
        "agent_b_tokens": ["Bob", "knows"],
    },
]


def find_token_positions(tokenizer, text, target_words):
    """Find positions of target words in tokenized text."""
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    token_strs = [tokenizer.decode([t]) for t in tokens]
    
    positions = []
    for word in target_words:
        for i, tok in enumerate(token_strs):
            if word.lower() in tok.lower():
                positions.append(i)
                break
    return positions


def main():
    print("\n[1/4] Loading model...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CFG.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CFG.model_name,
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        output_attentions=True,  # Get attention weights!
    )
    model.eval()
    print("  [OK]", flush=True)
    
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    print(f"  Model has {n_layers} layers x {n_heads} heads = {n_layers * n_heads} total heads")
    
    print("\n[2/4] Analyzing attention patterns...", flush=True)
    
    # Track cross-agent attention for each head
    cross_agent_attention = np.zeros((n_layers, n_heads))
    head_counts = np.zeros((n_layers, n_heads))
    
    for prompt_info in TEST_PROMPTS:
        text = prompt_info["text"]
        print(f"\n  Processing: '{text[:50]}...'", flush=True)
        
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        
        # outputs.attentions is tuple of (batch, heads, seq, seq) per layer
        attentions = outputs.attentions
        
        # Find agent token positions
        a_pos = find_token_positions(tokenizer, text, prompt_info["agent_a_tokens"])
        b_pos = find_token_positions(tokenizer, text, prompt_info["agent_b_tokens"])
        
        if not a_pos or not b_pos:
            print(f"    Could not find agent tokens, skipping")
            continue
        
        print(f"    Agent A tokens at positions: {a_pos}")
        print(f"    Agent B tokens at positions: {b_pos}")
        
        # For each layer and head, measure B→A attention
        for layer_idx, layer_attn in enumerate(attentions):
            attn = layer_attn[0].cpu().numpy()  # (heads, seq, seq)
            
            for head_idx in range(n_heads):
                head_attn = attn[head_idx]  # (seq, seq)
                
                # Measure how much B tokens attend to A tokens
                b_to_a_attn = 0
                for b in b_pos:
                    for a in a_pos:
                        if b > a:  # B can only attend to earlier A (causal)
                            b_to_a_attn += head_attn[b, a]
                
                if b_to_a_attn > 0:
                    cross_agent_attention[layer_idx, head_idx] += b_to_a_attn
                    head_counts[layer_idx, head_idx] += 1
    
    # Average
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_cross_attn = np.divide(cross_agent_attention, head_counts)
        avg_cross_attn = np.nan_to_num(avg_cross_attn, 0)
    
    print("\n[3/4] Finding top cross-agent attention heads...", flush=True)
    
    # Find top heads
    head_scores = []
    for layer in range(n_layers):
        for head in range(n_heads):
            score = avg_cross_attn[layer, head]
            if score > 0:
                head_scores.append({
                    "layer": layer,
                    "head": head,
                    "cross_agent_attn": float(score),
                })
    
    head_scores.sort(key=lambda x: x["cross_agent_attn"], reverse=True)
    
    print("\n  Top 20 Cross-Agent Attention Heads:")
    print("  " + "-" * 40)
    print(f"  {'Layer':<8} {'Head':<8} {'Cross-Attn Score':<15}")
    print("  " + "-" * 40)
    for h in head_scores[:20]:
        print(f"  {h['layer']:<8} {h['head']:<8} {h['cross_agent_attn']:.4f}")
    
    print("\n[4/4] Saving results...", flush=True)
    
    results = {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "top_heads": head_scores[:50],
        "cross_agent_attention_matrix": avg_cross_attn.tolist(),
    }
    
    with open(RESULTS_DIR / "attention_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Which layers have highest cross-agent attention?
    layer_totals = avg_cross_attn.sum(axis=1)
    top_layers = np.argsort(layer_totals)[::-1][:5]
    
    print("\n  Top 5 layers for cross-agent attention:")
    for l in top_layers:
        print(f"    Layer {l}: total score = {layer_totals[l]:.4f}")
    
    print(f"\n[OK] Results saved to {RESULTS_DIR / 'attention_analysis.json'}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()




















