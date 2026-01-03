"""
Attention Analysis: What Circuit Does the "Bridge" Activate?
=============================================================

We know that adding phrases like "X now knows" or "Therefore X believes"
transforms 18% -> 92-100% accuracy.

What's happening mechanistically?
- Which attention heads attend differently with vs without the bridge?
- Does adding the bridge cause attention to "communication" tokens?
- Or does it route differently to the "belief" representation?

This will tell us WHERE the weak connection is in the circuit.
"""

import json
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import random

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_attention_patterns(model, tokenizer, prompt: str) -> dict:
    """
    Get attention patterns for all heads at all layers.
    Focus on attention TO key tokens: agent name, locations, communicative verb.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # outputs.attentions is tuple of (batch, heads, seq, seq) per layer
    attention_data = {
        "tokens": tokens,
        "n_layers": len(outputs.attentions),
        "n_heads": outputs.attentions[0].shape[1],
    }
    
    # Get attention from last token to all others (this is what matters for prediction)
    last_token_attention = []
    for layer_idx, layer_attn in enumerate(outputs.attentions):
        # layer_attn: (batch, heads, seq, seq)
        # Get attention FROM last token TO all tokens
        last_to_all = layer_attn[0, :, -1, :].cpu().numpy()  # (heads, seq)
        last_token_attention.append(last_to_all)
    
    attention_data["last_token_attention"] = last_token_attention  # List of (heads, seq)
    
    return attention_data


def find_token_indices(tokens: list, keywords: list) -> dict:
    """Find indices of important tokens."""
    indices = {}
    tokens_lower = [t.lower().replace("_", "").replace("##", "") for t in tokens]
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        for i, token in enumerate(tokens_lower):
            if keyword_lower in token or token in keyword_lower:
                if keyword not in indices:
                    indices[keyword] = []
                indices[keyword].append(i)
    
    return indices


def compare_attention_patterns(attn_baseline: dict, attn_bridged: dict, agent: str, loc1: str, loc2: str) -> dict:
    """Compare attention patterns between baseline and bridged prompts."""
    
    # Find key token positions
    keywords = [agent.lower(), loc1.lower(), loc2.lower(), "tells", "says", "moved"]
    
    baseline_indices = find_token_indices(attn_baseline["tokens"], keywords)
    bridged_indices = find_token_indices(attn_bridged["tokens"], keywords)
    
    comparison = {
        "baseline_tokens": attn_baseline["tokens"],
        "bridged_tokens": attn_bridged["tokens"],
        "keyword_indices_baseline": baseline_indices,
        "keyword_indices_bridged": bridged_indices,
        "attention_differences": [],
    }
    
    n_layers = attn_baseline["n_layers"]
    n_heads = attn_baseline["n_heads"]
    
    # For each layer and head, compute attention to key tokens
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            baseline_attn = attn_baseline["last_token_attention"][layer_idx][head_idx]
            bridged_attn = attn_bridged["last_token_attention"][layer_idx][head_idx]
            
            # Attention to agent name (in baseline)
            agent_indices = baseline_indices.get(agent.lower(), [])
            if agent_indices:
                baseline_agent_attn = np.mean([baseline_attn[i] for i in agent_indices if i < len(baseline_attn)])
            else:
                baseline_agent_attn = 0.0
            
            # Attention to loc2 (the "correct" updated location)
            loc2_indices_base = baseline_indices.get(loc2.lower(), [])
            loc2_indices_bridge = bridged_indices.get(loc2.lower(), [])
            
            if loc2_indices_base:
                baseline_loc2_attn = np.mean([baseline_attn[i] for i in loc2_indices_base if i < len(baseline_attn)])
            else:
                baseline_loc2_attn = 0.0
                
            if loc2_indices_bridge:
                bridged_loc2_attn = np.mean([bridged_attn[i] for i in loc2_indices_bridge if i < len(bridged_attn)])
            else:
                bridged_loc2_attn = 0.0
            
            comparison["attention_differences"].append({
                "layer": layer_idx,
                "head": head_idx,
                "baseline_loc2_attn": float(baseline_loc2_attn),
                "bridged_loc2_attn": float(bridged_loc2_attn),
                "attn_change": float(bridged_loc2_attn - baseline_loc2_attn),
            })
    
    return comparison


def main():
    timer_start = time.perf_counter()
    
    print("=" * 70)
    print("ATTENTION ANALYSIS: What Circuit Does the Bridge Activate?")
    print("=" * 70)
    print("""
    Comparing attention patterns between:
    - Baseline (18% accuracy): "Eve tells Alice: 'moved to basket.' Where looks?"
    - Bridged (98% accuracy):  "...so Alice updated her belief. Where looks?"
    
    Looking for: Which heads attend more to the correct location (loc2) 
    when the bridge is present?
    """)
    
    # Load model
    print("[1/4] Loading model...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CFG.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CFG.model_name,
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        output_attentions=True,
    )
    model.eval()
    
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    print(f"  [OK] Model loaded: {n_layers} layers, {n_heads} heads", flush=True)
    
    # Generate test cases
    print("\n[2/4] Generating test cases...", flush=True)
    random.seed(42)
    
    test_cases = []
    for i in range(10):  # 10 cases for attention analysis
        agent = random.choice(["Alice", "Bob", "Carol"])
        informer = random.choice(["Eve", "Frank", "Grace"])
        obj = random.choice(["ball", "key", "book"])
        loc1, loc2 = random.sample(["drawer", "basket", "cupboard", "shelf"], 2)
        
        baseline = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2}.' "
            f"Where will {agent} look for the {obj}? {agent} will look in the"
        )
        
        bridged = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2},' "
            f"so {agent} updated their belief. "
            f"Where will {agent} look for the {obj}? {agent} will look in the"
        )
        
        test_cases.append({
            "agent": agent,
            "loc1": loc1,
            "loc2": loc2,
            "baseline": baseline,
            "bridged": bridged,
        })
    
    print(f"  Generated {len(test_cases)} test cases")
    
    # Collect attention patterns
    print("\n[3/4] Collecting attention patterns...", flush=True)
    
    all_comparisons = []
    head_attention_changes = {}  # (layer, head) -> list of changes
    
    for i, case in enumerate(test_cases):
        print(f"  [{i+1}/{len(test_cases)}] Processing...", flush=True)
        
        attn_baseline = get_attention_patterns(model, tokenizer, case["baseline"])
        attn_bridged = get_attention_patterns(model, tokenizer, case["bridged"])
        
        comparison = compare_attention_patterns(
            attn_baseline, attn_bridged,
            case["agent"], case["loc1"], case["loc2"]
        )
        all_comparisons.append(comparison)
        
        # Aggregate changes per head
        for diff in comparison["attention_differences"]:
            key = (diff["layer"], diff["head"])
            if key not in head_attention_changes:
                head_attention_changes[key] = []
            head_attention_changes[key].append(diff["attn_change"])
    
    # Analyze which heads change most
    print("\n[4/4] Analyzing results...\n", flush=True)
    
    # Compute mean change per head
    head_mean_changes = []
    for (layer, head), changes in head_attention_changes.items():
        mean_change = np.mean(changes)
        head_mean_changes.append((layer, head, mean_change))
    
    # Sort by absolute change (heads that change most)
    head_mean_changes.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print("=" * 70)
    print("TOP HEADS: Attention Change When Bridge Added")
    print("=" * 70)
    print("\n  (Positive = more attention to correct location with bridge)")
    print()
    
    print("  LAYER  HEAD   MEAN CHANGE")
    print("  " + "-" * 35)
    for layer, head, change in head_mean_changes[:20]:
        bar = "+" * int(change * 200) if change > 0 else "-" * int(abs(change) * 200)
        print(f"  L{layer:2d}    H{head:2d}    {change:+.4f}  {bar[:30]}")
    
    # Focus on our known ToM heads (L12H0, L23H0)
    print("\n" + "=" * 70)
    print("OUR KNOWN ToM HEADS (from ablation study)")
    print("=" * 70)
    
    tom_heads = [(12, 0), (23, 0)]
    for layer, head in tom_heads:
        if (layer, head) in head_attention_changes:
            changes = head_attention_changes[(layer, head)]
            mean_change = np.mean(changes)
            print(f"  L{layer}H{head}: mean attention change = {mean_change:+.4f}")
    
    # Find heads with biggest positive change (attend MORE to correct loc with bridge)
    positive_changes = [(l, h, c) for l, h, c in head_mean_changes if c > 0.01]
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if positive_changes:
        print(f"\n  Found {len(positive_changes)} heads that attend MORE to correct location with bridge:")
        for l, h, c in positive_changes[:5]:
            print(f"    L{l}H{h}: +{c:.4f}")
        
        print("""
    These heads are the "belief update bridge" circuit!
    When we add "so X updated their belief", these heads
    start attending more to the correct (updated) location.
        """)
    else:
        print("\n  No heads showed significant positive attention change.")
        print("  The bridge might work through a different mechanism.")
    
    # Save results
    output = {
        "head_mean_changes": [(l, h, float(c)) for l, h, c in head_mean_changes[:50]],
        "tom_heads_changes": {
            f"L{l}H{h}": float(np.mean(head_attention_changes.get((l, h), [0])))
            for l, h in tom_heads
        },
        "n_test_cases": len(test_cases),
    }
    
    with open(RESULTS_DIR / "attention_analysis_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    total_time = time.perf_counter() - timer_start
    print(f"\n[OK] Completed in {total_time:.1f}s")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


