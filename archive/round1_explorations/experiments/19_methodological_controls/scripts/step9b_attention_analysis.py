"""
ATTENTION PATTERN ANALYSIS: What do inhibitors vs enablers look at?
===================================================================

Fixed version that uses eager attention (not SDPA) to get attention weights.
"""

import json
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    print("=" * 70)
    print("ATTENTION PATTERN ANALYSIS")
    print("=" * 70)
    
    # Load fixed scenarios
    scenarios_file = RESULTS_DIR / "fixed_scenarios.json"
    with open(scenarios_file) as f:
        scenarios = json.load(f)
    print(f"Loaded {len(scenarios)} fixed scenarios")
    
    # Load model with EAGER attention
    print("\nLoading model with eager attention...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CFG.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CFG.model_name,
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",  # Force eager attention for attention weights
    )
    model.eval()
    print("  [OK]", flush=True)
    
    # Heads to analyze
    TOP_INHIBITORS = [(18, 11), (17, 4), (18, 14)]
    CRITICAL_ENABLERS = [(15, 9), (19, 2), (19, 15)]
    
    results = {
        "inhibitors": {},
        "enablers": {},
    }
    
    n_examples = 15  # Analyze first 15 scenarios
    
    print(f"\nAnalyzing {n_examples} scenarios...")
    print("-" * 50)
    
    for label, heads in [("INHIBITORS", TOP_INHIBITORS), ("ENABLERS", CRITICAL_ENABLERS)]:
        print(f"\n{label}:")
        
        for layer, head in heads:
            key = f"L{layer}H{head}"
            
            loc1_attns = []
            loc2_attns = []
            agent_attns = []
            verb_attns = []
            
            for scenario in scenarios[:n_examples]:
                inputs = tokenizer(scenario["prompt"], return_tensors="pt").to("cuda")
                
                with torch.no_grad():
                    outputs = model(**inputs, output_attentions=True)
                
                # Get attention from this layer
                # Shape: (batch, n_heads, seq, seq)
                attn = outputs.attentions[layer]
                n_heads_in_layer = attn.shape[1]
                
                # Handle if head index exceeds available heads (GQA)
                actual_head = min(head, n_heads_in_layer - 1)
                attn_pattern = attn[0, actual_head].cpu().numpy()  # (seq, seq)
                
                # Get tokens
                tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
                tokens_str = [str(t).lower().replace("##", "").replace("_", "").replace("Ġ", "") for t in tokens]
                
                # Last token's attention (prediction position)
                last_attn = attn_pattern[-1]
                
                # Find token indices for key elements
                loc1 = scenario["loc1"].lower()
                loc2 = scenario["loc2"].lower()
                agent = scenario["agent"].lower()
                
                loc1_idx = [i for i, t in enumerate(tokens_str) if loc1 in t]
                loc2_idx = [i for i, t in enumerate(tokens_str) if loc2 in t]
                agent_idx = [i for i, t in enumerate(tokens_str) if agent in t]
                
                # Sum attention to each type
                loc1_attn = sum(last_attn[i] for i in loc1_idx) if loc1_idx else 0
                loc2_attn = sum(last_attn[i] for i in loc2_idx) if loc2_idx else 0
                agent_attn = sum(last_attn[i] for i in agent_idx) if agent_idx else 0
                
                loc1_attns.append(float(loc1_attn))
                loc2_attns.append(float(loc2_attn))
                agent_attns.append(float(agent_attn))
            
            # Compute averages
            avg_loc1 = np.mean(loc1_attns)
            avg_loc2 = np.mean(loc2_attns)
            avg_agent = np.mean(agent_attns)
            
            ratio = avg_loc1 / (avg_loc2 + 1e-6)
            
            results[label.lower()][key] = {
                "avg_loc1_attention": float(avg_loc1),
                "avg_loc2_attention": float(avg_loc2),
                "avg_agent_attention": float(avg_agent),
                "ratio_loc1_loc2": float(ratio),
            }
            
            # Interpretation
            interp = ""
            if ratio > 2.0:
                interp = "STRONGLY anchors to original location"
            elif ratio > 1.3:
                interp = "Prefers original location"
            elif ratio < 0.5:
                interp = "FOCUSES on new location"
            elif ratio < 0.8:
                interp = "Prefers new location"
            else:
                interp = "Balanced attention"
            
            print(f"  {key}:")
            print(f"    Attention to original (loc1): {avg_loc1:.4f}")
            print(f"    Attention to new (loc2):      {avg_loc2:.4f}")
            print(f"    Attention to agent:           {avg_agent:.4f}")
            print(f"    Ratio loc1/loc2:              {ratio:.2f}")
            print(f"    --> {interp}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: What Explains the Inhibition?")
    print("=" * 70)
    
    print("\nHYPOTHESIS TEST:")
    print("  If inhibitors ANCHOR to original location, ratio > 1.5")
    print("  If enablers FOCUS on new location, ratio < 0.8")
    print()
    
    avg_inhibitor_ratio = np.mean([v["ratio_loc1_loc2"] for v in results["inhibitors"].values()])
    avg_enabler_ratio = np.mean([v["ratio_loc1_loc2"] for v in results["enablers"].values()])
    
    print(f"  Average INHIBITOR ratio: {avg_inhibitor_ratio:.2f}")
    print(f"  Average ENABLER ratio:   {avg_enabler_ratio:.2f}")
    
    if avg_inhibitor_ratio > avg_enabler_ratio * 1.5:
        print("\n  CONFIRMED: Inhibitors attend MORE to original location")
        print("  This explains their suppressive effect!")
    elif avg_inhibitor_ratio < avg_enabler_ratio * 0.7:
        print("\n  SURPRISING: Inhibitors attend LESS to original location")
        print("  Inhibition mechanism is NOT simple anchoring")
    else:
        print("\n  INCONCLUSIVE: Similar attention patterns")
        print("  Inhibition works through a different mechanism")
    
    # Save
    output_file = RESULTS_DIR / "attention_analysis_detailed.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Saved to {output_file}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

