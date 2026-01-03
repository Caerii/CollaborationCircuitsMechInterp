"""
Step 25: Attention Head-Level Steering

Step 19 showed that residual stream steering doesn't flip predictions.
Let's try steering at the attention HEAD level instead:

1. Identify which heads write to the ToM-relevant dimensions
2. Amplify/suppress specific heads to flip predictions
3. Compare to residual stream steering

OUTPUT: results/step25_head_steering.json, figures/step25_*.png
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
from analysis.signal_injection import HeadAmplifier

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def get_prediction_details(model, tokenizer, prompt, token1, token2):
    """Get detailed prediction info."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    id1 = tokenizer.encode(token1, add_special_tokens=False)[0]
    id2 = tokenizer.encode(token2, add_special_tokens=False)[0]
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    
    probs = F.softmax(logits, dim=-1)
    
    return {
        "logit_diff": float(logits[id1] - logits[id2]),
        "prob1": float(probs[id1]),
        "prob2": float(probs[id2]),
        "prediction": token1 if logits[id1] > logits[id2] else token2,
    }


def main():
    print("=" * 70)
    print("STEP 25: ATTENTION HEAD-LEVEL STEERING")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\nTrying to flip predictions by amplifying/suppressing specific heads")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "FB_alice_drawer",
            "prompt": "Alice put the ball in the drawer. Alice left. Bob moved it to the basket. Where does Alice think? Alice looks in the",
            "target": " drawer",  # FB correct
            "wrong": " basket",
            "goal": "Keep correct (drawer)",
        },
        {
            "name": "TB_alice_basket",
            "prompt": "Alice put the ball in the drawer. Alice watched Bob move it to the basket. Where does Alice think? Alice looks in the",
            "target": " basket",  # TB correct
            "wrong": " drawer",
            "goal": "Make correct (basket)",
        },
    ]
    
    # Critical heads from previous steps
    tom_heads = [(32, 0), (33, 4), (33, 16), (33, 28), (34, 0)]  # ToM enablers
    inhibitor = (18, 16)  # Multi-agent inhibitor
    
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
    
    amplifier = HeadAmplifier(model, tokenizer)
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"Goal: {scenario['goal']}")
        print(f"{'='*60}")
        sys.stdout.flush()
        
        scenario_result = {
            "name": scenario["name"],
            "goal": scenario["goal"],
            "experiments": [],
        }
        
        # Baseline
        baseline = get_prediction_details(model, tokenizer, scenario["prompt"], 
                                         scenario["target"], scenario["wrong"])
        print(f"\nBaseline: {baseline['prediction']} (diff={baseline['logit_diff']:.2f})")
        scenario_result["baseline"] = baseline
        
        # ========================================
        # EXPERIMENT 1: Amplify ToM heads
        # ========================================
        print("\n--- Amplifying ToM heads (L32-34) ---")
        for scale in [1.5, 2.0, 3.0]:
            try:
                result = amplifier.test_with_amplification(
                    prompt=scenario["prompt"],
                    heads=tom_heads,
                    scales=[scale],
                    target_token=scenario["target"],
                    contrast_token=scenario["wrong"],
                )[scale]
                
                flipped = (result["logit_diff"] > 0) != (baseline["logit_diff"] > 0)
                change = result["logit_diff"] - baseline["logit_diff"]
                
                exp_result = {
                    "type": "amplify_tom",
                    "scale": scale,
                    "logit_diff": result["logit_diff"],
                    "change": change,
                    "flipped": flipped,
                }
                scenario_result["experiments"].append(exp_result)
                
                status = "FLIP!" if flipped else "same"
                print(f"  {scale}x: diff={result['logit_diff']:.2f} (change={change:+.2f}, {status})")
            except Exception as e:
                print(f"  {scale}x: ERROR - {e}")
            sys.stdout.flush()
        
        # ========================================
        # EXPERIMENT 2: Suppress ToM heads
        # ========================================
        print("\n--- Suppressing ToM heads (scale < 1) ---")
        for scale in [0.5, 0.25, 0.0]:
            try:
                result = amplifier.test_with_amplification(
                    prompt=scenario["prompt"],
                    heads=tom_heads,
                    scales=[scale],
                    target_token=scenario["target"],
                    contrast_token=scenario["wrong"],
                )[scale]
                
                flipped = (result["logit_diff"] > 0) != (baseline["logit_diff"] > 0)
                change = result["logit_diff"] - baseline["logit_diff"]
                
                exp_result = {
                    "type": "suppress_tom",
                    "scale": scale,
                    "logit_diff": result["logit_diff"],
                    "change": change,
                    "flipped": flipped,
                }
                scenario_result["experiments"].append(exp_result)
                
                status = "FLIP!" if flipped else "same"
                print(f"  {scale}x: diff={result['logit_diff']:.2f} (change={change:+.2f}, {status})")
            except Exception as e:
                print(f"  {scale}x: ERROR - {e}")
            sys.stdout.flush()
        
        # ========================================
        # EXPERIMENT 3: Suppress inhibitor (L18H16)
        # ========================================
        print("\n--- Suppressing inhibitor (L18H16) ---")
        for scale in [0.5, 0.0]:
            try:
                result = amplifier.test_with_amplification(
                    prompt=scenario["prompt"],
                    heads=[inhibitor],
                    scales=[scale],
                    target_token=scenario["target"],
                    contrast_token=scenario["wrong"],
                )[scale]
                
                flipped = (result["logit_diff"] > 0) != (baseline["logit_diff"] > 0)
                change = result["logit_diff"] - baseline["logit_diff"]
                
                exp_result = {
                    "type": "suppress_inhibitor",
                    "scale": scale,
                    "logit_diff": result["logit_diff"],
                    "change": change,
                    "flipped": flipped,
                }
                scenario_result["experiments"].append(exp_result)
                
                status = "FLIP!" if flipped else "same"
                print(f"  {scale}x: diff={result['logit_diff']:.2f} (change={change:+.2f}, {status})")
            except Exception as e:
                print(f"  {scale}x: ERROR - {e}")
            sys.stdout.flush()
        
        results.append(scenario_result)
    
    # ========================================
    # SUMMARY
    # ========================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    total_flips = 0
    total_experiments = 0
    for r in results:
        for exp in r["experiments"]:
            total_experiments += 1
            if exp.get("flipped"):
                total_flips += 1
                print(f"FLIP: {r['name']} with {exp['type']} at {exp['scale']}x")
    
    print(f"\nTotal flips: {total_flips}/{total_experiments}")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {"model": config.model_name},
        "tom_heads": [{"layer": l, "head": h} for l, h in tom_heads],
        "inhibitor": {"layer": inhibitor[0], "head": inhibitor[1]},
        "results": results,
        "summary": {
            "total_flips": total_flips,
            "total_experiments": total_experiments,
        },
    }
    
    output_path = RESULTS_DIR / "step25_head_steering.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # Generate figure
    print("\nGenerating figure...")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, scenario_result in zip(axes, results):
        name = scenario_result["name"]
        baseline_diff = scenario_result["baseline"]["logit_diff"]
        
        # Extract data for plotting
        tom_amplify = [(e["scale"], e["logit_diff"]) for e in scenario_result["experiments"] 
                       if e["type"] == "amplify_tom"]
        tom_suppress = [(e["scale"], e["logit_diff"]) for e in scenario_result["experiments"] 
                        if e["type"] == "suppress_tom"]
        
        all_scales = [1.0]  # Baseline
        all_diffs = [baseline_diff]
        
        for scale, diff in tom_amplify + tom_suppress:
            all_scales.append(scale)
            all_diffs.append(diff)
        
        # Sort by scale
        sorted_data = sorted(zip(all_scales, all_diffs))
        scales = [s for s, _ in sorted_data]
        diffs = [d for _, d in sorted_data]
        
        colors = ['seagreen' if d > 0 else 'coral' for d in diffs]
        ax.bar([str(s) for s in scales], diffs, color=colors, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel("Scale Factor", fontsize=12)
        ax.set_ylabel("Logit Diff (target - wrong)", fontsize=12)
        ax.set_title(f"{name}\n(baseline={baseline_diff:.2f})", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step25_head_steering.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("STEP 25 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

