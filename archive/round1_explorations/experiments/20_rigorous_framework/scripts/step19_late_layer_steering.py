"""
Step 19: Late-Layer Steering at L28 (Peak Discriminability)

Step 14 failed because L12 is too early!
Step 15 found L28 has PEAK discriminability (10.94).

Let's try steering at L28 where the decision actually forms!

OUTPUT: results/step19_late_steering.json, figures/step19_*.png
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
from analysis.sae_analysis import SimpleSAE, SAEConfig, SAETrainer

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def get_prediction(model, tokenizer, prompt, target_token, contrast_token):
    """Get model prediction."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    target_ids = tokenizer.encode(target_token, add_special_tokens=False)
    contrast_ids = tokenizer.encode(contrast_token, add_special_tokens=False)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    
    target_logit = float(logits[target_ids[0]])
    contrast_logit = float(logits[contrast_ids[0]])
    
    return {
        "prediction": "target" if target_logit > contrast_logit else "contrast",
        "target_logit": target_logit,
        "contrast_logit": contrast_logit,
        "logit_diff": target_logit - contrast_logit,
    }


def collect_layer_activations(model, tokenizer, prompt, layer):
    """Collect layer output activation."""
    activation = []
    
    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        if hidden.dim() == 3:
            activation.append(hidden[0, -1, :].detach().cpu())
        else:
            activation.append(hidden[-1, :].detach().cpu())
    
    layer_module = model.model.layers[layer]
    handle = layer_module.register_forward_hook(hook)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs)
    
    handle.remove()
    return activation[0]


def run_with_modified_activation(model, tokenizer, prompt, layer, modified_activation, target_token, contrast_token):
    """Run model with modified layer activation."""
    
    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            new_hidden = hidden.clone()
            new_hidden[0, -1, :] = modified_activation.to(hidden.device).to(hidden.dtype)
            return (new_hidden,) + output[1:]
        else:
            new_output = output.clone()
            new_output[0, -1, :] = modified_activation.to(output.device).to(output.dtype)
            return new_output
    
    layer_module = model.model.layers[layer]
    handle = layer_module.register_forward_hook(hook)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    target_ids = tokenizer.encode(target_token, add_special_tokens=False)
    contrast_ids = tokenizer.encode(contrast_token, add_special_tokens=False)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    
    handle.remove()
    
    target_logit = float(logits[target_ids[0]])
    contrast_logit = float(logits[contrast_ids[0]])
    
    return {
        "prediction": "target" if target_logit > contrast_logit else "contrast",
        "logit_diff": target_logit - contrast_logit,
    }


def main():
    print("=" * 70)
    print("STEP 19: LATE-LAYER STEERING (L28)")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\nStep 14 failed at L12. Step 15 found L28 is the PEAK!")
    print("Let's try steering where the decision actually forms.")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    target_layer = 28  # PEAK discriminability!
    
    # Training scenarios
    fb_prompts = [
        "Alice put the ball in the drawer. Alice left. Bob moved it to the basket. Where does Alice think the ball is?",
        "Tom hid the key in the box. Tom left. Jerry moved the key to the drawer. Where does Tom think the key is?",
        "Chef put ingredients in cabinet A. Chef left. Waiter moved them to cabinet B. Where does Chef think they are?",
        "Sally put the toy in the basket. Sally went outside. Anne moved it to the box. Where does Sally think it is?",
        "Dad put cookies in the jar. Dad left. Mom moved them to the cupboard. Where does Dad think they are?",
        "Teacher put chalk in drawer 1. Teacher left. Student moved it to drawer 2. Where does Teacher think it is?",
    ]
    
    tb_prompts = [
        "Alice put the ball in the drawer. Alice stayed and watched. Bob moved it to the basket. Where does Alice think the ball is?",
        "Tom hid the key in the box. Tom watched Jerry move the key to the drawer. Where does Tom think the key is?",
        "Chef put ingredients in cabinet A. Chef saw Waiter move them to cabinet B. Where does Chef think they are?",
        "Sally put the toy in the basket. Sally watched Anne move it to the box. Where does Sally think it is?",
        "Dad put cookies in the jar. Dad watched Mom move them to the cupboard. Where does Dad think they are?",
        "Teacher put chalk in drawer 1. Teacher saw Student move it to drawer 2. Where does Teacher think it is?",
    ]
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "flip_fb_to_reality",
            "prompt": "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket. Where does Alice think the ball is? Alice thinks it's in the",
            "target": " drawer",
            "contrast": " basket",
            "goal": "Make model predict basket (as if Alice knew)",
        },
        {
            "name": "flip_tb_to_wrong",
            "prompt": "Alice put the ball in the drawer. Alice stayed and watched. Bob moved it to the basket. Where does Alice think the ball is? Alice thinks it's in the",
            "target": " basket",
            "contrast": " drawer",
            "goal": "Make model predict drawer (as if Alice didn't see)",
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
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded!")
    sys.stdout.flush()
    
    d_model = model.config.hidden_size
    
    # ========================================
    # COLLECT ACTIVATIONS AND COMPUTE DIRECTIONS
    # ========================================
    print(f"\n{'='*60}")
    print(f"COLLECTING L{target_layer} ACTIVATIONS")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    fb_acts = []
    for prompt in fb_prompts:
        act = collect_layer_activations(model, tokenizer, prompt, target_layer)
        fb_acts.append(act)
        print(".", end="")
        sys.stdout.flush()
    fb_acts = torch.stack(fb_acts).float()
    print(" FB done!")
    
    tb_acts = []
    for prompt in tb_prompts:
        act = collect_layer_activations(model, tokenizer, prompt, target_layer)
        tb_acts.append(act)
        print(".", end="")
        sys.stdout.flush()
    tb_acts = torch.stack(tb_acts).float()
    print(" TB done!")
    
    # Compute steering direction
    fb_mean = fb_acts.mean(dim=0)
    tb_mean = tb_acts.mean(dim=0)
    fb_to_tb = tb_mean - fb_mean
    tb_to_fb = fb_mean - tb_mean
    
    print(f"\nSteering vector norm: {fb_to_tb.norm():.2f}")
    sys.stdout.flush()
    
    # ========================================
    # STEERING EXPERIMENTS
    # ========================================
    print(f"\n{'='*60}")
    print("STEERING EXPERIMENTS")
    print(f"{'='*60}")
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"Goal: {scenario['goal']}")
        sys.stdout.flush()
        
        # Baseline
        baseline = get_prediction(model, tokenizer, scenario["prompt"], scenario["target"], scenario["contrast"])
        print(f"Baseline: {baseline['prediction']} (diff={baseline['logit_diff']:.2f})")
        
        # Get current activation
        current_act = collect_layer_activations(model, tokenizer, scenario["prompt"], target_layer).float()
        
        scenario_results = {
            "name": scenario["name"],
            "goal": scenario["goal"],
            "baseline": baseline,
            "steering_results": [],
        }
        
        # Try different steering strengths
        for strength in [0.5, 1.0, 2.0, 3.0]:
            if "fb_to_reality" in scenario["name"]:
                # Want to flip FB->TB (make Alice "know")
                steering_vec = fb_to_tb * strength
            else:
                # Want to flip TB->FB (make Alice "not know")
                steering_vec = tb_to_fb * strength
            
            modified_act = current_act + steering_vec
            
            result = run_with_modified_activation(
                model, tokenizer, scenario["prompt"], target_layer,
                modified_act, scenario["target"], scenario["contrast"]
            )
            
            flipped = baseline["prediction"] != result["prediction"]
            diff_change = result["logit_diff"] - baseline["logit_diff"]
            
            print(f"  Strength {strength:.1f}x: {result['prediction']} (diff={result['logit_diff']:.2f}, change={diff_change:+.2f}, {'FLIP!' if flipped else 'same'})")
            
            scenario_results["steering_results"].append({
                "strength": strength,
                "result": result,
                "flipped": flipped,
                "diff_change": diff_change,
            })
        
        results.append(scenario_results)
        sys.stdout.flush()
    
    # ========================================
    # SUMMARY
    # ========================================
    print(f"\n{'='*60}")
    print("STEERING SUMMARY")
    print(f"{'='*60}")
    
    total_flips = 0
    total_attempts = 0
    for r in results:
        for sr in r["steering_results"]:
            total_attempts += 1
            if sr["flipped"]:
                total_flips += 1
                print(f"FLIP at {sr['strength']}x: {r['name']}")
    
    print(f"\nTotal flips: {total_flips}/{total_attempts} ({total_flips/total_attempts:.1%})")
    
    # Compare to L12
    print(f"\nComparison to L12 steering (Step 14): 0% flips")
    print(f"L28 steering: {total_flips/total_attempts:.1%} flips")
    
    if total_flips > 0:
        print("\n*** SUCCESS: Late-layer steering works! ***")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {"model": config.model_name, "target_layer": target_layer},
        "steering_vector_norm": float(fb_to_tb.norm()),
        "results": results,
        "summary": {
            "total_flips": total_flips,
            "total_attempts": total_attempts,
            "flip_rate": total_flips / total_attempts if total_attempts > 0 else 0,
            "comparison_to_l12": "0% flips at L12 vs this",
        },
    }
    
    output_path = RESULTS_DIR / "step19_late_steering.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # Generate figure
    print("\nGenerating figure...")
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = []
    heights = []
    colors = []
    labels = []
    
    for i, r in enumerate(results):
        base_diff = r["baseline"]["logit_diff"]
        x.append(i * 5)
        heights.append(base_diff)
        colors.append("steelblue")
        labels.append(f"{r['name'][:15]}\nBaseline")
        
        for j, sr in enumerate(r["steering_results"]):
            x.append(i * 5 + j + 1)
            heights.append(sr["result"]["logit_diff"])
            colors.append("coral" if sr["flipped"] else "lightgray")
            labels.append(f"{sr['strength']}x")
    
    ax.bar(x, heights, color=colors, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel("Logit Difference", fontsize=12)
    ax.set_title(f"L{target_layer} Steering (Peak Discriminability Layer)", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    
    from matplotlib.patches import Patch
    legend = [Patch(facecolor='steelblue', label='Baseline'),
              Patch(facecolor='coral', label='FLIPPED'),
              Patch(facecolor='lightgray', label='No flip')]
    ax.legend(handles=legend, loc='upper right')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step19_late_steering.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("STEP 19 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

