"""
Step 14: Feature Steering - Can We Flip Predictions Using SAE Features?

THE BIG QUESTION: If feature #1979 encodes "outdated belief",
can we inject/suppress it to flip model predictions?

METHOD:
1. Train SAE on MLP activations (same as Step 13)
2. Identify "belief update" features
3. Patch activations with modified features
4. See if predictions flip!

This is CAUSAL EVIDENCE that features are meaningful.

OUTPUT: results/step14_steering.json, figures/step14_*.png
"""

import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from copy import deepcopy

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
    """Get model prediction (target vs contrast)."""
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


def collect_activations_with_hook(model, tokenizer, prompt, layer):
    """Collect MLP activation and return the input tensor."""
    activation = []
    
    def hook(module, input, output):
        activation.append(output[0, -1, :].detach().clone())
    
    mlp = model.model.layers[layer].mlp
    handle = mlp.register_forward_hook(hook)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs)
    
    handle.remove()
    return activation[0]


def run_with_modified_activation(model, tokenizer, prompt, layer, modified_activation, target_token, contrast_token):
    """Run model with modified MLP activation at specified layer."""
    
    def hook(module, input, output):
        # Replace the last token's activation
        new_output = output.clone()
        new_output[0, -1, :] = modified_activation.to(output.device).to(output.dtype)
        return new_output
    
    mlp = model.model.layers[layer].mlp
    handle = mlp.register_forward_hook(hook)
    
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
        "target_logit": target_logit,
        "contrast_logit": contrast_logit,
        "logit_diff": target_logit - contrast_logit,
    }


def main():
    print("=" * 70)
    print("STEP 14: FEATURE STEERING")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    target_layer = 12  # Best belief encoding layer
    
    # Steering scenarios
    # We'll try to flip FALSE BELIEF to TRUE BELIEF behavior
    scenarios = [
        {
            "name": "flip_fb_to_tb",
            "prompt": "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket. Where does Alice think the ball is? Alice thinks it's in the",
            "target": " drawer",  # FB answer (Alice doesn't know)
            "contrast": " basket",  # TB answer (reality)
            "expected_baseline": "target",  # Should predict drawer
            "goal": "Make model predict basket (as if Alice knew)",
        },
        {
            "name": "strengthen_fb",
            "prompt": "Tom hid the key in the box. Tom watched as Jerry moved the key to the drawer. Where does Tom think the key is? Tom thinks it's in the",
            "target": " drawer",  # TB answer (Tom saw the move)
            "contrast": " box",  # FB answer
            "expected_baseline": "target",  # Should predict drawer
            "goal": "Make model predict box (as if Tom didn't see)",
        },
    ]
    
    # Training scenarios for SAE
    train_prompts = [
        ("Alice put the ball in the drawer. Alice left. Bob moved it to the basket. Where does Alice think the ball is?", "false_belief"),
        ("Tom hid the key in the box. Tom left. Jerry moved the key to the drawer. Where does Tom think the key is?", "false_belief"),
        ("Chef put ingredients in cabinet A. Chef left. Waiter moved them to cabinet B. Where does Chef think they are?", "false_belief"),
        ("Sally put the toy in the basket. Sally went outside. Anne moved it to the box. Where does Sally think it is?", "false_belief"),
        ("Alice put the ball in the drawer. Alice stayed and watched. Bob moved it to the basket. Where does Alice think the ball is?", "true_belief"),
        ("Tom hid the key in the box. Tom watched Jerry move the key to the drawer. Where does Tom think the key is?", "true_belief"),
        ("Chef put ingredients in cabinet A. Chef saw Waiter move them to cabinet B. Where does Chef think they are?", "true_belief"),
        ("Sally put the toy in the basket. Sally watched Anne move it to the box. Where does Sally think it is?", "true_belief"),
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
    # TRAIN SAE
    # ========================================
    print(f"\n{'='*60}")
    print("TRAINING SAE")
    print(f"{'='*60}")
    
    # Collect training activations
    train_activations = []
    train_labels = []
    for prompt, label in train_prompts:
        act = collect_activations_with_hook(model, tokenizer, prompt, target_layer)
        train_activations.append(act.cpu())
        train_labels.append(label)
    
    train_activations = torch.stack(train_activations).float()
    
    # Train SAE
    sae_config = SAEConfig(d_model=d_model, d_sae=d_model * 4, l1_coeff=1e-3, lr=1e-3)
    sae = SimpleSAE(sae_config)
    trainer = SAETrainer(sae, lr=sae_config.lr)
    
    print("Training SAE...")
    for epoch in range(500):
        loss = trainer.step(train_activations)
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}: loss = {loss:.4f}")
    
    # Identify FB vs TB features
    sae.eval()
    with torch.no_grad():
        features = sae.get_feature_activations(train_activations)
    
    fb_mask = torch.tensor([l == "false_belief" for l in train_labels])
    tb_mask = torch.tensor([l == "true_belief" for l in train_labels])
    
    fb_features = features[fb_mask].mean(dim=0)
    tb_features = features[tb_mask].mean(dim=0)
    diff = fb_features - tb_features
    
    # Top FB features (positive diff = more active in FB)
    fb_top_vals, fb_top_idx = diff.topk(5)
    # Top TB features (negative diff = more active in TB)
    tb_top_vals, tb_top_idx = (-diff).topk(5)
    
    print("\nTop FALSE BELIEF features:")
    for idx, val in zip(fb_top_idx, fb_top_vals):
        print(f"  Feature #{idx.item()}: diff = {val:.3f}")
    
    print("\nTop TRUE BELIEF features:")
    for idx, val in zip(tb_top_idx, tb_top_vals):
        print(f"  Feature #{idx.item()}: diff = {-val:.3f}")
    
    # ========================================
    # STEERING EXPERIMENTS
    # ========================================
    print(f"\n{'='*60}")
    print("FEATURE STEERING EXPERIMENTS")
    print(f"{'='*60}")
    
    results = []
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"Goal: {scenario['goal']}")
        sys.stdout.flush()
        
        # Baseline prediction
        baseline = get_prediction(model, tokenizer, scenario["prompt"], scenario["target"], scenario["contrast"])
        print(f"Baseline: {baseline['prediction']} (diff={baseline['logit_diff']:.2f})")
        
        # Get activation for this prompt
        act = collect_activations_with_hook(model, tokenizer, scenario["prompt"], target_layer)
        
        # Encode to features (ensure same device)
        sae_device = next(sae.parameters()).device
        with torch.no_grad():
            feat = sae.encode(act.float().to(sae_device).unsqueeze(0))[0]
        
        scenario_results = {
            "name": scenario["name"],
            "goal": scenario["goal"],
            "baseline": baseline,
            "steering_results": [],
        }
        
        # STEERING 1: Suppress top FB features
        print("\nSteering 1: Suppress FB features")
        feat_modified = feat.clone()
        for idx in fb_top_idx[:3]:  # Top 3 FB features
            feat_modified[idx] = 0
        
        # Decode back to activation
        with torch.no_grad():
            act_modified = sae.decode(feat_modified.unsqueeze(0))[0].to(model.device)
        
        result_suppress_fb = run_with_modified_activation(
            model, tokenizer, scenario["prompt"], target_layer, 
            act_modified, scenario["target"], scenario["contrast"]
        )
        print(f"  Result: {result_suppress_fb['prediction']} (diff={result_suppress_fb['logit_diff']:.2f})")
        
        flipped = baseline["prediction"] != result_suppress_fb["prediction"]
        print(f"  FLIPPED: {flipped}")
        
        scenario_results["steering_results"].append({
            "method": "suppress_fb_features",
            "result": result_suppress_fb,
            "flipped": flipped,
        })
        
        # STEERING 2: Boost TB features
        print("\nSteering 2: Boost TB features")
        feat_modified = feat.clone()
        for idx in tb_top_idx[:3]:  # Top 3 TB features
            feat_modified[idx] = feat_modified[idx] * 3  # 3x boost
        
        with torch.no_grad():
            act_modified = sae.decode(feat_modified.unsqueeze(0))[0].to(model.device)
        
        result_boost_tb = run_with_modified_activation(
            model, tokenizer, scenario["prompt"], target_layer,
            act_modified, scenario["target"], scenario["contrast"]
        )
        print(f"  Result: {result_boost_tb['prediction']} (diff={result_boost_tb['logit_diff']:.2f})")
        
        flipped = baseline["prediction"] != result_boost_tb["prediction"]
        print(f"  FLIPPED: {flipped}")
        
        scenario_results["steering_results"].append({
            "method": "boost_tb_features",
            "result": result_boost_tb,
            "flipped": flipped,
        })
        
        # STEERING 3: Both (suppress FB + boost TB)
        print("\nSteering 3: Suppress FB + Boost TB")
        feat_modified = feat.clone()
        for idx in fb_top_idx[:3]:
            feat_modified[idx] = 0
        for idx in tb_top_idx[:3]:
            feat_modified[idx] = feat_modified[idx] * 3
        
        with torch.no_grad():
            act_modified = sae.decode(feat_modified.unsqueeze(0))[0].to(model.device)
        
        result_both = run_with_modified_activation(
            model, tokenizer, scenario["prompt"], target_layer,
            act_modified, scenario["target"], scenario["contrast"]
        )
        print(f"  Result: {result_both['prediction']} (diff={result_both['logit_diff']:.2f})")
        
        flipped = baseline["prediction"] != result_both["prediction"]
        print(f"  FLIPPED: {flipped}")
        
        scenario_results["steering_results"].append({
            "method": "suppress_fb_boost_tb",
            "result": result_both,
            "flipped": flipped,
        })
        
        results.append(scenario_results)
        sys.stdout.flush()
    
    # ========================================
    # SUMMARY
    # ========================================
    print(f"\n{'='*60}")
    print("STEERING SUMMARY")
    print(f"{'='*60}")
    
    total_attempts = 0
    total_flips = 0
    for r in results:
        for sr in r["steering_results"]:
            total_attempts += 1
            if sr["flipped"]:
                total_flips += 1
                print(f"FLIP: {r['name']} via {sr['method']}")
    
    print(f"\nTotal flips: {total_flips}/{total_attempts} ({total_flips/total_attempts:.1%})")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": config.model_name,
            "target_layer": target_layer,
        },
        "sae_features": {
            "fb_features": [int(x) for x in fb_top_idx.tolist()],
            "tb_features": [int(x) for x in tb_top_idx.tolist()],
        },
        "results": results,
        "summary": {
            "total_attempts": total_attempts,
            "total_flips": total_flips,
            "flip_rate": total_flips / total_attempts if total_attempts > 0 else 0,
        },
    }
    
    output_path = RESULTS_DIR / "step14_steering.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # ========================================
    # GENERATE FIGURE
    # ========================================
    print("\nGenerating figure...")
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Bar chart of logit diffs
    x = []
    heights = []
    colors = []
    labels = []
    
    for i, r in enumerate(results):
        base_diff = r["baseline"]["logit_diff"]
        x.append(i * 4)
        heights.append(base_diff)
        colors.append("steelblue")
        labels.append(f"{r['name']}\nBaseline")
        
        for j, sr in enumerate(r["steering_results"]):
            x.append(i * 4 + j + 1)
            heights.append(sr["result"]["logit_diff"])
            colors.append("coral" if sr["flipped"] else "lightgray")
            method_short = sr["method"].split("_")[0][:8]
            labels.append(f"{method_short}")
    
    bars = ax.bar(x, heights, color=colors, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel("Logit Difference (target - contrast)", fontsize=12)
    ax.set_title("Feature Steering: Can We Flip Predictions?", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='Baseline'),
        Patch(facecolor='coral', label='FLIPPED'),
        Patch(facecolor='lightgray', label='No flip'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step14_steering.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("STEP 14 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

