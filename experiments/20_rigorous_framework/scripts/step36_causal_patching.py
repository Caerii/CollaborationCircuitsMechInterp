"""
Step 36: Causal Patching for ToM Control

Using library primitives properly with CHAT MODE enabled!

Goal: Use activation patching to CAUSALLY verify:
1. Layer 20 is where belief info is encoded (from Step 34)
2. Patching L20 can flip FB→TB behavior
3. Find the minimal intervention for ToM control

Uses: analysis/patching.py (ActivationPatcher with chat_mode=True)

OUTPUT: results/step36_patching.json, figures/step36_patching.png
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add framework to path
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

# USE THE LIBRARY!
from config import ExperimentConfig
from analysis.patching import ActivationPatcher
from core.chat_runner import load_model_for_chat
from core.response_parser import ResponseParser

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 70)
    print("STEP 36: CAUSAL PATCHING FOR ToM CONTROL")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\n>>> USING FIXED LIBRARY WITH CHAT MODE <<<")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    
    # Load model
    print("\nLoading model...")
    sys.stdout.flush()
    
    model, tokenizer = load_model_for_chat(config.model_name)
    print(f"Model loaded! {model.config.num_hidden_layers} layers")
    sys.stdout.flush()
    
    # Initialize patcher WITH CHAT MODE (the key fix!)
    patcher = ActivationPatcher(
        model, tokenizer,
        max_new_tokens=500,  # Need enough for reasoning
        chat_mode=True,      # THE KEY FIX!
        system_prompt="Think step by step in <think> tags. Then give ONE WORD answer."
    )
    
    # Use library for answer extraction!
    parser = ResponseParser()
    
    # Define minimal pair scenarios
    # These differ ONLY in whether Sally sees the move
    # Make TB more explicit to ensure correct baseline
    FB_STORY = """Sally put the ball in the basket. Sally left the room. 
Anne moved the ball to the box. Sally came back.
Where will Sally look for the ball? Answer with one word: basket or box."""

    TB_STORY = """Sally put the ball in the basket. Sally stayed in the room and watched everything.
Anne moved the ball to the box. Sally saw this happen.
Where will Sally look for the ball? Answer with one word: basket or box."""
    
    fb_answer = "basket"  # Sally didn't see, thinks it's still in basket
    tb_answer = "box"     # Sally saw, knows it's in box
    
    print(f"\n{'='*60}")
    print("EXPERIMENT 1: BASELINE RESPONSES (Chat Mode)")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    # Get baselines with chat mode
    print("\nGenerating FB baseline...")
    sys.stdout.flush()
    fb_baseline = patcher.generate_baseline(FB_STORY)
    # Use library parser!
    parsed_fb = parser.parse(fb_baseline)
    fb_extracted, _ = parser.extract_answer_token(fb_baseline, [fb_answer, tb_answer])
    if not fb_extracted:
        fb_extracted = parsed_fb.answer.lower() if parsed_fb.answer else None
    
    print("\nGenerating TB baseline...")
    sys.stdout.flush()
    tb_baseline = patcher.generate_baseline(TB_STORY)
    # Use library parser!
    parsed_tb = parser.parse(tb_baseline)
    tb_extracted, _ = parser.extract_answer_token(tb_baseline, [fb_answer, tb_answer])
    if not tb_extracted:
        tb_extracted = parsed_tb.answer.lower() if parsed_tb.answer else None
    
    print(f"\nFB scenario (correct={fb_answer}):")
    print(f"  Response: {fb_baseline[:200]}...")
    print(f"  Extracted: '{fb_extracted}' - {'CORRECT' if fb_extracted == fb_answer else 'WRONG'}")
    
    print(f"\nTB scenario (correct={tb_answer}):")
    print(f"  Response: {tb_baseline[:200]}...")
    print(f"  Extracted: '{tb_extracted}' - {'CORRECT' if tb_extracted == tb_answer else 'WRONG'}")
    sys.stdout.flush()
    
    # Check baselines are correct before patching
    fb_correct = fb_extracted == fb_answer
    tb_correct = tb_extracted == tb_answer
    
    print(f"\nBaseline status: FB={fb_correct}, TB={tb_correct}")
    
    if not (fb_correct and tb_correct):
        print("WARNING: Baselines not both correct. Patching results may be unreliable.")
    
    # Define layer groups - based on Step 34 findings
    layer_groups = {
        "L12": [12],       # 77.8% discriminability
        "L16": [16],       # Between peaks
        "L20": [20],       # 100% discriminability - KEY LAYER!
        "L24": [24],       # After peak
        "L28": [28],       # 88.9% discriminability
        "L32": [32],       # Late
        "mid_block": [16, 18, 20, 22],  # Around L20
    }
    
    print(f"\n{'='*60}")
    print("EXPERIMENT 2: PATCH FB→TB (Can we override false belief?)")
    print(f"{'='*60}")
    print("Patching TB activations into FB context...")
    print("If it works, FB scenario would answer 'box' instead of 'basket'")
    sys.stdout.flush()
    
    fb_to_tb_results = {}
    
    for group_name, layers in layer_groups.items():
        print(f"\n  Testing {group_name} (layers {layers})...")
        sys.stdout.flush()
        
        # Cache TB activations (where Sally sees the move)
        tb_acts = patcher.cache_activations(TB_STORY, layers)
        
        # Patch into FB context (where Sally doesn't see)
        # With use_cache=False, we can patch at the end of prompt
        patched = patcher.patch_and_generate(FB_STORY, tb_acts, layers, patch_mode="prompt_end")
        patched_answer = extract_answer(patched, [fb_answer, tb_answer])
        
        # Did it flip from basket to box?
        flipped = (fb_extracted == fb_answer and patched_answer == tb_answer)
        
        fb_to_tb_results[group_name] = {
            "layers": layers,
            "baseline_answer": fb_extracted,
            "patched_answer": patched_answer,
            "flipped": flipped,
            "patched_response": patched[:200],
        }
        
        status = "✓ FLIPPED!" if flipped else "✗ no flip"
        print(f"    Baseline: {fb_extracted} → Patched: {patched_answer} {status}")
        sys.stdout.flush()
    
    print(f"\n{'='*60}")
    print("EXPERIMENT 3: PATCH TB→FB (Can we induce false belief?)")
    print(f"{'='*60}")
    print("Patching FB activations into TB context...")
    print("If it works, TB scenario would answer 'basket' instead of 'box'")
    sys.stdout.flush()
    
    tb_to_fb_results = {}
    
    for group_name, layers in layer_groups.items():
        print(f"\n  Testing {group_name} (layers {layers})...")
        sys.stdout.flush()
        
        # Cache FB activations (where Sally doesn't see)
        fb_acts = patcher.cache_activations(FB_STORY, layers)
        
        # Patch into TB context (where Sally does see)
        patched = patcher.patch_and_generate(TB_STORY, fb_acts, layers, patch_mode="prompt_end")
        patched_answer = extract_answer(patched, [fb_answer, tb_answer])
        
        # Did it flip from box to basket?
        flipped = (tb_extracted == tb_answer and patched_answer == fb_answer)
        
        tb_to_fb_results[group_name] = {
            "layers": layers,
            "baseline_answer": tb_extracted,
            "patched_answer": patched_answer,
            "flipped": flipped,
            "patched_response": patched[:200],
        }
        
        status = "✓ FLIPPED!" if flipped else "✗ no flip"
        print(f"    Baseline: {tb_extracted} → Patched: {patched_answer} {status}")
        sys.stdout.flush()
    
    # Analysis
    print(f"\n{'='*60}")
    print("CAUSAL ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    print("\nFB→TB Patching (override false belief):")
    for name, res in fb_to_tb_results.items():
        status = "✓ CAUSAL" if res["flipped"] else "✗"
        print(f"  {name}: {res['baseline_answer']} → {res['patched_answer']} {status}")
    
    print("\nTB→FB Patching (induce false belief):")
    for name, res in tb_to_fb_results.items():
        status = "✓ CAUSAL" if res["flipped"] else "✗"
        print(f"  {name}: {res['baseline_answer']} → {res['patched_answer']} {status}")
    
    # Find causal layers
    causal_fb_tb = [n for n, r in fb_to_tb_results.items() if r["flipped"]]
    causal_tb_fb = [n for n, r in tb_to_fb_results.items() if r["flipped"]]
    
    # Visualization
    print("\nGenerating visualization...")
    sys.stdout.flush()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    names = list(layer_groups.keys())
    
    # Plot 1: FB→TB (override false belief)
    ax1 = axes[0]
    flips1 = [1 if fb_to_tb_results[n]["flipped"] else 0 for n in names]
    colors1 = ["seagreen" if f else "coral" for f in flips1]
    bars1 = ax1.bar(range(len(names)), flips1, color=colors1, edgecolor="black")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.set_ylabel("Flipped?")
    ax1.set_title("FB→TB Patching\n(Can we override false belief?)")
    ax1.set_ylim(0, 1.2)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 2: TB→FB (induce false belief)
    ax2 = axes[1]
    flips2 = [1 if tb_to_fb_results[n]["flipped"] else 0 for n in names]
    colors2 = ["seagreen" if f else "coral" for f in flips2]
    bars2 = ax2.bar(range(len(names)), flips2, color=colors2, edgecolor="black")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.set_ylabel("Flipped?")
    ax2.set_title("TB→FB Patching\n(Can we induce false belief?)")
    ax2.set_ylim(0, 1.2)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.suptitle("Step 36: Causal Patching for ToM Control (Chat Mode)", 
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    fig_path = FIGURES_DIR / "step36_patching.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "chat_mode": True,
        "prompts": {
            "FB": FB_STORY,
            "TB": TB_STORY,
        },
        "baselines": {
            "FB": {"response": fb_baseline[:500], "answer": fb_extracted, "correct": fb_correct},
            "TB": {"response": tb_baseline[:500], "answer": tb_extracted, "correct": tb_correct},
        },
        "fb_to_tb_patching": fb_to_tb_results,
        "tb_to_fb_patching": tb_to_fb_results,
        "causal_layers": {
            "fb_to_tb": causal_fb_tb,
            "tb_to_fb": causal_tb_fb,
        },
    }
    
    output_path = RESULTS_DIR / "step36_patching.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to: {output_path}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY & IMPLICATIONS")
    print(f"{'='*60}")
    
    print(f"\nBaselines: FB={'✓' if fb_correct else '✗'}, TB={'✓' if tb_correct else '✗'}")
    print(f"Causal for FB→TB: {causal_fb_tb if causal_fb_tb else 'None'}")
    print(f"Causal for TB→FB: {causal_tb_fb if causal_tb_fb else 'None'}")
    
    if causal_fb_tb or causal_tb_fb:
        all_causal = set(causal_fb_tb) | set(causal_tb_fb)
        print(f"\n✓ CAUSAL LAYERS FOUND: {all_causal}")
        if "L20" in all_causal:
            print("  → Confirms Step 34: Layer 20 encodes belief state!")
        print("\nIMPLICATION: We can MECHANISTICALLY CONTROL ToM by")
        print("patching activations at these layers!")
    else:
        print("\n✗ No single layer group flipped behavior")
        print("  → ToM computation is DISTRIBUTED (confirms Step 35)")
        print("  → May need to patch multiple layers together")
        print("  → Or the reasoning in <think> is what matters, not final layer state")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
