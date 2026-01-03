"""
Step 7: Fine-Grained Analysis - Signal Extraction, MLP Neurons, Head Amplification

Using our specialized tools to get EXACT causal mechanisms:
1. SignalExtractor - Extract the "belief update signal" and inject it
2. MLPAnalyzer - Find which MLP neurons differ between conditions  
3. HeadAmplifier - Amplify critical heads to test their role

Based on findings from Steps 4-6:
- Critical layers: 32-34
- Critical heads: L32H0, L33H4, L33H16, L33H28, L34H0

OUTPUT: results/step7_fine_grained.json, figures/step7_*.png
"""

import sys
import json
import torch
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# Add framework to path
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig
from analysis.signal_injection import SignalExtractor, HeadAmplifier
from analysis.mlp_analysis import MLPAnalyzer

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 70)
    print("STEP 7: FINE-GRAINED ANALYSIS")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    
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
    
    # Critical layers from our findings
    critical_layers = [32, 33, 34]
    
    results = {}
    
    # ========================================
    # PART 1: SIGNAL EXTRACTION
    # ========================================
    print(f"\n{'='*60}")
    print("PART 1: SIGNAL EXTRACTION")
    print("Extract 'clean - corrupted' difference, inject to restore behavior")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    extractor = SignalExtractor(model, tokenizer)
    
    # Create minimal pairs for signal extraction
    # Clean = model should get it right (belief updated)
    # Corrupted = model might get it wrong (no belief update phrase)
    
    extraction_pairs = [
        {
            "name": "told_explicit",
            "clean": "Alice put ball in drawer. Bob moved it to basket. Alice was told about the move. Alice now knows the ball is in basket. Alice looks for the ball in the",
            "corrupted": "Alice put ball in drawer. Bob moved it to basket. Alice returns. Alice looks for the ball in the",
            "target": " basket",
            "contrast": " drawer",
        },
        {
            "name": "saw_explicit", 
            "clean": "Sally put toy in box. Anne moved toy to bin. Sally watched the whole time. Sally knows toy is in bin. Sally searches in the",
            "corrupted": "Sally put toy in box. Anne moved toy to bin. Sally comes back. Sally searches in the",
            "target": " bin",
            "contrast": " box",
        },
        {
            "name": "informed",
            "clean": "Mom left keys on table. Dad moved them to hook. Mom was informed of the move. Mom knows keys are on hook. Mom looks at the",
            "corrupted": "Mom left keys on table. Dad moved them to hook. Mom comes downstairs. Mom looks at the",
            "target": " hook",
            "contrast": " table",
        },
    ]
    
    signal_results = []
    
    for pair in extraction_pairs:
        print(f"\n--- {pair['name']} ---")
        sys.stdout.flush()
        
        # Extract signal
        try:
            signal = extractor.extract_signal(
                clean_prompt=pair["clean"],
                corrupted_prompt=pair["corrupted"],
                layers=critical_layers
            )
            
            # Test injection on the corrupted prompt itself
            injection_result = extractor.inject_and_test(
                prompt=pair["corrupted"],
                signal=signal,
                target_token=pair["target"],
                contrast_token=pair["contrast"]
            )
            
            # Calculate signal magnitude
            signal_magnitude = {
                layer: float(torch.norm(s).item()) 
                for layer, s in signal.items()
            }
            
            print(f"  Signal magnitude: L32={signal_magnitude.get(32, 0):.2f}, L33={signal_magnitude.get(33, 0):.2f}, L34={signal_magnitude.get(34, 0):.2f}")
            print(f"  Original correct: {injection_result.original_correct}")
            print(f"  After injection: {injection_result.injected_correct}")
            print(f"  FLIPPED: {injection_result.flipped}")
            sys.stdout.flush()
            
            signal_results.append({
                "name": pair["name"],
                "signal_magnitude": signal_magnitude,
                "original_correct": injection_result.original_correct,
                "injected_correct": injection_result.injected_correct,
                "flipped": injection_result.flipped,
                "original_prob": injection_result.original_prob,
                "injected_prob": injection_result.injected_prob,
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            signal_results.append({"name": pair["name"], "error": str(e)})
    
    results["signal_extraction"] = signal_results
    
    # ========================================
    # PART 2: MLP NEURON ANALYSIS
    # ========================================
    print(f"\n{'='*60}")
    print("PART 2: MLP NEURON ANALYSIS")
    print("Find which MLP neurons differ between ToM success/failure")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    mlp_analyzer = MLPAnalyzer(model, tokenizer)
    
    # Condition A: Model should predict CORRECT (belief updated)
    condition_a = [
        "Alice put ball in drawer. Bob told Alice he moved it to basket. Alice knows it's in basket. Alice looks in the",
        "Sally hid gem in cave. Tom informed Sally he moved it to forest. Sally knows it's in forest. Sally goes to the",
        "Mom left keys on table. Dad told Mom he moved them to hook. Mom knows they're on hook. Mom reaches for the",
    ]
    
    # Condition B: Model might predict WRONG (no explicit knowledge update)
    condition_b = [
        "Alice put ball in drawer. Bob moved it to basket. Alice returns. Alice looks in the",
        "Sally hid gem in cave. Tom moved it to forest. Sally returns. Sally goes to the",
        "Mom left keys on table. Dad moved them to hook. Mom comes back. Mom reaches for the",
    ]
    
    print("\nComparing conditions across critical layers...")
    sys.stdout.flush()
    
    try:
        mlp_results = mlp_analyzer.compare_conditions(
            condition_a_prompts=condition_a,
            condition_b_prompts=condition_b,
            layers=critical_layers,
            top_k=5
        )
        
        print("\nTop differing neurons per layer:")
        for layer_analysis in mlp_results:
            print(f"\n  Layer {layer_analysis.layer}:")
            print(f"    Total gate diff: {layer_analysis.gate_total_diff:.2f}")
            print(f"    Total down diff: {layer_analysis.down_total_diff:.2f}")
            if layer_analysis.gate_top_neurons:
                top_neuron = layer_analysis.gate_top_neurons[0]
                print(f"    Top gate neuron: #{top_neuron.neuron_idx} (diff={top_neuron.diff:.3f})")
            if layer_analysis.down_top_neurons:
                top_neuron = layer_analysis.down_top_neurons[0]
                print(f"    Top down neuron: #{top_neuron.neuron_idx} (diff={top_neuron.diff:.3f})")
        sys.stdout.flush()
        
        results["mlp_analysis"] = [
            {
                "layer": r.layer,
                "gate_total_diff": r.gate_total_diff,
                "down_total_diff": r.down_total_diff,
                "gate_top_neurons": [
                    {"idx": n.neuron_idx, "diff": n.diff} 
                    for n in r.gate_top_neurons[:3]
                ],
                "down_top_neurons": [
                    {"idx": n.neuron_idx, "diff": n.diff}
                    for n in r.down_top_neurons[:3]
                ],
            }
            for r in mlp_results
        ]
    except Exception as e:
        print(f"  ERROR: {e}")
        results["mlp_analysis"] = {"error": str(e)}
    
    # ========================================
    # PART 3: HEAD AMPLIFICATION
    # ========================================
    print(f"\n{'='*60}")
    print("PART 3: HEAD AMPLIFICATION")
    print("Amplify critical heads to test inhibitor vs enabler role")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    amplifier = HeadAmplifier(model, tokenizer)
    
    # Test prompts
    test_prompts = [
        {
            "prompt": "Alice put ball in drawer. Bob moved it to basket. Alice returns. Alice searches in the",
            "target": " drawer",  # False belief - should predict drawer
            "contrast": " basket",
        },
        {
            "prompt": "Sally left toy in box. Anne moved toy to bin. Sally comes back. Sally looks in the",
            "target": " box",
            "contrast": " bin",
        },
    ]
    
    # Critical heads from Step 5
    critical_heads = [(32, 0), (33, 4), (33, 16), (33, 28), (34, 0)]
    amplification_factors = [0.5, 1.0, 1.5, 2.0]  # 0.5 = reduce, 2.0 = amplify
    
    amplification_results = []
    
    for prompt_data in test_prompts:
        print(f"\n--- Testing: ...{prompt_data['prompt'][-50:]} ---")
        sys.stdout.flush()
        
        prompt_results = {"prompt": prompt_data["prompt"][-60:], "by_factor": {}}
        
        try:
            # Test with all amplification factors at once
            all_results = amplifier.test_with_amplification(
                prompt=prompt_data["prompt"],
                heads=critical_heads,
                scales=amplification_factors,
                target_token=prompt_data["target"],
                contrast_token=prompt_data["contrast"]
            )
            
            # Extract baseline (scale=1.0) - note: method returns "diff" not "logit_diff"
            baseline = all_results.get(1.0, {})
            baseline_correct = baseline.get("correct", False)
            baseline_diff = baseline.get("diff", 0)  # FIX: was "logit_diff"
            print(f"  Baseline (1.0x): {'correct' if baseline_correct else 'wrong'} (diff={baseline_diff:.2f})")
            prompt_results["baseline"] = {"correct": baseline_correct, "logit_diff": baseline_diff}
            
            # Print and store other factors
            for factor in amplification_factors:
                result = all_results.get(factor, {})
                is_correct = result.get("correct", False)  # FIX: use "correct" directly
                diff = result.get("diff", 0)  # FIX: was "logit_diff"
                flipped = is_correct != baseline_correct
                diff_change = diff - baseline_diff
                
                change = "FLIP" if flipped else "same"
                print(f"  Factor {factor:.1f}x: {'correct' if is_correct else 'wrong'} (diff={diff:.2f}, {change}, change={diff_change:+.2f})")
                
                prompt_results["by_factor"][str(factor)] = {
                    "correct": is_correct,
                    "logit_diff": diff,
                    "flipped": flipped,
                    "diff_change": diff_change,
                }
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            prompt_results["error"] = str(e)
        
        amplification_results.append(prompt_results)
        sys.stdout.flush()
    
    results["head_amplification"] = amplification_results
    
    # ========================================
    # SUMMARY
    # ========================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    # Signal extraction summary
    successful_injections = sum(1 for r in signal_results if r.get("flipped", False))
    print(f"\nSignal Extraction:")
    print(f"  Successful restorations: {successful_injections}/{len(signal_results)}")
    
    # MLP summary
    if isinstance(results.get("mlp_analysis"), list):
        max_layer = max(results["mlp_analysis"], key=lambda x: x["down_total_diff"])
        print(f"\nMLP Analysis:")
        print(f"  Most differentiating layer: {max_layer['layer']} (diff={max_layer['down_total_diff']:.2f})")
    
    # Amplification summary
    flip_counts = {f: 0 for f in amplification_factors}
    for prompt_results in amplification_results:
        for factor_str, result in prompt_results.get("by_factor", {}).items():
            if result.get("flipped"):
                flip_counts[float(factor_str)] += 1
    print(f"\nHead Amplification flips:")
    for factor, count in flip_counts.items():
        print(f"  {factor:.1f}x: {count}/{len(test_prompts)} flips")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    results["timestamp"] = datetime.now().isoformat()
    results["config"] = {
        "critical_layers": critical_layers,
        "critical_heads": [list(h) for h in critical_heads],
    }
    
    output_path = RESULTS_DIR / "step7_fine_grained.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # ========================================
    # GENERATE FIGURES
    # ========================================
    print("\nGenerating figures...")
    
    import matplotlib.pyplot as plt
    
    # Figure 1: MLP neuron differences across layers
    if isinstance(results.get("mlp_analysis"), list):
        fig, ax = plt.subplots(figsize=(10, 5))
        
        layers = [r["layer"] for r in results["mlp_analysis"]]
        gate_diffs = [r["gate_total_diff"] for r in results["mlp_analysis"]]
        down_diffs = [r["down_total_diff"] for r in results["mlp_analysis"]]
        
        x = np.arange(len(layers))
        width = 0.35
        
        ax.bar(x - width/2, gate_diffs, width, label='Gate (gating)', color='steelblue')
        ax.bar(x + width/2, down_diffs, width, label='Down (output)', color='coral')
        
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Total Neuron Difference", fontsize=12)
        ax.set_title("MLP Neuron Differences Between ToM Conditions", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in layers])
        ax.legend()
        
        plt.tight_layout()
        fig_path = FIGURES_DIR / "step7_mlp_differences.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Figure saved to: {fig_path}")
    
    # Figure 2: Amplification effects
    if amplification_results:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        x = np.arange(len(amplification_factors))
        width = 0.35
        
        for i, prompt_results in enumerate(amplification_results):
            baseline_diff = prompt_results.get("baseline", {}).get("logit_diff", 0)
            diffs = []
            for factor in amplification_factors:
                result = prompt_results.get("by_factor", {}).get(str(factor), {})
                diffs.append(result.get("logit_diff", baseline_diff))
            
            offset = (i - len(amplification_results)/2 + 0.5) * width
            bars = ax.bar(x + offset, diffs, width, label=f"Prompt {i+1}")
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel("Amplification Factor", fontsize=12)
        ax.set_ylabel("Logit Diff (target - contrast)", fontsize=12)
        ax.set_title("Head Amplification Effects on ToM", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{f:.1f}x" for f in amplification_factors])
        ax.legend()
        
        plt.tight_layout()
        fig_path = FIGURES_DIR / "step7_amplification_effects.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("STEP 7 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

