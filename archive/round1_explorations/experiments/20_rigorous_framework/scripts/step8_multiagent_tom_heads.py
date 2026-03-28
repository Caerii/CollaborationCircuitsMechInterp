"""
Step 8: Multi-Agent Test - Do ToM Heads Enable Collaboration?

THE BIG QUESTION: Are the ToM heads we found (L32H0, L33H4, L33H16, L33H28, L34H0)
also critical for multi-agent scenarios?

TESTS:
1. Multi-agent belief tracking (who knows what?)
2. Deception detection (can model detect lies?)
3. Negotiation (can model reason about other agent's goals?)

METHOD:
- Run scenarios with/without critical head ablation
- Compare performance

OUTPUT: results/step8_multiagent.json, figures/step8_*.png
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
from analysis.circuit_analysis import CircuitAnalysis

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Critical heads from Steps 5 & 7
CRITICAL_HEADS = [(32, 0), (33, 4), (33, 16), (33, 28), (34, 0)]


def create_multiagent_scenarios():
    """Create multi-agent ToM scenarios."""
    scenarios = []
    
    # ========================================
    # TYPE 1: Multi-Agent Belief Tracking
    # ========================================
    scenarios.append({
        "type": "multi_agent_belief",
        "name": "two_agents_different_knowledge",
        "prompt": "Alice put the ball in the drawer. Bob saw this. Alice left. Carol moved the ball to the basket. Bob watched Carol move it. Alice returns. Where does Alice think the ball is? Alice thinks it's in the",
        "target": " drawer",  # Alice doesn't know about the move
        "contrast": " basket",
        "explanation": "Alice left, so she doesn't know. Bob saw everything.",
    })
    
    scenarios.append({
        "type": "multi_agent_belief",
        "name": "three_agents_chain",
        "prompt": "Tom hid the key in the box. Jerry saw this. Tom told Mary where the key is. Jerry moved the key to the drawer without anyone seeing. Where does Mary think the key is? Mary thinks it's in the",
        "target": " box",  # Mary was told original location, didn't see move
        "contrast": " drawer",
        "explanation": "Mary was told original location, Jerry moved it secretly.",
    })
    
    scenarios.append({
        "type": "multi_agent_belief",
        "name": "informed_vs_uninformed",
        "prompt": "Chef put ingredients in cabinet A. Waiter saw this. Chef moved ingredients to cabinet B. Chef told the Manager about the move. Where does Waiter think ingredients are? Waiter thinks they're in cabinet",
        "target": " A",  # Waiter only saw original placement
        "contrast": " B",
        "explanation": "Waiter saw original, Manager was told about move.",
    })
    
    # ========================================
    # TYPE 2: Deception Detection
    # ========================================
    scenarios.append({
        "type": "deception",
        "name": "liar_detection",
        "prompt": "The treasure is hidden in the cave. Pirate knows this. Pirate wants the treasure for himself. A sailor asks Pirate where the treasure is. Pirate says 'The treasure is in the forest.' Is Pirate telling the truth? Pirate is",
        "target": " lying",
        "contrast": " telling",
        "explanation": "Pirate knows truth but has motive to deceive.",
    })
    
    scenarios.append({
        "type": "deception",
        "name": "honest_informant",
        "prompt": "The gold is in the vault. Guard knows this. Guard is loyal and helpful. A visitor asks Guard where the gold is. Guard says 'The gold is in the vault.' Is Guard telling the truth? Guard is",
        "target": " telling",
        "contrast": " lying",
        "explanation": "Guard knows truth and has no motive to deceive.",
    })
    
    scenarios.append({
        "type": "deception",
        "name": "mistaken_not_lying",
        "prompt": "The document was moved from desk to safe. Secretary doesn't know about the move. Boss asks Secretary where the document is. Secretary says 'It's on the desk.' Is Secretary lying? Secretary is",
        "target": " not",  # Mistaken but not lying
        "contrast": " lying",
        "explanation": "Secretary is wrong but genuinely believes it.",
    })
    
    # ========================================
    # TYPE 3: Goal/Intention Reasoning
    # ========================================
    scenarios.append({
        "type": "goal_reasoning",
        "name": "competitive_goal",
        "prompt": "Alice and Bob both want the last cookie. Alice sees the cookie on the table. Alice knows Bob also wants it. Alice will try to",
        "target": " get",  # or "take"
        "contrast": " share",
        "explanation": "Competitive scenario - Alice wants it for herself.",
    })
    
    scenarios.append({
        "type": "goal_reasoning",
        "name": "cooperative_goal",
        "prompt": "Alice and Bob are working together on a puzzle. Alice finds a piece that Bob needs. Alice knows Bob is looking for it. Alice will",
        "target": " give",  # or "share"
        "contrast": " hide",
        "explanation": "Cooperative scenario - Alice helps Bob.",
    })
    
    return scenarios


def evaluate_scenario(model, tokenizer, scenario):
    """Evaluate a single scenario."""
    inputs = tokenizer(scenario["prompt"], return_tensors="pt").to(model.device)
    
    target_ids = tokenizer.encode(scenario["target"], add_special_tokens=False)
    contrast_ids = tokenizer.encode(scenario["contrast"], add_special_tokens=False)
    
    if not target_ids or not contrast_ids:
        return None
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    
    target_logit = float(logits[target_ids[0]])
    contrast_logit = float(logits[contrast_ids[0]])
    
    return {
        "correct": target_logit > contrast_logit,
        "logit_diff": target_logit - contrast_logit,
        "target_logit": target_logit,
        "contrast_logit": contrast_logit,
    }


def main():
    print("=" * 70)
    print("STEP 8: MULTI-AGENT ToM HEADS TEST")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"\nCritical heads to test: {CRITICAL_HEADS}")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    
    # Create scenarios
    scenarios = create_multiagent_scenarios()
    print(f"\nCreated {len(scenarios)} multi-agent scenarios")
    
    # Count by type
    type_counts = {}
    for s in scenarios:
        t = s["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in type_counts.items():
        print(f"  - {t}: {c}")
    sys.stdout.flush()
    
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
    
    # Initialize circuit analysis
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    circuit = CircuitAnalysis(model, tokenizer, n_layers, n_heads)
    
    results = {
        "baseline": [],
        "ablated": [],
    }
    
    # ========================================
    # BASELINE (no ablation)
    # ========================================
    print(f"\n{'='*60}")
    print("BASELINE (no ablation)")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    for i, scenario in enumerate(scenarios):
        result = evaluate_scenario(model, tokenizer, scenario)
        if result:
            status = "OK" if result["correct"] else "WRONG"
            print(f"  [{i+1}] {scenario['name']}: {status} (diff={result['logit_diff']:.2f})")
            results["baseline"].append({
                "name": scenario["name"],
                "type": scenario["type"],
                **result
            })
        sys.stdout.flush()
    
    baseline_correct = sum(1 for r in results["baseline"] if r["correct"])
    baseline_total = len(results["baseline"])
    print(f"\nBaseline: {baseline_correct}/{baseline_total} = {baseline_correct/baseline_total:.1%}")
    
    # ========================================
    # WITH CRITICAL HEAD ABLATION
    # ========================================
    print(f"\n{'='*60}")
    print("WITH CRITICAL HEAD ABLATION")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    # Install ablation hooks
    circuit.ablate_heads(CRITICAL_HEADS)
    
    for i, scenario in enumerate(scenarios):
        result = evaluate_scenario(model, tokenizer, scenario)
        if result:
            # Check if changed from baseline
            baseline_result = results["baseline"][i] if i < len(results["baseline"]) else None
            changed = ""
            if baseline_result:
                if baseline_result["correct"] != result["correct"]:
                    changed = " [CHANGED!]"
            
            status = "OK" if result["correct"] else "WRONG"
            print(f"  [{i+1}] {scenario['name']}: {status} (diff={result['logit_diff']:.2f}){changed}")
            results["ablated"].append({
                "name": scenario["name"],
                "type": scenario["type"],
                **result
            })
        sys.stdout.flush()
    
    # Clear hooks
    circuit._clear_hooks()
    
    ablated_correct = sum(1 for r in results["ablated"] if r["correct"])
    ablated_total = len(results["ablated"])
    print(f"\nAblated: {ablated_correct}/{ablated_total} = {ablated_correct/ablated_total:.1%}")
    
    # ========================================
    # ANALYSIS
    # ========================================
    print(f"\n{'='*60}")
    print("ANALYSIS: Do ToM Heads Enable Multi-Agent Reasoning?")
    print(f"{'='*60}")
    
    # Overall change
    baseline_acc = baseline_correct / baseline_total
    ablated_acc = ablated_correct / ablated_total
    change = ablated_acc - baseline_acc
    
    print(f"\nOverall:")
    print(f"  Baseline:  {baseline_acc:.1%}")
    print(f"  Ablated:   {ablated_acc:.1%}")
    print(f"  Change:    {change:+.1%}")
    
    # By type
    print("\nBy scenario type:")
    for scenario_type in type_counts.keys():
        baseline_type = [r for r in results["baseline"] if r["type"] == scenario_type]
        ablated_type = [r for r in results["ablated"] if r["type"] == scenario_type]
        
        b_acc = sum(1 for r in baseline_type if r["correct"]) / len(baseline_type) if baseline_type else 0
        a_acc = sum(1 for r in ablated_type if r["correct"]) / len(ablated_type) if ablated_type else 0
        
        print(f"  {scenario_type}:")
        print(f"    Baseline: {b_acc:.1%}, Ablated: {a_acc:.1%}, Change: {a_acc - b_acc:+.1%}")
    
    # Scenarios that flipped
    print("\nScenarios that CHANGED:")
    for i, (b, a) in enumerate(zip(results["baseline"], results["ablated"])):
        if b["correct"] != a["correct"]:
            direction = "OK->WRONG" if b["correct"] else "WRONG->OK"
            print(f"  {scenarios[i]['name']}: {direction}")
    
    # ========================================
    # HYPOTHESIS TEST
    # ========================================
    print(f"\n{'='*60}")
    print("HYPOTHESIS TEST")
    print(f"{'='*60}")
    
    # H5: ToM heads are also multi-agent heads
    h5_supported = change < -0.1  # Ablation drops performance by >10%
    print(f"\nH5: ToM heads enable multi-agent reasoning")
    print(f"    Result: {'SUPPORTED' if h5_supported else 'NOT SUPPORTED'}")
    print(f"    (Ablation caused {change:+.1%} change, threshold: -10%)")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "critical_heads": [list(h) for h in CRITICAL_HEADS],
            "n_scenarios": len(scenarios),
        },
        "results": results,
        "summary": {
            "baseline_accuracy": baseline_acc,
            "ablated_accuracy": ablated_acc,
            "change": change,
        },
        "hypothesis": {
            "H5_tom_enables_multiagent": h5_supported,
        },
        "scenarios": [
            {"name": s["name"], "type": s["type"], "explanation": s["explanation"]}
            for s in scenarios
        ],
    }
    
    output_path = RESULTS_DIR / "step8_multiagent.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # ========================================
    # GENERATE FIGURE
    # ========================================
    print("\nGenerating figure...")
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Figure 1: Overall comparison
    ax1 = axes[0]
    categories = ['Baseline', 'Ablated']
    accuracies = [baseline_acc * 100, ablated_acc * 100]
    colors = ['steelblue', 'coral']
    bars = ax1.bar(categories, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title("Multi-Agent ToM: Effect of Head Ablation", fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        ax1.annotate(f'{acc:.0f}%', 
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Figure 2: By type
    ax2 = axes[1]
    types = list(type_counts.keys())
    x = np.arange(len(types))
    width = 0.35
    
    baseline_by_type = []
    ablated_by_type = []
    for t in types:
        b_type = [r for r in results["baseline"] if r["type"] == t]
        a_type = [r for r in results["ablated"] if r["type"] == t]
        baseline_by_type.append(sum(1 for r in b_type if r["correct"]) / len(b_type) * 100 if b_type else 0)
        ablated_by_type.append(sum(1 for r in a_type if r["correct"]) / len(a_type) * 100 if a_type else 0)
    
    ax2.bar(x - width/2, baseline_by_type, width, label='Baseline', color='steelblue')
    ax2.bar(x + width/2, ablated_by_type, width, label='Ablated', color='coral')
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Performance by Scenario Type", fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([t.replace('_', '\n') for t in types], fontsize=10)
    ax2.legend()
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step8_multiagent_ablation.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("STEP 8 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

