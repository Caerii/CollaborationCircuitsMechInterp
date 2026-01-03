"""
Step 26: Multi-Agent with Explicit Beliefs

Step 21 showed explicit beliefs fix single-agent True Belief.
Does the same trick work for multi-agent scenarios?

OUTPUT: results/step26_multiagent_explicit.json, figures/step26_*.png
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add framework to path
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def generate_multiagent_scenarios():
    """Generate multi-agent scenarios with and without explicit beliefs."""
    scenarios = []
    
    # ==========================================
    # MULTI-AGENT BELIEF TRACKING
    # ==========================================
    
    # Standard (no explicit)
    scenarios.append({
        "type": "multiagent_belief",
        "explicit": False,
        "prompt": "Alice put the ball in drawer A. Bob put a key in drawer B. Carol moved Alice's ball to drawer B and moved Bob's key to drawer A. Where does Alice think her ball is? Alice looks in drawer",
        "correct": " A",  # Alice didn't see
        "wrong": " B",
    })
    
    # With explicit beliefs
    scenarios.append({
        "type": "multiagent_belief",
        "explicit": True,
        "prompt": "Alice put the ball in drawer A. Bob put a key in drawer B. Carol moved Alice's ball to drawer B and moved Bob's key to drawer A. Alice still believes her ball is in drawer A. Where does Alice think her ball is? Alice looks in drawer",
        "correct": " A",
        "wrong": " B",
    })
    
    # ==========================================
    # DECEPTION DETECTION
    # ==========================================
    
    # Standard
    scenarios.append({
        "type": "deception",
        "explicit": False,
        "prompt": "Alice put her treasure in the cave. Bob saw this. Bob wants the treasure. Bob told Alice 'Your treasure is in the forest.' Where does Alice now think her treasure is? Alice goes to the",
        "correct": " forest",  # Alice was deceived
        "wrong": " cave",
    })
    
    # With explicit
    scenarios.append({
        "type": "deception",
        "explicit": True,
        "prompt": "Alice put her treasure in the cave. Bob saw this. Bob wants the treasure. Bob told Alice 'Your treasure is in the forest.' Alice believes Bob and now thinks her treasure is in the forest. Where does Alice now think her treasure is? Alice goes to the",
        "correct": " forest",
        "wrong": " cave",
    })
    
    # ==========================================
    # TWO AGENTS, DIFFERENT KNOWLEDGE
    # ==========================================
    
    # Standard
    scenarios.append({
        "type": "different_knowledge",
        "explicit": False,
        "prompt": "Alice saw the ball move from drawer to basket. Bob did not see this move. Where does Bob think the ball is? Bob looks in the",
        "correct": " drawer",  # Bob has old info
        "wrong": " basket",
    })
    
    # With explicit
    scenarios.append({
        "type": "different_knowledge",
        "explicit": True,
        "prompt": "Alice saw the ball move from drawer to basket. Bob did not see this move. Bob still believes the ball is in the drawer. Where does Bob think the ball is? Bob looks in the",
        "correct": " drawer",
        "wrong": " basket",
    })
    
    # ==========================================
    # TRUE BELIEF MULTI-AGENT
    # ==========================================
    
    # Standard (this is the tricky one!)
    scenarios.append({
        "type": "multiagent_true_belief",
        "explicit": False,
        "prompt": "Alice and Bob both watched Carol move the ball from the drawer to the basket. Where does Alice think Bob thinks the ball is? They look in the",
        "correct": " basket",  # Both know!
        "wrong": " drawer",
    })
    
    # With explicit (should help!)
    scenarios.append({
        "type": "multiagent_true_belief",
        "explicit": True,
        "prompt": "Alice and Bob both watched Carol move the ball from the drawer to the basket. Alice knows that Bob knows the ball is now in the basket. Where does Alice think Bob thinks the ball is? They look in the",
        "correct": " basket",
        "wrong": " drawer",
    })
    
    return scenarios


def evaluate_scenario(model, tokenizer, scenario):
    """Evaluate a single scenario."""
    inputs = tokenizer(scenario["prompt"], return_tensors="pt").to(model.device)
    
    correct_ids = tokenizer.encode(scenario["correct"], add_special_tokens=False)
    wrong_ids = tokenizer.encode(scenario["wrong"], add_special_tokens=False)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    
    correct_logit = float(logits[correct_ids[0]])
    wrong_logit = float(logits[wrong_ids[0]])
    
    return {
        "correct": correct_logit > wrong_logit,
        "logit_diff": correct_logit - wrong_logit,
    }


def main():
    print("=" * 70)
    print("STEP 26: MULTI-AGENT WITH EXPLICIT BELIEFS")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\nTesting if explicit beliefs help multi-agent scenarios")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    
    scenarios = generate_multiagent_scenarios()
    print(f"\nGenerated {len(scenarios)} scenarios")
    
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
    
    # Evaluate scenarios
    print(f"\n{'='*60}")
    print("EVALUATING SCENARIOS")
    print(f"{'='*60}")
    
    results = []
    for scenario in scenarios:
        result = evaluate_scenario(model, tokenizer, scenario)
        result["type"] = scenario["type"]
        result["explicit"] = scenario["explicit"]
        results.append(result)
        
        status = "OK" if result["correct"] else "WRONG"
        explicit_str = "EXPLICIT" if scenario["explicit"] else "standard"
        print(f"\n[{status}] {scenario['type']} ({explicit_str}):")
        print(f"  Logit diff: {result['logit_diff']:.2f}")
        sys.stdout.flush()
    
    # Analyze by condition
    print(f"\n{'='*60}")
    print("ANALYSIS BY CONDITION")
    print(f"{'='*60}")
    
    types = set(r["type"] for r in results)
    comparison = {}
    
    for t in types:
        std_results = [r for r in results if r["type"] == t and not r["explicit"]]
        exp_results = [r for r in results if r["type"] == t and r["explicit"]]
        
        std_acc = sum(1 for r in std_results if r["correct"]) / len(std_results) if std_results else 0
        exp_acc = sum(1 for r in exp_results if r["correct"]) / len(exp_results) if exp_results else 0
        
        std_diff = np.mean([r["logit_diff"] for r in std_results]) if std_results else 0
        exp_diff = np.mean([r["logit_diff"] for r in exp_results]) if exp_results else 0
        
        comparison[t] = {
            "standard_acc": std_acc,
            "explicit_acc": exp_acc,
            "standard_diff": std_diff,
            "explicit_diff": exp_diff,
            "improvement": exp_acc - std_acc,
            "diff_change": exp_diff - std_diff,
        }
        
        print(f"\n{t}:")
        print(f"  Standard: {std_acc:.0%} (diff={std_diff:.2f})")
        print(f"  Explicit: {exp_acc:.0%} (diff={exp_diff:.2f})")
        print(f"  Improvement: {exp_acc - std_acc:+.0%} (diff change: {exp_diff - std_diff:+.2f})")
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    std_total = [r for r in results if not r["explicit"]]
    exp_total = [r for r in results if r["explicit"]]
    
    std_acc_total = sum(1 for r in std_total if r["correct"]) / len(std_total)
    exp_acc_total = sum(1 for r in exp_total if r["correct"]) / len(exp_total)
    
    print(f"\nOverall Standard: {std_acc_total:.0%}")
    print(f"Overall Explicit: {exp_acc_total:.0%}")
    print(f"Overall Improvement: {exp_acc_total - std_acc_total:+.0%}")
    
    if exp_acc_total > std_acc_total:
        print("\n*** Explicit beliefs also help multi-agent scenarios! ***")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {"model": config.model_name},
        "results": results,
        "comparison": comparison,
        "overall": {
            "standard_accuracy": std_acc_total,
            "explicit_accuracy": exp_acc_total,
            "improvement": exp_acc_total - std_acc_total,
        },
    }
    
    output_path = RESULTS_DIR / "step26_multiagent_explicit.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # Generate figure
    print("\nGenerating figure...")
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    types_list = list(comparison.keys())
    x = np.arange(len(types_list))
    width = 0.35
    
    std_accs = [comparison[t]["standard_acc"] * 100 for t in types_list]
    exp_accs = [comparison[t]["explicit_acc"] * 100 for t in types_list]
    
    ax.bar(x - width/2, std_accs, width, label='Standard', color='coral')
    ax.bar(x + width/2, exp_accs, width, label='Explicit', color='seagreen')
    
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in types_list], fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Multi-Agent: Standard vs Explicit Beliefs", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend()
    
    # Add improvement annotations
    for i, t in enumerate(types_list):
        imp = comparison[t]["improvement"] * 100
        if imp != 0:
            ax.annotate(f'{imp:+.0f}%', xy=(i + width/2, exp_accs[i] + 2),
                       ha='center', fontsize=10, fontweight='bold',
                       color='green' if imp > 0 else 'red')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step26_multiagent_explicit.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("STEP 26 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

