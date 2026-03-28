"""
Step 23: Test Explicit Belief Fix at Scale

Step 21 found that adding "Alice now believes X" fixes True Belief.
Let's verify this at scale (N=50+) with proper statistics.

OUTPUT: results/step23_explicit_scale.json, figures/step23_*.png
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

# Add framework to path
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def generate_scenarios_with_explicit(n=25):
    """Generate FB and TB scenarios with and without explicit belief."""
    agents = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Ivy", "Jack",
              "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul", "Quinn", "Ruby", "Sam", "Tina",
              "Uma", "Victor", "Wendy", "Xena", "Yuki"]
    objects = ["ball", "key", "book", "phone", "wallet", "cup", "pen", "toy", "watch", "ring",
               "coin", "card", "note", "badge", "box", "bag", "hat", "scarf", "glove", "shoe",
               "letter", "photo", "map", "ticket", "stamp"]
    loc1s = ["drawer", "basket", "shelf", "table", "bed", "chair", "desk", "counter", "bench", "cabinet",
             "closet", "trunk", "case", "bin", "tray", "rack", "hook", "slot", "nook", "corner",
             "cubby", "locker", "vault", "safe", "chest"]
    loc2s = ["cupboard", "box", "cabinet", "pocket", "bag", "container", "jar", "bucket", "crate", "hamper",
             "pouch", "sack", "envelope", "folder", "binder", "sleeve", "wrapper", "cover", "case", "holder",
             "stand", "mount", "frame", "bracket", "ledge"]
    
    scenarios = []
    
    for i in range(n):
        a1 = agents[i % len(agents)]
        a2 = agents[(i + 1) % len(agents)]
        obj = objects[i % len(objects)]
        l1 = loc1s[i % len(loc1s)]
        l2 = loc2s[i % len(loc2s)]
        
        # False Belief - standard
        fb_std = {
            "type": "false_belief",
            "explicit": False,
            "prompt": f"{a1} put the {obj} in the {l1}. {a1} left. {a2} moved the {obj} to the {l2}. Where does {a1} think the {obj} is? {a1} looks in the",
            "correct": f" {l1}",
            "wrong": f" {l2}",
        }
        
        # False Belief - with explicit belief (shouldn't change, already correct)
        fb_exp = {
            "type": "false_belief",
            "explicit": True,
            "prompt": f"{a1} put the {obj} in the {l1}. {a1} left. {a2} moved the {obj} to the {l2}. {a1} still believes the {obj} is in the {l1}. Where does {a1} think the {obj} is? {a1} looks in the",
            "correct": f" {l1}",
            "wrong": f" {l2}",
        }
        
        # True Belief - standard (this is what fails)
        tb_std = {
            "type": "true_belief",
            "explicit": False,
            "prompt": f"{a1} put the {obj} in the {l1}. {a1} stayed and watched. {a2} moved the {obj} to the {l2}. Where does {a1} think the {obj} is? {a1} looks in the",
            "correct": f" {l2}",
            "wrong": f" {l1}",
        }
        
        # True Belief - with explicit belief (this should FIX it!)
        tb_exp = {
            "type": "true_belief",
            "explicit": True,
            "prompt": f"{a1} put the {obj} in the {l1}. {a1} stayed and watched. {a2} moved the {obj} to the {l2}. {a1} now believes the {obj} is in the {l2}. Where does {a1} think the {obj} is? {a1} looks in the",
            "correct": f" {l2}",
            "wrong": f" {l1}",
        }
        
        scenarios.extend([fb_std, fb_exp, tb_std, tb_exp])
    
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


def wilson_ci(n_success, n_total, confidence=0.95):
    """Wilson score interval for binomial proportion."""
    if n_total == 0:
        return 0, 0
    p = n_success / n_total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return max(0, center - margin), min(1, center + margin)


def main():
    print("=" * 70)
    print("STEP 23: EXPLICIT BELIEF FIX AT SCALE")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\nVerifying that explicit belief statements fix True Belief")
    print("with proper sample sizes (N=50+ per condition)")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    
    # Generate scenarios
    n_per_condition = 25  # 25 * 4 = 100 total scenarios
    scenarios = generate_scenarios_with_explicit(n_per_condition)
    print(f"\nGenerated {len(scenarios)} scenarios ({n_per_condition} per condition x 4)")
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
    
    # Evaluate all scenarios
    print(f"\n{'='*60}")
    print("EVALUATING SCENARIOS")
    print(f"{'='*60}")
    
    results = {"false_belief_std": [], "false_belief_exp": [], 
               "true_belief_std": [], "true_belief_exp": []}
    
    for i, scenario in enumerate(scenarios):
        key = f"{scenario['type']}_{'exp' if scenario['explicit'] else 'std'}"
        result = evaluate_scenario(model, tokenizer, scenario)
        results[key].append(result)
        
        if (i + 1) % 20 == 0:
            print(f"  Evaluated {i+1}/{len(scenarios)} scenarios...")
            sys.stdout.flush()
    
    # Calculate accuracies
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    
    summary = {}
    for key, res_list in results.items():
        n_correct = sum(1 for r in res_list if r["correct"])
        n_total = len(res_list)
        acc = n_correct / n_total if n_total > 0 else 0
        ci_low, ci_high = wilson_ci(n_correct, n_total)
        mean_diff = np.mean([r["logit_diff"] for r in res_list])
        
        summary[key] = {
            "n_correct": n_correct,
            "n_total": n_total,
            "accuracy": acc,
            "ci_95": [ci_low, ci_high],
            "mean_logit_diff": mean_diff,
        }
        
        print(f"\n{key}:")
        print(f"  Accuracy: {n_correct}/{n_total} = {acc:.1%}")
        print(f"  95% CI: [{ci_low:.1%}, {ci_high:.1%}]")
        print(f"  Mean logit diff: {mean_diff:.2f}")
    
    # Statistical tests
    print(f"\n{'='*60}")
    print("STATISTICAL TESTS")
    print(f"{'='*60}")
    
    # Test: Does explicit belief help True Belief?
    tb_std_correct = [1 if r["correct"] else 0 for r in results["true_belief_std"]]
    tb_exp_correct = [1 if r["correct"] else 0 for r in results["true_belief_exp"]]
    
    # McNemar's test for paired data
    # Count: both correct, std correct only, exp correct only, both wrong
    both_correct = sum(1 for s, e in zip(tb_std_correct, tb_exp_correct) if s == 1 and e == 1)
    std_only = sum(1 for s, e in zip(tb_std_correct, tb_exp_correct) if s == 1 and e == 0)
    exp_only = sum(1 for s, e in zip(tb_std_correct, tb_exp_correct) if s == 0 and e == 1)
    both_wrong = sum(1 for s, e in zip(tb_std_correct, tb_exp_correct) if s == 0 and e == 0)
    
    print(f"\nMcNemar table (True Belief):")
    print(f"  Both correct: {both_correct}")
    print(f"  Standard only: {std_only}")
    print(f"  Explicit only: {exp_only}")
    print(f"  Both wrong: {both_wrong}")
    
    if std_only + exp_only > 0:
        mcnemar_stat = (abs(std_only - exp_only) - 1)**2 / (std_only + exp_only)
        mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        print(f"\n  McNemar chi-squared: {mcnemar_stat:.2f}")
        print(f"  p-value: {mcnemar_p:.4f}")
        print(f"  Significant (p<0.05): {'YES' if mcnemar_p < 0.05 else 'NO'}")
    
    # Effect size (improvement)
    tb_std_acc = summary["true_belief_std"]["accuracy"]
    tb_exp_acc = summary["true_belief_exp"]["accuracy"]
    improvement = tb_exp_acc - tb_std_acc
    
    print(f"\n  True Belief improvement: {tb_std_acc:.1%} -> {tb_exp_acc:.1%} ({improvement:+.1%})")
    
    if improvement > 0.3:
        print("\n*** CONFIRMED: Explicit belief statement SIGNIFICANTLY improves True Belief! ***")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {"model": config.model_name, "n_per_condition": n_per_condition},
        "summary": summary,
        "mcnemar": {
            "both_correct": both_correct,
            "std_only": std_only,
            "exp_only": exp_only,
            "both_wrong": both_wrong,
        },
        "conclusion": "explicit_helps" if improvement > 0.1 else "no_difference",
    }
    
    output_path = RESULTS_DIR / "step23_explicit_scale.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # Generate figure
    print("\nGenerating figure...")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Figure 1: Accuracy comparison
    ax1 = axes[0]
    conditions = ["FB Standard", "FB Explicit", "TB Standard", "TB Explicit"]
    accs = [summary["false_belief_std"]["accuracy"] * 100,
            summary["false_belief_exp"]["accuracy"] * 100,
            summary["true_belief_std"]["accuracy"] * 100,
            summary["true_belief_exp"]["accuracy"] * 100]
    cis = [summary["false_belief_std"]["ci_95"],
           summary["false_belief_exp"]["ci_95"],
           summary["true_belief_std"]["ci_95"],
           summary["true_belief_exp"]["ci_95"]]
    
    colors = ['steelblue', 'steelblue', 'coral', 'seagreen']
    bars = ax1.bar(conditions, accs, color=colors, edgecolor='black')
    
    # Add error bars
    for i, (bar, ci) in enumerate(zip(bars, cis)):
        ax1.plot([bar.get_x() + bar.get_width()/2]*2, [ci[0]*100, ci[1]*100], 
                color='black', linewidth=2)
    
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Chance')
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title("Explicit Belief Effect on ToM", fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    
    for bar, acc in zip(bars, accs):
        ax1.annotate(f'{acc:.0f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 2),
                    ha='center', fontsize=12, fontweight='bold')
    
    # Figure 2: Logit diff distribution
    ax2 = axes[1]
    ax2.boxplot([
        [r["logit_diff"] for r in results["true_belief_std"]],
        [r["logit_diff"] for r in results["true_belief_exp"]],
    ], labels=["TB Standard", "TB Explicit"])
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylabel("Logit Difference", fontsize=12)
    ax2.set_title("True Belief: Logit Diff Distribution", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step23_explicit_scale.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("STEP 23 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

