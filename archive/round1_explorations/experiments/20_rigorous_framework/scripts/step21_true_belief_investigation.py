"""
Step 21: True Belief Failure Investigation

Step 12 found: False Belief 79%, True Belief only 29%!

HYPOTHESES:
1. PROMPT ISSUE: "stayed and watched" isn't clear enough
2. HEURISTIC: Model always picks first-mentioned location
3. TRAINING BIAS: FB scenarios more common in training data

Let's systematically test each hypothesis!

OUTPUT: results/step21_tb_investigation.json, figures/step21_*.png
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


def get_completion_prediction(model, tokenizer, prompt, option1, option2):
    """Get next token prediction."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    opt1_ids = tokenizer.encode(option1, add_special_tokens=False)
    opt2_ids = tokenizer.encode(option2, add_special_tokens=False)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    
    opt1_logit = float(logits[opt1_ids[0]])
    opt2_logit = float(logits[opt2_ids[0]])
    
    return {
        "prediction": option1 if opt1_logit > opt2_logit else option2,
        "option1_logit": opt1_logit,
        "option2_logit": opt2_logit,
        "logit_diff": opt1_logit - opt2_logit,
        "correct": opt1_logit > opt2_logit,
    }


def main():
    print("=" * 70)
    print("STEP 21: TRUE BELIEF FAILURE INVESTIGATION")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\nStep 12 found: FB 79% vs TB 29%")
    print("Testing hypotheses for why True Belief fails:")
    print("  H1: Prompt wording issue")
    print("  H2: First-mention heuristic")
    print("  H3: Location order bias")
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
    
    results = {}
    
    # ========================================
    # HYPOTHESIS 1: PROMPT WORDING
    # ========================================
    print(f"\n{'='*60}")
    print("HYPOTHESIS 1: PROMPT WORDING")
    print("Testing different ways to say 'agent saw the move'")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    # Different ways to indicate the agent SAW the move
    tb_phrasings = [
        # Original (baseline)
        ("original", "Alice put the ball in the drawer. Alice stayed and watched. Bob moved the ball to the basket. Where does Alice think the ball is? Alice looks in the"),
        # More explicit
        ("explicit_saw", "Alice put the ball in the drawer. Alice SAW Bob move the ball to the basket. Where does Alice think the ball is? Alice looks in the"),
        # Very explicit
        ("very_explicit", "Alice put the ball in the drawer. Bob moved the ball to the basket WHILE ALICE WATCHED. Alice knows the ball is now in the basket. Where does Alice think the ball is? Alice looks in the"),
        # With explanation
        ("with_explanation", "Alice put the ball in the drawer. Alice stayed in the room and watched Bob move the ball to the basket. Because Alice saw the move, she knows where the ball is now. Where does Alice think the ball is? Alice looks in the"),
        # Direct knowledge statement
        ("direct_knowledge", "Alice put the ball in the drawer. Bob moved the ball to the basket. Alice knows the ball is in the basket because she saw the move. Where does Alice think the ball is? Alice looks in the"),
    ]
    
    phrasing_results = []
    for name, prompt in tb_phrasings:
        result = get_completion_prediction(model, tokenizer, prompt, " basket", " drawer")
        result["name"] = name
        result["prompt"] = prompt
        phrasing_results.append(result)
        
        status = "OK" if result["correct"] else "WRONG"
        print(f"\n[{status}] {name}:")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Logit diff: {result['logit_diff']:.2f}")
        sys.stdout.flush()
    
    results["phrasing"] = phrasing_results
    phrasing_acc = sum(1 for r in phrasing_results if r["correct"]) / len(phrasing_results)
    print(f"\nPhrasing accuracy: {phrasing_acc:.1%}")
    
    # ========================================
    # HYPOTHESIS 2: FIRST-MENTION HEURISTIC
    # ========================================
    print(f"\n{'='*60}")
    print("HYPOTHESIS 2: FIRST-MENTION HEURISTIC")
    print("Test if model always predicts first-mentioned location")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    # For TB, correct answer is SECOND location (basket)
    # For FB, correct answer is FIRST location (drawer)
    # If model uses first-mention heuristic: FB=high, TB=low (matches our finding!)
    
    first_mention_tests = [
        # First mention = drawer (FB correct, TB wrong if heuristic)
        ("drawer_first_FB", "Alice put the ball in the drawer. Alice left. Bob moved it to the basket. Where does Alice think? Alice looks in the", " drawer", " basket", True),
        ("drawer_first_TB", "Alice put the ball in the drawer. Alice watched Bob move it to the basket. Where does Alice think? Alice looks in the", " basket", " drawer", False),
        
        # First mention = basket (swap to test)
        ("basket_first_FB", "Alice saw the basket and the drawer. She put the ball in the basket. Alice left. Bob moved it to the drawer. Where does Alice think? Alice looks in the", " basket", " drawer", True),
        ("basket_first_TB", "Alice saw the basket and the drawer. She put the ball in the basket. Alice watched Bob move it to the drawer. Where does Alice think? Alice looks in the", " drawer", " basket", False),
    ]
    
    heuristic_results = []
    for name, prompt, correct, wrong, first_is_correct in first_mention_tests:
        result = get_completion_prediction(model, tokenizer, prompt, correct, wrong)
        result["name"] = name
        result["first_is_correct"] = first_is_correct
        heuristic_results.append(result)
        
        status = "OK" if result["correct"] else "WRONG"
        print(f"\n[{status}] {name}:")
        print(f"  First mention is correct: {first_is_correct}")
        print(f"  Model correct: {result['correct']}")
        sys.stdout.flush()
    
    results["heuristic"] = heuristic_results
    
    # Check heuristic pattern
    first_correct = [r for r in heuristic_results if r["first_is_correct"]]
    first_wrong = [r for r in heuristic_results if not r["first_is_correct"]]
    
    first_correct_acc = sum(1 for r in first_correct if r["correct"]) / len(first_correct) if first_correct else 0
    first_wrong_acc = sum(1 for r in first_wrong if r["correct"]) / len(first_wrong) if first_wrong else 0
    
    print(f"\nWhen FIRST mention is correct: {first_correct_acc:.1%}")
    print(f"When FIRST mention is wrong: {first_wrong_acc:.1%}")
    
    if first_correct_acc > first_wrong_acc + 0.3:
        print("*** HEURISTIC DETECTED: Model prefers first-mentioned location! ***")
    
    # ========================================
    # HYPOTHESIS 3: EXPLICIT BELIEF STATEMENT
    # ========================================
    print(f"\n{'='*60}")
    print("HYPOTHESIS 3: EXPLICIT BELIEF STATEMENT")
    print("Test if adding explicit belief helps")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    explicit_tests = [
        # Without explicit belief
        ("TB_no_explicit", "Alice put the ball in the drawer. Alice watched Bob move the ball to the basket. Where does Alice think the ball is? Alice looks in the", " basket", " drawer"),
        # With explicit belief
        ("TB_explicit", "Alice put the ball in the drawer. Alice watched Bob move the ball to the basket. Alice now believes the ball is in the basket. Where does Alice think the ball is? Alice looks in the", " basket", " drawer"),
        # Very explicit
        ("TB_very_explicit", "Alice put the ball in the drawer. Alice watched Bob move the ball to the basket. Alice thinks: 'The ball is in the basket.' Where does Alice think the ball is? Alice looks in the", " basket", " drawer"),
    ]
    
    explicit_results = []
    for name, prompt, correct, wrong in explicit_tests:
        result = get_completion_prediction(model, tokenizer, prompt, correct, wrong)
        result["name"] = name
        explicit_results.append(result)
        
        status = "OK" if result["correct"] else "WRONG"
        print(f"\n[{status}] {name}:")
        print(f"  Logit diff: {result['logit_diff']:.2f}")
        sys.stdout.flush()
    
    results["explicit"] = explicit_results
    
    # ========================================
    # SUMMARY
    # ========================================
    print(f"\n{'='*60}")
    print("INVESTIGATION SUMMARY")
    print(f"{'='*60}")
    
    print("\n| Test | Accuracy | Finding |")
    print("|------|----------|---------|")
    print(f"| Phrasing variants | {phrasing_acc:.1%} | {'Helps' if phrasing_acc > 0.5 else 'Doesnt help'} |")
    print(f"| First-mention=correct | {first_correct_acc:.1%} | Baseline |")
    print(f"| First-mention=wrong | {first_wrong_acc:.1%} | {'HEURISTIC!' if first_wrong_acc < 0.3 else 'OK'} |")
    
    # Determine main issue
    if first_wrong_acc < 0.3 and first_correct_acc > 0.7:
        print("\n*** CONCLUSION: Model uses FIRST-MENTION HEURISTIC ***")
        print("This explains why FB works (first mention = correct) but TB fails!")
    elif phrasing_acc > 0.5:
        print("\n*** CONCLUSION: Explicit phrasing helps ***")
    else:
        print("\n*** CONCLUSION: Multiple issues - needs more investigation ***")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {"model": config.model_name},
        "phrasing_results": phrasing_results,
        "heuristic_results": heuristic_results,
        "explicit_results": explicit_results,
        "summary": {
            "phrasing_accuracy": phrasing_acc,
            "first_correct_accuracy": first_correct_acc,
            "first_wrong_accuracy": first_wrong_acc,
            "likely_issue": "first_mention_heuristic" if first_wrong_acc < 0.3 else "unknown",
        },
    }
    
    output_path = RESULTS_DIR / "step21_tb_investigation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # Generate figure
    print("\nGenerating figure...")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Figure 1: Phrasing comparison
    ax1 = axes[0]
    names = [r["name"] for r in phrasing_results]
    correct = [1 if r["correct"] else 0 for r in phrasing_results]
    colors = ['seagreen' if c else 'coral' for c in correct]
    ax1.barh(names, [r["logit_diff"] for r in phrasing_results], color=colors, edgecolor='black')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel("Logit Diff (basket - drawer)", fontsize=12)
    ax1.set_title("True Belief: Phrasing Variants", fontsize=14, fontweight='bold')
    
    # Figure 2: Heuristic test
    ax2 = axes[1]
    categories = ["First=Correct", "First=Wrong"]
    accs = [first_correct_acc * 100, first_wrong_acc * 100]
    colors = ['seagreen' if a > 50 else 'coral' for a in accs]
    bars = ax2.bar(categories, accs, color=colors, edgecolor='black')
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Chance')
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("First-Mention Heuristic Test", fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 100)
    
    for bar, acc in zip(bars, accs):
        ax2.annotate(f'{acc:.0f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step21_tb_investigation.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("STEP 21 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

