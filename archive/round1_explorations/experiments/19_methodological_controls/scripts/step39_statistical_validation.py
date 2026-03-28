"""
Step 39: Statistical Validation

ADDRESS METHODOLOGICAL ISSUES:
1. Small sample size → Test 50 scenarios per condition
2. No confidence intervals → Calculate Wilson intervals
3. No significance tests → Run Fisher's exact test

This script generates large-scale validation data.
"""

import torch
import json
import sys
import io
import random
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Diverse scenario elements
AGENTS = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", 
          "Iris", "Jack", "Kate", "Leo", "Mary", "Nick", "Olivia", "Paul"]

OBJECTS = ["ball", "book", "cup", "toy", "key", "wallet", "phone", "hat",
           "ring", "pen", "card", "coin", "watch", "glasses", "scarf", "bag"]

LOCATIONS = ["drawer", "basket", "box", "shelf", "table", "cupboard", "closet", "desk",
             "cabinet", "container", "bag", "pocket", "bowl", "tray", "bin", "case"]

MOVERS = ["someone", "another person", "a friend", "a neighbor", "a roommate"]

# Verb conditions (controlled)
ACTION_VERBS = ["searched in the", "looked in the", "will look in the", 
                "expects it in the", "went to the", "checked the"]

BELIEF_VERBS = ["thinks it is in the", "believes it is in the", 
                "assumes it is in the", "knows it is in the"]


def generate_scenario(agent1, agent2, obj, loc1, loc2, verb_completion, mover):
    """Generate a false belief scenario with controlled structure."""
    story = f"""{agent1} put the {obj} in the {loc1}. {agent1} left.
{mover} told {agent2} that they moved the {obj} to the {loc2}.
{agent1} returns. {agent1} {verb_completion}"""
    return story


def generate_n_scenarios(n, verb_type="action"):
    """Generate n unique scenarios for a verb type."""
    scenarios = []
    verbs = ACTION_VERBS if verb_type == "action" else BELIEF_VERBS
    
    used_combos = set()
    attempts = 0
    max_attempts = n * 10
    
    while len(scenarios) < n and attempts < max_attempts:
        attempts += 1
        
        # Pick random elements (ensure unique combos)
        agent1 = random.choice(AGENTS)
        agent2 = random.choice([a for a in AGENTS if a != agent1])
        obj = random.choice(OBJECTS)
        loc1 = random.choice(LOCATIONS)
        loc2 = random.choice([l for l in LOCATIONS if l != loc1])
        mover = random.choice(MOVERS)
        verb = random.choice(verbs)
        
        combo = (agent1, obj, loc1, loc2, verb)
        if combo in used_combos:
            continue
        used_combos.add(combo)
        
        scenario = generate_scenario(agent1, agent2, obj, loc1, loc2, verb, mover)
        scenarios.append({
            "prompt": scenario,
            "correct": loc1,  # Where agent1 last saw it
            "wrong": loc2,    # Where it actually is
            "verb": verb,
            "verb_type": verb_type
        })
    
    return scenarios


def wilson_ci(successes, total, confidence=0.95):
    """Calculate Wilson score confidence interval."""
    if total == 0:
        return 0, 0, 1
    
    p = successes / total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    
    lower = max(0, center - spread)
    upper = min(1, center + spread)
    
    return p, lower, upper


def test_scenarios(model, tokenizer, scenarios):
    """Test scenarios and return detailed results."""
    results = []
    
    for scenario in scenarios:
        inputs = tokenizer(scenario["prompt"], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits[0, -1, :]
        
        def get_logit(word):
            for prefix in [" ", ""]:
                tokens = tokenizer.encode(prefix + word, add_special_tokens=False)
                if tokens:
                    return logits[tokens[0]].item()
            return float('-inf')
        
        correct_logit = get_logit(scenario["correct"])
        wrong_logit = get_logit(scenario["wrong"])
        
        results.append({
            "correct": correct_logit > wrong_logit,
            "diff": correct_logit - wrong_logit,
            "verb_type": scenario["verb_type"],
            "verb": scenario["verb"]
        })
    
    return results


def run_statistical_validation():
    """Run large-scale statistical validation."""
    print("="*70)
    print("STEP 39: Statistical Validation")
    print("="*70)
    print(f"Random seed: {SEED}")
    
    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model.eval()
    
    # Generate scenarios
    N_PER_CONDITION = 50
    print(f"\nGenerating {N_PER_CONDITION} scenarios per condition...")
    
    action_scenarios = generate_n_scenarios(N_PER_CONDITION, "action")
    belief_scenarios = generate_n_scenarios(N_PER_CONDITION, "belief")
    
    print(f"  Action scenarios: {len(action_scenarios)}")
    print(f"  Belief scenarios: {len(belief_scenarios)}")
    
    # Test scenarios
    print("\nTesting scenarios...")
    action_results = test_scenarios(model, tokenizer, action_scenarios)
    belief_results = test_scenarios(model, tokenizer, belief_scenarios)
    
    # Calculate statistics
    action_correct = sum(1 for r in action_results if r["correct"])
    belief_correct = sum(1 for r in belief_results if r["correct"])
    
    action_n = len(action_results)
    belief_n = len(belief_results)
    
    # Wilson CIs
    action_acc, action_lo, action_hi = wilson_ci(action_correct, action_n)
    belief_acc, belief_lo, belief_hi = wilson_ci(belief_correct, belief_n)
    
    # Fisher's exact test
    table = [[action_correct, action_n - action_correct],
             [belief_correct, belief_n - belief_correct]]
    odds_ratio, p_value = stats.fisher_exact(table)
    
    # Effect size (Cohen's h)
    phi1 = 2 * np.arcsin(np.sqrt(action_acc))
    phi2 = 2 * np.arcsin(np.sqrt(belief_acc))
    cohens_h = abs(phi1 - phi2)
    
    # Print results
    print("\n" + "="*70)
    print("STATISTICAL RESULTS")
    print("="*70)
    
    print(f"\nAction verbs:")
    print(f"  Accuracy: {action_acc*100:.1f}% ({action_correct}/{action_n})")
    print(f"  95% CI: [{action_lo*100:.1f}%, {action_hi*100:.1f}%]")
    
    print(f"\nBelief verbs:")
    print(f"  Accuracy: {belief_acc*100:.1f}% ({belief_correct}/{belief_n})")
    print(f"  95% CI: [{belief_lo*100:.1f}%, {belief_hi*100:.1f}%]")
    
    print(f"\nStatistical comparison:")
    print(f"  Difference: {(action_acc - belief_acc)*100:.1f}%")
    print(f"  Fisher's exact p-value: {p_value:.4f}")
    print(f"  Odds ratio: {odds_ratio:.2f}")
    print(f"  Cohen's h (effect size): {cohens_h:.2f}")
    
    significance = "SIGNIFICANT" if p_value < 0.05 else "NOT SIGNIFICANT"
    effect_mag = "LARGE" if cohens_h > 0.8 else "MEDIUM" if cohens_h > 0.5 else "SMALL"
    
    print(f"\n  Conclusion: {significance} (p < 0.05)")
    print(f"  Effect size: {effect_mag}")
    
    # Logit difference analysis
    action_diffs = [r["diff"] for r in action_results]
    belief_diffs = [r["diff"] for r in belief_results]
    
    t_stat, t_pvalue = stats.ttest_ind(action_diffs, belief_diffs)
    
    print(f"\nLogit difference analysis:")
    print(f"  Action mean: {np.mean(action_diffs):.2f} (SD={np.std(action_diffs):.2f})")
    print(f"  Belief mean: {np.mean(belief_diffs):.2f} (SD={np.std(belief_diffs):.2f})")
    print(f"  t-test p-value: {t_pvalue:.4f}")
    
    # Save results
    results = {
        "seed": SEED,
        "n_per_condition": N_PER_CONDITION,
        "action": {
            "n": action_n,
            "correct": action_correct,
            "accuracy": action_acc,
            "ci_lower": action_lo,
            "ci_upper": action_hi,
            "mean_diff": np.mean(action_diffs),
            "std_diff": np.std(action_diffs)
        },
        "belief": {
            "n": belief_n,
            "correct": belief_correct,
            "accuracy": belief_acc,
            "ci_lower": belief_lo,
            "ci_upper": belief_hi,
            "mean_diff": np.mean(belief_diffs),
            "std_diff": np.std(belief_diffs)
        },
        "comparison": {
            "fisher_p": float(p_value),
            "odds_ratio": float(odds_ratio),
            "cohens_h": float(cohens_h),
            "ttest_p": float(t_pvalue),
            "significant": bool(p_value < 0.05)
        }
    }
    
    save_path = RESULTS_DIR / "statistical_validation_results.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {save_path}")
    
    return results


if __name__ == "__main__":
    run_statistical_validation()

