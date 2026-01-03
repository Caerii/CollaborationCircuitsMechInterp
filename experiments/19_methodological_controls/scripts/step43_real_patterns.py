"""
Step 43: Investigate Real Patterns

From our controlled tests, we found REAL effects:
1. "expects" (80%) vs "suspects" (0%) - huge difference!
2. "remembers" (100%) vs "forgets" (0%) - memory asymmetry
3. Future tense "will look" (20%) - tense matters

This script investigates these patterns with proper methodology.
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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Real patterns to investigate
VERB_PAIRS = {
    "expect_suspect": {
        "verb_a": "expects the ball in the",
        "verb_b": "suspects the ball is in the",
        "hypothesis": "Expectation vs suspicion framing"
    },
    "remember_forget": {
        "verb_a": "remembers the ball in the",
        "verb_b": "forgets the ball is in the",  
        "hypothesis": "Memory access vs memory loss framing"
    },
    "see_imagine": {
        "verb_a": "sees the ball in the",
        "verb_b": "imagines the ball in the",
        "hypothesis": "Perception vs imagination"
    },
    "know_wonder": {
        "verb_a": "knows the ball is in the",
        "verb_b": "wonders if the ball is in the",
        "hypothesis": "Certainty vs uncertainty"
    },
    "find_lose": {
        "verb_a": "will find the ball in the",
        "verb_b": "will lose the ball in the",
        "hypothesis": "Success vs failure framing"
    }
}

TENSE_TESTS = {
    "past": "looked for the ball in the",
    "present": "looks for the ball in the",
    "future": "will look for the ball in the",
    "perfect": "has looked for the ball in the",
    "progressive": "is looking for the ball in the",
}

# Diverse scenarios
SCENARIOS = [
    {"agent": "Alice", "obj": "ball", "loc1": "drawer", "loc2": "basket"},
    {"agent": "Bob", "obj": "book", "loc1": "shelf", "loc2": "desk"},
    {"agent": "Carol", "obj": "cup", "loc1": "table", "loc2": "cupboard"},
    {"agent": "David", "obj": "key", "loc1": "pocket", "loc2": "bag"},
    {"agent": "Emma", "obj": "toy", "loc1": "box", "loc2": "bin"},
    {"agent": "Frank", "obj": "hat", "loc1": "closet", "loc2": "chair"},
    {"agent": "Grace", "obj": "ring", "loc1": "drawer", "loc2": "box"},
    {"agent": "Henry", "obj": "pen", "loc1": "desk", "loc2": "bag"},
    {"agent": "Iris", "obj": "watch", "loc1": "nightstand", "loc2": "dresser"},
    {"agent": "Jack", "obj": "wallet", "loc1": "coat", "loc2": "table"},
]


def load_model():
    print("Loading Qwen3-4B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model.eval()
    return model, tokenizer


def wilson_ci(successes, total):
    if total == 0:
        return 0, 0, 1
    p = successes / total
    z = 1.96
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    return p, max(0, center - spread), min(1, center + spread)


def test_verb(model, tokenizer, verb_completion, scenarios):
    """Test a verb across all scenarios."""
    results = []
    
    for scenario in scenarios:
        story = f"""{scenario['agent']} put the {scenario['obj']} in the {scenario['loc1']}. {scenario['agent']} left.
Someone moved the {scenario['obj']} to the {scenario['loc2']} while {scenario['agent']} was away.
{scenario['agent']} returns. {scenario['agent']} {verb_completion}"""
        
        inputs = tokenizer(story, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits[0, -1, :]
        
        def get_logit(word):
            for prefix in [" ", ""]:
                tokens = tokenizer.encode(prefix + word, add_special_tokens=False)
                if tokens:
                    return logits[tokens[0]].item()
            return float('-inf')
        
        correct_logit = get_logit(scenario["loc1"])
        wrong_logit = get_logit(scenario["loc2"])
        diff = correct_logit - wrong_logit
        
        results.append({
            "correct": diff > 0,
            "diff": diff
        })
    
    n_correct = sum(r["correct"] for r in results)
    acc, ci_lo, ci_hi = wilson_ci(n_correct, len(results))
    mean_diff = np.mean([r["diff"] for r in results])
    
    return {
        "accuracy": float(acc),
        "ci_lower": float(ci_lo),
        "ci_upper": float(ci_hi),
        "n_correct": n_correct,
        "n_total": len(results),
        "mean_diff": float(mean_diff),
        "std_diff": float(np.std([r["diff"] for r in results]))
    }


def run_real_patterns():
    """Investigate real patterns with proper methodology."""
    print("="*70)
    print("STEP 43: Investigate Real Patterns")
    print("="*70)
    print(f"Seed: {SEED}")
    print(f"N scenarios: {len(SCENARIOS)}")
    
    model, tokenizer = load_model()
    
    all_results = {}
    
    # ================================================================
    # TEST 1: Verb Pairs
    # ================================================================
    print("\n" + "="*70)
    print("TEST 1: VERB PAIR COMPARISONS")
    print("="*70)
    
    pair_results = {}
    
    for pair_name, pair_data in VERB_PAIRS.items():
        print(f"\n--- {pair_name}: {pair_data['hypothesis']} ---")
        
        result_a = test_verb(model, tokenizer, pair_data["verb_a"], SCENARIOS)
        result_b = test_verb(model, tokenizer, pair_data["verb_b"], SCENARIOS)
        
        # Fisher's exact test
        table = [[result_a["n_correct"], result_a["n_total"] - result_a["n_correct"]],
                 [result_b["n_correct"], result_b["n_total"] - result_b["n_correct"]]]
        odds_ratio, p_value = stats.fisher_exact(table)
        
        status_a = "[OK]" if result_a["accuracy"] > 0.5 else "[FAIL]"
        status_b = "[OK]" if result_b["accuracy"] > 0.5 else "[FAIL]"
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        
        print(f"  Verb A: {result_a['accuracy']*100:.0f}% (CI: {result_a['ci_lower']*100:.0f}-{result_a['ci_upper']*100:.0f}%) {status_a}")
        print(f"  Verb B: {result_b['accuracy']*100:.0f}% (CI: {result_b['ci_lower']*100:.0f}-{result_b['ci_upper']*100:.0f}%) {status_b}")
        print(f"  Difference: {(result_a['accuracy'] - result_b['accuracy'])*100:+.0f}%, p={p_value:.4f} {sig}")
        
        pair_results[pair_name] = {
            "verb_a": pair_data["verb_a"],
            "verb_b": pair_data["verb_b"],
            "hypothesis": pair_data["hypothesis"],
            "result_a": result_a,
            "result_b": result_b,
            "p_value": float(p_value),
            "odds_ratio": float(odds_ratio),
            "significant": bool(p_value < 0.05)
        }
    
    all_results["verb_pairs"] = pair_results
    
    # ================================================================
    # TEST 2: Tense Effects
    # ================================================================
    print("\n" + "="*70)
    print("TEST 2: TENSE EFFECTS")
    print("="*70)
    
    tense_results = {}
    
    print(f"\n{'Tense':<15} {'Completion':<40} {'Acc':>6} {'CI':>15}")
    print("-"*80)
    
    for tense_name, completion in TENSE_TESTS.items():
        result = test_verb(model, tokenizer, completion, SCENARIOS)
        tense_results[tense_name] = result
        
        status = "[OK]" if result["accuracy"] > 0.5 else "[FAIL]"
        ci_str = f"[{result['ci_lower']*100:.0f}-{result['ci_upper']*100:.0f}%]"
        print(f"{tense_name:<15} {completion:<40} {result['accuracy']*100:>5.0f}% {ci_str:>15} {status}")
    
    all_results["tense"] = tense_results
    
    # Statistical comparison: past vs future
    past_result = tense_results["past"]
    future_result = tense_results["future"]
    table = [[past_result["n_correct"], past_result["n_total"] - past_result["n_correct"]],
             [future_result["n_correct"], future_result["n_total"] - future_result["n_correct"]]]
    _, p_tense = stats.fisher_exact(table)
    
    print(f"\nPast vs Future comparison: p={p_tense:.4f}")
    
    # ================================================================
    # TEST 3: Semantic Categories
    # ================================================================
    print("\n" + "="*70)
    print("TEST 3: SEMANTIC CATEGORIES")
    print("="*70)
    
    semantic_tests = {
        # Mental state verbs
        "thinks": "thinks the ball is in the",
        "believes": "believes the ball is in the",
        "knows": "knows the ball is in the",
        "assumes": "assumes the ball is in the",
        
        # Perceptual verbs
        "sees": "sees the ball in the",
        "perceives": "perceives the ball in the",
        
        # Memory verbs
        "remembers": "remembers the ball in the",
        "recalls": "recalls the ball being in the",
        
        # Action verbs
        "searches": "searches for the ball in the",
        "looks": "looks for the ball in the",
        "reaches": "reaches for the ball in the",
        "goes": "goes to the",
    }
    
    semantic_results = {}
    
    print(f"\n{'Category':<12} {'Verb':<12} {'Completion':<35} {'Acc':>6}")
    print("-"*70)
    
    for verb_name, completion in semantic_tests.items():
        result = test_verb(model, tokenizer, completion, SCENARIOS)
        semantic_results[verb_name] = result
        
        # Categorize
        if verb_name in ["thinks", "believes", "knows", "assumes"]:
            cat = "mental"
        elif verb_name in ["sees", "perceives"]:
            cat = "perceptual"
        elif verb_name in ["remembers", "recalls"]:
            cat = "memory"
        else:
            cat = "action"
        
        status = "[OK]" if result["accuracy"] > 0.5 else "[FAIL]"
        print(f"{cat:<12} {verb_name:<12} {completion:<35} {result['accuracy']*100:>5.0f}% {status}")
    
    all_results["semantic"] = semantic_results
    
    # Category averages
    print("\n--- Category Averages ---")
    categories = {
        "mental": ["thinks", "believes", "knows", "assumes"],
        "perceptual": ["sees", "perceives"],
        "memory": ["remembers", "recalls"],
        "action": ["searches", "looks", "reaches", "goes"]
    }
    
    for cat, verbs in categories.items():
        accs = [semantic_results[v]["accuracy"] for v in verbs if v in semantic_results]
        mean_acc = np.mean(accs) if accs else 0
        print(f"  {cat}: {mean_acc*100:.1f}% (n={len(accs)} verbs)")
    
    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "="*70)
    print("SUMMARY OF REAL PATTERNS")
    print("="*70)
    
    print("\n1. SIGNIFICANT VERB PAIR DIFFERENCES:")
    for pair_name, data in pair_results.items():
        if data["significant"]:
            diff = (data["result_a"]["accuracy"] - data["result_b"]["accuracy"]) * 100
            print(f"   {pair_name}: {diff:+.0f}% difference (p={data['p_value']:.4f})")
    
    print("\n2. TENSE EFFECTS:")
    best_tense = max(tense_results.items(), key=lambda x: x[1]["accuracy"])
    worst_tense = min(tense_results.items(), key=lambda x: x[1]["accuracy"])
    print(f"   Best: {best_tense[0]} ({best_tense[1]['accuracy']*100:.0f}%)")
    print(f"   Worst: {worst_tense[0]} ({worst_tense[1]['accuracy']*100:.0f}%)")
    
    print("\n3. SEMANTIC CATEGORY RANKING:")
    for cat, verbs in sorted(categories.items(), 
                             key=lambda x: np.mean([semantic_results.get(v, {}).get("accuracy", 0) for v in x[1]]),
                             reverse=True):
        accs = [semantic_results[v]["accuracy"] for v in verbs if v in semantic_results]
        mean_acc = np.mean(accs) if accs else 0
        print(f"   {cat}: {mean_acc*100:.1f}%")
    
    # Save
    save_path = RESULTS_DIR / "real_patterns_results.json"
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {save_path}")
    
    return all_results


if __name__ == "__main__":
    run_real_patterns()

