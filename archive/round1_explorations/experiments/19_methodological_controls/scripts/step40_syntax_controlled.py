"""
Step 40: Syntax-Controlled Verb Comparison

ADDRESS CONFOUND: Previous comparison mixed verb type with syntax.

"searched in the" vs "thinks it is in the" differ in:
1. Verb type (our hypothesis)
2. Syntax structure (confound)
3. Length (confound)

THIS SCRIPT: Tests verbs with IDENTICAL syntax structure.

Controlled comparisons:
1. "[Agent] expects it to be in the ___" (action-like)
2. "[Agent] assumes it to be in the ___" (belief-like)

Both have identical syntax but different semantic frames.
"""

import torch
import json
import sys
import io
import random
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path(__file__).parent.parent / "results"

# SYNTAX-MATCHED VERB PAIRS
# Same structure: "[Agent] [VERB] it to be in the"
SYNTAX_MATCHED_TESTS = {
    "action_frame": [
        "expects it to be in the",    # Expectation-based
        "remembers it being in the",  # Memory-based
        "pictures it in the",         # Mental action
    ],
    "belief_frame": [
        "believes it to be in the",   # Belief-based
        "assumes it to be in the",    # Assumption-based  
        "considers it to be in the",  # Consideration
    ]
}

# ADDITIONAL CONTROLS
# Same verb, different tense
TENSE_CONTROLS = {
    "present": "looks for it in the",
    "past": "looked for it in the", 
    "future": "will look for it in the"
}

# Same structure, minimal verb change
MINIMAL_PAIRS = [
    ("expects the ball in the", "suspects the ball is in the"),      # expect/suspect
    ("remembers it in the", "forgets it is in the"),                  # remember/forget  
    ("sees it in the", "imagines it is in the"),                      # see/imagine
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


def create_prompt(agent1, agent2, obj, loc1, loc2, verb_completion):
    """Create standardized prompt with verb completion."""
    return f"""{agent1} put the {obj} in the {loc1}. {agent1} left.
Someone told {agent2} that they moved the {obj} to the {loc2}.
{agent1} returns. {agent1} {verb_completion}"""


def test_verb(model, tokenizer, verb_completion):
    """Test a verb completion across multiple scenarios."""
    scenarios = [
        ("Alice", "Bob", "ball", "drawer", "basket"),
        ("Carol", "David", "book", "shelf", "desk"),
        ("Emma", "Frank", "key", "pocket", "bag"),
        ("Grace", "Henry", "cup", "table", "cupboard"),
        ("Iris", "Jack", "toy", "box", "bin"),
    ]
    
    results = []
    for agent1, agent2, obj, loc1, loc2 in scenarios:
        prompt = create_prompt(agent1, agent2, obj, loc1, loc2, verb_completion)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits[0, -1, :]
        
        def get_logit(word):
            for prefix in [" ", ""]:
                tokens = tokenizer.encode(prefix + word, add_special_tokens=False)
                if tokens:
                    return logits[tokens[0]].item()
            return float('-inf')
        
        loc1_logit = get_logit(loc1)
        loc2_logit = get_logit(loc2)
        diff = loc1_logit - loc2_logit
        
        results.append({
            "correct": diff > 0,
            "diff": diff
        })
    
    accuracy = sum(r["correct"] for r in results) / len(results)
    mean_diff = np.mean([r["diff"] for r in results])
    
    return {
        "accuracy": accuracy,
        "mean_diff": mean_diff,
        "n": len(results),
        "results": results
    }


def run_syntax_controlled_test():
    """Run syntax-controlled verb comparison."""
    print("="*70)
    print("STEP 40: Syntax-Controlled Verb Comparison")
    print("="*70)
    
    model, tokenizer = load_model()
    
    all_results = {}
    
    # TEST 1: Syntax-matched verbs
    print("\n" + "-"*70)
    print("TEST 1: SYNTAX-MATCHED VERBS")
    print("-"*70)
    print("Testing verbs with IDENTICAL syntactic structure")
    print("Structure: '[Agent] [VERB] it to be in the'\n")
    
    print(f"{'Frame':<15} {'Verb Completion':<35} {'Acc':<8} {'Diff':<8}")
    print("-"*66)
    
    for frame, verbs in SYNTAX_MATCHED_TESTS.items():
        frame_results = {}
        for verb in verbs:
            result = test_verb(model, tokenizer, verb)
            frame_results[verb] = result
            status = "[OK]" if result["accuracy"] > 0.5 else "[FAIL]"
            print(f"{frame:<15} {verb:<35} {result['accuracy']*100:>5.0f}% {result['mean_diff']:>+6.2f} {status}")
        
        all_results[frame] = frame_results
    
    # Analyze frame differences
    action_accs = [r["accuracy"] for r in all_results["action_frame"].values()]
    belief_accs = [r["accuracy"] for r in all_results["belief_frame"].values()]
    
    print(f"\nAction frame mean: {np.mean(action_accs)*100:.1f}%")
    print(f"Belief frame mean: {np.mean(belief_accs)*100:.1f}%")
    print(f"Difference: {(np.mean(action_accs) - np.mean(belief_accs))*100:.1f}%")
    
    # TEST 2: Tense controls
    print("\n" + "-"*70)
    print("TEST 2: TENSE CONTROLS")
    print("-"*70)
    print("Testing same verb across tenses\n")
    
    print(f"{'Tense':<15} {'Verb Completion':<35} {'Acc':<8} {'Diff':<8}")
    print("-"*66)
    
    tense_results = {}
    for tense, verb in TENSE_CONTROLS.items():
        result = test_verb(model, tokenizer, verb)
        tense_results[tense] = result
        status = "[OK]" if result["accuracy"] > 0.5 else "[FAIL]"
        print(f"{tense:<15} {verb:<35} {result['accuracy']*100:>5.0f}% {result['mean_diff']:>+6.2f} {status}")
    
    all_results["tense_controls"] = tense_results
    
    # TEST 3: Minimal pairs
    print("\n" + "-"*70)
    print("TEST 3: MINIMAL VERB PAIRS")
    print("-"*70)
    print("Testing verbs that differ minimally\n")
    
    print(f"{'Verb A':<30} {'Acc A':<8} {'Verb B':<30} {'Acc B':<8} {'Diff'}")
    print("-"*86)
    
    pair_results = []
    for verb_a, verb_b in MINIMAL_PAIRS:
        result_a = test_verb(model, tokenizer, verb_a)
        result_b = test_verb(model, tokenizer, verb_b)
        
        diff = result_a["accuracy"] - result_b["accuracy"]
        pair_results.append({
            "verb_a": verb_a,
            "verb_b": verb_b,
            "acc_a": result_a["accuracy"],
            "acc_b": result_b["accuracy"],
            "diff": diff
        })
        
        status_a = "[OK]" if result_a["accuracy"] > 0.5 else "[FAIL]"
        status_b = "[OK]" if result_b["accuracy"] > 0.5 else "[FAIL]"
        print(f"{verb_a:<30} {result_a['accuracy']*100:>5.0f}% {status_a}  "
              f"{verb_b:<30} {result_b['accuracy']*100:>5.0f}% {status_b}  {diff*100:+.0f}%")
    
    all_results["minimal_pairs"] = pair_results
    
    # SUMMARY
    print("\n" + "="*70)
    print("SUMMARY: Is it VERB TYPE or SYNTAX?")
    print("="*70)
    
    # Check if syntax-controlled comparison shows same pattern
    action_mean = np.mean(action_accs)
    belief_mean = np.mean(belief_accs)
    
    if abs(action_mean - belief_mean) > 0.2:
        print(f"\n[CONFIRMED] Effect persists with controlled syntax!")
        print(f"   Action-frame verbs: {action_mean*100:.1f}%")
        print(f"   Belief-frame verbs: {belief_mean*100:.1f}%")
        print(f"   This suggests SEMANTIC FRAME matters, not just syntax.")
    else:
        print(f"\n[REVISED] Effect disappears with controlled syntax!")
        print(f"   Action-frame verbs: {action_mean*100:.1f}%")
        print(f"   Belief-frame verbs: {belief_mean*100:.1f}%")
        print(f"   Original findings may have been confounded by syntax.")
    
    # Save
    save_path = RESULTS_DIR / "syntax_controlled_results.json"
    
    # Convert to serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            return obj
    
    with open(save_path, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\nSaved: {save_path}")
    
    return all_results


if __name__ == "__main__":
    run_syntax_controlled_test()


