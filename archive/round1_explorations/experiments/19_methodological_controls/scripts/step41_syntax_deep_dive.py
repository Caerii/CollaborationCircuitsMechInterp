"""
Step 41: Deep Dive on Syntax

Investigate exactly which syntactic features cause ToM failure.

Key question: Why does "thinks it is in the" fail but "believes it to be in the" succeed?

Syntactic differences to test:
1. Embedded clause ("that X is in") vs. infinitive ("to be in")
2. Pronoun reference ("it" vs. explicit object)
3. Copula presence ("is" vs. "be")
4. Complementizer ("that" explicit vs. implicit)
5. Verb argument structure
"""

import torch
import json
import sys
import io
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RESULTS_DIR = Path(__file__).parent.parent / "results"

# BASE STORY (constant across all tests)
BASE_STORY = """Alice put the ball in the drawer. Alice left the room.
Someone told Bob that they moved the ball to the basket.
Alice returns. Alice"""

CORRECT = "drawer"
WRONG = "basket"


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


def test_completion(model, tokenizer, completion):
    """Test a specific completion and return logit diff."""
    prompt = BASE_STORY + " " + completion
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
    
    correct_logit = get_logit(CORRECT)
    wrong_logit = get_logit(WRONG)
    diff = correct_logit - wrong_logit
    
    return {
        "diff": diff,
        "correct": diff > 0,
        "correct_logit": correct_logit,
        "wrong_logit": wrong_logit
    }


def run_syntax_deep_dive():
    """Run comprehensive syntax analysis."""
    print("="*70)
    print("STEP 41: Deep Dive on Syntax")
    print("="*70)
    
    model, tokenizer = load_model()
    
    all_results = {}
    
    # ================================================================
    # TEST 1: Clause Type
    # ================================================================
    print("\n" + "="*70)
    print("TEST 1: CLAUSE TYPE")
    print("="*70)
    print("Comparing different clause structures with SAME verb root 'think/believe'\n")
    
    clause_tests = {
        # Finite clauses (tensed verb)
        "finite_that_explicit": "thinks that the ball is in the",
        "finite_that_implicit": "thinks the ball is in the",
        "finite_present": "thinks it is in the",
        
        # Non-finite clauses (infinitive)
        "infinitive_full": "thinks it to be in the",
        "infinitive_simple": "thinks of it in the",
        
        # Same patterns with "believe"
        "believe_finite_that": "believes that the ball is in the",
        "believe_finite_implicit": "believes the ball is in the",
        "believe_finite_it": "believes it is in the",
        "believe_infinitive": "believes it to be in the",
        
        # Gerund/progressive
        "thinks_being": "thinks of it as being in the",
    }
    
    print(f"{'Completion':<45} {'Diff':>8} {'Result':<8}")
    print("-"*65)
    
    clause_results = {}
    for name, completion in clause_tests.items():
        result = test_completion(model, tokenizer, completion)
        clause_results[name] = result
        status = "[OK]" if result["correct"] else "[FAIL]"
        print(f"{completion:<45} {result['diff']:>+7.2f} {status}")
    
    all_results["clause_type"] = clause_results
    
    # ================================================================
    # TEST 2: Pronoun vs. Explicit Reference
    # ================================================================
    print("\n" + "="*70)
    print("TEST 2: PRONOUN vs. EXPLICIT REFERENCE")
    print("="*70)
    print("Testing if pronoun 'it' causes issues\n")
    
    pronoun_tests = {
        "pronoun_it": "thinks it is in the",
        "explicit_ball": "thinks the ball is in the",
        "explicit_object": "thinks the object is in the",
        "demonstrative_that": "thinks that is in the",
        "no_object": "thinks in the",  # Weird but diagnostic
    }
    
    print(f"{'Completion':<45} {'Diff':>8} {'Result':<8}")
    print("-"*65)
    
    pronoun_results = {}
    for name, completion in pronoun_tests.items():
        result = test_completion(model, tokenizer, completion)
        pronoun_results[name] = result
        status = "[OK]" if result["correct"] else "[FAIL]"
        print(f"{completion:<45} {result['diff']:>+7.2f} {status}")
    
    all_results["pronoun_reference"] = pronoun_results
    
    # ================================================================
    # TEST 3: Copula and Tense
    # ================================================================
    print("\n" + "="*70)
    print("TEST 3: COPULA AND TENSE")
    print("="*70)
    print("Testing the role of 'is' vs 'be' vs 'was'\n")
    
    copula_tests = {
        "present_is": "thinks it is in the",
        "infinitive_be": "thinks it to be in the",
        "past_was": "thinks it was in the",
        "future_will_be": "thinks it will be in the",
        "subjunctive_were": "thinks it were in the",
        "no_copula": "thinks it in the",  # Ungrammatical but diagnostic
    }
    
    print(f"{'Completion':<45} {'Diff':>8} {'Result':<8}")
    print("-"*65)
    
    copula_results = {}
    for name, completion in copula_tests.items():
        result = test_completion(model, tokenizer, completion)
        copula_results[name] = result
        status = "[OK]" if result["correct"] else "[FAIL]"
        print(f"{completion:<45} {result['diff']:>+7.2f} {status}")
    
    all_results["copula_tense"] = copula_results
    
    # ================================================================
    # TEST 4: Word Order
    # ================================================================
    print("\n" + "="*70)
    print("TEST 4: WORD ORDER")
    print("="*70)
    print("Testing position of location reference\n")
    
    order_tests = {
        "standard_SVO": "thinks it is in the",
        "fronted_location": "thinks in the",
        "postposed_it": "thinks the ball in the",
        "cleft": "thinks what is there is in the",
        "topicalized": "thinks the location is the",
    }
    
    print(f"{'Completion':<45} {'Diff':>8} {'Result':<8}")
    print("-"*65)
    
    order_results = {}
    for name, completion in order_tests.items():
        result = test_completion(model, tokenizer, completion)
        order_results[name] = result
        status = "[OK]" if result["correct"] else "[FAIL]"
        print(f"{completion:<45} {result['diff']:>+7.2f} {status}")
    
    all_results["word_order"] = order_results
    
    # ================================================================
    # TEST 5: Action vs. State Framing (Controlled Syntax)
    # ================================================================
    print("\n" + "="*70)
    print("TEST 5: ACTION vs. STATE FRAMING (Syntax Controlled)")
    print("="*70)
    print("All use 'the ball in the' structure\n")
    
    framing_tests = {
        # Action framing
        "searches_for": "searches for the ball in the",
        "looks_for": "looks for the ball in the",
        "expects": "expects the ball in the",
        "remembers": "remembers the ball in the",
        "pictures": "pictures the ball in the",
        
        # State framing
        "believes": "believes the ball is in the",  # Has 'is'
        "assumes": "assumes the ball is in the",    # Has 'is'
        "knows": "knows the ball is in the",        # Has 'is'
        
        # State without 'is' (to isolate)
        "believes_no_is": "believes the ball to be in the",
        "assumes_no_is": "assumes the ball to be in the",
        "knows_no_is": "knows the ball to be in the",
    }
    
    print(f"{'Completion':<45} {'Diff':>8} {'Result':<8}")
    print("-"*65)
    
    framing_results = {}
    for name, completion in framing_tests.items():
        result = test_completion(model, tokenizer, completion)
        framing_results[name] = result
        status = "[OK]" if result["correct"] else "[FAIL]"
        print(f"{completion:<45} {result['diff']:>+7.2f} {status}")
    
    all_results["action_vs_state"] = framing_results
    
    # ================================================================
    # ANALYSIS
    # ================================================================
    print("\n" + "="*70)
    print("ANALYSIS: What Syntactic Features Predict Failure?")
    print("="*70)
    
    # Analyze patterns
    all_completions = []
    for category, results in all_results.items():
        for name, result in results.items():
            all_completions.append({
                "category": category,
                "name": name,
                "correct": result["correct"],
                "diff": result["diff"]
            })
    
    # Find patterns
    successes = [c for c in all_completions if c["correct"]]
    failures = [c for c in all_completions if not c["correct"]]
    
    print(f"\nOverall: {len(successes)} successes, {len(failures)} failures")
    
    print("\n--- FAILURES ---")
    for f in sorted(failures, key=lambda x: x["diff"]):
        print(f"  {f['name']}: diff={f['diff']:+.2f}")
    
    print("\n--- SUCCESSES (top 10) ---")
    for s in sorted(successes, key=lambda x: x["diff"], reverse=True)[:10]:
        print(f"  {s['name']}: diff={s['diff']:+.2f}")
    
    # Look for "is" pattern
    print("\n--- 'is' vs NO 'is' comparison ---")
    is_pattern_results = {}
    for name, result in all_results.get("action_vs_state", {}).items():
        if "no_is" in name or name in ["believes", "assumes", "knows"]:
            is_pattern_results[name] = result
    
    for name, result in is_pattern_results.items():
        status = "[OK]" if result["correct"] else "[FAIL]"
        has_is = "no_is" not in name and name in ["believes", "assumes", "knows"]
        is_marker = "(has 'is')" if has_is else "(no 'is')"
        print(f"  {name:<20} {is_marker:<12} {result['diff']:>+7.2f} {status}")
    
    # Save
    save_path = RESULTS_DIR / "syntax_deep_dive_results.json"
    
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, bool):
            return bool(obj)
        else:
            return obj
    
    with open(save_path, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\nSaved: {save_path}")
    
    return all_results


if __name__ == "__main__":
    run_syntax_deep_dive()


