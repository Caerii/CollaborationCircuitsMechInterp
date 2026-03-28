"""
Step 33: Minimal Pair Analysis - WHY Does the Simple Format Fail?

We know:
- Working:  "Alice returned. Alice searched in the"  
- Failing:  "Alice returns. Alice thinks the ball is in the"

This script systematically varies ONE element at a time to find
the critical difference that causes failure.

Hypotheses to test:
1. Tense: "returned" vs "returns"
2. Action verb: "searched" vs "thinks"
3. Explicit object mention: "the ball" vs no mention
4. Question framing: implicit vs explicit
5. Narrative length/detail
"""

import torch
import json
import sys
import io
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"


def load_model():
    print("Loading Qwen3-4B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model.eval()
    return model, tokenizer


# Base scenario (the story part stays constant)
BASE_STORY = """Alice put the ball in the drawer. Alice then left the room.
While Alice was away, Bob told Carol that he moved the ball to the basket."""

# Minimal pair completions - varying ONE thing at a time
MINIMAL_PAIRS = {
    # ========== BASELINE COMPARISONS ==========
    "working_baseline": {
        "completion": "When Alice returned, Alice looked for the ball. Alice searched in the",
        "hypothesis": "Full working format (control)"
    },
    "failing_baseline": {
        "completion": "Alice returns. Alice thinks the ball is in the",
        "hypothesis": "Minimal failing format (control)"
    },
    
    # ========== HYPOTHESIS 1: TENSE ==========
    "tense_past_thinks": {
        "completion": "Alice returned. Alice thinks the ball is in the",
        "hypothesis": "Past tense + thinks"
    },
    "tense_present_searched": {
        "completion": "Alice returns. Alice searches in the",
        "hypothesis": "Present tense + searches"
    },
    "tense_present_looked": {
        "completion": "Alice returns. Alice looks in the",
        "hypothesis": "Present tense + looks"
    },
    
    # ========== HYPOTHESIS 2: ACTION VERB ==========
    "verb_searched": {
        "completion": "Alice returns. Alice searched in the",
        "hypothesis": "Returns + searched (mixed tense)"
    },
    "verb_looks_for": {
        "completion": "Alice returns. Alice looks for the ball in the",
        "hypothesis": "Returns + looks for the ball"
    },
    "verb_will_look": {
        "completion": "Alice returns. Alice will look in the",
        "hypothesis": "Returns + will look"
    },
    "verb_believes": {
        "completion": "Alice returns. Alice believes the ball is in the",
        "hypothesis": "Returns + believes"
    },
    "verb_expects": {
        "completion": "Alice returns. Alice expects to find the ball in the",
        "hypothesis": "Returns + expects"
    },
    
    # ========== HYPOTHESIS 3: EXPLICIT OBJECT ==========
    "object_no_mention": {
        "completion": "Alice returns. Alice thinks it is in the",
        "hypothesis": "No explicit 'ball' mention"
    },
    "object_with_ball_searched": {
        "completion": "Alice returns. Alice searched for the ball in the",
        "hypothesis": "Returns + searched for the ball"
    },
    
    # ========== HYPOTHESIS 4: NARRATIVE DETAIL ==========
    "detail_minimal": {
        "completion": "Alice returns. Ball location:",
        "hypothesis": "Ultra-minimal (just label)"
    },
    "detail_with_when": {
        "completion": "When Alice returned, Alice thinks the ball is in the",
        "hypothesis": "Add 'When' + thinks"
    },
    "detail_full_thinks": {
        "completion": "When Alice returned, Alice looked for the ball. Alice thinks it is in the",
        "hypothesis": "Full narrative + thinks"
    },
    "detail_question": {
        "completion": "Alice returns. Where does Alice think the ball is? In the",
        "hypothesis": "Explicit question format"
    },
    
    # ========== HYPOTHESIS 5: PUNCTUATION/STRUCTURE ==========
    "structure_comma": {
        "completion": "Alice returns, and thinks the ball is in the",
        "hypothesis": "Comma instead of period"
    },
    "structure_semicolon": {
        "completion": "Alice returns; she thinks the ball is in the",
        "hypothesis": "Semicolon + pronoun"
    },
    "structure_newline": {
        "completion": "Alice returns.\nAlice thinks the ball is in the",
        "hypothesis": "Newline between sentences"
    },
    
    # ========== HYPOTHESIS 6: SUBJECT REFERENCE ==========
    "subject_she": {
        "completion": "Alice returns. She thinks the ball is in the",
        "hypothesis": "Pronoun 'she' instead of 'Alice'"
    },
    "subject_alice_believes": {
        "completion": "Alice returned. Alice believes it's in the",
        "hypothesis": "Past + believes + contraction"
    },
    
    # ========== HYPOTHESIS 7: COGNITIVE VERB ALTERNATIVES ==========
    "cognitive_remembers": {
        "completion": "Alice returns. Alice remembers the ball being in the",
        "hypothesis": "Memory-based verb"
    },
    "cognitive_knows": {
        "completion": "Alice returns. Alice knows the ball is in the",
        "hypothesis": "'Knows' (factive verb)"
    },
    "cognitive_assumes": {
        "completion": "Alice returns. Alice assumes the ball is in the",
        "hypothesis": "'Assumes' verb"
    },
}


def test_prompt(model, tokenizer, story, completion):
    """Test a prompt and return detailed results."""
    prompt = story + "\n" + completion
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits[0, -1, :]
    
    # Get drawer and basket logits
    drawer_id = tokenizer.encode(" drawer", add_special_tokens=False)[0]
    basket_id = tokenizer.encode(" basket", add_special_tokens=False)[0]
    
    drawer_logit = logits[drawer_id].item()
    basket_logit = logits[basket_id].item()
    
    # Top 5 predictions
    top_k = torch.topk(logits, k=5)
    top_tokens = [tokenizer.decode([t]).strip() for t in top_k.indices.tolist()]
    top_logits = top_k.values.tolist()
    
    return {
        "drawer": drawer_logit,
        "basket": basket_logit,
        "diff": drawer_logit - basket_logit,
        "correct": drawer_logit > basket_logit,
        "top_5": top_tokens,
        "top_logits": top_logits[:5]
    }


def run_minimal_pair_analysis():
    """Run all minimal pair tests."""
    model, tokenizer = load_model()
    
    results = {}
    
    print("\n" + "="*80)
    print("MINIMAL PAIR ANALYSIS: Finding the Critical Difference")
    print("="*80)
    print(f"\nBase story (constant):\n{BASE_STORY}")
    print("\n" + "-"*80)
    
    # Group results by hypothesis
    working = []
    failing = []
    
    for name, config in MINIMAL_PAIRS.items():
        result = test_prompt(model, tokenizer, BASE_STORY, config["completion"])
        
        status = "[OK]" if result["correct"] else "[FAIL]"
        print(f"\n{status} {name}")
        print(f"  Hypothesis: {config['hypothesis']}")
        print(f"  Completion: \"{config['completion'][-50:]}...\"")
        print(f"  Diff: {result['diff']:+.2f} | Top: {result['top_5'][:3]}")
        
        results[name] = {
            **config,
            **result
        }
        
        if result["correct"]:
            working.append((name, result["diff"]))
        else:
            failing.append((name, result["diff"]))
    
    # Analysis summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\n✓ WORKING ({len(working)}/{len(MINIMAL_PAIRS)}):")
    working.sort(key=lambda x: x[1], reverse=True)
    for name, diff in working:
        print(f"  {name}: diff={diff:+.2f}")
    
    print(f"\n✗ FAILING ({len(failing)}/{len(MINIMAL_PAIRS)}):")
    failing.sort(key=lambda x: x[1], reverse=True)
    for name, diff in failing:
        print(f"  {name}: diff={diff:+.2f}")
    
    # Identify the critical factor
    print("\n" + "="*80)
    print("CRITICAL FACTOR IDENTIFICATION")
    print("="*80)
    
    # Check each hypothesis
    hypotheses = {
        "TENSE": ["tense_past_thinks", "tense_present_searched", "tense_present_looked"],
        "VERB": ["verb_searched", "verb_looks_for", "verb_will_look", "verb_believes", "verb_expects"],
        "DETAIL": ["detail_minimal", "detail_with_when", "detail_full_thinks", "detail_question"],
        "STRUCTURE": ["structure_comma", "structure_semicolon", "structure_newline"],
        "SUBJECT": ["subject_she", "subject_alice_believes"],
        "COGNITIVE": ["cognitive_remembers", "cognitive_knows", "cognitive_assumes"]
    }
    
    for hyp_name, variants in hypotheses.items():
        correct_count = sum(1 for v in variants if results.get(v, {}).get("correct", False))
        total = len([v for v in variants if v in results])
        if total > 0:
            print(f"\n{hyp_name}: {correct_count}/{total} variants work")
            for v in variants:
                if v in results:
                    r = results[v]
                    status = "[OK]" if r["correct"] else "[FAIL]"
                    print(f"  {status} {v}: diff={r['diff']:+.2f}")
    
    # Save results
    save_path = RESULTS_DIR / "minimal_pairs_results.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {save_path}")
    
    return results


def main():
    print("="*80)
    print("STEP 33: Minimal Pair Analysis")
    print("="*80)
    print("\nGoal: Find the EXACT factor that causes ToM failure")
    
    results = run_minimal_pair_analysis()


if __name__ == "__main__":
    main()


