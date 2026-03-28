"""
Step 54: Controlled Test - Account for All Confounds

Key findings from step53:
1. STRONG location bias (drawer > basket, box > shelf)
2. Completion phrase MATTERS A LOT
3. Order effects are confounded with location bias

Let's control for everything.
"""

import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_model():
    print("Loading Qwen3-4B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    model.eval()
    return model, tokenizer


def get_probs(model, tokenizer, prompt, loc_a, loc_b):
    """Get probabilities for two locations."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    
    probs = torch.softmax(logits, dim=-1)
    
    a_ids = tokenizer.encode(" " + loc_a, add_special_tokens=False)
    b_ids = tokenizer.encode(" " + loc_b, add_special_tokens=False)
    
    a_prob = probs[a_ids[0]].item() if a_ids else 0
    b_prob = probs[b_ids[0]].item() if b_ids else 0
    
    return a_prob, b_prob


def test_controlled_tom(model, tokenizer):
    """
    CONTROLLED TEST: Use SAME locations, vary ONLY presence.
    Use counterbalanced location order to cancel bias.
    """
    print("\n" + "="*70)
    print("CONTROLLED ToM TEST")
    print("="*70)
    
    print("""
    Design:
    - Same two locations used throughout (drawer, basket)
    - COUNTERBALANCED: test both orderings
    - ONLY variable: whether agent saw/was told
    - Use "looks in the" (best performing completion)
    """)
    
    results = {
        "left": [],
        "stayed": [],
        "told": [],
    }
    
    # Run with both orderings to cancel location bias
    orderings = [
        ("drawer", "basket"),
        ("basket", "drawer"),
    ]
    
    for first_loc, second_loc in orderings:
        # Condition 1: Agent LEFT (didn't see)
        prompt_left = f"Alice put the ball in the {first_loc}. Alice left. Bob moved the ball to the {second_loc}. Alice returned. Alice looks in the"
        first_p, second_p = get_probs(model, tokenizer, prompt_left, first_loc, second_loc)
        
        # ToM correct = first_loc (agent's belief)
        results["left"].append({
            "first_loc": first_loc,
            "second_loc": second_loc,
            "first_prob": first_p,
            "second_prob": second_p,
            "tom_correct": first_p > second_p,
            "correct_answer": first_loc
        })
        
        # Condition 2: Agent STAYED (saw)
        prompt_stayed = f"Alice put the ball in the {first_loc}. Bob moved the ball to the {second_loc}. Alice looks in the"
        first_p, second_p = get_probs(model, tokenizer, prompt_stayed, first_loc, second_loc)
        
        # ToM correct = second_loc (agent SAW the move)
        results["stayed"].append({
            "first_loc": first_loc,
            "second_loc": second_loc,
            "first_prob": first_p,
            "second_prob": second_p,
            "tom_correct": second_p > first_p,
            "correct_answer": second_loc
        })
        
        # Condition 3: Agent TOLD
        prompt_told = f"Alice put the ball in the {first_loc}. Alice left. Bob moved the ball to the {second_loc}. Bob told Alice he moved it. Alice looks in the"
        first_p, second_p = get_probs(model, tokenizer, prompt_told, first_loc, second_loc)
        
        # ToM correct = second_loc (agent was TOLD)
        results["told"].append({
            "first_loc": first_loc,
            "second_loc": second_loc,
            "first_prob": first_p,
            "second_prob": second_p,
            "tom_correct": second_p > first_p,
            "correct_answer": second_loc
        })
    
    print("\n[Results by condition (counterbalanced)]")
    
    for cond in ["left", "stayed", "told"]:
        print(f"\n  === {cond.upper()} ===")
        for r in results[cond]:
            status = "CORRECT" if r["tom_correct"] else "WRONG"
            print(f"    {r['first_loc']} -> {r['second_loc']}: {r['first_loc']}={r['first_prob']*100:.1f}%, {r['second_loc']}={r['second_prob']*100:.1f}% | ToM: {status}")
        
        correct = sum(1 for r in results[cond] if r["tom_correct"])
        print(f"    TOTAL: {correct}/2 = {correct/2*100:.0f}%")
    
    return results


def test_presence_effect_magnitude(model, tokenizer):
    """
    Measure the MAGNITUDE of presence effect.
    If ToM exists, "left" should strongly differ from "stayed".
    """
    print("\n" + "="*70)
    print("PRESENCE EFFECT MAGNITUDE")
    print("="*70)
    
    # Use neutral locations
    first_loc, second_loc = "container A", "container B"
    
    # Wait, let's use made-up locations to eliminate all bias
    print("\n[Using made-up location names to eliminate bias]")
    
    prompts = {
        "left": f"Alice put the object in location X. Alice left the room. Bob moved the object to location Y. Alice returned. Alice looks in location",
        "stayed": f"Alice put the object in location X. Bob moved the object to location Y. Alice looks in location",
        "told": f"Alice put the object in location X. Alice left. Bob moved the object to location Y. Bob told Alice where it is. Alice looks in location",
    }
    
    for cond, prompt in prompts.items():
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
        
        probs = torch.softmax(logits, dim=-1)
        
        # Get top 5 predictions
        top5 = torch.topk(probs, 5)
        
        print(f"\n  [{cond.upper()}]")
        print(f"    Top predictions:")
        for prob, idx in zip(top5.values, top5.indices):
            token = tokenizer.decode([idx])
            print(f"      '{token}': {prob.item()*100:.1f}%")


def test_the_key_question(model, tokenizer):
    """
    THE KEY QUESTION: Does "Alice left" actually change the model's behavior?
    """
    print("\n" + "="*70)
    print("THE KEY QUESTION: Does 'Alice left' Matter?")
    print("="*70)
    
    # Minimal pair: ONLY difference is "Alice left"
    base = "Alice put the ball in the drawer."
    move = "Bob moved the ball to the basket."
    question = "Alice looks in the"
    
    prompt_with_left = f"{base} Alice left. {move} Alice returned. {question}"
    prompt_without_left = f"{base} {move} {question}"
    
    print(f"\n  [WITH 'Alice left']")
    print(f"    Prompt: '{prompt_with_left}'")
    drawer_p, basket_p = get_probs(model, tokenizer, prompt_with_left, "drawer", "basket")
    print(f"    drawer: {drawer_p*100:.1f}%, basket: {basket_p*100:.1f}%")
    print(f"    ToM: {'CORRECT' if drawer_p > basket_p else 'WRONG'} (should be drawer)")
    
    print(f"\n  [WITHOUT 'Alice left']")
    print(f"    Prompt: '{prompt_without_left}'")
    drawer_p2, basket_p2 = get_probs(model, tokenizer, prompt_without_left, "drawer", "basket")
    print(f"    drawer: {drawer_p2*100:.1f}%, basket: {basket_p2*100:.1f}%")
    print(f"    ToM: {'CORRECT' if basket_p2 > drawer_p2 else 'WRONG'} (should be basket - Alice saw!)")
    
    # The critical comparison
    print(f"\n  [CRITICAL COMPARISON]")
    print(f"    Effect of 'Alice left' on drawer probability:")
    print(f"      With 'left': {drawer_p*100:.1f}%")
    print(f"      Without 'left': {drawer_p2*100:.1f}%")
    print(f"      Difference: {(drawer_p - drawer_p2)*100:+.1f}%")
    
    if drawer_p > drawer_p2 + 0.1:
        print(f"\n    [EVIDENCE FOR ToM] 'Alice left' INCREASES drawer probability!")
        print(f"    The model IS tracking presence!")
    elif drawer_p < drawer_p2 - 0.1:
        print(f"\n    [PARADOX] 'Alice left' DECREASES drawer probability!")
    else:
        print(f"\n    [NO EFFECT] 'Alice left' doesn't change behavior much")
    
    return {
        "with_left": {"drawer": drawer_p, "basket": basket_p},
        "without_left": {"drawer": drawer_p2, "basket": basket_p2},
        "effect": drawer_p - drawer_p2
    }


def main():
    print("="*70)
    print("STEP 54: Controlled Analysis")
    print("="*70)
    
    model, tokenizer = load_model()
    
    controlled_results = test_controlled_tom(model, tokenizer)
    test_presence_effect_magnitude(model, tokenizer)
    key_results = test_the_key_question(model, tokenizer)
    
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    # Analyze controlled results
    left_correct = sum(1 for r in controlled_results["left"] if r["tom_correct"])
    stayed_correct = sum(1 for r in controlled_results["stayed"] if r["tom_correct"])
    told_correct = sum(1 for r in controlled_results["told"] if r["tom_correct"])
    
    print(f"""
    [CONTROLLED RESULTS (counterbalanced for location bias)]:
    
    - LEFT (should predict first loc): {left_correct}/2 = {left_correct/2*100:.0f}%
    - STAYED (should predict second loc): {stayed_correct}/2 = {stayed_correct/2*100:.0f}%
    - TOLD (should predict second loc): {told_correct}/2 = {told_correct/2*100:.0f}%
    
    [KEY QUESTION RESULT]:
    
    Effect of 'Alice left' on drawer probability: {key_results['effect']*100:+.1f}%
    """)
    
    if key_results['effect'] > 0.1:
        print("""
    ============================================================
    REVISED CONCLUSION: Model DOES have SOME ToM capability!
    ============================================================
    
    Evidence:
    - 'Alice left' INCREASES probability of original location
    - This suggests presence tracking IS happening
    - BUT it's weak and confounded by other factors
    
    The truth is NUANCED:
    - Some ToM exists (presence affects prediction)
    - But it's unreliable (confounded by location bias, completion phrase)
    - Not the simple "no ToM" conclusion we made before
        """)
    else:
        print("""
    ============================================================
    CONFIRMED: Model does NOT reliably track presence
    ============================================================
        """)


if __name__ == "__main__":
    main()


