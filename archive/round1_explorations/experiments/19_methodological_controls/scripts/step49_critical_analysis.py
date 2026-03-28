"""
Step 49: Critical Self-Analysis

Critique our own speedrun findings:
1. Why is baseline 76.7% now vs 22.5% originally?
2. Is the probe task trivial (just detecting token words)?
3. Are we comparing apples to apples?
"""

import torch
import json
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from datetime import datetime

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_model():
    print("Loading Qwen3-4B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
        output_hidden_states=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    model.eval()
    return model, tokenizer


def critique_1_baseline_discrepancy(model, tokenizer):
    """
    CRITIQUE 1: Why did baseline change from 22.5% to 76.7%?
    
    Test the EXACT prompts from original experiments vs our new ones.
    """
    print("\n" + "="*70)
    print("CRITIQUE 1: Baseline Discrepancy Analysis")
    print("="*70)
    
    # ORIGINAL format (from early experiments - low accuracy)
    original_prompts = [
        ("Alice put the ball in the drawer. Bob told Alice: 'I moved the ball to the basket.' Where will Alice look for the ball? Alice will look in the", "drawer", "basket"),
        ("Sally puts the ball in the basket. Sally leaves. Anne moves the ball to the box. Sally returns. Where will Sally look? Sally will look in the", "basket", "box"),
        ("John hid the toy in the cabinet. Mary told John: 'I put it in the drawer.' John will look in the", "drawer", "cabinet"),
    ]
    
    # OUR format (from step45 - high accuracy)  
    our_prompts = [
        ("Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket. Alice returned. Alice looks for the ball in the", "drawer", "basket"),
        ("Bob put the key in the shelf. Bob left. Carol moved the key to the cabinet. Bob returned. Bob looks for the key in the", "shelf", "cabinet"),
        ("Carol put the book in the box. Carol left. Dave moved the book to the desk. Carol returned. Carol looks for the book in the", "box", "desk"),
    ]
    
    def test_prompts(prompts, label):
        correct = 0
        for prompt, correct_loc, incorrect_loc in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1, :]
            
            c_ids = tokenizer.encode(" " + correct_loc, add_special_tokens=False)
            i_ids = tokenizer.encode(" " + incorrect_loc, add_special_tokens=False)
            
            if c_ids and i_ids:
                c_logit = logits[c_ids[0]].item()
                i_logit = logits[i_ids[0]].item()
                is_correct = c_logit > i_logit
                correct += int(is_correct)
                print(f"  {label}: '{prompt[:50]}...' -> {correct_loc if is_correct else incorrect_loc} ({'OK' if is_correct else 'FAIL'})")
        
        return correct / len(prompts) * 100 if prompts else 0
    
    print("\n[Testing ORIGINAL format prompts]")
    orig_acc = test_prompts(original_prompts, "ORIG")
    
    print("\n[Testing OUR format prompts]")
    our_acc = test_prompts(our_prompts, "OURS")
    
    print(f"\n  Original format accuracy: {orig_acc:.1f}%")
    print(f"  Our format accuracy: {our_acc:.1f}%")
    print(f"  Difference: {our_acc - orig_acc:+.1f}%")
    
    if our_acc > orig_acc + 20:
        print("\n  [WARNING] Our prompts are EASIER - we changed the task difficulty!")
        print("  The 'syntax controlled' prompts may have made ToM trivially easy.")
    
    return {"original_acc": orig_acc, "our_acc": our_acc}


def critique_2_probe_task_triviality(model, tokenizer):
    """
    CRITIQUE 2: Is the 100% probe accuracy just detecting literal tokens?
    
    Test with randomized entity names to see if it's conceptual or lexical.
    """
    print("\n" + "="*70)
    print("CRITIQUE 2: Is Probe Task Trivial?")
    print("="*70)
    
    # Test: Can we get 100% just from the literal name tokens?
    print("\n[Analysis] If probe works at 100%, is it detecting:")
    print("  A) Conceptual 'role' representations, or")
    print("  B) Just the literal tokens 'User', 'Assistant', 'Agent B'?")
    
    # Create dialogues with SAME role words but different positions
    test_cases = [
        # Normal case
        {
            "dialogue": "User: Hello\nAssistant: Hi there\nAgent B: Good morning",
            "expected": "Trivially separable by name tokens"
        },
        # Swap names but keep roles
        {
            "dialogue": "Alex: Hello\nHelper: Hi there\nPartner: Good morning", 
            "expected": "Still separable if role is encoded"
        },
        # Use same name for all
        {
            "dialogue": "Person: [as user] Hello\nPerson: [as self] Hi\nPerson: [as other] Morning",
            "expected": "Only works if conceptual role is encoded"
        }
    ]
    
    print("\n  The 100% probe accuracy is likely TRIVIAL because:")
    print("  - Each entity has a UNIQUE name (User, Assistant, Agent B)")
    print("  - The probe just learns to detect these tokens")
    print("  - This does NOT prove conceptual role separation!")
    
    print("\n  [PROPER TEST NEEDED]: Use same names, vary only contextual role")
    
    return {"verdict": "Probe task is likely trivial - detecting tokens not concepts"}


def critique_3_explicit_implicit_comparison(model, tokenizer):
    """
    CRITIQUE 3: Is the explicit vs implicit comparison fair?
    
    The formats are very different - are we testing ToM or format sensitivity?
    """
    print("\n" + "="*70)
    print("CRITIQUE 3: Explicit vs Implicit Comparison Fairness")
    print("="*70)
    
    # The formats we compared:
    print("\n[IMPLICIT format we used]:")
    print('  "Alice put ball in drawer. Alice left. Bob moved ball to basket.')
    print('   Alice returned. Alice looks in the"')
    
    print("\n[EXPLICIT format we used]:")
    print('  "Alice believes the ball is in the drawer. The ball is actually')
    print('   in the basket. Alice looks in the"')
    
    print("\n[PROBLEM]: These differ in MORE than just explicit/implicit:")
    print("  1. Narrative structure (events vs statements)")
    print("  2. Sentence length and complexity")  
    print("  3. Use of 'believes' vs action verbs")
    print("  4. Presence/absence of temporal markers")
    
    # Better test: MINIMAL difference
    print("\n[BETTER COMPARISON - minimal pairs]:")
    
    minimal_pairs = [
        # Pair 1: Only difference is explicit belief statement
        {
            "implicit": "Alice put the ball in drawer. Alice left. Bob moved ball to basket. Alice returned. Alice looks in the",
            "explicit": "Alice put the ball in drawer. Alice left. Bob moved ball to basket. Alice returned. Alice still believes the ball is in the drawer. Alice looks in the",
            "correct": "drawer",
            "incorrect": "basket"
        },
        # Pair 2: Add "doesn't know"
        {
            "implicit": "Bob hid key in cabinet. Bob left. Carol moved key to desk. Bob returned. Bob looks in the",
            "explicit": "Bob hid key in cabinet. Bob left. Carol moved key to desk. Bob returned. Bob doesn't know the key was moved. Bob looks in the",
            "correct": "cabinet",
            "incorrect": "desk"
        },
    ]
    
    print("\n[Testing MINIMAL pairs]:")
    results = {"implicit": [], "explicit": []}
    
    for pair in minimal_pairs:
        for cond in ["implicit", "explicit"]:
            inputs = tokenizer(pair[cond], return_tensors="pt").to("cuda")
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1, :]
            
            c_ids = tokenizer.encode(" " + pair["correct"], add_special_tokens=False)
            i_ids = tokenizer.encode(" " + pair["incorrect"], add_special_tokens=False)
            
            if c_ids and i_ids:
                diff = logits[c_ids[0]].item() - logits[i_ids[0]].item()
                results[cond].append(diff > 0)
                status = "OK" if diff > 0 else "FAIL"
                print(f"  {cond.upper()}: logit_diff={diff:+.2f} -> {status}")
    
    impl_acc = sum(results["implicit"]) / len(results["implicit"]) * 100 if results["implicit"] else 0
    expl_acc = sum(results["explicit"]) / len(results["explicit"]) * 100 if results["explicit"] else 0
    
    print(f"\n  Minimal pair results:")
    print(f"    Implicit: {impl_acc:.0f}%")
    print(f"    Explicit (with belief hint): {expl_acc:.0f}%")
    
    return {"implicit_minimal": impl_acc, "explicit_minimal": expl_acc}


def critique_4_what_is_actually_happening(model, tokenizer):
    """
    CRITIQUE 4: What's REALLY going on?
    
    Let's trace the actual token predictions step by step.
    """
    print("\n" + "="*70)
    print("CRITIQUE 4: What's Actually Happening - Token-Level Analysis")
    print("="*70)
    
    # The key question: Does the model even "understand" the narrative?
    
    test_prompt = "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket. Alice returned. Alice looks for the ball in the"
    
    print(f"\n[Prompt]: {test_prompt}")
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits[0, -1, :]
    
    # Get top predictions
    top_k = 10
    top_probs = torch.softmax(logits, dim=-1)
    top_indices = torch.topk(top_probs, top_k).indices
    
    print(f"\n[Top {top_k} predictions]:")
    for idx in top_indices:
        token = tokenizer.decode([idx])
        prob = top_probs[idx].item()
        print(f"  '{token}': {prob*100:.1f}%")
    
    # Get specific logits
    drawer_ids = tokenizer.encode(" drawer", add_special_tokens=False)
    basket_ids = tokenizer.encode(" basket", add_special_tokens=False)
    
    if drawer_ids and basket_ids:
        drawer_prob = top_probs[drawer_ids[0]].item()
        basket_prob = top_probs[basket_ids[0]].item()
        
        print(f"\n[Key comparison]:")
        print(f"  'drawer' (correct ToM): {drawer_prob*100:.2f}%")
        print(f"  'basket' (reality): {basket_prob*100:.2f}%")
        print(f"  Ratio: {drawer_prob/basket_prob:.2f}x")
    
    # Check if model is just completing based on recency
    print("\n[Recency analysis]:")
    print("  Last mentioned location: 'basket' (most recent)")
    print("  First mentioned location: 'drawer' (Alice's belief)")
    print("  If model uses recency heuristic -> predicts 'basket' (WRONG ToM)")
    print("  If model uses ToM -> predicts 'drawer' (CORRECT)")
    
    return {}


def critique_5_the_real_question(model, tokenizer):
    """
    CRITIQUE 5: What should we actually be measuring?
    """
    print("\n" + "="*70)
    print("CRITIQUE 5: What Should We Actually Measure?")
    print("="*70)
    
    print("""
    [THE REAL QUESTIONS]:
    
    1. Does the model TRACK who knows what?
       - Not just "can it complete sentences"
       - But "does it maintain separate knowledge states"
    
    2. Does the model INFER beliefs from presence/absence?
       - "Alice left" -> Alice didn't see what happened
       - This requires causal reasoning, not pattern matching
    
    3. Is explicit framing actually helping ToM, or just cueing?
       - "Alice believes X" might just be a lexical trigger
       - vs genuine belief state tracking
    
    4. What's the BASELINE for these tasks?
       - A model with NO ToM would predict... what?
       - Recency? First-mention? Random?
    
    [OUR SPEEDRUN PROBLEMS]:
    
    - We changed prompt format -> changed task difficulty
    - We tested with unique entity names -> trivial probe task
    - We compared very different formats -> confounded comparison
    - We didn't establish a proper baseline -> no null model
    
    [WHAT WE SHOULD DO]:
    
    1. Use IDENTICAL prompts across conditions
    2. Test with controlled entity naming
    3. Establish null model baseline (recency, first-mention)
    4. Use larger sample sizes with proper statistics
    """)
    
    return {}


def main():
    print("="*70)
    print("STEP 49: Critical Self-Analysis of Speedrun Findings")
    print("="*70)
    
    model, tokenizer = load_model()
    
    results = {}
    
    # Run all critiques
    results["critique_1"] = critique_1_baseline_discrepancy(model, tokenizer)
    results["critique_2"] = critique_2_probe_task_triviality(model, tokenizer)
    results["critique_3"] = critique_3_explicit_implicit_comparison(model, tokenizer)
    results["critique_4"] = critique_4_what_is_actually_happening(model, tokenizer)
    results["critique_5"] = critique_5_the_real_question(model, tokenizer)
    
    # Summary
    print("\n" + "="*70)
    print("CRITICAL SELF-ANALYSIS SUMMARY")
    print("="*70)
    
    print("""
    [VERDICT ON SPEEDRUN FINDINGS]:
    
    1. CIRCUIT RE-VALIDATION: Inconclusive
       - Baseline changed from 22.5% to 76.7% 
       - We made the task EASIER, not tested the same thing
    
    2. EXPLICIT vs IMPLICIT: Likely Confounded
       - Formats differed in many ways, not just explicit/implicit
       - Need minimal pairs with ONLY the belief statement added
    
    3. SELF/OTHER/USER SEPARATION: Likely Trivial
       - 100% accuracy is suspicious
       - Probe may just detect literal name tokens
       - Need to test with controlled naming
    
    [THE TRUTH]:
    
    We rushed to conclusions. The findings need:
    - Controlled comparisons
    - Proper baselines  
    - Larger samples
    - Minimal pair designs
    
    Science requires rigor. We cut corners.
    """)
    
    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "verdict": "Speedrun findings are methodologically flawed"
    }
    
    output_path = RESULTS_DIR / "step49_critical_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()


