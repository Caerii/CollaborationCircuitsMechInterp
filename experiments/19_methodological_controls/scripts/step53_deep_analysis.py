"""
Step 53: Deep Analysis - Question Everything

Before concluding "no ToM", let's examine:
1. Is there a location bias (drawer > basket always)?
2. Do different completion phrases change behavior?
3. Is it really first-mention or something else?
4. What about the reversed scenarios inconsistency?
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


def get_location_probs(model, tokenizer, prompt, locations):
    """Get probabilities for multiple locations."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    
    probs = torch.softmax(logits, dim=-1)
    
    result = {}
    for loc in locations:
        for prefix in [" ", ""]:
            ids = tokenizer.encode(prefix + loc, add_special_tokens=False)
            if ids:
                result[loc] = probs[ids[0]].item()
                break
        if loc not in result:
            result[loc] = 0.0
    
    return result


def test_1_location_bias(model, tokenizer):
    """TEST 1: Is there an inherent bias for certain locations?"""
    print("\n" + "="*70)
    print("TEST 1: Location Bias Analysis")
    print("="*70)
    
    # Test with neutral prompt - no ToM involved
    neutral_prompt = "The person looks in the"
    locations = ["drawer", "basket", "box", "cabinet", "closet", "bag", "shelf", "desk"]
    
    probs = get_location_probs(model, tokenizer, neutral_prompt, locations)
    
    print("\n[Neutral prompt - no ToM context]")
    print(f"  Prompt: '{neutral_prompt}'")
    print("\n  Location probabilities:")
    for loc, p in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"    {loc}: {p*100:.1f}%")
    
    # Check if drawer has inherent advantage
    drawer_prob = probs.get("drawer", 0)
    basket_prob = probs.get("basket", 0)
    
    print(f"\n  Drawer/Basket ratio: {drawer_prob/basket_prob:.2f}x")
    
    if drawer_prob > basket_prob * 2:
        print("  [WARNING] Strong inherent bias toward 'drawer'!")
    
    return probs


def test_2_completion_phrases(model, tokenizer):
    """TEST 2: Do different completion phrases change ToM behavior?"""
    print("\n" + "="*70)
    print("TEST 2: Completion Phrase Effects")
    print("="*70)
    
    base = "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket. Alice returned."
    
    completions = [
        ("Alice looks in the", "present simple"),
        ("Alice will look in the", "future"),
        ("Alice looked in the", "past"),
        ("Alice searches in the", "searches"),
        ("Alice checks the", "checks"),
        ("Where does Alice think the ball is? The", "question form"),
        ("Alice expects to find the ball in the", "expects"),
        ("Alice goes to the", "goes to"),
    ]
    
    print("\n[Same scenario, different completions]")
    print(f"  Base: '{base[:50]}...'")
    
    results = {}
    for completion, label in completions:
        prompt = base + " " + completion
        probs = get_location_probs(model, tokenizer, prompt, ["drawer", "basket"])
        
        drawer_p = probs.get("drawer", 0)
        basket_p = probs.get("basket", 0)
        correct = drawer_p > basket_p  # ToM answer is drawer
        
        results[label] = {
            "drawer": drawer_p,
            "basket": basket_p,
            "tom_correct": correct
        }
        
        status = "OK" if correct else "FAIL"
        print(f"\n  [{label}]: {completion}")
        print(f"    drawer: {drawer_p*100:.1f}%, basket: {basket_p*100:.1f}% -> {status}")
    
    return results


def test_3_location_order_effect(model, tokenizer):
    """TEST 3: Does the ORDER of locations matter?"""
    print("\n" + "="*70)
    print("TEST 3: Location Order Effects")
    print("="*70)
    
    # Test all combinations
    scenarios = [
        # (first_loc, second_loc, agent_left)
        ("drawer", "basket", True, "drawer first, basket second, agent LEFT"),
        ("basket", "drawer", True, "basket first, drawer second, agent LEFT"),
        ("box", "shelf", True, "box first, shelf second, agent LEFT"),
        ("shelf", "box", True, "shelf first, box second, agent LEFT"),
        ("cabinet", "desk", True, "cabinet first, desk second, agent LEFT"),
        ("desk", "cabinet", True, "desk first, cabinet second, agent LEFT"),
    ]
    
    print("\n[Testing order effects with various location pairs]")
    
    results = []
    for first, second, left, label in scenarios:
        if left:
            prompt = f"Alice put the ball in the {first}. Alice left. Bob moved the ball to the {second}. Alice returned. Alice looks in the"
        else:
            prompt = f"Alice put the ball in the {first}. Bob moved the ball to the {second}. Alice looks in the"
        
        probs = get_location_probs(model, tokenizer, prompt, [first, second])
        
        first_p = probs.get(first, 0)
        second_p = probs.get(second, 0)
        
        # ToM correct = first location (agent's belief)
        tom_correct = first_p > second_p
        
        results.append({
            "first_loc": first,
            "second_loc": second,
            "first_prob": first_p,
            "second_prob": second_p,
            "tom_correct": tom_correct,
            "predicts_first": first_p > second_p
        })
        
        status = "OK" if tom_correct else "FAIL"
        print(f"\n  [{label}]")
        print(f"    {first}: {first_p*100:.1f}%, {second}: {second_p*100:.1f}% -> {status}")
    
    # Summary
    first_wins = sum(1 for r in results if r["predicts_first"])
    print(f"\n  Summary: Model predicts FIRST location {first_wins}/{len(results)} times")
    
    return results


def test_4_presence_tracking_detailed(model, tokenizer):
    """TEST 4: Detailed presence tracking analysis."""
    print("\n" + "="*70)
    print("TEST 4: Detailed Presence Tracking")
    print("="*70)
    
    # Use multiple location pairs to control for bias
    location_pairs = [
        ("drawer", "basket"),
        ("basket", "drawer"),
        ("box", "shelf"),
        ("shelf", "box"),
        ("cabinet", "desk"),
    ]
    
    conditions = [
        ("left", "Alice left the room.", False),  # Alice doesn't know
        ("stayed", "", True),  # Alice knows (saw)
        ("told", "Bob told Alice where he put it.", True),  # Alice knows (told)
        ("explicit", "Bob said 'I put the ball in the {new}'.", True),  # Alice knows (explicit)
    ]
    
    results = {cond[0]: [] for cond in conditions}
    
    print("\n[Testing presence conditions across location pairs]")
    
    for first, second in location_pairs:
        for cond_name, cond_text, alice_knows in conditions:
            if cond_name == "left":
                prompt = f"Alice put the ball in the {first}. Alice left the room. Bob moved the ball to the {second}. Alice returned. Alice looks in the"
            elif cond_name == "stayed":
                prompt = f"Alice put the ball in the {first}. Bob moved the ball to the {second}. Alice looks in the"
            elif cond_name == "told":
                prompt = f"Alice put the ball in the {first}. Alice left. Bob moved the ball to the {second}. Bob told Alice where he put it. Alice looks in the"
            else:  # explicit
                prompt = f"Alice put the ball in the {first}. Alice left. Bob moved the ball to the {second}. Bob said 'I put the ball in the {second}'. Alice looks in the"
            
            probs = get_location_probs(model, tokenizer, prompt, [first, second])
            first_p = probs.get(first, 0)
            second_p = probs.get(second, 0)
            
            # What's correct?
            if alice_knows:
                correct_loc = second  # Alice knows new location
            else:
                correct_loc = first  # Alice believes original
            
            tom_correct = (correct_loc == first and first_p > second_p) or \
                         (correct_loc == second and second_p > first_p)
            
            results[cond_name].append({
                "first": first,
                "second": second,
                "first_prob": first_p,
                "second_prob": second_p,
                "tom_correct": tom_correct,
                "correct_loc": correct_loc
            })
    
    # Summary by condition
    print("\n[Summary by condition]")
    for cond_name in ["left", "stayed", "told", "explicit"]:
        correct = sum(1 for r in results[cond_name] if r["tom_correct"])
        total = len(results[cond_name])
        print(f"  {cond_name:10s}: {correct}/{total} = {correct/total*100:.0f}% ToM correct")
    
    return results


def test_5_what_is_the_model_doing(model, tokenizer):
    """TEST 5: Let's see what the model is ACTUALLY computing."""
    print("\n" + "="*70)
    print("TEST 5: What Is The Model Actually Doing?")
    print("="*70)
    
    # Let's ask it directly
    direct_questions = [
        "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket. Alice returned. Where does Alice THINK the ball is?",
        "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket. Alice returned. Where is the ball ACTUALLY?",
        "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket. Alice returned. Did Alice see Bob move the ball?",
        "Alice put the ball in the drawer. Bob moved the ball to the basket. Did Alice see Bob move the ball?",
    ]
    
    print("\n[Direct questions to model]")
    
    for q in direct_questions:
        inputs = tokenizer(q, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"\n  Q: {q[:70]}...")
        print(f"  A: {response.strip()[:100]}")
    
    return {}


def main():
    print("="*70)
    print("STEP 53: Deep Analysis - Question Everything")
    print("="*70)
    
    model, tokenizer = load_model()
    
    results = {}
    
    results["location_bias"] = test_1_location_bias(model, tokenizer)
    results["completion_phrases"] = test_2_completion_phrases(model, tokenizer)
    results["location_order"] = test_3_location_order_effect(model, tokenizer)
    results["presence_tracking"] = test_4_presence_tracking_detailed(model, tokenizer)
    results["direct_questions"] = test_5_what_is_the_model_doing(model, tokenizer)
    
    print("\n" + "="*70)
    print("DEEP ANALYSIS SUMMARY")
    print("="*70)
    
    # Analyze location bias
    if results["location_bias"]:
        drawer_p = results["location_bias"].get("drawer", 0)
        basket_p = results["location_bias"].get("basket", 0)
        if drawer_p > basket_p * 1.5:
            print("\n  [FINDING 1] There IS a location bias (drawer preferred)")
        else:
            print("\n  [FINDING 1] No strong location bias")
    
    # Analyze completion effect
    if results["completion_phrases"]:
        correct = sum(1 for v in results["completion_phrases"].values() if v.get("tom_correct", False))
        total = len(results["completion_phrases"])
        print(f"\n  [FINDING 2] Completion phrase effect: {correct}/{total} ToM correct")
    
    # Analyze order effect
    if results["location_order"]:
        first_wins = sum(1 for r in results["location_order"] if r.get("predicts_first", False))
        total = len(results["location_order"])
        print(f"\n  [FINDING 3] First-location bias: {first_wins}/{total} predict first")
    
    # Analyze presence tracking
    if results["presence_tracking"]:
        for cond in ["left", "stayed", "told", "explicit"]:
            if cond in results["presence_tracking"]:
                correct = sum(1 for r in results["presence_tracking"][cond] if r.get("tom_correct", False))
                total = len(results["presence_tracking"][cond])
                print(f"\n  [FINDING 4] {cond}: {correct}/{total} = {correct/total*100:.0f}%")
    
    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": "Deep analysis complete - see detailed results"
    }
    
    output_path = RESULTS_DIR / "step53_deep_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n\nSaved to: {output_path}")


if __name__ == "__main__":
    main()


