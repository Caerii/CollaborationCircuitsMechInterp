"""
Step 37: Deep Cross-Model Validation

Test the verb-type effect across multiple model families:
1. Qwen family (Qwen3-4B, Qwen2.5-1.5B, Qwen2.5-0.5B)
2. Microsoft Phi family (Phi-3-mini if available)
3. Other small models

For each model, we test:
- ToMi benchmark (action vs belief phrasing)
- Verb type sensitivity
- Layer divergence patterns (where possible)
"""

import torch
import json
import sys
import io
import gc
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Models to test (small enough to fit in VRAM)
MODELS = [
    {"name": "Qwen3-4B", "id": "Qwen/Qwen3-4B"},
    {"name": "Qwen2.5-1.5B", "id": "Qwen/Qwen2.5-1.5B"},
    {"name": "Qwen2.5-0.5B", "id": "Qwen/Qwen2.5-0.5B"},
]

# ToMi scenarios
TOMI_SCENARIOS = [
    {
        "name": "Sally-Anne",
        "story": "Sally puts the ball in the basket. Sally leaves. Anne moves the ball to the box. Sally returns.",
        "belief_q": "Where does Sally think the ball is? Sally thinks it is in the",
        "action_q": "Where will Sally look for the ball? Sally will look in the",
        "correct": "basket",
        "wrong": "box"
    },
    {
        "name": "Maxi",
        "story": "Maxi puts chocolate in the cupboard. Maxi leaves. Mother moves the chocolate to the drawer. Maxi returns.",
        "belief_q": "Where does Maxi think the chocolate is? Maxi thinks it is in the",
        "action_q": "Where will Maxi look for the chocolate? Maxi will look in the",
        "correct": "cupboard",
        "wrong": "drawer"
    },
    {
        "name": "Emma teddy",
        "story": "Emma puts teddy on the bed. Emma goes to school. Dad moves teddy to the closet. Emma returns.",
        "belief_q": "Where does Emma think teddy is? Emma thinks it is on the",
        "action_q": "Where will Emma look for teddy? Emma will look on the",
        "correct": "bed",
        "wrong": "closet"
    },
]

# Verb type test
VERB_TEST = {
    "story": """Alice put the ball in the drawer. Alice left the room.
While Alice was away, Bob told Carol that he moved the ball to the basket.
Alice returns. Alice""",
    "action_verbs": ["searched in the", "looks in the", "will look in the"],
    "belief_verbs": ["thinks the ball is in the", "believes the ball is in the", "knows the ball is in the"],
    "correct": "drawer",
    "wrong": "basket"
}


def load_model(model_id):
    """Load a model with error handling."""
    print(f"  Loading {model_id}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"  Error loading {model_id}: {e}")
        return None, None


def test_prompt(model, tokenizer, prompt, correct, wrong):
    """Test a prompt and return results."""
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
    
    correct_logit = get_logit(correct)
    wrong_logit = get_logit(wrong)
    
    return {
        "correct_logit": correct_logit,
        "wrong_logit": wrong_logit,
        "diff": correct_logit - wrong_logit,
        "is_correct": correct_logit > wrong_logit
    }


def test_model(model_config):
    """Run all tests on a single model."""
    model_name = model_config["name"]
    model_id = model_config["id"]
    
    print(f"\n{'='*60}")
    print(f"TESTING: {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer = load_model(model_id)
    if model is None:
        return {"error": "Failed to load model"}
    
    results = {
        "model": model_name,
        "tomi": {"belief": {}, "action": {}},
        "verb_type": {"action": {}, "belief": {}}
    }
    
    # Test ToMi scenarios
    print("\n--- ToMi Benchmark ---")
    tomi_belief_correct = 0
    tomi_action_correct = 0
    
    for scenario in TOMI_SCENARIOS:
        # Belief phrasing
        belief_result = test_prompt(
            model, tokenizer,
            scenario["story"] + "\n" + scenario["belief_q"],
            scenario["correct"], scenario["wrong"]
        )
        
        # Action phrasing
        action_result = test_prompt(
            model, tokenizer,
            scenario["story"] + "\n" + scenario["action_q"],
            scenario["correct"], scenario["wrong"]
        )
        
        b_status = "[OK]" if belief_result["is_correct"] else "[FAIL]"
        a_status = "[OK]" if action_result["is_correct"] else "[FAIL]"
        
        print(f"  {scenario['name']}: Belief {b_status} ({belief_result['diff']:+.2f}), "
              f"Action {a_status} ({action_result['diff']:+.2f})")
        
        results["tomi"]["belief"][scenario["name"]] = belief_result
        results["tomi"]["action"][scenario["name"]] = action_result
        
        if belief_result["is_correct"]:
            tomi_belief_correct += 1
        if action_result["is_correct"]:
            tomi_action_correct += 1
    
    results["tomi"]["belief_accuracy"] = tomi_belief_correct / len(TOMI_SCENARIOS)
    results["tomi"]["action_accuracy"] = tomi_action_correct / len(TOMI_SCENARIOS)
    
    # Test verb type sensitivity
    print("\n--- Verb Type Sensitivity ---")
    
    print("  Action verbs:")
    action_correct = 0
    for verb_completion in VERB_TEST["action_verbs"]:
        prompt = VERB_TEST["story"] + " " + verb_completion
        result = test_prompt(model, tokenizer, prompt, 
                           VERB_TEST["correct"], VERB_TEST["wrong"])
        status = "[OK]" if result["is_correct"] else "[FAIL]"
        print(f"    {status} '{verb_completion[:20]}...' (diff={result['diff']:+.2f})")
        results["verb_type"]["action"][verb_completion] = result
        if result["is_correct"]:
            action_correct += 1
    
    print("  Belief verbs:")
    belief_correct = 0
    for verb_completion in VERB_TEST["belief_verbs"]:
        prompt = VERB_TEST["story"] + " " + verb_completion
        result = test_prompt(model, tokenizer, prompt,
                           VERB_TEST["correct"], VERB_TEST["wrong"])
        status = "[OK]" if result["is_correct"] else "[FAIL]"
        print(f"    {status} '{verb_completion[:20]}...' (diff={result['diff']:+.2f})")
        results["verb_type"]["belief"][verb_completion] = result
        if result["is_correct"]:
            belief_correct += 1
    
    results["verb_type"]["action_accuracy"] = action_correct / len(VERB_TEST["action_verbs"])
    results["verb_type"]["belief_accuracy"] = belief_correct / len(VERB_TEST["belief_verbs"])
    
    # Summary
    print(f"\n--- Summary for {model_name} ---")
    print(f"  ToMi: Belief={results['tomi']['belief_accuracy']*100:.0f}%, "
          f"Action={results['tomi']['action_accuracy']*100:.0f}%")
    print(f"  Verb Type: Action={results['verb_type']['action_accuracy']*100:.0f}%, "
          f"Belief={results['verb_type']['belief_accuracy']*100:.0f}%")
    
    # Cleanup
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


def run_cross_model_validation():
    """Run validation across all models."""
    all_results = {}
    
    print("="*70)
    print("CROSS-MODEL DEEP VALIDATION")
    print("="*70)
    
    for model_config in MODELS:
        results = test_model(model_config)
        all_results[model_config["name"]] = results
    
    # Cross-model summary
    print("\n" + "="*70)
    print("CROSS-MODEL SUMMARY")
    print("="*70)
    
    print("\n" + "-"*70)
    print(f"{'Model':<20} | ToMi Belief | ToMi Action | Verb Action | Verb Belief")
    print("-"*70)
    
    for model_name, results in all_results.items():
        if "error" in results:
            print(f"{model_name:<20} | ERROR")
            continue
        
        tb = results["tomi"]["belief_accuracy"] * 100
        ta = results["tomi"]["action_accuracy"] * 100
        va = results["verb_type"]["action_accuracy"] * 100
        vb = results["verb_type"]["belief_accuracy"] * 100
        
        print(f"{model_name:<20} | {tb:>10.0f}% | {ta:>10.0f}% | {va:>10.0f}% | {vb:>10.0f}%")
    
    print("-"*70)
    
    # Save
    save_path = RESULTS_DIR / "cross_model_deep_results.json"
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {save_path}")
    
    return all_results


def main():
    print("="*70)
    print("STEP 37: Deep Cross-Model Validation")
    print("="*70)
    
    results = run_cross_model_validation()


if __name__ == "__main__":
    main()


