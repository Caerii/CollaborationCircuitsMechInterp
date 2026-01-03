"""
Step 2: Behavioral Test - Does the Model Pass False Belief?
============================================================

Test if Qwen3-4B correctly answers:
- Where does the agent THINK the object is? (belief)
- Where IS the object actually? (reality)
"""

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STEP 2: BEHAVIORAL FALSE BELIEF TEST")
print("=" * 60)


def test_completion(model, tokenizer, prompt, expected_answer):
    """Test if model completes with expected answer."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    completion = tokenizer.decode(output[0], skip_special_tokens=True)
    generated = completion[len(prompt):].strip().lower()
    
    # Check if expected answer appears in completion
    correct = expected_answer.lower() in generated
    
    return {
        "generated": generated,
        "expected": expected_answer,
        "correct": correct,
    }


def main():
    print("\n[1/3] Loading data...", flush=True)
    with open(DATA_DIR / "prompts.json") as f:
        prompts = json.load(f)
    
    # Take a subset for speed
    prompts = prompts[:40]  # 20 false belief, 20 true belief (approximately)
    print(f"  Testing {len(prompts)} scenarios")
    
    print("\n[2/3] Loading model...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CFG.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CFG.model_name,
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    print("  [OK]", flush=True)
    
    print("\n[3/3] Running behavioral tests...", flush=True)
    
    results = {
        "false_belief": {"belief_correct": 0, "reality_correct": 0, "total": 0},
        "true_belief": {"belief_correct": 0, "reality_correct": 0, "total": 0},
    }
    
    detailed_results = []
    
    for i, p in enumerate(prompts):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(prompts)}]", flush=True)
        
        scenario_type = "false_belief" if p["is_false_belief"] else "true_belief"
        
        # Test belief question
        belief_result = test_completion(model, tokenizer, p["belief_prompt"], p["belief_answer"])
        
        # Test reality question
        reality_result = test_completion(model, tokenizer, p["reality_prompt"], p["reality_answer"])
        
        results[scenario_type]["total"] += 1
        if belief_result["correct"]:
            results[scenario_type]["belief_correct"] += 1
        if reality_result["correct"]:
            results[scenario_type]["reality_correct"] += 1
        
        detailed_results.append({
            "id": p["id"],
            "type": scenario_type,
            "belief_generated": belief_result["generated"],
            "belief_expected": belief_result["expected"],
            "belief_correct": belief_result["correct"],
            "reality_generated": reality_result["generated"],
            "reality_expected": reality_result["expected"],
            "reality_correct": reality_result["correct"],
        })
    
    # Save results
    with open(RESULTS_DIR / "behavioral_results.json", "w") as f:
        json.dump({
            "summary": results,
            "detailed": detailed_results,
        }, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BEHAVIORAL TEST RESULTS")
    print("=" * 60)
    
    print("\nFALSE BELIEF scenarios (agent has wrong belief):")
    fb = results["false_belief"]
    if fb["total"] > 0:
        print(f"  Belief question correct:  {fb['belief_correct']}/{fb['total']} = {fb['belief_correct']/fb['total']:.0%}")
        print(f"  Reality question correct: {fb['reality_correct']}/{fb['total']} = {fb['reality_correct']/fb['total']:.0%}")
    
    print("\nTRUE BELIEF scenarios (agent has correct belief):")
    tb = results["true_belief"]
    if tb["total"] > 0:
        print(f"  Belief question correct:  {tb['belief_correct']}/{tb['total']} = {tb['belief_correct']/tb['total']:.0%}")
        print(f"  Reality question correct: {tb['reality_correct']}/{tb['total']} = {tb['reality_correct']/tb['total']:.0%}")
    
    # Key metric: Does model distinguish belief from reality in FALSE belief cases?
    print("\n" + "-" * 60)
    print("KEY METRIC: False Belief Understanding")
    print("-" * 60)
    
    if fb["total"] > 0:
        belief_acc = fb['belief_correct'] / fb['total']
        reality_acc = fb['reality_correct'] / fb['total']
        
        if belief_acc > 0.7 and reality_acc > 0.7:
            print("\n>>> PASS: Model correctly tracks BOTH belief AND reality <<<")
            print("    This suggests genuine Theory of Mind capability!")
        elif belief_acc > 0.5:
            print("\n>>> PARTIAL: Model shows some belief tracking <<<")
        else:
            print("\n>>> FAIL: Model conflates belief with reality <<<")
            print("    Model may not have robust Theory of Mind")
    
    print(f"\n[OK] Results saved to {RESULTS_DIR / 'behavioral_results.json'}")
    
    # Show some examples
    print("\n" + "=" * 60)
    print("EXAMPLE OUTPUTS")
    print("=" * 60)
    
    for r in detailed_results[:3]:
        print(f"\n  [{r['type']}]")
        print(f"  Belief: expected '{r['belief_expected']}', got '{r['belief_generated'][:30]}' -> {'CORRECT' if r['belief_correct'] else 'WRONG'}")
        print(f"  Reality: expected '{r['reality_expected']}', got '{r['reality_generated'][:30]}' -> {'CORRECT' if r['reality_correct'] else 'WRONG'}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()





















