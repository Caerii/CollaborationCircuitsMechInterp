"""
Step 27: Complex Multi-Agent Scenario Testing

Test if the late-circuit ablation generalizes to:
1. Multi-turn dialogues with belief changes
2. 3+ agents with different knowledge states
3. Nested beliefs ("Alice thinks Bob thinks...")
4. Different domains (not just object location)
"""

import torch
import json
import sys
import io
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# The 10 critical late-layer heads (from our discovery)
LATE_CIRCUIT_HEADS = [
    (32, 6), (32, 31),
    (33, 6), (33, 13), (33, 17), (33, 31),
    (34, 17),
    (35, 0), (35, 1), (35, 17)
]

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_model():
    """Load model."""
    print("Loading Qwen3-4B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model.eval()
    return model, tokenizer


def create_complex_scenarios():
    """Create diverse complex ToM scenarios."""
    scenarios = []
    
    # Category 1: Multi-turn dialogue
    scenarios.append({
        "category": "multi_turn",
        "name": "Two belief updates",
        "prompt": """Alice put the cake in the fridge. Alice left.
Bob moved the cake to the counter. Bob told Carol about the move.
Carol then moved the cake to the pantry. Carol did not tell anyone.
Alice returned. Alice will look for the cake in the""",
        "correct": "fridge",
        "wrong": ["counter", "pantry"]
    })
    
    scenarios.append({
        "category": "multi_turn",
        "name": "Witnessed vs told",
        "prompt": """The treasure was in the cave. Alice saw Bob hide the treasure in the forest.
Bob told Carol he hid the treasure in the mountain.
Where does Carol think the treasure is? Carol thinks it is in the""",
        "correct": "mountain",
        "wrong": ["cave", "forest"]
    })
    
    # Category 2: 3+ agents
    scenarios.append({
        "category": "three_agents",
        "name": "Three agents different knowledge",
        "prompt": """The toy started on the shelf.
Alice moved the toy to the box. Only Bob saw this.
Carol stayed in another room the whole time.
Where does Carol think the toy is? Carol thinks the toy is on the""",
        "correct": "shelf",
        "wrong": ["box"]
    })
    
    scenarios.append({
        "category": "three_agents",
        "name": "Chain of communication",
        "prompt": """The keys were on the table.
Alice moved the keys to the drawer. Alice told Bob.
Bob told Carol the keys are in the drawer.
Where does Carol think the keys are? Carol thinks they are in the""",
        "correct": "drawer",
        "wrong": ["table"]
    })
    
    scenarios.append({
        "category": "three_agents",
        "name": "Partial communication",
        "prompt": """The document was in folder A.
Alice moved it to folder B. Alice told Bob.
Carol was not informed of any changes.
Where does Carol think the document is? Carol thinks it is in folder""",
        "correct": "A",
        "wrong": ["B"]
    })
    
    # Category 3: Nested beliefs (second-order ToM)
    scenarios.append({
        "category": "nested_beliefs",
        "name": "Alice thinks Bob thinks",
        "prompt": """Bob put his phone in his jacket.
Alice saw Bob put the phone there.
Bob then moved his phone to his bag when Alice wasn't looking.
What does Alice think Bob thinks about where the phone is?
Alice thinks Bob believes the phone is in the""",
        "correct": "bag",
        "wrong": ["jacket"]
    })
    
    scenarios.append({
        "category": "nested_beliefs", 
        "name": "False belief about false belief",
        "prompt": """The gift was hidden in the closet.
Mom moved it to under the bed. She told Dad but not the kids.
The kids still think it's in the closet.
What does Dad think the kids believe?
Dad thinks the kids believe the gift is in the""",
        "correct": "closet",
        "wrong": ["bed", "under the bed"]
    })
    
    # Category 4: Different domains (not object location)
    scenarios.append({
        "category": "different_domain",
        "name": "Secret knowledge",
        "prompt": """Alice knew the password was "blue123".
The IT department changed the password to "red456".
They emailed Bob but not Alice.
What password will Alice try to use? Alice will try""",
        "correct": "blue123",
        "wrong": ["red456"]
    })
    
    scenarios.append({
        "category": "different_domain",
        "name": "Schedule belief",
        "prompt": """The meeting was scheduled for 2pm.
The organizer changed it to 3pm and told Bob.
Alice did not receive the update.
What time does Alice think the meeting is? Alice thinks it is at""",
        "correct": "2pm",
        "wrong": ["3pm"]
    })
    
    scenarios.append({
        "category": "different_domain",
        "name": "Price knowledge",
        "prompt": """The car was listed at $20,000.
The dealer lowered the price to $18,000 and told the manager.
The salesperson doesn't know about the price change.
What price will the salesperson quote? The salesperson will say""",
        "correct": "$20,000",
        "wrong": ["$18,000"]
    })
    
    # Category 5: With "told" verb (our known failure case)
    scenarios.append({
        "category": "told_verb",
        "name": "Standard told scenario",
        "prompt": """Alice put the ball in the drawer.
Bob told Carol that he moved the ball to the basket.
Where does Alice think the ball is? Alice thinks the ball is in the""",
        "correct": "drawer",
        "wrong": ["basket"]
    })
    
    scenarios.append({
        "category": "told_verb",
        "name": "Complex told scenario",
        "prompt": """The report was on desk A. Only Alice knew this.
Bob told Carol he put the report on desk B.
Alice was in a meeting and didn't hear this.
Where will Alice look for the report? Alice will look on desk""",
        "correct": "A",
        "wrong": ["B"]
    })
    
    return scenarios


def register_ablation_hooks(model, heads_to_ablate):
    """Register hooks to ablate specified heads."""
    hooks = []
    
    for layer_idx, head_idx in heads_to_ablate:
        layer = model.model.layers[layer_idx]
        
        def make_hook(l_idx, h_idx):
            def hook(module, input, output):
                # output from o_proj is (batch, seq, hidden)
                hidden = output
                batch, seq_len, hidden_size = hidden.shape
                n_heads = 32  # Qwen3-4B has 32 heads
                head_dim = hidden_size // n_heads
                
                # Reshape to access individual heads
                hidden = hidden.view(batch, seq_len, n_heads, head_dim)
                hidden[:, :, h_idx, :] = 0  # Zero out the head
                hidden = hidden.view(batch, seq_len, hidden_size)
                return hidden
            return hook
        
        hook = layer.self_attn.o_proj.register_forward_hook(make_hook(layer_idx, head_idx))
        hooks.append(hook)
    
    return hooks


def clear_hooks(hooks):
    """Remove all hooks."""
    for hook in hooks:
        hook.remove()


def test_scenario(model, tokenizer, prompt, correct, wrong):
    """Test a single scenario and return prediction info."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits[0, -1, :]  # Last token logits
    
    # Get logits for correct and wrong answers
    def get_token_logit(answer):
        # Try with and without space prefix
        for prefix in [" ", ""]:
            tokens = tokenizer.encode(prefix + str(answer), add_special_tokens=False)
            if tokens:
                return logits[tokens[0]].item()
        return float('-inf')
    
    correct_logit = get_token_logit(correct)
    wrong_logits = [get_token_logit(w) for w in wrong]
    max_wrong_logit = max(wrong_logits) if wrong_logits else float('-inf')
    
    # Get top prediction
    top_token_id = torch.argmax(logits).item()
    top_token = tokenizer.decode([top_token_id]).strip()
    
    is_correct = correct_logit > max_wrong_logit
    
    return {
        "correct_logit": correct_logit,
        "max_wrong_logit": max_wrong_logit,
        "logit_diff": correct_logit - max_wrong_logit,
        "top_prediction": top_token,
        "is_correct": is_correct
    }


def run_complex_evaluation():
    """Run evaluation on all complex scenarios."""
    model, tokenizer = load_model()
    scenarios = create_complex_scenarios()
    
    results = {
        "baseline": {},
        "ablated": {},
        "by_category": {}
    }
    
    print("\n" + "="*70)
    print("TESTING COMPLEX SCENARIOS")
    print("="*70)
    
    # Test baseline
    print("\n--- BASELINE (No Intervention) ---\n")
    
    for scenario in scenarios:
        result = test_scenario(model, tokenizer, scenario["prompt"], 
                              scenario["correct"], scenario["wrong"])
        
        cat = scenario["category"]
        name = scenario["name"]
        key = f"{cat}_{name}"
        
        results["baseline"][key] = {
            **result,
            "category": cat,
            "name": name,
            "correct_answer": scenario["correct"]
        }
        
        status = "[OK]" if result["is_correct"] else "[FAIL]"
        print(f"{status} {cat}/{name}")
        print(f"    Correct: '{scenario['correct']}' ({result['correct_logit']:.2f})")
        print(f"    Top pred: '{result['top_prediction']}', Diff: {result['logit_diff']:+.2f}")
    
    # Test with ablation
    print("\n--- WITH LATE CIRCUIT ABLATION (10 heads) ---\n")
    
    hooks = register_ablation_hooks(model, LATE_CIRCUIT_HEADS)
    
    for scenario in scenarios:
        result = test_scenario(model, tokenizer, scenario["prompt"],
                              scenario["correct"], scenario["wrong"])
        
        cat = scenario["category"]
        name = scenario["name"]
        key = f"{cat}_{name}"
        
        results["ablated"][key] = {
            **result,
            "category": cat,
            "name": name,
            "correct_answer": scenario["correct"]
        }
        
        baseline_correct = results["baseline"][key]["is_correct"]
        ablated_correct = result["is_correct"]
        
        if ablated_correct and not baseline_correct:
            status = "[FIXED]"
        elif ablated_correct:
            status = "[OK]"
        elif baseline_correct and not ablated_correct:
            status = "[BROKEN]"
        else:
            status = "[STILL FAIL]"
        
        print(f"{status} {cat}/{name}")
        print(f"    Correct: '{scenario['correct']}' ({result['correct_logit']:.2f})")
        print(f"    Top pred: '{result['top_prediction']}', Diff: {result['logit_diff']:+.2f}")
    
    clear_hooks(hooks)
    
    # Summary by category
    print("\n" + "="*70)
    print("SUMMARY BY CATEGORY")
    print("="*70)
    
    categories = set(s["category"] for s in scenarios)
    
    for cat in sorted(categories):
        cat_scenarios = [s for s in scenarios if s["category"] == cat]
        
        baseline_correct = sum(1 for s in cat_scenarios 
                              if results["baseline"][f"{s['category']}_{s['name']}"]["is_correct"])
        ablated_correct = sum(1 for s in cat_scenarios
                             if results["ablated"][f"{s['category']}_{s['name']}"]["is_correct"])
        
        total = len(cat_scenarios)
        baseline_pct = baseline_correct / total * 100
        ablated_pct = ablated_correct / total * 100
        
        print(f"\n{cat}:")
        print(f"  Baseline: {baseline_correct}/{total} ({baseline_pct:.0f}%)")
        print(f"  Ablated:  {ablated_correct}/{total} ({ablated_pct:.0f}%)")
        print(f"  Change:   {ablated_pct - baseline_pct:+.0f}%")
        
        results["by_category"][cat] = {
            "baseline_correct": baseline_correct,
            "ablated_correct": ablated_correct,
            "total": total,
            "baseline_pct": baseline_pct,
            "ablated_pct": ablated_pct
        }
    
    # Overall summary
    total = len(scenarios)
    baseline_total = sum(1 for k in results["baseline"] if results["baseline"][k]["is_correct"])
    ablated_total = sum(1 for k in results["ablated"] if results["ablated"][k]["is_correct"])
    
    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)
    print(f"\nBaseline: {baseline_total}/{total} ({baseline_total/total*100:.0f}%)")
    print(f"Ablated:  {ablated_total}/{total} ({ablated_total/total*100:.0f}%)")
    print(f"Change:   {(ablated_total-baseline_total)/total*100:+.0f}%")
    
    results["overall"] = {
        "baseline_correct": baseline_total,
        "ablated_correct": ablated_total,
        "total": total
    }
    
    # Save results
    save_path = RESULTS_DIR / "complex_scenarios_results.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {save_path}")
    
    return results


def main():
    print("="*70)
    print("STEP 27: Complex Multi-Agent Scenario Testing")
    print("="*70)
    print("\nTesting generalization to:")
    print("  1. Multi-turn dialogues with belief changes")
    print("  2. 3+ agents with different knowledge states")
    print("  3. Nested beliefs (second-order ToM)")
    print("  4. Different domains (not just object location)")
    print("  5. Our known failure case ('told' verb)")
    
    results = run_complex_evaluation()


if __name__ == "__main__":
    main()


