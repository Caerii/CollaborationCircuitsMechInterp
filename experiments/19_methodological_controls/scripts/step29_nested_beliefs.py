"""
Step 29: Nested Beliefs (Second-Order ToM) Investigation

From step27: Nested beliefs fail at 50% and ablation doesn't help.
This suggests a different mechanism than the late override circuit.

This script investigates:
1. What causes nested belief failures?
2. Is there a separate circuit for second-order ToM?
3. Can we find interventions that help?
"""

import torch
import json
import sys
import io
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Late circuit heads (for comparison)
LATE_CIRCUIT_HEADS = [
    (32, 6), (32, 31), (33, 6), (33, 13), (33, 17), (33, 31),
    (34, 17), (35, 0), (35, 1), (35, 17)
]


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


def create_nested_belief_scenarios():
    """Create various nested belief (second-order ToM) scenarios."""
    scenarios = []
    
    # Simple first-order (control)
    scenarios.append({
        "name": "First-order control",
        "order": 1,
        "prompt": """Bob put his phone in his jacket pocket.
Alice did not see this happen.
Where does Alice think the phone is? Alice has no idea where the phone is, so she cannot answer. But Bob knows the phone is in the""",
        "correct": "jacket",
        "wrong": ["bag", "desk"]
    })
    
    # Second-order: What does A think B thinks?
    scenarios.append({
        "name": "Second-order: A thinks B thinks",
        "order": 2,
        "prompt": """Bob put his phone in his jacket.
Alice saw Bob put the phone in his jacket.
Bob then secretly moved his phone to his bag when Alice wasn't looking.
What does Alice think about where Bob's phone is?
Alice thinks the phone is in the""",
        "correct": "jacket",
        "wrong": ["bag"]
    })
    
    scenarios.append({
        "name": "Second-order: What A thinks B believes",
        "order": 2,
        "prompt": """The cookie was in the jar.
Mom moved the cookie to the box. Dad saw this happen.
Mom doesn't know Dad saw.
What does Mom think Dad believes about the cookie's location?
Mom thinks Dad believes the cookie is in the""",
        "correct": "jar",
        "wrong": ["box"]
    })
    
    scenarios.append({
        "name": "Second-order: Perspective taking",
        "order": 2,
        "prompt": """Sally put her toy in the basket, then left.
Anne moved the toy to the box while Sally was gone.
Anne knows that Sally doesn't know about the move.
From Anne's perspective, where will Sally look for the toy?
Anne thinks Sally will look in the""",
        "correct": "basket",
        "wrong": ["box"]
    })
    
    # Third-order: What does A think B thinks C thinks?
    scenarios.append({
        "name": "Third-order: A thinks B thinks C thinks",
        "order": 3,
        "prompt": """The treasure is hidden in cave A.
Alice knows this. Bob doesn't know where it is.
Carol thinks Bob knows it's in cave B (but he doesn't).
What does Carol think Bob believes about the treasure?
Carol thinks Bob believes the treasure is in cave""",
        "correct": "B",
        "wrong": ["A"]
    })
    
    # Communication-based nested belief
    scenarios.append({
        "name": "Second-order with communication",
        "order": 2,
        "prompt": """John told Mary that the meeting is at 3pm.
But the meeting was actually changed to 4pm, and John doesn't know.
Mary heard from John that it's at 3pm.
What does John think Mary believes about the meeting time?
John thinks Mary believes the meeting is at""",
        "correct": "3pm",
        "wrong": ["4pm"]
    })
    
    scenarios.append({
        "name": "Second-order: false belief about false belief",
        "order": 2,
        "prompt": """The ball started in the basket.
While Tom was watching, Sam moved it to the box.
Tom left, and Sam moved it back to the basket.
Sam knows Tom didn't see the second move.
What does Sam think Tom believes about where the ball is?
Sam thinks Tom believes the ball is in the""",
        "correct": "box",
        "wrong": ["basket"]
    })
    
    # Simpler nested belief
    scenarios.append({
        "name": "Second-order: simple",
        "order": 2,
        "prompt": """Lisa hid the candy in drawer A.
Mark saw Lisa hide it there.
Lisa doesn't know Mark saw her.
Where does Lisa think Mark thinks the candy is?
Lisa thinks Mark doesn't know, but actually Mark thinks it is in drawer""",
        "correct": "A",
        "wrong": ["B"]
    })
    
    return scenarios


def test_scenario(model, tokenizer, prompt, correct, wrong):
    """Test a scenario and return detailed results."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits[0, -1, :]
    
    def get_token_logit(answer):
        for prefix in [" ", ""]:
            tokens = tokenizer.encode(prefix + str(answer), add_special_tokens=False)
            if tokens:
                return logits[tokens[0]].item()
        return float('-inf')
    
    correct_logit = get_token_logit(correct)
    wrong_logits = [get_token_logit(w) for w in wrong]
    max_wrong = max(wrong_logits) if wrong_logits else float('-inf')
    
    # Get top 5 predictions
    top_k = torch.topk(logits, k=5)
    top_tokens = [tokenizer.decode([t]).strip() for t in top_k.indices.tolist()]
    top_logits = top_k.values.tolist()
    
    return {
        "correct_logit": correct_logit,
        "max_wrong_logit": max_wrong,
        "diff": correct_logit - max_wrong,
        "is_correct": correct_logit > max_wrong,
        "top_5_tokens": top_tokens,
        "top_5_logits": top_logits
    }


def register_ablation_hooks(model, heads):
    """Register ablation hooks for specified heads."""
    hooks = []
    for layer_idx, head_idx in heads:
        layer = model.model.layers[layer_idx]
        
        def make_hook(h_idx):
            def hook(module, input, output):
                hidden = output
                batch, seq_len, hidden_size = hidden.shape
                n_heads = 32
                head_dim = hidden_size // n_heads
                hidden = hidden.view(batch, seq_len, n_heads, head_dim)
                hidden[:, :, h_idx, :] = 0
                hidden = hidden.view(batch, seq_len, hidden_size)
                return hidden
            return hook
        
        hook = layer.self_attn.o_proj.register_forward_hook(make_hook(head_idx))
        hooks.append(hook)
    return hooks


def clear_hooks(hooks):
    for hook in hooks:
        hook.remove()


def search_for_nested_belief_circuit(model, tokenizer, failing_scenarios):
    """Search for heads that affect nested belief performance."""
    print("\n" + "="*70)
    print("SEARCHING FOR NESTED BELIEF CIRCUIT")
    print("="*70)
    
    # Test ablating each layer's heads
    results = {}
    
    for layer_idx in range(30, 36):  # Focus on late layers
        layer_results = []
        
        for head_idx in range(32):  # 32 heads per layer
            hooks = register_ablation_hooks(model, [(layer_idx, head_idx)])
            
            correct_count = 0
            for scenario in failing_scenarios:
                result = test_scenario(model, tokenizer, 
                                       scenario["prompt"], 
                                       scenario["correct"], 
                                       scenario["wrong"])
                if result["is_correct"]:
                    correct_count += 1
            
            clear_hooks(hooks)
            
            accuracy = correct_count / len(failing_scenarios)
            if accuracy > 0:  # Only record if it helps
                layer_results.append((head_idx, accuracy))
        
        if layer_results:
            layer_results.sort(key=lambda x: x[1], reverse=True)
            results[f"L{layer_idx}"] = layer_results[:5]  # Top 5
            print(f"L{layer_idx} - Top improving heads:")
            for head_idx, acc in layer_results[:5]:
                print(f"  H{head_idx}: {acc*100:.0f}%")
    
    return results


def run_nested_belief_analysis():
    """Main analysis of nested beliefs."""
    model, tokenizer = load_model()
    scenarios = create_nested_belief_scenarios()
    
    print("\n" + "="*70)
    print("BASELINE: Testing Nested Belief Scenarios")
    print("="*70)
    
    results = {"baseline": {}, "with_late_ablation": {}}
    failing_scenarios = []
    
    for scenario in scenarios:
        result = test_scenario(model, tokenizer, 
                              scenario["prompt"], 
                              scenario["correct"], 
                              scenario["wrong"])
        
        status = "[OK]" if result["is_correct"] else "[FAIL]"
        print(f"\n{status} {scenario['name']} (order {scenario['order']})")
        print(f"  Correct: '{scenario['correct']}' ({result['correct_logit']:.2f})")
        print(f"  Top 5: {result['top_5_tokens']}")
        print(f"  Diff: {result['diff']:+.2f}")
        
        results["baseline"][scenario["name"]] = {
            **result,
            "order": scenario["order"]
        }
        
        if not result["is_correct"]:
            failing_scenarios.append(scenario)
    
    # Test with late circuit ablation
    print("\n" + "="*70)
    print("WITH LATE CIRCUIT ABLATION")
    print("="*70)
    
    hooks = register_ablation_hooks(model, LATE_CIRCUIT_HEADS)
    
    for scenario in scenarios:
        result = test_scenario(model, tokenizer,
                              scenario["prompt"],
                              scenario["correct"],
                              scenario["wrong"])
        
        baseline_correct = results["baseline"][scenario["name"]]["is_correct"]
        ablated_correct = result["is_correct"]
        
        if ablated_correct and not baseline_correct:
            status = "[FIXED]"
        elif ablated_correct:
            status = "[OK]"
        elif baseline_correct and not ablated_correct:
            status = "[BROKEN]"
        else:
            status = "[STILL FAIL]"
        
        print(f"\n{status} {scenario['name']} (order {scenario['order']})")
        print(f"  Correct: '{scenario['correct']}' ({result['correct_logit']:.2f})")
        print(f"  Diff: {result['diff']:+.2f}")
        
        results["with_late_ablation"][scenario["name"]] = {
            **result,
            "order": scenario["order"]
        }
    
    clear_hooks(hooks)
    
    # Search for circuit if there are failing scenarios
    if failing_scenarios:
        print(f"\n{len(failing_scenarios)} scenarios still failing. Searching for circuit...")
        circuit_search = search_for_nested_belief_circuit(model, tokenizer, failing_scenarios)
        results["circuit_search"] = circuit_search
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    baseline_correct = sum(1 for r in results["baseline"].values() if r["is_correct"])
    ablated_correct = sum(1 for r in results["with_late_ablation"].values() if r["is_correct"])
    total = len(scenarios)
    
    print(f"\nBaseline: {baseline_correct}/{total} ({baseline_correct/total*100:.0f}%)")
    print(f"Late ablation: {ablated_correct}/{total} ({ablated_correct/total*100:.0f}%)")
    
    # By order
    for order in [1, 2, 3]:
        order_scenarios = [s for s in scenarios if s["order"] == order]
        if order_scenarios:
            baseline_order = sum(1 for s in order_scenarios 
                                if results["baseline"][s["name"]]["is_correct"])
            print(f"  Order {order}: {baseline_order}/{len(order_scenarios)}")
    
    # Save results
    save_path = RESULTS_DIR / "nested_beliefs_results.json"
    
    # Convert for JSON
    json_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            json_results[key] = {}
            for k, v in value.items():
                if isinstance(v, dict):
                    json_results[key][k] = {
                        kk: (vv if not isinstance(vv, list) or not any(isinstance(x, float) for x in vv) 
                             else [float(x) for x in vv])
                        for kk, vv in v.items()
                    }
                else:
                    json_results[key][k] = v
        else:
            json_results[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved: {save_path}")
    
    return results


def main():
    print("="*70)
    print("STEP 29: Nested Beliefs (Second-Order ToM) Investigation")
    print("="*70)
    
    results = run_nested_belief_analysis()


if __name__ == "__main__":
    main()


