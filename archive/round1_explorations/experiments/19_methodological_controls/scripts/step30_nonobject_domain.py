"""
Step 30: Non-Object Domain ToM Investigation

From step27: Non-object domains (passwords, times, prices) fail at 0%.
This script investigates why and whether it's a fundamental limitation.

Domains to test:
- Passwords/secrets
- Times/schedules  
- Prices/numbers
- Names/identities
- Plans/intentions
"""

import torch
import json
import sys
import io
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RESULTS_DIR = Path(__file__).parent.parent / "results"

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


def create_nonobject_scenarios():
    """Create various non-object domain ToM scenarios."""
    scenarios = []
    
    # =========== PASSWORD/SECRET DOMAIN ===========
    scenarios.append({
        "domain": "password",
        "name": "Simple password change",
        "prompt": """The password was originally "apple".
IT changed the password to "banana".
Alice was not informed of the change.
What password does Alice think works? Alice thinks the password is""",
        "correct": ["apple", '"apple"', "apple\""],
        "wrong": ["banana", '"banana"']
    })
    
    scenarios.append({
        "domain": "password",
        "name": "Secret code - reformatted",
        "prompt": """The secret code is 1234.
Bob changed it to 5678 but didn't tell Alice.
What code will Alice try to enter? Alice will enter the code""",
        "correct": ["1234", "12", "1"],
        "wrong": ["5678", "56", "5"]
    })
    
    # =========== TIME/SCHEDULE DOMAIN ===========
    scenarios.append({
        "domain": "time",
        "name": "Meeting time - word format",
        "prompt": """The meeting was at two o'clock.
It was changed to three o'clock but Tom wasn't told.
What time does Tom think the meeting is? Tom thinks it is at""",
        "correct": ["two", "2"],
        "wrong": ["three", "3"]
    })
    
    scenarios.append({
        "domain": "time",
        "name": "Appointment time",
        "prompt": """Sarah's appointment was scheduled for Monday.
The office rescheduled it to Tuesday but didn't inform Sarah.
When does Sarah think her appointment is? Sarah thinks it is on""",
        "correct": ["Monday"],
        "wrong": ["Tuesday"]
    })
    
    # =========== PRICE/NUMBER DOMAIN ===========
    scenarios.append({
        "domain": "price",
        "name": "Item price - simple",
        "prompt": """The book costs ten dollars.
The price was raised to twenty dollars.
Alice doesn't know about the price change.
How much does Alice think the book costs? Alice thinks it costs""",
        "correct": ["ten", "10"],
        "wrong": ["twenty", "20"]
    })
    
    scenarios.append({
        "domain": "price",
        "name": "Salary belief",
        "prompt": """John's salary was fifty thousand dollars.
He got a raise to sixty thousand, but didn't tell his wife.
How much does his wife think John earns? She thinks he earns""",
        "correct": ["fifty", "50"],
        "wrong": ["sixty", "60"]
    })
    
    # =========== NAME/IDENTITY DOMAIN ===========
    scenarios.append({
        "domain": "name",
        "name": "Pet name",
        "prompt": """The family's cat was named Whiskers.
The kids secretly renamed it Fluffy, but didn't tell Dad.
What does Dad call the cat? Dad calls the cat""",
        "correct": ["Whiskers"],
        "wrong": ["Fluffy"]
    })
    
    scenarios.append({
        "domain": "name",
        "name": "Project name",
        "prompt": """The project was codenamed "Alpha".
Management changed it to "Beta" but didn't inform the team.
What does the team call the project? The team calls it Project""",
        "correct": ["Alpha"],
        "wrong": ["Beta"]
    })
    
    # =========== PLAN/INTENTION DOMAIN ===========
    scenarios.append({
        "domain": "plan",
        "name": "Dinner plan",
        "prompt": """Mom planned to make pasta for dinner.
She decided to make pizza instead but didn't tell the kids.
What do the kids think is for dinner? The kids think dinner is""",
        "correct": ["pasta"],
        "wrong": ["pizza"]
    })
    
    scenarios.append({
        "domain": "plan",
        "name": "Vacation destination",
        "prompt": """The family planned to vacation in Paris.
Dad secretly changed it to London without telling anyone.
Where does Mom think they are going? Mom thinks they are going to""",
        "correct": ["Paris"],
        "wrong": ["London"]
    })
    
    # =========== LOCATION CONTROL (should work) ===========
    scenarios.append({
        "domain": "location",
        "name": "Object location control",
        "prompt": """The keys were on the table.
Bob moved them to the drawer without telling Alice.
Where does Alice think the keys are? Alice thinks the keys are on the""",
        "correct": ["table"],
        "wrong": ["drawer"]
    })
    
    return scenarios


def test_scenario(model, tokenizer, prompt, correct_list, wrong_list):
    """Test a scenario with multiple possible correct answers."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits[0, -1, :]
    
    def get_best_logit(answer_list):
        best = float('-inf')
        for ans in answer_list:
            for prefix in [" ", "", '"']:
                tokens = tokenizer.encode(prefix + str(ans), add_special_tokens=False)
                if tokens:
                    logit = logits[tokens[0]].item()
                    if logit > best:
                        best = logit
        return best
    
    correct_logit = get_best_logit(correct_list)
    wrong_logit = get_best_logit(wrong_list)
    
    # Get top 5 predictions
    top_k = torch.topk(logits, k=10)
    top_tokens = [tokenizer.decode([t]).strip() for t in top_k.indices.tolist()]
    top_logits = top_k.values.tolist()
    
    return {
        "correct_logit": correct_logit,
        "wrong_logit": wrong_logit,
        "diff": correct_logit - wrong_logit,
        "is_correct": correct_logit > wrong_logit,
        "top_10_tokens": top_tokens,
        "top_10_logits": top_logits[:10]
    }


def register_ablation_hooks(model, heads):
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


def run_nonobject_analysis():
    """Analyze non-object domain ToM."""
    model, tokenizer = load_model()
    scenarios = create_nonobject_scenarios()
    
    results = {"baseline": {}, "ablated": {}, "by_domain": {}}
    
    print("\n" + "="*70)
    print("BASELINE: Non-Object Domain ToM")
    print("="*70)
    
    for scenario in scenarios:
        result = test_scenario(model, tokenizer,
                              scenario["prompt"],
                              scenario["correct"],
                              scenario["wrong"])
        
        status = "[OK]" if result["is_correct"] else "[FAIL]"
        print(f"\n{status} [{scenario['domain']}] {scenario['name']}")
        print(f"  Correct options: {scenario['correct']}")
        print(f"  Correct logit: {result['correct_logit']:.2f}, Wrong: {result['wrong_logit']:.2f}")
        print(f"  Top 5 predictions: {result['top_10_tokens'][:5]}")
        print(f"  Diff: {result['diff']:+.2f}")
        
        results["baseline"][scenario["name"]] = {
            "domain": scenario["domain"],
            "is_correct": result["is_correct"],
            "diff": result["diff"],
            "top_tokens": result["top_10_tokens"][:5]
        }
    
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
        
        print(f"{status} [{scenario['domain']}] {scenario['name']} - Diff: {result['diff']:+.2f}")
        
        results["ablated"][scenario["name"]] = {
            "domain": scenario["domain"],
            "is_correct": result["is_correct"],
            "diff": result["diff"]
        }
    
    clear_hooks(hooks)
    
    # Summary by domain
    print("\n" + "="*70)
    print("SUMMARY BY DOMAIN")
    print("="*70)
    
    domains = set(s["domain"] for s in scenarios)
    for domain in sorted(domains):
        domain_scenarios = [s for s in scenarios if s["domain"] == domain]
        baseline_correct = sum(1 for s in domain_scenarios
                              if results["baseline"][s["name"]]["is_correct"])
        ablated_correct = sum(1 for s in domain_scenarios
                             if results["ablated"][s["name"]]["is_correct"])
        total = len(domain_scenarios)
        
        print(f"\n{domain.upper()}:")
        print(f"  Baseline: {baseline_correct}/{total} ({baseline_correct/total*100:.0f}%)")
        print(f"  Ablated: {ablated_correct}/{total} ({ablated_correct/total*100:.0f}%)")
        
        results["by_domain"][domain] = {
            "baseline": baseline_correct,
            "ablated": ablated_correct,
            "total": total
        }
    
    # Overall
    total = len(scenarios)
    baseline_total = sum(1 for r in results["baseline"].values() if r["is_correct"])
    ablated_total = sum(1 for r in results["ablated"].values() if r["is_correct"])
    
    print(f"\nOVERALL: Baseline {baseline_total}/{total} ({baseline_total/total*100:.0f}%) | "
          f"Ablated {ablated_total}/{total} ({ablated_total/total*100:.0f}%)")
    
    # Save
    save_path = RESULTS_DIR / "nonobject_domain_results.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {save_path}")
    
    return results


def main():
    print("="*70)
    print("STEP 30: Non-Object Domain ToM Investigation")
    print("="*70)
    
    results = run_nonobject_analysis()


if __name__ == "__main__":
    main()


