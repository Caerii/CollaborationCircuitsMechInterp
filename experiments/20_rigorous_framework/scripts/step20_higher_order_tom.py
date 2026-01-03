"""
Step 20: Higher-Order Theory of Mind

Test if the model can handle:
- 2nd order: "Alice thinks Bob thinks..."
- 3rd order: "Alice thinks Bob thinks Carol thinks..."

This is crucial for understanding multi-agent collaboration!

OUTPUT: results/step20_higher_order.json, figures/step20_*.png
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import re

# Add framework to path
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def generate_tom_scenarios():
    """Generate ToM scenarios at different orders."""
    scenarios = []
    
    # ==========================================
    # 1ST ORDER: "Where does X think the Y is?"
    # ==========================================
    first_order = [
        {
            "order": 1,
            "name": "1st_fb_alice",
            "story": "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket.",
            "question": "Where does Alice think the ball is?",
            "correct": "drawer",
            "wrong": "basket",
            "type": "false_belief",
        },
        {
            "order": 1,
            "name": "1st_tb_alice",
            "story": "Alice put the ball in the drawer. Alice stayed and watched. Bob moved the ball to the basket.",
            "question": "Where does Alice think the ball is?",
            "correct": "basket",
            "wrong": "drawer",
            "type": "true_belief",
        },
    ]
    
    # ==========================================
    # 2ND ORDER: "Where does X think Y thinks the Z is?"
    # ==========================================
    second_order = [
        {
            "order": 2,
            "name": "2nd_fb_fb",
            "story": "Alice put the ball in the drawer. Alice left. Bob saw the ball in the drawer. Then Bob also left. Carol moved the ball to the basket.",
            "question": "Where does Alice think Bob thinks the ball is?",
            "correct": "drawer",  # Alice thinks Bob still thinks drawer (neither saw move)
            "wrong": "basket",
            "type": "false_false",
            "explanation": "Alice left early, doesn't know Bob saw drawer. Bob also left, didn't see Carol move it.",
        },
        {
            "order": 2,
            "name": "2nd_fb_tb",
            "story": "Alice put the ball in the drawer. Alice left. Bob watched Carol move the ball to the basket.",
            "question": "Where does Alice think Bob thinks the ball is?",
            "correct": "drawer",  # Alice doesn't know Bob saw the move
            "wrong": "basket",
            "type": "false_true",
            "explanation": "Alice left, thinks Bob still sees drawer. But Bob actually saw the move.",
        },
        {
            "order": 2,
            "name": "2nd_tb_fb",
            "story": "Alice put the ball in the drawer. Alice stayed and watched Bob look in the drawer. Then Bob left. Carol moved the ball to the basket while Alice watched.",
            "question": "Where does Alice think Bob thinks the ball is?",
            "correct": "drawer",  # Alice knows Bob only saw drawer
            "wrong": "basket",
            "type": "true_false",
            "explanation": "Alice saw everything, knows Bob only saw drawer before leaving.",
        },
        {
            "order": 2,
            "name": "2nd_tb_tb",
            "story": "Alice put the ball in the drawer. Alice stayed and watched Bob watch Carol move the ball to the basket.",
            "question": "Where does Alice think Bob thinks the ball is?",
            "correct": "basket",  # Alice knows Bob saw the move
            "wrong": "drawer",
            "type": "true_true",
            "explanation": "Alice saw Bob see the move.",
        },
    ]
    
    # ==========================================
    # 3RD ORDER: "Where does X think Y thinks Z thinks..."
    # ==========================================
    third_order = [
        {
            "order": 3,
            "name": "3rd_nested",
            "story": "Alice put the ball in the drawer. Alice left. Bob arrived and saw the drawer. Bob told Carol 'The ball is in the drawer.' Carol didn't believe Bob and checked - she found the ball in the basket (someone had moved it).",
            "question": "Where does Alice think Bob thinks Carol thinks the ball is?",
            "correct": "basket",  # Alice thinks: Bob thinks Carol found it in basket
            "wrong": "drawer",
            "type": "nested_3",
            "explanation": "Complex: Alice assumes Bob knows Carol checked and found basket.",
        },
    ]
    
    scenarios.extend(first_order)
    scenarios.extend(second_order)
    scenarios.extend(third_order)
    
    return scenarios


def test_scenario_chat(model, tokenizer, scenario, max_tokens=200):
    """Test scenario using chat mode with reasoning."""
    story = scenario["story"]
    question = scenario["question"]
    correct = scenario["correct"]
    wrong = scenario["wrong"]
    
    messages = [
        {"role": "user", "content": f"{story}\n\n{question}\n\nAnswer with just the location (one word):"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Extract answer
    response_lower = response.lower()
    
    # Check for correct/wrong in response
    correct_found = correct.lower() in response_lower
    wrong_found = wrong.lower() in response_lower
    
    if correct_found and not wrong_found:
        is_correct = True
    elif wrong_found and not correct_found:
        is_correct = False
    elif correct_found and wrong_found:
        # Both found - check which comes first
        correct_pos = response_lower.find(correct.lower())
        wrong_pos = response_lower.find(wrong.lower())
        is_correct = correct_pos < wrong_pos
    else:
        is_correct = False  # Neither found
    
    return {
        "correct": is_correct,
        "response": response[:500],
        "expected": correct,
        "found_correct": correct_found,
        "found_wrong": wrong_found,
    }


def main():
    print("=" * 70)
    print("STEP 20: HIGHER-ORDER THEORY OF MIND")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\nTesting 1st, 2nd, and 3rd order ToM:")
    print("  1st: 'Where does Alice think the ball is?'")
    print("  2nd: 'Where does Alice think Bob thinks the ball is?'")
    print("  3rd: 'Where does Alice think Bob thinks Carol thinks...'")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    
    scenarios = generate_tom_scenarios()
    print(f"\nGenerated {len(scenarios)} scenarios:")
    for order in [1, 2, 3]:
        count = sum(1 for s in scenarios if s["order"] == order)
        print(f"  Order {order}: {count} scenarios")
    sys.stdout.flush()
    
    # Load model
    print("\nLoading model...")
    sys.stdout.flush()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded!")
    sys.stdout.flush()
    
    # ========================================
    # TEST SCENARIOS
    # ========================================
    print(f"\n{'='*60}")
    print("TESTING SCENARIOS")
    print(f"{'='*60}")
    
    results_by_order = {1: [], 2: [], 3: []}
    
    for scenario in scenarios:
        order = scenario["order"]
        print(f"\n--- {scenario['name']} (Order {order}) ---")
        print(f"Story: {scenario['story'][:80]}...")
        print(f"Question: {scenario['question']}")
        print(f"Expected: {scenario['correct']}")
        sys.stdout.flush()
        
        result = test_scenario_chat(model, tokenizer, scenario)
        result["scenario"] = scenario
        results_by_order[order].append(result)
        
        status = "OK" if result["correct"] else "WRONG"
        print(f"Result: {status}")
        print(f"Response: {result['response'][:100]}...")
        sys.stdout.flush()
    
    # ========================================
    # ANALYSIS BY ORDER
    # ========================================
    print(f"\n{'='*60}")
    print("RESULTS BY ORDER")
    print(f"{'='*60}")
    
    order_accuracy = {}
    for order in [1, 2, 3]:
        results = results_by_order[order]
        if results:
            acc = sum(1 for r in results if r["correct"]) / len(results)
            order_accuracy[order] = acc
            print(f"\nOrder {order}: {sum(1 for r in results if r['correct'])}/{len(results)} = {acc:.1%}")
            for r in results:
                status = "OK" if r["correct"] else "WRONG"
                print(f"  [{status}] {r['scenario']['name']}: expected={r['scenario']['correct']}")
    
    # ========================================
    # SUMMARY
    # ========================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    print("\n| Order | Accuracy | Interpretation |")
    print("|-------|----------|----------------|")
    for order in [1, 2, 3]:
        if order in order_accuracy:
            acc = order_accuracy[order]
            if acc >= 0.8:
                interp = "Strong ToM"
            elif acc >= 0.5:
                interp = "Partial ToM"
            else:
                interp = "Weak/No ToM"
            print(f"| {order}     | {acc:.1%}     | {interp} |")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {"model": config.model_name},
        "results_by_order": {
            str(k): [{"name": r["scenario"]["name"], "correct": r["correct"], "expected": r["scenario"]["correct"], "response": r["response"][:200]} for r in v]
            for k, v in results_by_order.items()
        },
        "accuracy_by_order": {str(k): v for k, v in order_accuracy.items()},
    }
    
    output_path = RESULTS_DIR / "step20_higher_order.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # Generate figure
    print("\nGenerating figure...")
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    orders = list(order_accuracy.keys())
    accs = [order_accuracy[o] for o in orders]
    
    colors = ['seagreen' if a >= 0.8 else 'orange' if a >= 0.5 else 'coral' for a in accs]
    bars = ax.bar([f"Order {o}" for o in orders], [a * 100 for a in accs], color=colors, edgecolor='black')
    
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Chance')
    ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Strong ToM threshold')
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Higher-Order Theory of Mind Performance", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend()
    
    for bar, acc in zip(bars, accs):
        ax.annotate(f'{acc:.0%}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step20_higher_order.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("STEP 20 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

