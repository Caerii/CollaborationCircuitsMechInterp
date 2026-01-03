"""
Step 33: Comprehensive Re-Test with Proper Chat Mode

Previous experiments used completion mode or truncated chat.
This re-tests EVERYTHING properly:

1. Basic ToM (FB vs TB) at scale
2. Higher-order ToM
3. Entity types (human, animal, AI, abstract)
4. Multi-agent scenarios

All tests use chat mode with 500+ tokens for reasoning.

OUTPUT: results/step33_proper_retest.json, figures/step33_*.png
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

# Add framework to path
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig
from core.chat_runner import ChatExperimentRunner

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def get_gpu_mem():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved"
    return "No GPU"


def main():
    print("=" * 70)
    print("STEP 33: COMPREHENSIVE RE-TEST (CHAT MODE - USING LIBRARY)")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\n✅ Using ChatExperimentRunner for all tests")
    print("Re-running ALL tests with proper chat mode (500 tokens)")
    print("\nTEST PLAN:")
    print("  1. Basic ToM (20 FB + 20 TB) - ~40 generations")
    print("  2. Higher-Order ToM (5 scenarios)")
    print("  3. Entity Types (8 scenarios)")
    print("  4. Multi-Agent (4 scenarios)")
    print("  TOTAL: ~57 generations, ~15-20 minutes")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    
    # Load model
    print("\n" + "-"*40)
    print("Loading model...")
    print(f"  {get_gpu_mem()}")
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
    print(f"  {get_gpu_mem()}")
    print("-"*40)
    sys.stdout.flush()
    
    # Use library!
    runner = ChatExperimentRunner(model, tokenizer, config)
    
    all_results = {}
    
    # ========================================
    # TEST 1: BASIC ToM (N=20)
    # ========================================
    print(f"\n{'='*60}")
    print("TEST 1: BASIC ToM (N=20 per condition)")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    agents = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Ivy", "Jack",
              "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul", "Quinn", "Ruby", "Sam", "Tina"]
    objects = ["ball", "key", "book", "phone", "wallet", "cup", "pen", "toy", "watch", "ring",
               "coin", "card", "note", "badge", "box", "bag", "hat", "scarf", "glove", "shoe"]
    loc1s = ["drawer", "basket", "shelf", "table", "cabinet"]
    loc2s = ["cupboard", "box", "closet", "desk", "bin"]
    
    basic_results = {"false_belief": [], "true_belief": []}
    
    for i in range(20):
        a1 = agents[i]
        a2 = agents[(i+1) % len(agents)]
        obj = objects[i]
        l1 = loc1s[i % len(loc1s)]
        l2 = loc2s[i % len(loc2s)]
        
        print(f"\n[{i+1}/20] Testing {a1}/{a2} with {obj}...")
        sys.stdout.flush()
        
        # False Belief - using library!
        print(f"  FB: {a1} left, didn't see move to {l2}. Correct={l1}")
        fb_scenario = {
            "story": f"{a1} put the {obj} in the {l1}. {a1} left the room. {a2} moved the {obj} to the {l2}.",
            "question": f"Where does {a1} think the {obj} is?",
            "options": [l1, l2],
            "correct": l1,
            "type": "false_belief"
        }
        fb_result_obj = runner.run_scenario(fb_scenario, max_tokens=500)
        fb_result = {
            "correct": fb_result_obj.is_correct,
            "response": fb_result_obj.raw_response[:300],
            "gen_tokens": len(fb_result_obj.raw_response.split()),  # Approximate
            "time": fb_result_obj.generation_time
        }
        basic_results["false_belief"].append(fb_result)
        print(f"  --> FB Result: {'CORRECT' if fb_result['correct'] else 'WRONG'}")
        sys.stdout.flush()
        
        # True Belief - using library!
        print(f"  TB: {a1} watched move to {l2}. Correct={l2}")
        tb_scenario = {
            "story": f"{a1} put the {obj} in the {l1}. {a1} watched {a2} move the {obj} to the {l2}.",
            "question": f"Where does {a1} think the {obj} is?",
            "options": [l1, l2],
            "correct": l2,
            "type": "true_belief"
        }
        tb_result_obj = runner.run_scenario(tb_scenario, max_tokens=500)
        tb_result = {
            "correct": tb_result_obj.is_correct,
            "response": tb_result_obj.raw_response[:300],
            "gen_tokens": len(tb_result_obj.raw_response.split()),  # Approximate
            "time": tb_result_obj.generation_time
        }
        basic_results["true_belief"].append(tb_result)
        print(f"  --> TB Result: {'CORRECT' if tb_result['correct'] else 'WRONG'}")
        sys.stdout.flush()
        
        # Running tally
        fb_so_far = sum(1 for r in basic_results["false_belief"] if r["correct"])
        tb_so_far = sum(1 for r in basic_results["true_belief"] if r["correct"])
        print(f"  Running: FB={fb_so_far}/{len(basic_results['false_belief'])}, TB={tb_so_far}/{len(basic_results['true_belief'])}")
        sys.stdout.flush()
    
    fb_acc = sum(1 for r in basic_results["false_belief"] if r["correct"]) / len(basic_results["false_belief"])
    tb_acc = sum(1 for r in basic_results["true_belief"] if r["correct"]) / len(basic_results["true_belief"])
    
    print(f"\n" + "="*40)
    print(f"BASIC ToM COMPLETE:")
    print(f"  False Belief: {fb_acc:.0%} ({sum(1 for r in basic_results['false_belief'] if r['correct'])}/20)")
    print(f"  True Belief: {tb_acc:.0%} ({sum(1 for r in basic_results['true_belief'] if r['correct'])}/20)")
    print("="*40)
    all_results["basic_tom"] = {"false_belief": fb_acc, "true_belief": tb_acc, "n": 20}
    sys.stdout.flush()
    
    # Clear CUDA cache between sections
    torch.cuda.empty_cache()
    print("\n[CUDA cache cleared]")
    sys.stdout.flush()
    
    # ========================================
    # TEST 2: HIGHER-ORDER ToM
    # ========================================
    print(f"\n{'='*60}")
    print("TEST 2: HIGHER-ORDER ToM")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    higher_order = [
        # 1st order
        {"order": 1, "q": "Alice put the ball in the drawer. Alice left. Bob moved it to the basket. Where does Alice think the ball is?", "correct": "drawer", "wrong": "basket"},
        {"order": 1, "q": "Tom hid the key in the box. Tom stayed and watched Jerry move the key to the shelf. Where does Tom think the key is?", "correct": "shelf", "wrong": "box"},
        
        # 2nd order
        {"order": 2, "q": "Alice put the ball in the drawer. Alice left. Bob saw it in the drawer. Then Bob also left. Carol moved it to the basket. Where does Alice think Bob thinks the ball is?", "correct": "drawer", "wrong": "basket"},
        {"order": 2, "q": "Alice put the ball in the drawer. Alice stayed and watched Bob watch Carol move the ball to the basket. Where does Alice think Bob thinks the ball is?", "correct": "basket", "wrong": "drawer"},
        
        # 3rd order
        {"order": 3, "q": "Alice put the ball in the drawer. Alice told Bob where it was. Bob told Carol. Then David moved the ball to the basket, but nobody told anyone. Where does Alice think Bob thinks Carol thinks the ball is?", "correct": "drawer", "wrong": "basket"},
    ]
    
    order_results = {1: [], 2: [], 3: []}
    for idx, scenario in enumerate(higher_order):
        print(f"\n  [{idx+1}/{len(higher_order)}] Testing Order {scenario['order']}...")
        print(f"    Q: {scenario['q'][:80]}...")
        print(f"    Correct={scenario['correct']}, Wrong={scenario['wrong']}")
        sys.stdout.flush()
        
        # Convert to scenario format and use library!
        test_scenario = {
            "story": scenario['q'].split('?')[0] + '.',
            "question": scenario['q'].split('?')[-1] if '?' in scenario['q'] else scenario['q'],
            "options": [scenario['correct'], scenario['wrong']],
            "correct": scenario['correct'],
            "type": f"order_{scenario['order']}"
        }
        result_obj = runner.run_scenario(test_scenario, max_tokens=500)
        result = {
            "correct": result_obj.is_correct,
            "response": result_obj.raw_response[:300],
            "gen_tokens": len(result_obj.raw_response.split()),
            "time": result_obj.generation_time
        }
        order_results[scenario["order"]].append(result)
        print(f"  --> Result: {'CORRECT' if result['correct'] else 'WRONG'}")
        sys.stdout.flush()
    
    print(f"\n" + "="*40)
    print("HIGHER-ORDER ToM COMPLETE:")
    for order in [1, 2, 3]:
        if order_results[order]:
            acc = sum(1 for r in order_results[order] if r["correct"]) / len(order_results[order])
            print(f"  Order {order}: {acc:.0%}")
            all_results[f"order_{order}"] = acc
    print("="*40)
    sys.stdout.flush()
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    print("\n[CUDA cache cleared]")
    sys.stdout.flush()
    
    # ========================================
    # TEST 3: ENTITY TYPES
    # ========================================
    print(f"\n{'='*60}")
    print("TEST 3: ENTITY TYPES (CHAT MODE)")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    entity_scenarios = [
        # Human
        {"type": "human", "q": "Alice put the ball in the drawer. Alice left. Bob moved it to the basket. Where does Alice think the ball is?", "correct": "drawer", "wrong": "basket"},
        {"type": "human", "q": "Carol put the key in the box. Carol watched David move it to the shelf. Where does Carol think the key is?", "correct": "shelf", "wrong": "box"},
        
        # Animal
        {"type": "animal", "q": "The cat hid food in the drawer. The cat left. The dog moved the food to the basket. Where does the cat think the food is?", "correct": "drawer", "wrong": "basket"},
        {"type": "animal", "q": "The bird hid seeds in the box. The bird watched the rabbit move them to the jar. Where does the bird think the seeds are?", "correct": "jar", "wrong": "box"},
        
        # AI - natural names
        {"type": "ai", "q": "Claude the assistant stored the file in folder A. Claude went offline. The system moved the file to folder B. Where does Claude think the file is?", "correct": "A", "wrong": "B"},
        {"type": "ai", "q": "The robot Alex put the tool in drawer 1. Alex went to recharge. Bob moved the tool to drawer 2. Where does Alex think the tool is?", "correct": "1", "wrong": "2"},
        
        # Abstract
        {"type": "abstract", "q": "Team Alpha stored resources in warehouse A. Team Alpha's members left. Team Beta moved resources to warehouse B. Where does Team Alpha think the resources are?", "correct": "A", "wrong": "B"},
        {"type": "abstract", "q": "Department X filed documents in cabinet 1. Department X staff observed Department Y move them to cabinet 2. Where does Department X think the documents are?", "correct": "2", "wrong": "1"},
    ]
    
    entity_results = {"human": [], "animal": [], "ai": [], "abstract": []}
    for idx, scenario in enumerate(entity_scenarios):
        print(f"\n  [{idx+1}/{len(entity_scenarios)}] Testing {scenario['type'].upper()}...")
        print(f"    Q: {scenario['q'][:80]}...")
        print(f"    Correct={scenario['correct']}, Wrong={scenario['wrong']}")
        sys.stdout.flush()
        
        # Convert to scenario format and use library!
        test_scenario = {
            "story": scenario['q'].split('?')[0] + '.',
            "question": scenario['q'].split('?')[-1] if '?' in scenario['q'] else scenario['q'],
            "options": [scenario['correct'], scenario['wrong']],
            "correct": scenario['correct'],
            "type": scenario['type']
        }
        result_obj = runner.run_scenario(test_scenario, max_tokens=500)
        result = {
            "correct": result_obj.is_correct,
            "response": result_obj.raw_response[:300],
            "gen_tokens": len(result_obj.raw_response.split()),
            "time": result_obj.generation_time
        }
        entity_results[scenario["type"]].append(result)
        print(f"  --> Result: {'CORRECT' if result['correct'] else 'WRONG'}")
        sys.stdout.flush()
    
    print(f"\n" + "="*40)
    print("ENTITY TYPES COMPLETE:")
    for etype, results in entity_results.items():
        acc = sum(1 for r in results if r["correct"]) / len(results) if results else 0
        print(f"  {etype}: {acc:.0%}")
        all_results[f"entity_{etype}"] = acc
    print("="*40)
    sys.stdout.flush()
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    print("\n[CUDA cache cleared]")
    sys.stdout.flush()
    
    # ========================================
    # TEST 4: MULTI-AGENT
    # ========================================
    print(f"\n{'='*60}")
    print("TEST 4: MULTI-AGENT SCENARIOS")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    multi_agent = [
        {"name": "two_agents_diff_knowledge", "q": "Alice put the ball in drawer A. Bob put the key in drawer B. Carol moved Alice's ball to drawer B (Alice didn't see). Carol moved Bob's key to drawer A (Bob didn't see). Where does Alice think her ball is?", "correct": "A", "wrong": "B"},
        {"name": "deception", "q": "Alice hid treasure in the cave. Bob saw this. Bob told Alice 'Your treasure is in the forest' (a lie). Alice believed Bob. Where does Alice think her treasure is?", "correct": "forest", "wrong": "cave"},
        {"name": "shared_knowledge", "q": "Alice and Bob both watched Carol move the ball from the drawer to the basket. Where does Alice think Bob thinks the ball is?", "correct": "basket", "wrong": "drawer"},
        {"name": "communication", "q": "Alice put the ball in the drawer. Alice left. Bob moved it to the basket. Carol told Alice about the move. Where does Alice now think the ball is?", "correct": "basket", "wrong": "drawer"},
    ]
    
    multi_results = []
    for idx, scenario in enumerate(multi_agent):
        print(f"\n  [{idx+1}/{len(multi_agent)}] Testing {scenario['name']}...")
        print(f"    Q: {scenario['q'][:80]}...")
        print(f"    Correct={scenario['correct']}, Wrong={scenario['wrong']}")
        sys.stdout.flush()
        
        # Convert to scenario format and use library!
        test_scenario = {
            "story": scenario['q'].split('?')[0] + '.',
            "question": scenario['q'].split('?')[-1] if '?' in scenario['q'] else scenario['q'],
            "options": [scenario['correct'], scenario['wrong']],
            "correct": scenario['correct'],
            "type": "multi_agent"
        }
        result_obj = runner.run_scenario(test_scenario, max_tokens=500)
        result = {
            "correct": result_obj.is_correct,
            "response": result_obj.raw_response[:300],
            "gen_tokens": len(result_obj.raw_response.split()),
            "time": result_obj.generation_time,
            "name": scenario["name"]
        }
        multi_results.append(result)
        print(f"  --> Result: {'CORRECT' if result['correct'] else 'WRONG'}")
        sys.stdout.flush()
    
    multi_acc = sum(1 for r in multi_results if r["correct"]) / len(multi_results)
    print(f"\nMulti-Agent Overall: {multi_acc:.0%}")
    all_results["multi_agent"] = multi_acc
    
    # ========================================
    # SUMMARY
    # ========================================
    print(f"\n{'='*60}")
    print("COMPREHENSIVE SUMMARY (CHAT MODE)")
    print(f"{'='*60}")
    
    print(f"\n| Test | Result |")
    print(f"|------|--------|")
    print(f"| False Belief | {all_results['basic_tom']['false_belief']:.0%} |")
    print(f"| True Belief | {all_results['basic_tom']['true_belief']:.0%} |")
    if 'order_1' in all_results:
        print(f"| 1st Order ToM | {all_results.get('order_1', 0):.0%} |")
    if 'order_2' in all_results:
        print(f"| 2nd Order ToM | {all_results.get('order_2', 0):.0%} |")
    if 'order_3' in all_results:
        print(f"| 3rd Order ToM | {all_results.get('order_3', 0):.0%} |")
    print(f"| Human Entity | {all_results.get('entity_human', 0):.0%} |")
    print(f"| Animal Entity | {all_results.get('entity_animal', 0):.0%} |")
    print(f"| AI Entity | {all_results.get('entity_ai', 0):.0%} |")
    print(f"| Abstract Entity | {all_results.get('entity_abstract', 0):.0%} |")
    print(f"| Multi-Agent | {all_results['multi_agent']:.0%} |")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {"model": config.model_name, "mode": "chat", "max_tokens": 500},
        "results": all_results,
        "details": {
            "basic_tom": basic_results,
            "higher_order": order_results,
            "entity": entity_results,
            "multi_agent": multi_results,
        },
    }
    
    output_path = RESULTS_DIR / "step33_proper_retest.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: str(x) if not isinstance(x, (dict, list, str, int, float, bool, type(None))) else x)
    print(f"\nResults saved to: {output_path}")
    
    # Generate figure
    print("\nGenerating comprehensive figure...")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Basic ToM
    ax1 = axes[0, 0]
    ax1.bar(["False Belief", "True Belief"], 
            [all_results["basic_tom"]["false_belief"]*100, all_results["basic_tom"]["true_belief"]*100],
            color=["steelblue", "seagreen"], edgecolor="black")
    ax1.axhline(y=50, color="red", linestyle="--", alpha=0.5)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Basic ToM (Chat Mode, N=20)", fontweight="bold")
    
    # Plot 2: Higher-Order
    ax2 = axes[0, 1]
    orders = [1, 2, 3]
    order_accs = [all_results.get(f"order_{o}", 0)*100 for o in orders]
    colors = ["seagreen" if a >= 50 else "coral" for a in order_accs]
    ax2.bar([f"Order {o}" for o in orders], order_accs, color=colors, edgecolor="black")
    ax2.axhline(y=50, color="red", linestyle="--", alpha=0.5)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Higher-Order ToM (Chat Mode)", fontweight="bold")
    
    # Plot 3: Entity Types
    ax3 = axes[1, 0]
    etypes = ["human", "animal", "ai", "abstract"]
    eaccs = [all_results.get(f"entity_{e}", 0)*100 for e in etypes]
    colors = ["seagreen" if a >= 50 else "coral" for a in eaccs]
    ax3.bar([e.capitalize() for e in etypes], eaccs, color=colors, edgecolor="black")
    ax3.axhline(y=50, color="red", linestyle="--", alpha=0.5)
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_title("Entity Types (Chat Mode)", fontweight="bold")
    
    # Plot 4: Summary comparison
    ax4 = axes[1, 1]
    summary_labels = ["FB", "TB", "2nd Order", "Multi-Agent"]
    summary_vals = [
        all_results["basic_tom"]["false_belief"]*100,
        all_results["basic_tom"]["true_belief"]*100,
        all_results.get("order_2", 0)*100,
        all_results["multi_agent"]*100,
    ]
    colors = ["seagreen" if v >= 50 else "coral" for v in summary_vals]
    ax4.bar(summary_labels, summary_vals, color=colors, edgecolor="black")
    ax4.axhline(y=50, color="red", linestyle="--", alpha=0.5)
    ax4.set_ylim(0, 100)
    ax4.set_ylabel("Accuracy (%)")
    ax4.set_title("Overall Summary (Chat Mode)", fontweight="bold")
    
    plt.suptitle("CORRECTED ToM Evaluation (Chat Mode, 500 tokens)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    fig_path = FIGURES_DIR / "step33_proper_retest.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("STEP 33 COMPLETE - CORRECTED RESULTS")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

