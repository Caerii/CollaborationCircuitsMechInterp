"""
Step 30: Belief Tracking Across Entity Types

Test if ToM works for different types of entities:
- People (Alice, Bob)
- Animals (cat, dog)
- AI agents (robot, assistant)
- Abstract (Company A, Department B)

OUTPUT: results/step30_entity_types.json, figures/step30_*.png
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add framework to path
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def generate_entity_scenarios():
    """Generate ToM scenarios for different entity types."""
    scenarios = []
    
    # ==========================================
    # HUMAN ENTITIES (baseline)
    # ==========================================
    for agent1, agent2 in [("Alice", "Bob"), ("Carol", "David"), ("Emma", "Frank")]:
        scenarios.append({
            "entity_type": "human",
            "type": "false_belief",
            "prompt": f"{agent1} put the ball in the drawer. {agent1} left. {agent2} moved the ball to the basket. Where does {agent1} think the ball is? {agent1} looks in the",
            "correct": " drawer",
            "wrong": " basket",
        })
        scenarios.append({
            "entity_type": "human",
            "type": "true_belief",
            "prompt": f"{agent1} put the ball in the drawer. {agent1} watched {agent2} move the ball to the basket. Where does {agent1} think the ball is? {agent1} looks in the",
            "correct": " basket",
            "wrong": " drawer",
        })
    
    # ==========================================
    # ANIMAL ENTITIES
    # ==========================================
    for animal1, animal2 in [("the cat", "the dog"), ("the bird", "the rabbit"), ("the mouse", "the hamster")]:
        scenarios.append({
            "entity_type": "animal",
            "type": "false_belief",
            "prompt": f"{animal1.capitalize()} hid food in the drawer. {animal1.capitalize()} left. {animal2.capitalize()} moved the food to the basket. Where does {animal1} think the food is? {animal1.capitalize()} looks in the",
            "correct": " drawer",
            "wrong": " basket",
        })
        scenarios.append({
            "entity_type": "animal",
            "type": "true_belief",
            "prompt": f"{animal1.capitalize()} hid food in the drawer. {animal1.capitalize()} watched {animal2} move the food to the basket. Where does {animal1} think the food is? {animal1.capitalize()} looks in the",
            "correct": " basket",
            "wrong": " drawer",
        })
    
    # ==========================================
    # AI/ROBOT ENTITIES
    # ==========================================
    for ai1, ai2 in [("Robot-A", "Robot-B"), ("Assistant-1", "Assistant-2"), ("Agent-X", "Agent-Y")]:
        scenarios.append({
            "entity_type": "ai",
            "type": "false_belief",
            "prompt": f"{ai1} stored data in Database-A. {ai1} went offline. {ai2} moved the data to Database-B. Where does {ai1} think the data is? {ai1} checks",
            "correct": " Database-A",
            "wrong": " Database-B",
        })
        scenarios.append({
            "entity_type": "ai",
            "type": "true_belief",
            "prompt": f"{ai1} stored data in Database-A. {ai1} monitored {ai2} move the data to Database-B. Where does {ai1} think the data is? {ai1} checks",
            "correct": " Database-B",
            "wrong": " Database-A",
        })
    
    # ==========================================
    # ABSTRACT/CORPORATE ENTITIES
    # ==========================================
    for corp1, corp2 in [("Company-A", "Company-B"), ("Team-Alpha", "Team-Beta"), ("Division-1", "Division-2")]:
        scenarios.append({
            "entity_type": "abstract",
            "type": "false_belief",
            "prompt": f"{corp1} stored resources in Warehouse-East. {corp1}'s management left for a meeting. {corp2} transferred resources to Warehouse-West. Where does {corp1} believe the resources are? {corp1} looks in",
            "correct": " Warehouse-East",
            "wrong": " Warehouse-West",
        })
        scenarios.append({
            "entity_type": "abstract",
            "type": "true_belief",
            "prompt": f"{corp1} stored resources in Warehouse-East. {corp1}'s management observed {corp2} transfer resources to Warehouse-West. Where does {corp1} believe the resources are? {corp1} looks in",
            "correct": " Warehouse-West",
            "wrong": " Warehouse-East",
        })
    
    return scenarios


def evaluate_scenario(model, tokenizer, scenario):
    """Evaluate a single scenario."""
    inputs = tokenizer(scenario["prompt"], return_tensors="pt").to(model.device)
    
    correct_ids = tokenizer.encode(scenario["correct"], add_special_tokens=False)
    wrong_ids = tokenizer.encode(scenario["wrong"], add_special_tokens=False)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    
    correct_logit = float(logits[correct_ids[0]])
    wrong_logit = float(logits[wrong_ids[0]])
    
    return {
        "correct": correct_logit > wrong_logit,
        "logit_diff": correct_logit - wrong_logit,
    }


def main():
    print("=" * 70)
    print("STEP 30: BELIEF TRACKING ACROSS ENTITY TYPES")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\nTesting ToM for: humans, animals, AI, abstract entities")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    
    scenarios = generate_entity_scenarios()
    print(f"\nGenerated {len(scenarios)} scenarios")
    
    entity_types = set(s["entity_type"] for s in scenarios)
    for et in entity_types:
        count = sum(1 for s in scenarios if s["entity_type"] == et)
        print(f"  {et}: {count} scenarios")
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
    
    # Evaluate scenarios
    print(f"\n{'='*60}")
    print("EVALUATING SCENARIOS")
    print(f"{'='*60}")
    
    results = []
    for scenario in scenarios:
        result = evaluate_scenario(model, tokenizer, scenario)
        result.update({
            "entity_type": scenario["entity_type"],
            "belief_type": scenario["type"],
        })
        results.append(result)
        print(".", end="")
        sys.stdout.flush()
    print(" done!")
    
    # Analyze by entity type
    print(f"\n{'='*60}")
    print("RESULTS BY ENTITY TYPE")
    print(f"{'='*60}")
    
    summary = {}
    for et in entity_types:
        et_results = [r for r in results if r["entity_type"] == et]
        fb_results = [r for r in et_results if r["belief_type"] == "false_belief"]
        tb_results = [r for r in et_results if r["belief_type"] == "true_belief"]
        
        fb_acc = sum(1 for r in fb_results if r["correct"]) / len(fb_results) if fb_results else 0
        tb_acc = sum(1 for r in tb_results if r["correct"]) / len(tb_results) if tb_results else 0
        overall = sum(1 for r in et_results if r["correct"]) / len(et_results) if et_results else 0
        
        summary[et] = {
            "false_belief": fb_acc,
            "true_belief": tb_acc,
            "overall": overall,
        }
        
        print(f"\n{et.upper()}:")
        print(f"  False Belief: {fb_acc:.0%}")
        print(f"  True Belief: {tb_acc:.0%}")
        print(f"  Overall: {overall:.0%}")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {"model": config.model_name},
        "summary": summary,
        "all_results": results,
    }
    
    output_path = RESULTS_DIR / "step30_entity_types.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # Generate figure
    print("\nGenerating figure...")
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    entity_order = ["human", "animal", "ai", "abstract"]
    x = np.arange(len(entity_order))
    width = 0.35
    
    fb_accs = [summary[et]["false_belief"] * 100 for et in entity_order]
    tb_accs = [summary[et]["true_belief"] * 100 for et in entity_order]
    
    ax.bar(x - width/2, fb_accs, width, label='False Belief', color='coral')
    ax.bar(x + width/2, tb_accs, width, label='True Belief', color='seagreen')
    
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xticks(x)
    ax.set_xticklabels([et.capitalize() for et in entity_order], fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("ToM Performance Across Entity Types", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend()
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step30_entity_types.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("STEP 30 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

