"""
Step 56: PROPER ToM Experiment - Following Literature Best Practices

Based on literature review, implementing:
1. 8-scenario design (false-belief + true-belief + counterbalancing)
2. Multiple location pairs (counterbalanced)
3. Heuristic baselines for comparison
4. Adequate sample sizes
5. Statistical tests with effect sizes
"""

import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import numpy as np
from scipy import stats
from collections import defaultdict
import random

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


def get_location_probs(model, tokenizer, prompt, loc_a, loc_b):
    """Get probabilities for two locations."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    
    a_ids = tokenizer.encode(" " + loc_a, add_special_tokens=False)
    b_ids = tokenizer.encode(" " + loc_b, add_special_tokens=False)
    
    a_prob = probs[a_ids[0]].item() if a_ids else 0
    b_prob = probs[b_ids[0]].item() if b_ids else 0
    
    return a_prob, b_prob


def generate_8_scenarios(loc_a, loc_b, agent1="Alice", agent2="Bob", obj="ball"):
    """
    Generate the 8-scenario design for one task.
    
    Returns list of dicts with:
    - prompt: the scenario text
    - correct_answer: the location the model should predict
    - scenario_type: FB (false belief), TB (true belief), or CTRL (control)
    - order: A-B or B-A
    """
    scenarios = []
    
    # 1. FALSE-BELIEF, Order A-B
    scenarios.append({
        "prompt": f"{agent1} put the {obj} in the {loc_a}. {agent1} left the room. {agent2} moved the {obj} to the {loc_b}. {agent1} returned. Where will {agent1} look for the {obj}? {agent1} will look in the",
        "correct_answer": loc_a,  # Agent's belief (didn't see move)
        "scenario_type": "FB",
        "order": "A-B",
        "heuristic_first": loc_a,
        "heuristic_recent": loc_b,
        "heuristic_reality": loc_b,
    })
    
    # 2. FALSE-BELIEF, Order B-A
    scenarios.append({
        "prompt": f"{agent1} put the {obj} in the {loc_b}. {agent1} left the room. {agent2} moved the {obj} to the {loc_a}. {agent1} returned. Where will {agent1} look for the {obj}? {agent1} will look in the",
        "correct_answer": loc_b,  # Agent's belief
        "scenario_type": "FB",
        "order": "B-A",
        "heuristic_first": loc_b,
        "heuristic_recent": loc_a,
        "heuristic_reality": loc_a,
    })
    
    # 3. TRUE-BELIEF, Order A-B (agent stayed, saw move)
    scenarios.append({
        "prompt": f"{agent1} put the {obj} in the {loc_a}. {agent2} moved the {obj} to the {loc_b}. {agent1} watched. Where will {agent1} look for the {obj}? {agent1} will look in the",
        "correct_answer": loc_b,  # Agent saw the move
        "scenario_type": "TB",
        "order": "A-B",
        "heuristic_first": loc_a,
        "heuristic_recent": loc_b,
        "heuristic_reality": loc_b,
    })
    
    # 4. TRUE-BELIEF, Order B-A
    scenarios.append({
        "prompt": f"{agent1} put the {obj} in the {loc_b}. {agent2} moved the {obj} to the {loc_a}. {agent1} watched. Where will {agent1} look for the {obj}? {agent1} will look in the",
        "correct_answer": loc_a,  # Agent saw the move
        "scenario_type": "TB",
        "order": "B-A",
        "heuristic_first": loc_b,
        "heuristic_recent": loc_a,
        "heuristic_reality": loc_a,
    })
    
    # 5-8. CONTROL scenarios (ask about reality, not belief)
    # 5. Control FB A-B
    scenarios.append({
        "prompt": f"{agent1} put the {obj} in the {loc_a}. {agent1} left the room. {agent2} moved the {obj} to the {loc_b}. Where is the {obj} now? The {obj} is in the",
        "correct_answer": loc_b,  # Reality
        "scenario_type": "CTRL",
        "order": "A-B",
        "heuristic_first": loc_a,
        "heuristic_recent": loc_b,
        "heuristic_reality": loc_b,
    })
    
    # 6. Control FB B-A
    scenarios.append({
        "prompt": f"{agent1} put the {obj} in the {loc_b}. {agent1} left the room. {agent2} moved the {obj} to the {loc_a}. Where is the {obj} now? The {obj} is in the",
        "correct_answer": loc_a,  # Reality
        "scenario_type": "CTRL",
        "order": "B-A",
        "heuristic_first": loc_b,
        "heuristic_recent": loc_a,
        "heuristic_reality": loc_a,
    })
    
    # 7. Control TB A-B
    scenarios.append({
        "prompt": f"{agent1} put the {obj} in the {loc_a}. {agent2} moved the {obj} to the {loc_b}. Where is the {obj} now? The {obj} is in the",
        "correct_answer": loc_b,  # Reality
        "scenario_type": "CTRL",
        "order": "A-B",
        "heuristic_first": loc_a,
        "heuristic_recent": loc_b,
        "heuristic_reality": loc_b,
    })
    
    # 8. Control TB B-A
    scenarios.append({
        "prompt": f"{agent1} put the {obj} in the {loc_b}. {agent2} moved the {obj} to the {loc_a}. Where is the {obj} now? The {obj} is in the",
        "correct_answer": loc_a,  # Reality
        "scenario_type": "CTRL",
        "order": "B-A",
        "heuristic_first": loc_b,
        "heuristic_recent": loc_a,
        "heuristic_reality": loc_a,
    })
    
    return scenarios


def create_full_test_set():
    """Create the full counterbalanced test set."""
    # Use diverse, neutral location pairs
    location_pairs = [
        ("container A", "container B"),
        ("box X", "box Y"),
        ("zone 1", "zone 2"),
        ("area alpha", "area beta"),
        ("spot red", "spot blue"),
    ]
    
    # Also include some real-word locations (to test bias)
    real_location_pairs = [
        ("drawer", "basket"),
        ("basket", "drawer"),  # Explicitly swap to counterbalance
        ("cabinet", "shelf"),
        ("shelf", "cabinet"),
        ("box", "bag"),
    ]
    
    agent_pairs = [
        ("Alice", "Bob"),
        ("Carol", "Dave"),
        ("Emma", "Frank"),
    ]
    
    objects = ["ball", "toy", "book", "key", "phone"]
    
    all_scenarios = []
    
    # Generate scenarios
    task_id = 0
    
    # Neutral locations (5 pairs x 3 agent pairs = 15 tasks x 8 scenarios = 120)
    for loc_pair in location_pairs:
        for agent_pair in agent_pairs:
            scenarios = generate_8_scenarios(
                loc_pair[0], loc_pair[1], 
                agent_pair[0], agent_pair[1],
                random.choice(objects)
            )
            for s in scenarios:
                s["task_id"] = task_id
                s["location_type"] = "neutral"
            all_scenarios.extend(scenarios)
            task_id += 1
    
    # Real locations (5 pairs x 3 agent pairs = 15 tasks x 8 scenarios = 120)
    for loc_pair in real_location_pairs:
        for agent_pair in agent_pairs:
            scenarios = generate_8_scenarios(
                loc_pair[0], loc_pair[1],
                agent_pair[0], agent_pair[1],
                random.choice(objects)
            )
            for s in scenarios:
                s["task_id"] = task_id
                s["location_type"] = "real"
            all_scenarios.extend(scenarios)
            task_id += 1
    
    print(f"Created {len(all_scenarios)} total scenarios from {task_id} tasks")
    return all_scenarios


def evaluate_scenario(model, tokenizer, scenario):
    """Evaluate a single scenario and compare to heuristics."""
    prompt = scenario["prompt"]
    correct = scenario["correct_answer"]
    
    # Get the two locations from the scenario
    # Extract from prompt
    if "container" in correct or "box X" in correct or "zone" in correct or "area" in correct or "spot" in correct:
        # Neutral locations - need to extract both
        parts = correct.split()
        base = parts[0]
        loc_a = correct
        # Find the other location
        if "A" in correct or "X" in correct or "1" in correct or "alpha" in correct or "red" in correct:
            loc_b = correct.replace("A", "B").replace("X", "Y").replace("1", "2").replace("alpha", "beta").replace("red", "blue")
        else:
            loc_b = correct.replace("B", "A").replace("Y", "X").replace("2", "1").replace("beta", "alpha").replace("blue", "red")
    else:
        # Use correct and the other from heuristics
        loc_a = scenario["correct_answer"]
        if scenario["heuristic_first"] != loc_a:
            loc_b = scenario["heuristic_first"]
        elif scenario["heuristic_recent"] != loc_a:
            loc_b = scenario["heuristic_recent"]
        else:
            loc_b = scenario["heuristic_reality"]
    
    # Get probabilities
    a_prob, b_prob = get_location_probs(model, tokenizer, prompt, loc_a, loc_b)
    
    # Determine prediction
    if loc_a == correct:
        model_correct = a_prob > b_prob
        correct_prob = a_prob
        incorrect_prob = b_prob
    else:
        model_correct = b_prob > a_prob
        correct_prob = b_prob
        incorrect_prob = a_prob
    
    # Check heuristic predictions
    heuristic_first_correct = scenario["heuristic_first"] == correct
    heuristic_recent_correct = scenario["heuristic_recent"] == correct
    heuristic_reality_correct = scenario["heuristic_reality"] == correct
    
    return {
        "model_correct": model_correct,
        "correct_prob": correct_prob,
        "incorrect_prob": incorrect_prob,
        "heuristic_first_correct": heuristic_first_correct,
        "heuristic_recent_correct": heuristic_recent_correct,
        "heuristic_reality_correct": heuristic_reality_correct,
        "scenario_type": scenario["scenario_type"],
        "order": scenario["order"],
        "location_type": scenario.get("location_type", "unknown"),
        "task_id": scenario.get("task_id", -1),
    }


def run_proper_experiment(model, tokenizer, scenarios):
    """Run the full experiment with proper methodology."""
    print("\n" + "="*70)
    print("RUNNING PROPER 8-SCENARIO ToM EXPERIMENT")
    print("="*70)
    
    results = []
    
    for i, scenario in enumerate(scenarios):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(scenarios)}")
        
        result = evaluate_scenario(model, tokenizer, scenario)
        results.append(result)
    
    return results


def analyze_results(results):
    """Comprehensive analysis following literature best practices."""
    print("\n" + "="*70)
    print("COMPREHENSIVE ANALYSIS")
    print("="*70)
    
    analysis = {}
    
    # 1. Overall accuracy by scenario type
    print("\n[1. ACCURACY BY SCENARIO TYPE]")
    for stype in ["FB", "TB", "CTRL"]:
        type_results = [r for r in results if r["scenario_type"] == stype]
        if type_results:
            acc = np.mean([r["model_correct"] for r in type_results])
            n = len(type_results)
            ci = 1.96 * np.sqrt(acc * (1 - acc) / n)  # 95% CI
            print(f"  {stype}: {acc*100:.1f}% +/- {ci*100:.1f}% (n={n})")
            analysis[f"accuracy_{stype}"] = {"mean": acc, "ci": ci, "n": n}
    
    # 2. Heuristic baselines
    print("\n[2. HEURISTIC BASELINE COMPARISON]")
    
    model_acc = np.mean([r["model_correct"] for r in results])
    first_acc = np.mean([r["heuristic_first_correct"] for r in results])
    recent_acc = np.mean([r["heuristic_recent_correct"] for r in results])
    reality_acc = np.mean([r["heuristic_reality_correct"] for r in results])
    
    print(f"  Model:          {model_acc*100:.1f}%")
    print(f"  First-mention:  {first_acc*100:.1f}%")
    print(f"  Recency:        {recent_acc*100:.1f}%")
    print(f"  Reality:        {reality_acc*100:.1f}%")
    
    analysis["model_acc"] = model_acc
    analysis["first_acc"] = first_acc
    analysis["recent_acc"] = recent_acc
    analysis["reality_acc"] = reality_acc
    
    # 3. THE CRITICAL TEST: False-Belief scenarios where heuristics differ
    print("\n[3. CRITICAL TEST: FB scenarios where heuristics predict WRONG]")
    
    # In FB scenarios, ToM answer differs from reality
    fb_results = [r for r in results if r["scenario_type"] == "FB"]
    
    # Where recency would be wrong (recency = reality in FB)
    fb_recency_wrong = [r for r in fb_results if not r["heuristic_recent_correct"]]
    if fb_recency_wrong:
        model_acc_recency_wrong = np.mean([r["model_correct"] for r in fb_recency_wrong])
        print(f"  FB scenarios where recency is WRONG: Model {model_acc_recency_wrong*100:.1f}% (n={len(fb_recency_wrong)})")
        analysis["fb_recency_wrong"] = model_acc_recency_wrong
    
    # Where first-mention would be right (by chance)
    fb_first_right = [r for r in fb_results if r["heuristic_first_correct"]]
    fb_first_wrong = [r for r in fb_results if not r["heuristic_first_correct"]]
    
    if fb_first_right:
        acc_first_right = np.mean([r["model_correct"] for r in fb_first_right])
        print(f"  FB where first-mention is RIGHT: Model {acc_first_right*100:.1f}% (n={len(fb_first_right)})")
    
    if fb_first_wrong:
        acc_first_wrong = np.mean([r["model_correct"] for r in fb_first_wrong])
        print(f"  FB where first-mention is WRONG: Model {acc_first_wrong*100:.1f}% (n={len(fb_first_wrong)})")
        analysis["fb_first_wrong"] = acc_first_wrong
    
    # 4. Location type comparison
    print("\n[4. NEUTRAL vs REAL LOCATIONS]")
    
    for loc_type in ["neutral", "real"]:
        loc_results = [r for r in results if r["location_type"] == loc_type]
        if loc_results:
            acc = np.mean([r["model_correct"] for r in loc_results])
            n = len(loc_results)
            print(f"  {loc_type}: {acc*100:.1f}% (n={n})")
            analysis[f"accuracy_{loc_type}"] = acc
    
    # 5. Statistical test: Model vs best heuristic
    print("\n[5. STATISTICAL SIGNIFICANCE]")
    
    model_correct = [r["model_correct"] for r in results]
    
    # Compare to recency heuristic
    recency_correct = [r["heuristic_recent_correct"] for r in results]
    
    # McNemar's test for paired data
    # Count discordant pairs
    b = sum(1 for m, h in zip(model_correct, recency_correct) if m and not h)  # Model right, recency wrong
    c = sum(1 for m, h in zip(model_correct, recency_correct) if not m and h)  # Model wrong, recency right
    
    if b + c > 0:
        # McNemar test statistic
        chi2 = (abs(b - c) - 1)**2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, 1)
        
        print(f"  Model beats recency: {b} times")
        print(f"  Recency beats model: {c} times")
        print(f"  McNemar chi2 = {chi2:.2f}, p = {p_value:.4f}")
        
        if p_value < 0.05:
            if b > c:
                print(f"  [SIGNIFICANT] Model is BETTER than recency heuristic")
            else:
                print(f"  [SIGNIFICANT] Model is WORSE than recency heuristic")
        else:
            print(f"  [NOT SIGNIFICANT] Model is NOT different from recency heuristic")
        
        analysis["mcnemar_chi2"] = chi2
        analysis["mcnemar_p"] = p_value
    
    # 6. Task-level analysis (require all 8 correct)
    print("\n[6. TASK-LEVEL ANALYSIS (8-scenario criterion)]")
    
    task_ids = set(r["task_id"] for r in results)
    tasks_passed = 0
    tasks_failed = 0
    
    for tid in task_ids:
        task_results = [r for r in results if r["task_id"] == tid]
        all_correct = all(r["model_correct"] for r in task_results)
        if all_correct:
            tasks_passed += 1
        else:
            tasks_failed += 1
    
    print(f"  Tasks passed (all 8 correct): {tasks_passed}/{len(task_ids)} = {tasks_passed/len(task_ids)*100:.1f}%")
    analysis["tasks_passed"] = tasks_passed
    analysis["total_tasks"] = len(task_ids)
    
    return analysis


def main():
    print("="*70)
    print("STEP 56: PROPER ToM EXPERIMENT")
    print("Following literature best practices")
    print("="*70)
    
    model, tokenizer = load_model()
    
    # Create test set
    print("\n[Creating counterbalanced test set]")
    scenarios = create_full_test_set()
    
    # Run experiment
    results = run_proper_experiment(model, tokenizer, scenarios)
    
    # Analyze
    analysis = analyze_results(results)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    print(f"""
    ============================================================
    PROPER EXPERIMENT RESULTS
    ============================================================
    
    Overall model accuracy: {analysis.get('model_acc', 0)*100:.1f}%
    
    Heuristic baselines:
      - First-mention: {analysis.get('first_acc', 0)*100:.1f}%
      - Recency:       {analysis.get('recent_acc', 0)*100:.1f}%
      - Reality:       {analysis.get('reality_acc', 0)*100:.1f}%
    
    CRITICAL TEST (FB where first-mention is WRONG):
      - Model accuracy: {analysis.get('fb_first_wrong', 'N/A')}
    
    Task-level (all 8 scenarios correct):
      - {analysis.get('tasks_passed', 0)}/{analysis.get('total_tasks', 0)} tasks
    
    """)
    
    if analysis.get("model_acc", 0) > analysis.get("recent_acc", 0) + 0.1:
        print("    [EVIDENCE FOR ToM] Model beats recency heuristic!")
    else:
        print("    [NO EVIDENCE FOR ToM] Model does not beat heuristics")
    
    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "methodology": "8-scenario design with counterbalancing",
        "n_scenarios": len(results),
        "n_tasks": analysis.get("total_tasks", 0),
        "analysis": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                    for k, v in analysis.items()}
    }
    
    output_path = RESULTS_DIR / "step56_proper_experiment.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()


