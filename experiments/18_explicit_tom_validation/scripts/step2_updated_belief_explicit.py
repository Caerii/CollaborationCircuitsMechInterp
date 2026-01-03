"""
Updated Belief: Explicit vs Implicit Test
==========================================

Key insight from previous test: Model got 90% on implicit Sally-Anne
BUT we found 2% on updated beliefs in dialogue.

The difference:
- Sally-Anne: Sally's belief IS the first-mentioned location (put ball there)
- Dialogue update: Agent's belief should UPDATE to a LATER location

This test compares:
1. IMPLICIT update: Agent learns new info through dialogue/observation
2. EXPLICIT update: We explicitly state the agent's updated belief

If explicit update works but implicit fails, confirms:
- Model can't COMPUTE belief updates
- Model CAN process explicitly stated belief updates
"""

import json
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import random

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_update_scenarios(n: int = 100) -> list:
    """Generate scenarios where agent's belief must UPDATE."""
    random.seed(42)
    
    AGENTS = ["Alice", "Bob", "Carol", "David"]
    INFORMERS = ["Eve", "Frank", "Grace", "Henry"]
    OBJECTS = ["ball", "key", "book", "toy"]
    LOCATIONS = ["drawer", "basket", "cupboard", "shelf"]
    
    scenarios = []
    
    for i in range(n):
        agent = random.choice(AGENTS)
        informer = random.choice(INFORMERS)
        obj = random.choice(OBJECTS)
        original_loc, updated_loc = random.sample(LOCATIONS, 2)
        
        # IMPLICIT UPDATE: Agent learns through dialogue
        # Model must infer that agent's belief CHANGED
        implicit_update = (
            f"{agent} initially put the {obj} in the {original_loc}. "
            f"Later, {informer} tells {agent}: 'I moved the {obj} to the {updated_loc}.' "
            f"{agent} now knows the new location. "
            f"Where will {agent} look for the {obj}? {agent} will look in the"
        )
        
        # EXPLICIT UPDATE: We state the agent's new belief directly
        explicit_update = (
            f"{agent} initially believed the {obj} was in the {original_loc}. "
            f"After receiving new information, {agent} now believes the {obj} is in the {updated_loc}. "
            f"Where will {agent} look for the {obj}? {agent} will look in the"
        )
        
        # STRUCTURED UPDATE: Very explicit format
        structured_update = (
            f"[{agent.upper()}'S INITIAL BELIEF]: {obj} is in {original_loc}\n"
            f"[UPDATE]: {agent} learned the {obj} was moved to {updated_loc}\n"
            f"[{agent.upper()}'S CURRENT BELIEF]: {obj} is in {updated_loc}\n"
            f"[QUESTION]: Where will {agent} look?\n"
            f"[ANSWER]: {agent} will look in the"
        )
        
        # UNCHANGED agent (for comparison) - should work well
        unchanged_story = (
            f"{agent} put the {obj} in the {original_loc}. "
            f"{agent} left the room. "
            f"{informer} moved the {obj} to the {updated_loc}, but {agent} doesn't know. "
            f"Where will {agent} look for the {obj}? {agent} will look in the"
        )
        
        scenarios.append({
            "id": f"update_{i}",
            "agent": agent,
            "informer": informer,
            "object": obj,
            "original_loc": original_loc,
            "updated_loc": updated_loc,
            "versions": {
                "implicit_update": {
                    "prompt": implicit_update,
                    "correct": f" {updated_loc}",  # Agent learned new location
                    "wrong": f" {original_loc}",
                },
                "explicit_update": {
                    "prompt": explicit_update,
                    "correct": f" {updated_loc}",  # Explicitly stated new belief
                    "wrong": f" {original_loc}",
                },
                "structured_update": {
                    "prompt": structured_update,
                    "correct": f" {updated_loc}",  # Very explicit
                    "wrong": f" {original_loc}",
                },
                "unchanged_agent": {
                    "prompt": unchanged_story,
                    "correct": f" {original_loc}",  # Agent didn't learn
                    "wrong": f" {updated_loc}",
                },
            },
        })
    
    return scenarios


def test_scenario(model, tokenizer, prompt: str, correct: str, wrong: str) -> dict:
    """Test if model prefers correct over wrong."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    
    correct_id = tokenizer.encode(correct, add_special_tokens=False)[0]
    wrong_id = tokenizer.encode(wrong, add_special_tokens=False)[0]
    
    correct_logit = logits[correct_id].item()
    wrong_logit = logits[wrong_id].item()
    
    max_logit = max(correct_logit, wrong_logit)
    correct_prob = np.exp(correct_logit - max_logit) / (np.exp(correct_logit - max_logit) + np.exp(wrong_logit - max_logit))
    
    return {
        "correct_logit": correct_logit,
        "wrong_logit": wrong_logit,
        "correct_prob": float(correct_prob),
        "chose_correct": correct_logit > wrong_logit,
    }


def main():
    timer_start = time.perf_counter()
    
    print("=" * 70)
    print("UPDATED BELIEF: EXPLICIT vs IMPLICIT TEST")
    print("=" * 70)
    print("\nKey question: Can model track belief UPDATES?")
    print("- Previous Sally-Anne (unchanged belief): 90% - but first-mentioned heuristic works!")
    print("- Dialogue updates (changed belief): 2% - heuristic fails!")
    print("- This test: Does EXPLICIT update statement help?\n")
    
    # Load model
    print("[1/4] Loading model...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CFG.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CFG.model_name,
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    print("  [OK] Model loaded", flush=True)
    
    # Generate scenarios
    print("\n[2/4] Generating belief update scenarios...", flush=True)
    scenarios = generate_update_scenarios(100)
    print(f"  Generated {len(scenarios)} scenarios (4 versions each)")
    
    # Test all versions
    print("\n[3/4] Testing all versions...", flush=True)
    
    results = {
        "implicit_update": {"correct": 0, "total": 0, "probs": []},
        "explicit_update": {"correct": 0, "total": 0, "probs": []},
        "structured_update": {"correct": 0, "total": 0, "probs": []},
        "unchanged_agent": {"correct": 0, "total": 0, "probs": []},
    }
    
    for i, scenario in enumerate(scenarios):
        if i % 20 == 0:
            print(f"  [{i}/{len(scenarios)}]", flush=True)
        
        for version_name, version_data in scenario["versions"].items():
            result = test_scenario(
                model, tokenizer,
                version_data["prompt"],
                version_data["correct"],
                version_data["wrong"]
            )
            
            results[version_name]["total"] += 1
            if result["chose_correct"]:
                results[version_name]["correct"] += 1
            results[version_name]["probs"].append(result["correct_prob"])
    
    # Compute and display results
    print("\n[4/4] Computing results...\n", flush=True)
    
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    summary = {}
    for version_name, data in results.items():
        acc = data["correct"] / data["total"]
        mean_prob = np.mean(data["probs"])
        std_prob = np.std(data["probs"])
        
        summary[version_name] = {
            "accuracy": float(acc),
            "mean_prob": float(mean_prob),
            "std_prob": float(std_prob),
            "n": data["total"],
        }
        
        bar = "=" * int(acc * 40)
        marker = " <-- CRITICAL TEST" if version_name == "implicit_update" else ""
        print(f"\n  {version_name.upper():20s}: {acc:5.1%} |{bar}|{marker}")
        print(f"  {'':20s}  Mean prob: {mean_prob:.3f} (+/- {std_prob:.3f})")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    implicit_upd = summary["implicit_update"]["accuracy"]
    explicit_upd = summary["explicit_update"]["accuracy"]
    structured_upd = summary["structured_update"]["accuracy"]
    unchanged = summary["unchanged_agent"]["accuracy"]
    
    print(f"\n  BELIEF UPDATES:")
    print(f"    Implicit (infer from narrative): {implicit_upd:.1%}")
    print(f"    Explicit (stated directly):      {explicit_upd:.1%}")
    print(f"    Structured (clear format):       {structured_upd:.1%}")
    
    print(f"\n  UNCHANGED (baseline):")
    print(f"    Agent doesn't learn:             {unchanged:.1%}")
    
    # Key comparison
    update_gap = explicit_upd - implicit_upd
    print(f"\n  UPDATE GAP: {update_gap:+.1%}")
    print(f"  (Explicit - Implicit for belief updates)")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if explicit_upd > 0.85 and implicit_upd < 0.60:
        print("""
  [+++] STRONG CONFIRMATION!
  
  The model CAN track updated beliefs when they're EXPLICIT
  but FAILS when it has to INFER updates from narrative.
  
  This is the smoking gun:
  - Explicit statement: "Alice now believes X" → Model follows
  - Implicit inference: "Eve told Alice X" → Model ignores
  
  The "belief computation" circuit is weak/missing.
  The "belief parsing" circuit is strong.
        """)
    elif explicit_upd > implicit_upd + 0.15:
        print("""
  [++] MODERATE confirmation
  
  Explicit updates work better than implicit, but both
  may be partially successful. The gap suggests the model
  benefits from explicit belief statements.
        """)
    elif unchanged > 0.85 and implicit_upd < 0.60:
        print("""
  [!] INTERESTING PATTERN
  
  Model succeeds when belief = first-mentioned location (unchanged)
  Model fails when belief = later location (update)
  
  This is the "first-mentioned-location" heuristic in action!
        """)
    else:
        print(f"""
  Results: implicit={implicit_upd:.1%}, explicit={explicit_upd:.1%}
  Need further analysis of the pattern.
        """)
    
    # Save results
    output = {
        "summary": summary,
        "hypothesis": "Model can track EXPLICIT belief updates but fails at IMPLICIT updates",
        "key_finding": f"Implicit update: {implicit_upd:.1%}, Explicit update: {explicit_upd:.1%}",
    }
    
    with open(RESULTS_DIR / "belief_update_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    total_time = time.perf_counter() - timer_start
    print(f"\n[OK] Completed in {total_time:.1f}s")
    print(f"     Saved to {RESULTS_DIR / 'belief_update_results.json'}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


