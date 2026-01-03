"""
Pure Inference Test: Absolutely No Explicit Cues
=================================================

Previous tests had subtle explicit cues like "Alice now knows the new location."

This test removes ALL explicit cues - model must truly INFER belief updates.

The earlier dialogue test that got 2% accuracy:
"Alice tells Bob: 'The ball is in the drawer.' Bob leaves. 
 Carol tells Alice: 'I moved the ball to the basket.' 
 Where will Alice look?"

No statement that Alice "believes" or "knows" the new location.
Model must infer: hearing information → believing it.
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


def generate_pure_inference_scenarios(n: int = 100) -> list:
    """
    Generate scenarios with ZERO explicit belief cues.
    Model must infer from communicative acts alone.
    """
    random.seed(42)
    
    AGENTS = ["Alice", "Bob", "Carol", "David"]
    OTHERS = ["Eve", "Frank", "Grace", "Henry"]
    OBJECTS = ["ball", "key", "book", "toy"]
    LOCATIONS = ["drawer", "basket", "cupboard", "shelf"]
    
    scenarios = []
    
    for i in range(n):
        agent = random.choice(AGENTS)
        informer = random.choice(OTHERS)
        obj = random.choice(OBJECTS)
        loc1, loc2 = random.sample(LOCATIONS, 2)
        
        # === PURE INFERENCE: Agent UPDATED (heard new info) ===
        # No explicit "believes", "knows", "thinks" - just communication
        pure_update = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} says to {agent}: 'I moved the {obj} to the {loc2}.' "
            f"Where will {agent} look for the {obj}? {agent} will look in the"
        )
        
        # === PURE INFERENCE: Agent UNCHANGED (didn't hear) ===
        # Agent left before being told
        pure_unchanged = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{agent} leaves the room. "
            f"{informer} moves the {obj} to the {loc2}. "
            f"{agent} returns. "
            f"Where will {agent} look for the {obj}? {agent} will look in the"
        )
        
        # === EXPLICIT: Agent's belief is stated ===
        # For comparison - this should work well
        explicit_update = (
            f"{agent} believes the {obj} is in the {loc2}. "
            f"Where will {agent} look for the {obj}? {agent} will look in the"
        )
        
        explicit_unchanged = (
            f"{agent} believes the {obj} is in the {loc1}. "
            f"The {obj} is actually in the {loc2}. "
            f"Where will {agent} look for the {obj}? {agent} will look in the"
        )
        
        # === DIALOGUE FORMAT (like our failing test) ===
        dialogue_update = (
            f"{agent}: 'I'll put the {obj} in the {loc1}.'\n"
            f"{informer}: 'I moved the {obj} to the {loc2}.'\n"
            f"Question: Where will {agent} look for the {obj}?\n"
            f"Answer: {agent} will look in the"
        )
        
        dialogue_unchanged = (
            f"{agent}: 'I'll put the {obj} in the {loc1}.'\n"
            f"[{agent} leaves]\n"
            f"{informer}: 'I'll move the {obj} to the {loc2}.'\n"
            f"[{agent} returns]\n"
            f"Question: Where will {agent} look for the {obj}?\n"
            f"Answer: {agent} will look in the"
        )
        
        scenarios.append({
            "id": f"pure_{i}",
            "agent": agent,
            "informer": informer,
            "object": obj,
            "loc1": loc1,  # Original/believed location
            "loc2": loc2,  # Updated/reality location
            "versions": {
                "pure_update": {
                    "prompt": pure_update,
                    "correct": f" {loc2}",  # Should look where told
                    "wrong": f" {loc1}",
                    "description": "Agent heard update, no explicit cues",
                },
                "pure_unchanged": {
                    "prompt": pure_unchanged,
                    "correct": f" {loc1}",  # Didn't hear, old belief
                    "wrong": f" {loc2}",
                    "description": "Agent didn't hear, should use old belief",
                },
                "explicit_update": {
                    "prompt": explicit_update,
                    "correct": f" {loc2}",  # Explicit belief = loc2
                    "wrong": f" {loc1}",
                    "description": "Explicit belief statement (updated)",
                },
                "explicit_unchanged": {
                    "prompt": explicit_unchanged,
                    "correct": f" {loc1}",  # Explicit belief = loc1
                    "wrong": f" {loc2}",
                    "description": "Explicit belief statement (unchanged)",
                },
                "dialogue_update": {
                    "prompt": dialogue_update,
                    "correct": f" {loc2}",  # Heard in dialogue
                    "wrong": f" {loc1}",
                    "description": "Dialogue format, agent heard",
                },
                "dialogue_unchanged": {
                    "prompt": dialogue_unchanged,
                    "correct": f" {loc1}",  # Absent during dialogue
                    "wrong": f" {loc2}",
                    "description": "Dialogue format, agent absent",
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
    print("PURE INFERENCE TEST: No Explicit Belief Cues")
    print("=" * 70)
    print("\nThis test removes ALL explicit cues like 'believes', 'knows', etc.")
    print("Model must INFER that hearing information leads to believing it.\n")
    
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
    print("\n[2/4] Generating scenarios...", flush=True)
    scenarios = generate_pure_inference_scenarios(100)
    print(f"  Generated {len(scenarios)} scenarios (6 versions each)")
    
    # Test all versions
    print("\n[3/4] Testing all versions...", flush=True)
    
    version_names = [
        "pure_update", "pure_unchanged", 
        "explicit_update", "explicit_unchanged",
        "dialogue_update", "dialogue_unchanged"
    ]
    results = {v: {"correct": 0, "total": 0, "probs": []} for v in version_names}
    
    for i, scenario in enumerate(scenarios):
        if i % 20 == 0:
            print(f"  [{i}/{len(scenarios)}]", flush=True)
        
        for version_name in version_names:
            version_data = scenario["versions"][version_name]
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
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    summary = {}
    for version_name, data in results.items():
        acc = data["correct"] / data["total"]
        mean_prob = np.mean(data["probs"])
        
        summary[version_name] = {
            "accuracy": float(acc),
            "mean_prob": float(mean_prob),
        }
    
    # Group by condition
    print("\n  === UPDATED agents (should look at NEW location) ===")
    for v in ["pure_update", "explicit_update", "dialogue_update"]:
        acc = summary[v]["accuracy"]
        bar = "=" * int(acc * 30)
        marker = " ***" if acc < 0.5 else ""
        print(f"    {v:20s}: {acc:5.1%} |{bar}|{marker}")
    
    print("\n  === UNCHANGED agents (should look at OLD location) ===")
    for v in ["pure_unchanged", "explicit_unchanged", "dialogue_unchanged"]:
        acc = summary[v]["accuracy"]
        bar = "=" * int(acc * 30)
        marker = " ***" if acc < 0.5 else ""
        print(f"    {v:20s}: {acc:5.1%} |{bar}|{marker}")
    
    # Key analysis
    print("\n" + "=" * 70)
    print("KEY COMPARISON: Updated Agent Performance")
    print("=" * 70)
    
    pure_upd = summary["pure_update"]["accuracy"]
    explicit_upd = summary["explicit_update"]["accuracy"]
    dialogue_upd = summary["dialogue_update"]["accuracy"]
    
    pure_unch = summary["pure_unchanged"]["accuracy"]
    explicit_unch = summary["explicit_unchanged"]["accuracy"]
    dialogue_unch = summary["dialogue_unchanged"]["accuracy"]
    
    print(f"""
    WHEN BELIEF SHOULD UPDATE (agent heard new info):
      Pure inference:     {pure_upd:5.1%}  (no explicit cues)
      Explicit belief:    {explicit_upd:5.1%}  (belief stated)
      Dialogue format:    {dialogue_upd:5.1%}  (like multi-agent chat)
    
    WHEN BELIEF UNCHANGED (agent absent):
      Pure inference:     {pure_unch:5.1%}
      Explicit belief:    {explicit_unch:5.1%}
      Dialogue format:    {dialogue_unch:5.1%}
    """)
    
    # The key metric
    update_vs_unchanged_gap = pure_upd - pure_unch
    print(f"    GAP (pure updated - pure unchanged): {update_vs_unchanged_gap:+.1%}")
    
    if pure_upd < 0.6 and pure_unch > 0.8:
        print("""
    [!!!] CRITICAL FINDING:
    
    Model FAILS to track belief updates (updated agent) but
    SUCCEEDS on unchanged agents (first-mentioned heuristic works).
    
    This is exactly what we found in dialogue tracking!
    Confirms: Model can't infer belief updates from communication.
        """)
    elif explicit_upd - pure_upd > 0.20:
        print("""
    [!!!] EXPLICIT vs IMPLICIT gap confirmed!
    
    Model does much better when beliefs are explicitly stated.
    Can't compute belief updates from communicative acts alone.
        """)
    
    # Save
    output = {
        "summary": summary,
        "key_finding": f"Pure update: {pure_upd:.1%}, Pure unchanged: {pure_unch:.1%}",
    }
    
    with open(RESULTS_DIR / "pure_inference_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    total_time = time.perf_counter() - timer_start
    print(f"\n[OK] Completed in {total_time:.1f}s")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

