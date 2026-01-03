"""
Explicit Structure ToM Test
============================

Hypothesis: Model succeeds at ToM when beliefs are EXPLICIT, fails when IMPLICIT.

This test compares:
1. NARRATIVE (implicit): "Sally left. Anne moved ball. Sally returns. Where looks?"
2. EXPLICIT BELIEF: "Sally believes ball is in basket. Ball is in box. Where looks?"
3. STRUCTURED: "[SALLY_BELIEF]: basket [REALITY]: box. Where will Sally look?"

If model succeeds on (2) and (3) but fails on (1), confirms:
- The model CAN process belief information
- The model CANNOT compute beliefs from narrative

This explains why multi-agent software dev works (explicit) but Sally-Anne fails (implicit).
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


def generate_scenarios(n: int = 100) -> list:
    """Generate matched pairs: implicit vs explicit belief framing."""
    random.seed(42)
    
    AGENTS = ["Sally", "Alice", "Bob", "Carol", "David", "Emma"]
    OTHERS = ["Anne", "Frank", "Grace", "Henry", "Ivy", "Jack"]
    OBJECTS = ["ball", "key", "book", "toy", "cup", "phone"]
    LOCATIONS = ["basket", "box", "drawer", "cupboard", "shelf", "desk"]
    
    scenarios = []
    
    for i in range(n):
        agent = random.choice(AGENTS)
        other = random.choice(OTHERS)
        obj = random.choice(OBJECTS)
        loc_belief, loc_reality = random.sample(LOCATIONS, 2)
        
        # VERSION 1: IMPLICIT (standard false belief narrative)
        # Model must INFER Sally's belief from events
        implicit_story = (
            f"{agent} puts the {obj} in the {loc_belief}. "
            f"{agent} leaves the room. "
            f"{other} moves the {obj} from the {loc_belief} to the {loc_reality}. "
            f"{agent} returns. "
            f"Where will {agent} look for the {obj}? {agent} will look in the"
        )
        
        # VERSION 2: EXPLICIT BELIEF STATEMENT
        # Model just needs to READ Sally's belief
        explicit_belief = (
            f"{agent} believes the {obj} is in the {loc_belief}. "
            f"However, the {obj} is actually in the {loc_reality}. "
            f"Where will {agent} look for the {obj}? {agent} will look in the"
        )
        
        # VERSION 3: STRUCTURED FORMAT
        # Even more explicit - like multi-agent protocol
        structured = (
            f"[{agent.upper()}'S BELIEF]: The {obj} is in the {loc_belief}.\n"
            f"[REALITY]: The {obj} is in the {loc_reality}.\n"
            f"[QUESTION]: Where will {agent} look for the {obj}?\n"
            f"[ANSWER]: {agent} will look in the"
        )
        
        # VERSION 4: MULTI-AGENT PROTOCOL STYLE
        # Simulating software dev communication
        protocol = (
            f"AGENT_{agent.upper()}_KNOWLEDGE: I last saw the {obj} in the {loc_belief}.\n"
            f"AGENT_{other.upper()}_ACTION: I moved the {obj} to the {loc_reality}.\n"
            f"NOTE: {agent} was not present for {other}'s action.\n"
            f"QUERY: Based on {agent}'s knowledge, where will {agent} look?\n"
            f"RESPONSE: {agent} will look in the"
        )
        
        correct = f" {loc_belief}"  # Belief-based answer
        wrong = f" {loc_reality}"   # Reality-based answer
        
        scenarios.append({
            "id": f"pair_{i}",
            "agent": agent,
            "other": other,
            "object": obj,
            "belief_loc": loc_belief,
            "reality_loc": loc_reality,
            "versions": {
                "implicit": implicit_story,
                "explicit_belief": explicit_belief,
                "structured": structured,
                "protocol": protocol,
            },
            "correct_completion": correct,
            "wrong_completion": wrong,
        })
    
    return scenarios


def test_scenario(model, tokenizer, prompt: str, correct: str, wrong: str) -> dict:
    """Test if model prefers correct (belief) over wrong (reality)."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    
    correct_id = tokenizer.encode(correct, add_special_tokens=False)[0]
    wrong_id = tokenizer.encode(wrong, add_special_tokens=False)[0]
    
    correct_logit = logits[correct_id].item()
    wrong_logit = logits[wrong_id].item()
    
    # Softmax for probabilities
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
    print("EXPLICIT vs IMPLICIT ToM STRUCTURE TEST")
    print("=" * 70)
    print("\nHypothesis: Model succeeds when beliefs are EXPLICIT, fails when IMPLICIT")
    print("This explains why multi-agent software dev works but Sally-Anne fails.\n")
    
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
    print("\n[2/4] Generating matched scenarios...", flush=True)
    scenarios = generate_scenarios(100)
    print(f"  Generated {len(scenarios)} scenario pairs (4 versions each)")
    
    # Test all versions
    print("\n[3/4] Testing all versions...", flush=True)
    
    results = {
        "implicit": {"correct": 0, "total": 0, "probs": []},
        "explicit_belief": {"correct": 0, "total": 0, "probs": []},
        "structured": {"correct": 0, "total": 0, "probs": []},
        "protocol": {"correct": 0, "total": 0, "probs": []},
    }
    
    for i, scenario in enumerate(scenarios):
        if i % 20 == 0:
            print(f"  [{i}/{len(scenarios)}]", flush=True)
        
        for version_name, prompt in scenario["versions"].items():
            result = test_scenario(
                model, tokenizer, prompt,
                scenario["correct_completion"],
                scenario["wrong_completion"]
            )
            
            results[version_name]["total"] += 1
            if result["chose_correct"]:
                results[version_name]["correct"] += 1
            results[version_name]["probs"].append(result["correct_prob"])
    
    # Compute statistics
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
        print(f"\n  {version_name.upper():20s}: {acc:5.1%} |{bar}|")
        print(f"  {'':20s}  Mean prob: {mean_prob:.3f} (+/- {std_prob:.3f})")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    implicit_acc = summary["implicit"]["accuracy"]
    explicit_acc = summary["explicit_belief"]["accuracy"]
    structured_acc = summary["structured"]["accuracy"]
    protocol_acc = summary["protocol"]["accuracy"]
    
    print(f"\n  Implicit (narrative):  {implicit_acc:.1%}")
    print(f"  Explicit (stated):     {explicit_acc:.1%}")
    print(f"  Structured (format):   {structured_acc:.1%}")
    print(f"  Protocol (multi-agent):{protocol_acc:.1%}")
    
    # Key finding
    explicit_avg = (explicit_acc + structured_acc + protocol_acc) / 3
    gap = explicit_avg - implicit_acc
    
    print(f"\n  EXPLICIT avg: {explicit_avg:.1%}")
    print(f"  IMPLICIT:     {implicit_acc:.1%}")
    print(f"  GAP:          {gap:+.1%}")
    
    if gap > 0.15:
        print("\n  [+++] STRONG CONFIRMATION!")
        print("        Model handles EXPLICIT beliefs much better than IMPLICIT.")
        print("        This explains why multi-agent software dev works!")
    elif gap > 0.05:
        print("\n  [++] MODERATE gap")
        print("       Some advantage for explicit framing.")
    else:
        print("\n  [~] No significant gap")
        print("      Model handles both similarly (may indicate different issue).")
    
    # Implications
    print("\n" + "=" * 70)
    print("IMPLICATIONS FOR MULTI-AGENT CIRCUITS")
    print("=" * 70)
    
    print("""
  IF explicit >> implicit:
  
    1. L12H0 + L23H0 = "Belief Statement Parser"
       - Extracts explicitly stated beliefs
       - Routes to correct prediction
       - THIS is the collaboration circuit!
    
    2. Missing: "Belief Computation Circuit"
       - Would need to infer beliefs from events
       - Track who was present/absent
       - Compute information states
    
    3. Multi-agent software dev works because:
       - All information is EXPLICIT in the prompt
       - Agents COMMUNICATE their knowledge
       - No inference from narrative needed
    
    4. For TRUE multi-agent ToM, would need:
       - Training on belief computation
       - Circuits for information flow tracking
       - Or: Always make beliefs explicit (current workaround)
    """)
    
    # Save results
    output = {
        "summary": summary,
        "hypothesis": "Model succeeds when beliefs are EXPLICIT, fails when IMPLICIT",
        "finding": f"Explicit avg: {explicit_avg:.1%}, Implicit: {implicit_acc:.1%}, Gap: {gap:+.1%}",
        "scenarios": [{
            "id": s["id"],
            "agent": s["agent"],
            "belief_loc": s["belief_loc"],
            "reality_loc": s["reality_loc"],
        } for s in scenarios[:10]],  # Save sample
    }
    
    with open(RESULTS_DIR / "explicit_vs_implicit_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    total_time = time.perf_counter() - timer_start
    print(f"\n[OK] Completed in {total_time:.1f}s")
    print(f"     Saved to {RESULTS_DIR / 'explicit_vs_implicit_results.json'}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


