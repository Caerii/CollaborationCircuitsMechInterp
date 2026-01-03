"""
Methodological Controls: Is This a Real Limitation?
====================================================

Before claiming the model "can't" infer belief updates, we need to rule out:

1. CONTEXT: What if we tell it this is a social scenario?
2. VERBS: Are we using the "wrong" communicative verbs?
3. FRAMING: Does explicit ToM framing help?
4. CHAIN OF THOUGHT: Can it reason through if prompted?
5. FEW-SHOT: Does showing examples help?
6. DIRECT QUESTION: What if we just ask "what does X now believe?"

If ANY of these help significantly, it's a prompt issue, not an architectural limit.
If NONE help, it's likely a genuine capability gap.
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


def generate_control_scenarios(n: int = 50) -> list:
    """Generate scenarios with multiple control conditions."""
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
        loc1, loc2 = random.sample(LOCATIONS, 2)
        
        # ============================================
        # BASELINE: Our original failing prompt
        # ============================================
        baseline = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} says to {agent}: 'I moved the {obj} to the {loc2}.' "
            f"Where will {agent} look for the {obj}? {agent} will look in the"
        )
        
        # ============================================
        # CONTROL 1: Social context system prompt
        # ============================================
        social_context = (
            f"In this social scenario, when someone tells another person information, "
            f"the listener updates their beliefs based on what they heard.\n\n"
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2}.' "
            f"Where will {agent} look for the {obj}? {agent} will look in the"
        )
        
        # ============================================
        # CONTROL 2: Different communicative verbs
        # ============================================
        verb_informs = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} informs {agent} that the {obj} is now in the {loc2}. "
            f"Where will {agent} look for the {obj}? {agent} will look in the"
        )
        
        verb_lets_know = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} lets {agent} know: 'The {obj} is now in the {loc2}.' "
            f"Where will {agent} look for the {obj}? {agent} will look in the"
        )
        
        verb_updates = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} updates {agent}: 'I moved the {obj} to the {loc2}.' "
            f"Where will {agent} look for the {obj}? {agent} will look in the"
        )
        
        # ============================================
        # CONTROL 3: Explicit ToM framing
        # ============================================
        tom_framing = (
            f"Theory of Mind Task:\n"
            f"{agent} initially put the {obj} in the {loc1}. "
            f"{informer} then told {agent} that the {obj} had been moved to the {loc2}. "
            f"Based on {agent}'s updated knowledge, where will {agent} look? {agent} will look in the"
        )
        
        # ============================================
        # CONTROL 4: Chain of thought prompt
        # ============================================
        cot_prompt = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2}.' "
            f"Let's think step by step about what {agent} now believes:\n"
            f"1. {agent} originally knew the {obj} was in the {loc1}.\n"
            f"2. {informer} told {agent} about the new location.\n"
            f"3. {agent} heard this information.\n"
            f"4. Therefore, {agent} now believes the {obj} is in the"
        )
        
        # ============================================
        # CONTROL 5: Few-shot example
        # ============================================
        few_shot = (
            f"Example: John put the pen in the drawer. Mary tells John: "
            f"'I moved the pen to the desk.' Where will John look? "
            f"John will look in the desk.\n\n"
            f"Now: {agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2}.' "
            f"Where will {agent} look? {agent} will look in the"
        )
        
        # ============================================
        # CONTROL 6: Direct belief question
        # ============================================
        direct_belief = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2}.' "
            f"What does {agent} now believe about where the {obj} is? "
            f"{agent} believes the {obj} is in the"
        )
        
        # ============================================
        # CONTROL 7: Explicit "heard and understood"
        # ============================================
        heard_understood = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2}.' "
            f"{agent} heard and understood this information. "
            f"Where will {agent} look for the {obj}? {agent} will look in the"
        )
        
        # ============================================
        # CONTROL 8: Narrative consequence
        # ============================================
        narrative_consequence = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2}.' "
            f"Thanks to this information, {agent} knows where to find the {obj}. "
            f"{agent} will look in the"
        )
        
        # ============================================
        # CONTROL 9: Question about informer's action effect
        # ============================================
        informer_effect = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2}.' "
            f"As a result of {informer} telling {agent} this, "
            f"{agent} will look in the"
        )
        
        # ============================================
        # EXPLICIT BASELINE: Should work (for comparison)
        # ============================================
        explicit_baseline = (
            f"{agent} believes the {obj} is in the {loc2}. "
            f"Where will {agent} look? {agent} will look in the"
        )
        
        scenarios.append({
            "id": f"control_{i}",
            "agent": agent,
            "informer": informer,
            "object": obj,
            "loc1": loc1,
            "loc2": loc2,
            "correct": f" {loc2}",  # Updated belief
            "wrong": f" {loc1}",    # Original (wrong if updated)
            "versions": {
                "baseline": baseline,
                "social_context": social_context,
                "verb_informs": verb_informs,
                "verb_lets_know": verb_lets_know,
                "verb_updates": verb_updates,
                "tom_framing": tom_framing,
                "chain_of_thought": cot_prompt,
                "few_shot": few_shot,
                "direct_belief": direct_belief,
                "heard_understood": heard_understood,
                "narrative_consequence": narrative_consequence,
                "informer_effect": informer_effect,
                "explicit_baseline": explicit_baseline,
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
    print("METHODOLOGICAL CONTROLS: Is This a Real Limitation?")
    print("=" * 70)
    print("""
    Testing whether belief update failure is:
    A) A true architectural limitation
    B) A prompt/framing sensitivity issue
    
    If controls help significantly -> prompt issue
    If controls don't help -> architectural limitation
    """)
    
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
    print("\n[2/4] Generating control scenarios...", flush=True)
    scenarios = generate_control_scenarios(50)
    print(f"  Generated {len(scenarios)} scenarios with 13 control conditions each")
    
    # Test all versions
    print("\n[3/4] Testing all control conditions...", flush=True)
    
    version_names = list(scenarios[0]["versions"].keys())
    results = {v: {"correct": 0, "total": 0, "probs": []} for v in version_names}
    
    for i, scenario in enumerate(scenarios):
        if i % 10 == 0:
            print(f"  [{i}/{len(scenarios)}]", flush=True)
        
        for version_name, prompt in scenario["versions"].items():
            result = test_scenario(
                model, tokenizer, prompt,
                scenario["correct"],
                scenario["wrong"]
            )
            
            results[version_name]["total"] += 1
            if result["chose_correct"]:
                results[version_name]["correct"] += 1
            results[version_name]["probs"].append(result["correct_prob"])
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS: All Control Conditions")
    print("=" * 70)
    
    summary = {}
    sorted_results = []
    
    for version_name, data in results.items():
        acc = data["correct"] / data["total"]
        mean_prob = np.mean(data["probs"])
        
        summary[version_name] = {
            "accuracy": float(acc),
            "mean_prob": float(mean_prob),
        }
        sorted_results.append((version_name, acc))
    
    # Sort by accuracy
    sorted_results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n  CONDITION                      ACCURACY")
    print("  " + "-" * 50)
    
    for version_name, acc in sorted_results:
        bar = "=" * int(acc * 30)
        marker = ""
        if version_name == "baseline":
            marker = " <-- BASELINE"
        elif version_name == "explicit_baseline":
            marker = " <-- EXPLICIT (should work)"
        elif acc > 0.80:
            marker = " ***"
        print(f"  {version_name:28s}: {acc:5.1%} |{bar}|{marker}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    baseline_acc = summary["baseline"]["accuracy"]
    explicit_acc = summary["explicit_baseline"]["accuracy"]
    
    # Find best control
    best_control = max(
        [(k, v["accuracy"]) for k, v in summary.items() if k not in ["baseline", "explicit_baseline"]],
        key=lambda x: x[1]
    )
    
    print(f"\n  Baseline (original prompt):     {baseline_acc:.1%}")
    print(f"  Explicit baseline (should work): {explicit_acc:.1%}")
    print(f"  Best control condition:          {best_control[0]} ({best_control[1]:.1%})")
    
    improvement = best_control[1] - baseline_acc
    print(f"\n  Improvement from best control: {improvement:+.1%}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if best_control[1] >= 0.80:
        print(f"""
    [PROMPT SENSITIVITY CONFIRMED]
    
    The control '{best_control[0]}' achieves {best_control[1]:.1%} accuracy!
    
    This means the model CAN infer belief updates when properly prompted.
    Our earlier failure was due to prompt framing, not architectural limits.
    
    Key insight: The capability EXISTS but requires the right framing.
    
    For multi-agent systems: Use prompts similar to '{best_control[0]}'
        """)
    elif best_control[1] >= 0.50:
        print(f"""
    [PARTIAL IMPROVEMENT]
    
    Best control ({best_control[0]}) achieves {best_control[1]:.1%}
    vs baseline of {baseline_acc:.1%}
    
    Some improvement, but not full success. Suggests:
    - Capability is weak but present
    - Better prompting helps but doesn't solve completely
        """)
    else:
        print(f"""
    [ARCHITECTURAL LIMITATION CONFIRMED]
    
    Even with extensive controls, best accuracy is only {best_control[1]:.1%}
    
    None of these helped significantly:
    - Social context framing
    - Different verbs
    - Chain of thought
    - Few-shot examples
    - Direct belief questions
    
    The model genuinely struggles to infer belief updates from communication.
    This appears to be a fundamental capability gap.
        """)
    
    # Which controls helped most?
    print("\n  CONTROL EFFECTIVENESS RANKING:")
    for i, (name, acc) in enumerate(sorted_results[:5]):
        if name != "explicit_baseline":
            delta = acc - baseline_acc
            print(f"    {i+1}. {name}: {acc:.1%} ({delta:+.1%} vs baseline)")
    
    # Save results
    output = {
        "summary": summary,
        "baseline_accuracy": baseline_acc,
        "explicit_accuracy": explicit_acc,
        "best_control": {"name": best_control[0], "accuracy": best_control[1]},
        "improvement": improvement,
    }
    
    with open(RESULTS_DIR / "methodological_controls_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    total_time = time.perf_counter() - timer_start
    print(f"\n[OK] Completed in {total_time:.1f}s")
    print(f"     Saved to {RESULTS_DIR / 'methodological_controls_results.json'}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


