"""
What Specifically Helps? Decomposing the Successful Prompts
============================================================

We found that chain-of-thought and few-shot get 100% accuracy.
But what SPECIFICALLY helps?

1. Is it the STRUCTURE of CoT, or the actual REASONING content?
2. Is it just providing ANY scaffolding?
3. Can we find a minimal intervention that helps?

This will tell us:
- Where in the processing pipeline the "belief update" inference happens
- What circuit/computation needs to be "activated"
- How to design multi-agent prompts effectively
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


def generate_decomposition_scenarios(n: int = 50) -> list:
    """Generate scenarios with decomposed interventions."""
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
        
        # Baseline (fails)
        baseline = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} says to {agent}: 'I moved the {obj} to the {loc2}.' "
            f"Where will {agent} look for the {obj}? {agent} will look in the"
        )
        
        # Full CoT (works)
        full_cot = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2}.' "
            f"Let's think step by step about what {agent} now believes:\n"
            f"1. {agent} originally knew the {obj} was in the {loc1}.\n"
            f"2. {informer} told {agent} about the new location.\n"
            f"3. {agent} heard this information.\n"
            f"4. Therefore, {agent} now believes the {obj} is in the"
        )
        
        # ==========================================
        # DECOMPOSITION: What part of CoT helps?
        # ==========================================
        
        # Just "Let's think step by step" (no actual steps)
        cot_prefix_only = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2}.' "
            f"Let's think step by step. "
            f"Where will {agent} look? {agent} will look in the"
        )
        
        # Steps without "therefore" conclusion
        cot_steps_no_conclusion = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2}.' "
            f"1. {agent} originally knew the {obj} was in the {loc1}.\n"
            f"2. {informer} told {agent} about the new location.\n"
            f"3. {agent} heard this information.\n"
            f"Where will {agent} look? {agent} will look in the"
        )
        
        # Only the conclusion part ("therefore X believes")
        cot_conclusion_only = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2}.' "
            f"Therefore, {agent} now believes the {obj} is in the"
        )
        
        # Single key step: "X heard this information"
        single_step_heard = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2}.' "
            f"{agent} heard this. "
            f"Where will {agent} look? {agent} will look in the"
        )
        
        # Single key step: "X now knows"
        single_step_knows = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2}.' "
            f"{agent} now knows the new location. "
            f"Where will {agent} look? {agent} will look in the"
        )
        
        # Minimal: just add "and believed it"
        minimal_believed = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2},' "
            f"and {agent} believed it. "
            f"Where will {agent} look? {agent} will look in the"
        )
        
        # Minimal: "so X updated their belief"
        minimal_updated = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2},' "
            f"so {agent} updated their belief. "
            f"Where will {agent} look? {agent} will look in the"
        )
        
        # Causal: "because X heard this, X will..."
        causal_because = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2}.' "
            f"Because {agent} heard this, {agent} will look in the"
        )
        
        # Contrastive: "X no longer thinks it's in loc1"
        contrastive = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2}.' "
            f"{agent} no longer thinks the {obj} is in the {loc1}. "
            f"Where will {agent} look? {agent} will look in the"
        )
        
        # Question-based: "Given that X was told..."
        given_that = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2}.' "
            f"Given that {agent} was told this information, "
            f"where will {agent} look? {agent} will look in the"
        )
        
        scenarios.append({
            "id": f"decomp_{i}",
            "agent": agent,
            "informer": informer,
            "object": obj,
            "loc1": loc1,
            "loc2": loc2,
            "correct": f" {loc2}",
            "wrong": f" {loc1}",
            "versions": {
                "baseline": baseline,
                "full_cot": full_cot,
                "cot_prefix_only": cot_prefix_only,
                "cot_steps_no_conclusion": cot_steps_no_conclusion,
                "cot_conclusion_only": cot_conclusion_only,
                "single_step_heard": single_step_heard,
                "single_step_knows": single_step_knows,
                "minimal_believed": minimal_believed,
                "minimal_updated": minimal_updated,
                "causal_because": causal_because,
                "contrastive": contrastive,
                "given_that": given_that,
            },
        })
    
    return scenarios


def test_scenario(model, tokenizer, prompt: str, correct: str, wrong: str) -> dict:
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
        "chose_correct": correct_logit > wrong_logit,
        "correct_prob": float(correct_prob),
    }


def main():
    timer_start = time.perf_counter()
    
    print("=" * 70)
    print("DECOMPOSITION: What Specifically Helps?")
    print("=" * 70)
    print("""
    Full CoT works. But what PART of it helps?
    - The "let's think step by step" prefix?
    - The numbered steps?
    - The "therefore" conclusion?
    - A single phrase like "X heard this"?
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
    print("\n[2/4] Generating decomposition scenarios...", flush=True)
    scenarios = generate_decomposition_scenarios(50)
    print(f"  Generated {len(scenarios)} scenarios with 12 variants each")
    
    # Test all versions
    print("\n[3/4] Testing all variants...", flush=True)
    
    version_names = list(scenarios[0]["versions"].keys())
    results = {v: {"correct": 0, "total": 0} for v in version_names}
    
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
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS: Decomposition Analysis")
    print("=" * 70)
    
    summary = {}
    sorted_results = []
    
    for version_name, data in results.items():
        acc = data["correct"] / data["total"]
        summary[version_name] = {"accuracy": float(acc)}
        sorted_results.append((version_name, acc))
    
    sorted_results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n  VARIANT                        ACCURACY    INTERPRETATION")
    print("  " + "-" * 65)
    
    for version_name, acc in sorted_results:
        bar = "=" * int(acc * 20)
        
        # Add interpretation
        if acc >= 0.95:
            interp = "WORKS"
        elif acc >= 0.70:
            interp = "Helps a lot"
        elif acc >= 0.50:
            interp = "Helps somewhat"
        else:
            interp = "Doesn't help"
            
        print(f"  {version_name:28s}: {acc:5.1%} |{bar:20s}| {interp}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    baseline_acc = summary["baseline"]["accuracy"]
    full_cot_acc = summary["full_cot"]["accuracy"]
    
    # Find minimal successful intervention
    minimal_success = None
    for name, acc in sorted_results:
        if acc >= 0.90 and name not in ["full_cot", "baseline"]:
            word_count = len(name.split("_"))
            if minimal_success is None or word_count < len(minimal_success[0].split("_")):
                minimal_success = (name, acc)
    
    print(f"\n  Baseline: {baseline_acc:.1%}")
    print(f"  Full CoT: {full_cot_acc:.1%}")
    
    if minimal_success:
        print(f"\n  MINIMAL SUCCESSFUL INTERVENTION: '{minimal_success[0]}' ({minimal_success[1]:.1%})")
    
    # What helps?
    helps = [(n, a) for n, a in sorted_results if a > baseline_acc + 0.30]
    helps_slightly = [(n, a) for n, a in sorted_results if baseline_acc + 0.10 < a <= baseline_acc + 0.30]
    doesnt_help = [(n, a) for n, a in sorted_results if a <= baseline_acc + 0.10]
    
    if helps:
        print(f"\n  INTERVENTIONS THAT WORK:")
        for n, a in helps:
            print(f"    - {n}: {a:.1%}")
    
    if helps_slightly:
        print(f"\n  PARTIALLY HELP:")
        for n, a in helps_slightly:
            print(f"    - {n}: {a:.1%}")
    
    if doesnt_help:
        print(f"\n  DON'T HELP MUCH:")
        for n, a in doesnt_help[:3]:
            print(f"    - {n}: {a:.1%}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION: What Activates Belief Update Processing?")
    print("=" * 70)
    
    # Check specific patterns
    cot_prefix = summary.get("cot_prefix_only", {}).get("accuracy", 0)
    single_knows = summary.get("single_step_knows", {}).get("accuracy", 0)
    single_heard = summary.get("single_step_heard", {}).get("accuracy", 0)
    minimal_believed = summary.get("minimal_believed", {}).get("accuracy", 0)
    
    print(f"""
    Analysis of what triggers belief update inference:
    
    - Just "let's think step by step": {cot_prefix:.1%}
      -> {'STRUCTURE alone helps' if cot_prefix > 0.5 else 'Structure alone not enough'}
    
    - "X heard this": {single_heard:.1%}
      -> {'Hearing acknowledgment helps' if single_heard > 0.5 else 'Hearing acknowledgment not enough'}
    
    - "X now knows": {single_knows:.1%}
      -> {'Knowledge state mention helps' if single_knows > 0.5 else 'Knowledge state not enough'}
    
    - "and X believed it": {minimal_believed:.1%}
      -> {'Explicit belief update helps' if minimal_believed > 0.5 else 'Explicit belief not enough'}
    """)
    
    # Save results
    output = {
        "summary": summary,
        "sorted_results": sorted_results,
        "baseline": baseline_acc,
        "full_cot": full_cot_acc,
        "minimal_success": minimal_success,
    }
    
    with open(RESULTS_DIR / "decomposition_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    total_time = time.perf_counter() - timer_start
    print(f"\n[OK] Completed in {total_time:.1f}s")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


