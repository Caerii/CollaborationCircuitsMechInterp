"""
Step 24: Combined Circuit Ablation - Maximum ToM Improvement

Test the combined ablation of:
- Early circuit (L17-L21): 5 heads
- Late circuit (L32-L35): 10 heads

Total: 15 heads for maximum ToM improvement.
"""

import torch
import json
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from tqdm import tqdm

# Configuration
MODEL_NAME = "Qwen/Qwen3-4B"

# The TWO circuits
EARLY_CIRCUIT = [(17, 4), (18, 11), (18, 14), (19, 30), (21, 17)]  # Original discovery
LATE_CIRCUIT = [
    (35, 0), (35, 1), (35, 17),     # L35
    (33, 6), (33, 13), (33, 17), (33, 31),  # L33
    (32, 6), (32, 31),              # L32
    (34, 17)                        # L34
]
COMBINED_CIRCUIT = EARLY_CIRCUIT + LATE_CIRCUIT

# Comprehensive test verbs
TEST_VERBS = {
    "bad_verbs": ["told", "said", "mentioned", "stated", "informed", "notified", "wrote", "emailed", "texted"],
    "good_verbs": ["announced", "asked", "explained", "hinted", "shouted", "questioned", "declared"],
    "mid_verbs": ["communicated", "indicated", "signaled", "reported"]
}

CORRECT_ANSWER = " drawer"
WRONG_ANSWER = " basket"

# Output paths
OUTPUT_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_prompt(verb: str) -> str:
    return f"""Alice puts the ball in the drawer. Alice leaves.
Bob {verb} Carol that he moved the ball to the basket.
Alice returns. Alice will look for the ball in the"""


def test_with_ablation(model, tokenizer, prompt: str, heads_to_ablate: list) -> dict:
    """Test prediction with specified heads ablated."""
    
    hooks = []
    
    for layer_idx, head_idx in heads_to_ablate:
        def make_hook(target_layer, target_head):
            def hook(module, input, output):
                hidden = output if isinstance(output, torch.Tensor) else output[0]
                original_shape = hidden.shape
                
                if hidden.dim() == 2:
                    hidden = hidden.unsqueeze(0)
                
                hidden_size = hidden.shape[-1]
                n_heads = model.config.num_attention_heads
                head_dim = hidden_size // n_heads
                
                hidden = hidden.view(-1, hidden.shape[1], n_heads, head_dim)
                hidden[:, :, target_head, :] = 0
                hidden = hidden.view(-1, hidden.shape[1], hidden_size)
                
                if len(original_shape) == 2:
                    hidden = hidden.squeeze(0)
                
                if isinstance(output, torch.Tensor):
                    return hidden
                return (hidden,) + output[1:]
            return hook
        
        layer = model.model.layers[layer_idx].self_attn.o_proj
        hook = layer.register_forward_hook(make_hook(layer_idx, head_idx))
        hooks.append(hook)
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        correct_id = tokenizer.encode(CORRECT_ANSWER, add_special_tokens=False)[0]
        wrong_id = tokenizer.encode(WRONG_ANSWER, add_special_tokens=False)[0]
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]
        
        return {
            "diff": float(logits[correct_id] - logits[wrong_id]),
            "correct": logits[correct_id] > logits[wrong_id]
        }
    finally:
        for hook in hooks:
            hook.remove()


def main():
    print("=" * 70)
    print("COMBINED CIRCUIT ABLATION TEST")
    print("=" * 70)
    print(f"\nEarly Circuit (L17-L21): {len(EARLY_CIRCUIT)} heads")
    print(f"Late Circuit (L32-L35): {len(LATE_CIRCUIT)} heads")
    print(f"Combined: {len(COMBINED_CIRCUIT)} heads total")
    print()
    
    model, tokenizer = load_model()
    
    # Flatten all verbs
    all_verbs = []
    for category, verbs in TEST_VERBS.items():
        for verb in verbs:
            all_verbs.append((verb, category))
    
    # Test configurations
    configs = [
        ("Baseline", []),
        ("Early Only", EARLY_CIRCUIT),
        ("Late Only", LATE_CIRCUIT),
        ("Combined", COMBINED_CIRCUIT)
    ]
    
    results = {}
    
    print("Testing all verbs with different ablation configs...")
    print("-" * 70)
    
    for verb, category in tqdm(all_verbs, desc="Verbs"):
        prompt = generate_prompt(verb)
        verb_results = {"category": category}
        
        for config_name, heads in configs:
            result = test_with_ablation(model, tokenizer, prompt, heads)
            verb_results[config_name] = result
        
        results[verb] = verb_results
    
    # Analyze results
    print("\n" + "=" * 70)
    print("RESULTS BY VERB CATEGORY")
    print("=" * 70)
    
    for category in ["bad_verbs", "good_verbs", "mid_verbs"]:
        category_results = {verb: r for verb, r in results.items() if r["category"] == category}
        
        print(f"\n{category.upper()} ({len(category_results)} verbs):")
        print("-" * 50)
        
        for config_name in ["Baseline", "Early Only", "Late Only", "Combined"]:
            correct_count = sum(1 for r in category_results.values() if r[config_name]["correct"])
            accuracy = correct_count / len(category_results) * 100
            print(f"  {config_name:12s}: {accuracy:5.1f}% ({correct_count}/{len(category_results)})")
    
    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL ACCURACY")
    print("=" * 70)
    
    for config_name in ["Baseline", "Early Only", "Late Only", "Combined"]:
        correct_count = sum(1 for r in results.values() if r[config_name]["correct"])
        accuracy = correct_count / len(results) * 100
        print(f"  {config_name:12s}: {accuracy:5.1f}% ({correct_count}/{len(results)})")
    
    # Per-verb details
    print("\n" + "=" * 70)
    print("PER-VERB DETAILS")
    print("=" * 70)
    
    for verb, r in sorted(results.items()):
        baseline = "OK" if r["Baseline"]["correct"] else "FAIL"
        early = "OK" if r["Early Only"]["correct"] else "FAIL"
        late = "OK" if r["Late Only"]["correct"] else "FAIL"
        combined = "OK" if r["Combined"]["correct"] else "FAIL"
        
        # Highlight fixed cases
        fixed_by = []
        if not r["Baseline"]["correct"]:
            if r["Early Only"]["correct"]:
                fixed_by.append("early")
            if r["Late Only"]["correct"]:
                fixed_by.append("late")
            if r["Combined"]["correct"]:
                fixed_by.append("combined")
        
        fix_str = f" -> Fixed by: {', '.join(fixed_by)}" if fixed_by else ""
        print(f"  {verb:15s}: baseline={baseline}, early={early}, late={late}, combined={combined}{fix_str}")
    
    # Save results
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "circuits": {
            "early": EARLY_CIRCUIT,
            "late": LATE_CIRCUIT,
            "combined": COMBINED_CIRCUIT
        },
        "results": results,
        "summary": {
            config_name: {
                "overall_accuracy": sum(1 for r in results.values() if r[config_name]["correct"]) / len(results),
                "by_category": {
                    cat: sum(1 for v, r in results.items() if r["category"] == cat and r[config_name]["correct"]) / 
                         len([v for v, r in results.items() if r["category"] == cat])
                    for cat in ["bad_verbs", "good_verbs", "mid_verbs"]
                }
            }
            for config_name in ["Baseline", "Early Only", "Late Only", "Combined"]
        }
    }
    
    with open(OUTPUT_DIR / "combined_circuit_ablation_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    
    print(f"\nSaved to: {OUTPUT_DIR / 'combined_circuit_ablation_results.json'}")


if __name__ == "__main__":
    main()


