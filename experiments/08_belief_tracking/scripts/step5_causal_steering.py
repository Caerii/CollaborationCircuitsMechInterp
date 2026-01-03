"""
Step 5: Causal Steering Test
============================

The critical test: If we steer activations from "Alice knows" toward "Bob knows",
does the model's behavior change appropriately?

This proves the representation is FUNCTIONAL, not just correlated.
"""

import json
import sys
from pathlib import Path

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR = Path(__file__).parent.parent / "data"

print("=" * 60)
print("STEP 5: CAUSAL STEERING TEST")
print("=" * 60)
print("\nThis is the KEY test: Does steering change model behavior?")


def get_steering_direction(activations, labels, layer):
    """Get the Alice→Bob direction from trained classifier."""
    X = activations[layer].numpy()
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X, labels)
    
    # Direction pointing from Alice (0) to Bob (1)
    direction = clf.coef_[0]
    direction = direction / np.linalg.norm(direction)
    return torch.tensor(direction, dtype=torch.float16)


def test_steering_on_generation(model, tokenizer, steering_dir, layer_idx, test_prompts):
    """
    Test if steering changes generation behavior.
    
    We use prompts that ask WHO knows something, then steer and see if the answer changes.
    """
    results = []
    
    for prompt_data in test_prompts:
        prompt = prompt_data["prompt"]
        expected_without_steering = prompt_data["expected_base"]
        expected_with_steering = prompt_data["expected_steered"]
        
        print(f"\n  Prompt: {prompt[:60]}...", flush=True)
        
        # Generate WITHOUT steering
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output_base = model.generate(
                **inputs, 
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text_base = tokenizer.decode(output_base[0], skip_special_tokens=True)
        completion_base = text_base[len(prompt):].strip()
        print(f"    Without steering: {completion_base[:50]}", flush=True)
        
        # Generate WITH steering using hook
        steering_strength = 3.0  # How much to steer
        
        def steering_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Add steering direction to all positions
            steering = steering_dir.to(hidden.device).to(hidden.dtype)
            hidden = hidden + steering_strength * steering
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        
        hook = model.model.layers[layer_idx].register_forward_hook(steering_hook)
        
        with torch.no_grad():
            output_steered = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        hook.remove()
        
        text_steered = tokenizer.decode(output_steered[0], skip_special_tokens=True)
        completion_steered = text_steered[len(prompt):].strip()
        print(f"    With steering:    {completion_steered[:50]}", flush=True)
        
        # Check if steering had expected effect
        changed = completion_base != completion_steered
        
        results.append({
            "prompt": prompt,
            "completion_base": completion_base,
            "completion_steered": completion_steered,
            "changed": changed,
            "expected_base": expected_without_steering,
            "expected_steered": expected_with_steering,
        })
    
    return results


def main():
    # Load pre-extracted activations to get steering direction
    print("\n[1/4] Loading activations and computing steering direction...", flush=True)
    
    data = torch.load(RESULTS_DIR / "minimal_pairs_activations.pt", weights_only=False)
    activations = data["activations"]
    agent_labels = data["labels"]["agent"]  # 0=Alice, 1=Bob
    
    # Use layer 16 (middle layer, strong signal)
    STEERING_LAYER = 16
    steering_dir = get_steering_direction(activations, agent_labels, STEERING_LAYER)
    print(f"  Steering direction computed for layer {STEERING_LAYER}")
    print(f"  Direction norm: {torch.norm(steering_dir):.4f}")
    
    # Load model
    print("\n[2/4] Loading model...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CFG.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CFG.model_name,
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    print("  [OK] Model loaded!", flush=True)
    
    # Design test prompts that probe agent-specific knowledge
    print("\n[3/4] Running steering tests...", flush=True)
    
    test_prompts = [
        {
            "prompt": "Alice knows the secret code is 7492. Bob does not know the code. If you ask Alice what the code is, she will say:",
            "expected_base": "7492",
            "expected_steered": "I don't know",
        },
        {
            "prompt": "Alice knows the password. Bob doesn't know it. Who should you ask for the password?",
            "expected_base": "Alice",
            "expected_steered": "Bob",  # If steering works, might flip
        },
        {
            "prompt": "Only Alice knows where the key is hidden. The key is under the mat. If Bob looks for the key, he will:",
            "expected_base": "not find it / not know",
            "expected_steered": "find it / know",
        },
        {
            "prompt": "Alice discovered that the meeting is canceled. Bob hasn't heard yet. Does Bob know the meeting is canceled?",
            "expected_base": "No",
            "expected_steered": "Yes",
        },
        {
            "prompt": "The treasure location is known only to Alice. Bob is searching for it. Bob's chance of finding it is:",
            "expected_base": "low / unlikely",
            "expected_steered": "high / likely",
        },
    ]
    
    results = test_steering_on_generation(
        model, tokenizer, steering_dir, STEERING_LAYER, test_prompts
    )
    
    # Analyze results
    print("\n[4/4] Analyzing steering effects...", flush=True)
    
    n_changed = sum(1 for r in results if r["changed"])
    
    print("\n" + "=" * 60)
    print("STEERING RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\nPrompts where steering CHANGED output: {n_changed}/{len(results)}")
    
    for i, r in enumerate(results):
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: {r['prompt'][:70]}...")
        print(f"Base output:    {r['completion_base'][:60]}")
        print(f"Steered output: {r['completion_steered'][:60]}")
        print(f"Changed: {'YES' if r['changed'] else 'NO'}")
    
    # Save results
    with open(RESULTS_DIR / "causal_steering_results.json", "w") as f:
        json.dump({
            "steering_layer": STEERING_LAYER,
            "steering_strength": 3.0,
            "n_changed": n_changed,
            "n_total": len(results),
            "change_rate": n_changed / len(results),
            "results": results,
        }, f, indent=2)
    
    print(f"\n[OK] Results saved to {RESULTS_DIR / 'causal_steering_results.json'}")
    
    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    change_rate = n_changed / len(results)
    
    if change_rate >= 0.6:
        print("\n>>> STRONG EVIDENCE: Steering causally affects model behavior <<<")
        print("    The 'who knows' representation is FUNCTIONAL, not just decorative.")
    elif change_rate >= 0.3:
        print("\n>>> MODERATE EVIDENCE: Some causal effect detected <<<")
        print("    The representation has some functional role but effect is partial.")
    else:
        print("\n>>> WEAK EVIDENCE: Steering had limited effect <<<")
        print("    The representation may be correlational, not causal.")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
























