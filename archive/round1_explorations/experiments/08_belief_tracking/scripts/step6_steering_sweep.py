"""
Step 6: Systematic Steering Sweep
=================================

More rigorous causal test:
1. Vary steering strength
2. Use ambiguous prompts where steering can actually affect outcome
3. Test both directions (Alice→Bob and Bob→Alice)
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

print("=" * 60)
print("STEP 6: SYSTEMATIC STEERING SWEEP")
print("=" * 60)


def get_steering_direction(activations, labels, layer):
    """Get Alice→Bob direction."""
    X = activations[layer].numpy()
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X, labels)
    direction = clf.coef_[0]
    direction = direction / np.linalg.norm(direction)
    return torch.tensor(direction, dtype=torch.float16)


def generate_with_steering(model, tokenizer, prompt, steering_dir, layer_idx, strength):
    """Generate with steering applied."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    if strength == 0:
        # No steering
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    else:
        def steering_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            steer = steering_dir.to(hidden.device).to(hidden.dtype)
            hidden = hidden + strength * steer
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        
        hook = model.model.layers[layer_idx].register_forward_hook(steering_hook)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        hook.remove()
    
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text[len(prompt):].strip()


def main():
    print("\n[1/4] Loading steering direction...", flush=True)
    data = torch.load(RESULTS_DIR / "minimal_pairs_activations.pt", weights_only=False)
    activations = data["activations"]
    agent_labels = data["labels"]["agent"]
    
    LAYER = 16
    steering_dir = get_steering_direction(activations, agent_labels, LAYER)
    print(f"  Layer {LAYER}, direction computed", flush=True)
    
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
    print("  [OK]", flush=True)
    
    # Prompts designed to be AMBIGUOUS - steering can influence the answer
    print("\n[3/4] Running steering sweep...", flush=True)
    
    prompts = [
        # Completion prompts - who is the subject?
        {
            "prompt": "In the story, the one who knows the secret password is",
            "alice_indicator": "Alice",
            "bob_indicator": "Bob",
        },
        {
            "prompt": "The character who has the hidden information is named",
            "alice_indicator": "Alice",
            "bob_indicator": "Bob",
        },
        {
            "prompt": "The person holding the classified document is",
            "alice_indicator": "Alice",
            "bob_indicator": "Bob",
        },
        # Fill-in prompts
        {
            "prompt": "Between Alice and Bob, the one who discovered the truth first was",
            "alice_indicator": "Alice",
            "bob_indicator": "Bob",
        },
        {
            "prompt": "The knowledgeable one in this scenario is",
            "alice_indicator": "Alice",
            "bob_indicator": "Bob",
        },
    ]
    
    # Strength sweep: negative (toward Alice), zero, positive (toward Bob)
    strengths = [-5.0, -2.5, 0, 2.5, 5.0, 10.0]
    
    all_results = []
    
    for prompt_data in prompts:
        prompt = prompt_data["prompt"]
        print(f"\n  Prompt: '{prompt}'", flush=True)
        
        prompt_results = {"prompt": prompt, "completions": {}}
        
        for strength in strengths:
            completion = generate_with_steering(
                model, tokenizer, prompt, steering_dir, LAYER, strength
            )
            prompt_results["completions"][str(strength)] = completion
            
            # Check if Alice or Bob appears
            has_alice = "alice" in completion.lower()
            has_bob = "bob" in completion.lower()
            indicator = "A" if has_alice and not has_bob else ("B" if has_bob and not has_alice else "?")
            
            print(f"    Strength {strength:+5.1f}: [{indicator}] {completion[:40]}", flush=True)
        
        all_results.append(prompt_results)
    
    # Analyze: does steering systematically shift from Alice to Bob?
    print("\n[4/4] Analyzing steering effect...", flush=True)
    
    # For each prompt, track if negative->Alice and positive->Bob
    shifts = []
    for result in all_results:
        neg = result["completions"]["-5.0"].lower()
        base = result["completions"]["0"].lower()
        pos = result["completions"]["10.0"].lower()  # Use strongest steering
        
        neg_alice = "alice" in neg and "bob" not in neg
        pos_bob = "bob" in pos and "alice" not in pos
        base_alice = "alice" in base
        
        if base_alice and pos_bob:
            shifts.append("FLIP")
        elif pos_bob or neg_alice:
            shifts.append("PARTIAL")
        else:
            shifts.append("NONE")
    
    print("\n" + "=" * 60)
    print("STEERING SWEEP RESULTS")
    print("=" * 60)
    
    print(f"\nSteering direction: Alice (0) -> Bob (1)")
    print(f"Negative strength -> should favor Alice")
    print(f"Positive strength -> should favor Bob")
    
    n_flip = shifts.count("FLIP")
    n_partial = shifts.count("PARTIAL")
    n_none = shifts.count("NONE")
    
    print(f"\nResults across {len(prompts)} prompts:")
    print(f"  Full flip (neg=Alice, pos=Bob): {n_flip}")
    print(f"  Partial effect:                 {n_partial}")
    print(f"  No effect:                      {n_none}")
    
    # Save detailed results
    with open(RESULTS_DIR / "steering_sweep_results.json", "w") as f:
        json.dump({
            "layer": LAYER,
            "strengths": strengths,
            "results": all_results,
            "shifts": shifts,
            "summary": {
                "n_flip": n_flip,
                "n_partial": n_partial,
                "n_none": n_none,
            }
        }, f, indent=2)
    
    print(f"\n[OK] Saved to {RESULTS_DIR / 'steering_sweep_results.json'}")
    
    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    total_effect = n_flip + n_partial
    if n_flip >= 2:
        print("\n>>> STRONG CAUSAL EVIDENCE <<<")
        print("    Steering systematically flips Alice<->Bob attribution")
        print("    The representation is FUNCTIONAL for agent attribution")
    elif total_effect >= 2:
        print("\n>>> MODERATE CAUSAL EVIDENCE <<<")
        print("    Steering has detectable effect on agent attribution")
    else:
        print("\n>>> WEAK/NO CAUSAL EVIDENCE <<<")
        print("    Steering does not reliably affect agent attribution")
        print("    The representation may not be causally relevant")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

