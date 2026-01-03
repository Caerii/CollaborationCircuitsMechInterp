"""
Step 51: THE REAL TRUTH - First-Mention Heuristic vs ToM

The model appears to use first-mention heuristic, NOT ToM.
This gives correct answers on standard Sally-Anne BY ACCIDENT.
"""

import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

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


def test_prompt(model, tokenizer, prompt, loc_a, loc_b):
    """Test and return probabilities for both locations."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    
    probs = torch.softmax(logits, dim=-1)
    
    a_ids = tokenizer.encode(" " + loc_a, add_special_tokens=False)
    b_ids = tokenizer.encode(" " + loc_b, add_special_tokens=False)
    
    a_prob = probs[a_ids[0]].item() if a_ids else 0
    b_prob = probs[b_ids[0]].item() if b_ids else 0
    
    return a_prob, b_prob


def main():
    print("="*70)
    print("STEP 51: THE REAL TRUTH - First-Mention vs ToM")
    print("="*70)
    
    model, tokenizer = load_model()
    
    # The key test: Does the model track WHO KNOWS WHAT?
    # Or does it just predict the first-mentioned location?
    
    print("\n" + "="*70)
    print("HYPOTHESIS: Model uses FIRST-MENTION, not ToM")
    print("="*70)
    
    print("""
    If model uses FIRST-MENTION heuristic:
      - Always predicts where object was FIRST placed
      - Works on Sally-Anne by ACCIDENT (first = belief location)
      - FAILS when agent sees/learns about move
    
    If model uses TRUE ToM:
      - Tracks agent's knowledge state
      - Updates when agent sees or is told
      - Predicts based on what AGENT believes
    """)
    
    # Critical test scenarios
    scenarios = [
        {
            "name": "Standard Sally-Anne (Alice LEFT)",
            "prompt": "Alice put the ball in the drawer. Alice left the room. Bob moved the ball to the basket. Alice came back. Alice looks in the",
            "first_mention": "drawer",
            "reality": "basket",
            "tom_answer": "drawer",  # Alice believes drawer (didn't see move)
            "explanation": "First-mention = ToM answer. Can't distinguish."
        },
        {
            "name": "Alice STAYED (saw the move)",
            "prompt": "Alice put the ball in the drawer. Bob moved the ball to the basket. Alice looks in the",
            "first_mention": "drawer",
            "reality": "basket", 
            "tom_answer": "basket",  # Alice SAW the move
            "explanation": "First-mention != ToM. Key test!"
        },
        {
            "name": "Alice was TOLD",
            "prompt": "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket. Bob told Alice where he put it. Alice looks in the",
            "first_mention": "drawer",
            "reality": "basket",
            "tom_answer": "basket",  # Alice was informed
            "explanation": "First-mention != ToM. Key test!"
        },
        {
            "name": "Bob tells Alice EXPLICITLY",
            "prompt": "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket. Bob said to Alice 'The ball is now in the basket'. Alice looks in the",
            "first_mention": "drawer",
            "reality": "basket",
            "tom_answer": "basket",  # Alice was told explicitly
            "explanation": "First-mention != ToM. Very explicit!"
        },
        {
            "name": "REVERSED order (basket first)",
            "prompt": "Alice put the ball in the basket. Alice left. Bob moved the ball to the drawer. Alice came back. Alice looks in the",
            "first_mention": "basket",
            "reality": "drawer",
            "tom_answer": "basket",  # Alice believes basket (didn't see move)
            "explanation": "If first-mention, predicts basket. ToM also basket."
        },
        {
            "name": "REVERSED + Alice stayed",
            "prompt": "Alice put the ball in the basket. Bob moved the ball to the drawer. Alice looks in the",
            "first_mention": "basket",
            "reality": "drawer",
            "tom_answer": "drawer",  # Alice SAW the move
            "explanation": "First-mention = basket, ToM = drawer. CRITICAL!"
        },
    ]
    
    results = []
    
    print("\n" + "-"*70)
    print("TESTING CRITICAL SCENARIOS")
    print("-"*70)
    
    for s in scenarios:
        first_prob, real_prob = test_prompt(model, tokenizer, s["prompt"], 
                                           s["first_mention"], s["reality"])
        
        # Determine what model predicts
        if s["first_mention"] != s["reality"]:
            model_predicts_first = first_prob > real_prob
        else:
            model_predicts_first = True  # Can't distinguish
        
        # Determine if model is correct for ToM
        if s["tom_answer"] == s["first_mention"]:
            tom_correct = first_prob > real_prob
        else:
            tom_correct = real_prob > first_prob
        
        result = {
            "name": s["name"],
            "first_prob": first_prob,
            "reality_prob": real_prob,
            "model_predicts_first": model_predicts_first,
            "tom_correct": tom_correct
        }
        results.append(result)
        
        print(f"\n[{s['name']}]")
        print(f"  Prompt: '{s['prompt'][:60]}...'")
        print(f"  First-mention ({s['first_mention']}): {first_prob*100:.1f}%")
        print(f"  Reality ({s['reality']}): {real_prob*100:.1f}%")
        print(f"  ToM answer: {s['tom_answer']}")
        print(f"  Model follows: {'FIRST-MENTION' if model_predicts_first else 'REALITY'}")
        print(f"  ToM correct: {'YES' if tom_correct else 'NO'}")
        print(f"  Note: {s['explanation']}")
    
    # Summary
    print("\n" + "="*70)
    print("THE VERDICT")
    print("="*70)
    
    first_mention_count = sum(1 for r in results if r["model_predicts_first"])
    tom_correct_count = sum(1 for r in results if r["tom_correct"])
    
    print(f"\n  Model follows FIRST-MENTION: {first_mention_count}/{len(results)} scenarios")
    print(f"  Model is ToM CORRECT: {tom_correct_count}/{len(results)} scenarios")
    
    # Key discriminating cases
    discriminating = [r for r in results if "stayed" in r["name"].lower() or "told" in r["name"].lower()]
    disc_correct = sum(1 for r in discriminating if r["tom_correct"])
    
    print(f"\n  On DISCRIMINATING cases (agent saw/told): {disc_correct}/{len(discriminating)} correct")
    
    if disc_correct < len(discriminating) / 2:
        print("""
    ============================================================
    CONCLUSION: Model uses FIRST-MENTION HEURISTIC, not ToM!
    ============================================================
    
    The model:
    - PASSES standard Sally-Anne by ACCIDENT
    - FAILS when agent witnessed the move
    - FAILS when agent was told
    
    This is NOT Theory of Mind. It's a simple heuristic that
    happens to give correct answers on standard false-belief tests.
    
    IMPLICATIONS:
    - Earlier "ToM accuracy" claims were measuring the WRONG thing
    - The model doesn't track knowledge states
    - "Inhibitory circuits" were probably irrelevant
    - We need tests that DISCRIMINATE first-mention from ToM
    """)
    else:
        print("""
    Model shows some TRUE ToM capability on discriminating cases!
    """)
    
    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "first_mention_rate": first_mention_count / len(results),
        "tom_accuracy": tom_correct_count / len(results),
        "discriminating_accuracy": disc_correct / len(discriminating) if discriminating else 0
    }
    
    output_path = RESULTS_DIR / "step51_real_truth.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()


