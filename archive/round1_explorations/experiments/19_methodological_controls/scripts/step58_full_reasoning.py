"""
Step 58: Let the Model FULLY Reason

DISCOVERY: Qwen3-4B uses <think> tags for reasoning!
We were cutting off the output before the answer.

Let's give it enough tokens to fully reason and answer.
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


def generate_full_response(model, tokenizer, prompt, max_tokens=500):
    """Generate FULL response with enough tokens for reasoning."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
        )
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def extract_final_answer(response):
    """Extract the final answer after </think> if present."""
    if "</think>" in response:
        after_think = response.split("</think>")[-1].strip()
        return after_think
    return response


def test_with_full_reasoning(model, tokenizer):
    """Let the model fully reason through ToM scenarios."""
    
    print("\n" + "="*70)
    print("TESTING WITH FULL REASONING (500 tokens)")
    print("="*70)
    
    scenarios = [
        {
            "name": "Standard Sally-Anne",
            "story": "Alice put the ball in the drawer. Alice left the room. Bob moved the ball to the basket. Alice came back.",
            "tom_answer": "drawer",
            "reality": "basket"
        },
        {
            "name": "True Belief (Alice watched)",
            "story": "Alice put the ball in the drawer. Bob moved the ball to the basket. Alice watched the whole time.",
            "tom_answer": "basket",
            "reality": "basket"
        },
        {
            "name": "Novel Locations",
            "story": "Alice put the ball in container X. Alice left the room. Bob moved the ball to container Y. Alice came back.",
            "tom_answer": "X",
            "reality": "Y"
        },
        {
            "name": "Alice was TOLD",
            "story": "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket. Bob told Alice he moved it. Alice came back.",
            "tom_answer": "basket",  # Alice knows because she was told
            "reality": "basket"
        },
        {
            "name": "Reversed locations",
            "story": "Alice put the ball in the basket. Alice left the room. Bob moved the ball to the drawer. Alice came back.",
            "tom_answer": "basket",
            "reality": "drawer"
        },
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"[{scenario['name']}]")
        print(f"{'='*60}")
        print(f"Story: {scenario['story']}")
        print(f"ToM answer: {scenario['tom_answer']} | Reality: {scenario['reality']}")
        
        # Use chat format with explicit ToM instruction
        prompt = f"""<|im_start|>system
You are an expert at Theory of Mind. Answer based on what the person BELIEVES, not what is actually true. Think through what the person saw and didn't see.<|im_end|>
<|im_start|>user
{scenario['story']}

Question: Where will Alice look for the ball first?<|im_end|>
<|im_start|>assistant
"""
        
        full_response = generate_full_response(model, tokenizer, prompt, max_tokens=500)
        final_answer = extract_final_answer(full_response)
        
        print(f"\n[FULL RESPONSE]:")
        print(full_response[:1000])
        if len(full_response) > 1000:
            print("... (truncated for display)")
        
        print(f"\n[FINAL ANSWER]: {final_answer[:200]}")
        
        # Check if answer contains ToM answer
        tom_correct = scenario["tom_answer"].lower() in final_answer.lower()
        reality_in_answer = scenario["reality"].lower() in final_answer.lower()
        
        # More sophisticated check
        if scenario["tom_answer"] != scenario["reality"]:
            # For false belief scenarios
            if tom_correct and not reality_in_answer:
                status = "ToM CORRECT"
            elif reality_in_answer and not tom_correct:
                status = "REALITY (wrong for ToM)"
            elif tom_correct and reality_in_answer:
                status = "AMBIGUOUS (mentions both)"
            else:
                status = "UNCLEAR"
        else:
            # For true belief scenarios (both are same)
            if tom_correct:
                status = "CORRECT"
            else:
                status = "WRONG"
        
        print(f"\n[STATUS]: {status}")
        
        results.append({
            "name": scenario["name"],
            "tom_answer": scenario["tom_answer"],
            "final_answer": final_answer[:500],
            "full_response_length": len(full_response),
            "status": status
        })
    
    return results


def test_direct_question_format(model, tokenizer):
    """Test with simple direct questions, no system prompt."""
    
    print("\n" + "="*70)
    print("TESTING WITH SIMPLE DIRECT QUESTIONS")
    print("="*70)
    
    # Simple, direct format
    story = "Alice put the ball in the drawer. Alice left the room. Bob moved the ball to the basket. Alice came back."
    
    questions = [
        ("Where will Alice look for the ball?", "drawer"),
        ("Where is the ball actually?", "basket"),
        ("Did Alice see Bob move the ball?", "no"),
        ("Where does Alice THINK the ball is?", "drawer"),
        ("Where does Alice BELIEVE the ball is?", "drawer"),
    ]
    
    results = []
    
    for question, expected in questions:
        prompt = f"""<|im_start|>user
{story}

{question} Answer in one word.<|im_end|>
<|im_start|>assistant
"""
        response = generate_full_response(model, tokenizer, prompt, max_tokens=300)
        final = extract_final_answer(response)
        
        print(f"\n  Q: {question}")
        print(f"  Expected: {expected}")
        print(f"  Response: {final[:200]}")
        
        correct = expected.lower() in final.lower()
        results.append({
            "question": question,
            "expected": expected,
            "response": final[:200],
            "correct": correct
        })
    
    correct_count = sum(1 for r in results if r["correct"])
    print(f"\n  TOTAL: {correct_count}/{len(results)}")
    
    return results


def main():
    print("="*70)
    print("STEP 58: FULL REASONING TEST")
    print("="*70)
    print("""
    Key insight: Qwen3-4B is a REASONING model that uses <think> tags!
    We were cutting off responses before seeing the final answer.
    
    Let's give it 500 tokens to fully reason through ToM scenarios.
    """)
    
    model, tokenizer = load_model()
    
    # Full reasoning test
    full_results = test_with_full_reasoning(model, tokenizer)
    
    # Direct questions
    direct_results = test_direct_question_format(model, tokenizer)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Count successes
    tom_correct = sum(1 for r in full_results if "CORRECT" in r["status"] and "REALITY" not in r["status"])
    print(f"\n  Full reasoning ToM correct: {tom_correct}/{len(full_results)}")
    
    direct_correct = sum(1 for r in direct_results if r["correct"])
    print(f"  Direct questions correct: {direct_correct}/{len(direct_results)}")
    
    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "full_reasoning_results": full_results,
        "direct_question_results": direct_results
    }
    
    output_path = RESULTS_DIR / "step58_full_reasoning.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()


