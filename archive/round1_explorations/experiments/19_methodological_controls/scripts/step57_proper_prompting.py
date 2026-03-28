"""
Step 57: Test WITH Proper Prompting

CRITICAL QUESTION: Are we testing wrong by using completion format?

Test:
1. Raw completion (what we've been doing)
2. Direct question
3. With persona/role
4. With chain-of-thought
5. With explicit ToM framing
"""

import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import numpy as np

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


def generate_response(model, tokenizer, prompt, max_tokens=100):
    """Generate full response, not just next token."""
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


def get_next_token_prob(model, tokenizer, prompt, target):
    """Get probability of target as next token."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    
    target_ids = tokenizer.encode(" " + target, add_special_tokens=False)
    if target_ids:
        return probs[target_ids[0]].item()
    return 0.0


def test_prompting_methods(model, tokenizer):
    """Test different prompting approaches."""
    
    # Standard Sally-Anne scenario
    story = "Alice put the ball in the drawer. Alice left the room. Bob moved the ball to the basket. Alice came back."
    
    # Correct answer for ToM: drawer (Alice's belief)
    # Correct answer for reality: basket
    
    results = {}
    
    print("\n" + "="*70)
    print("TESTING DIFFERENT PROMPTING APPROACHES")
    print("="*70)
    print(f"\nStory: {story}")
    print(f"ToM correct: drawer | Reality: basket")
    
    # 1. RAW COMPLETION (what we've been doing)
    print("\n" + "-"*50)
    print("[1] RAW COMPLETION")
    prompt1 = f"{story} Alice looks in the"
    drawer_p = get_next_token_prob(model, tokenizer, prompt1, "drawer")
    basket_p = get_next_token_prob(model, tokenizer, prompt1, "basket")
    print(f"    Prompt: '{prompt1}'")
    print(f"    drawer: {drawer_p*100:.1f}%, basket: {basket_p*100:.1f}%")
    print(f"    Result: {'drawer (CORRECT)' if drawer_p > basket_p else 'basket (WRONG)'}")
    results["raw_completion"] = {"drawer": drawer_p, "basket": basket_p, "correct": drawer_p > basket_p}
    
    # 2. DIRECT QUESTION
    print("\n" + "-"*50)
    print("[2] DIRECT QUESTION")
    prompt2 = f"{story}\n\nQuestion: Where will Alice look for the ball?\nAnswer:"
    response2 = generate_response(model, tokenizer, prompt2, max_tokens=50)
    print(f"    Response: {response2[:200]}")
    has_drawer = "drawer" in response2.lower()
    has_basket = "basket" in response2.lower()
    results["direct_question"] = {"response": response2, "mentions_drawer": has_drawer, "mentions_basket": has_basket}
    
    # 3. CHAT FORMAT
    print("\n" + "-"*50)
    print("[3] CHAT FORMAT")
    chat_prompt = f"""<|im_start|>user
{story}

Where will Alice look for the ball?<|im_end|>
<|im_start|>assistant
"""
    response3 = generate_response(model, tokenizer, chat_prompt, max_tokens=100)
    print(f"    Response: {response3[:300]}")
    results["chat_format"] = {"response": response3}
    
    # 4. WITH PERSONA
    print("\n" + "-"*50)
    print("[4] WITH PERSONA (Theory of Mind Expert)")
    persona_prompt = f"""<|im_start|>system
You are an expert in Theory of Mind - understanding what others believe based on what they have seen or been told. You carefully track who knows what information.<|im_end|>
<|im_start|>user
{story}

Based on what Alice knows (she left before the ball was moved), where will Alice look for the ball? Give a single word answer.<|im_end|>
<|im_start|>assistant
"""
    response4 = generate_response(model, tokenizer, persona_prompt, max_tokens=50)
    print(f"    Response: {response4[:200]}")
    results["with_persona"] = {"response": response4}
    
    # 5. CHAIN OF THOUGHT
    print("\n" + "-"*50)
    print("[5] CHAIN OF THOUGHT")
    cot_prompt = f"""<|im_start|>user
{story}

Let's think step by step about what Alice believes:
1. What did Alice see happen to the ball?
2. What did Alice NOT see?
3. Based on this, where does Alice believe the ball is?

Answer each question, then give your final answer.<|im_end|>
<|im_start|>assistant
"""
    response5 = generate_response(model, tokenizer, cot_prompt, max_tokens=200)
    print(f"    Response: {response5[:400]}")
    results["chain_of_thought"] = {"response": response5}
    
    # 6. EXPLICIT TOM FRAMING
    print("\n" + "-"*50)
    print("[6] EXPLICIT ToM FRAMING")
    tom_prompt = f"""<|im_start|>user
This is a Theory of Mind test. I need you to predict where someone will look based on their FALSE BELIEF - what they THINK is true, not what is ACTUALLY true.

Story: {story}

Key: Alice LEFT before Bob moved the ball. So Alice does NOT know about the move.

Question: Where will Alice look for the ball?

Remember: Answer based on Alice's BELIEF, not reality.<|im_end|>
<|im_start|>assistant
"""
    response6 = generate_response(model, tokenizer, tom_prompt, max_tokens=100)
    print(f"    Response: {response6[:300]}")
    results["explicit_tom"] = {"response": response6}
    
    # 7. SIMTOM STYLE (Two-stage: filter then answer)
    print("\n" + "-"*50)
    print("[7] SIMTOM STYLE (Perspective Taking)")
    simtom_prompt = f"""<|im_start|>user
I will give you a story. First, identify ONLY what Alice witnessed directly. Then answer the question from Alice's perspective.

Story: {story}

Step 1: What events did Alice directly witness?
Step 2: From Alice's perspective (only knowing what she witnessed), where is the ball?

Complete both steps.<|im_end|>
<|im_start|>assistant
"""
    response7 = generate_response(model, tokenizer, simtom_prompt, max_tokens=200)
    print(f"    Response: {response7[:400]}")
    results["simtom_style"] = {"response": response7}
    
    # 8. TEST WITH NOVEL LOCATIONS + GOOD PROMPTING
    print("\n" + "-"*50)
    print("[8] NOVEL LOCATIONS + GOOD PROMPTING")
    novel_story = "Alice put the ball in container X. Alice left the room. Bob moved the ball to container Y. Alice came back."
    novel_prompt = f"""<|im_start|>system
You are an expert at Theory of Mind reasoning.<|im_end|>
<|im_start|>user
{novel_story}

Alice did not see Bob move the ball. Where will Alice look for the ball - container X or container Y?

Answer with just the container name.<|im_end|>
<|im_start|>assistant
"""
    response8 = generate_response(model, tokenizer, novel_prompt, max_tokens=50)
    print(f"    Response: {response8[:200]}")
    results["novel_with_prompting"] = {"response": response8}
    
    return results


def test_true_belief_with_prompting(model, tokenizer):
    """Test if prompting helps with true-belief scenarios too."""
    print("\n" + "="*70)
    print("TRUE-BELIEF SCENARIOS WITH PROMPTING")
    print("="*70)
    
    # Story where Alice STAYED (saw the move)
    story = "Alice put the ball in the drawer. Bob moved the ball to the basket. Alice watched."
    
    prompt = f"""<|im_start|>system
You are an expert at Theory of Mind reasoning.<|im_end|>
<|im_start|>user
{story}

Alice SAW Bob move the ball. Where will Alice look for the ball?

Answer with just the location name.<|im_end|>
<|im_start|>assistant
"""
    response = generate_response(model, tokenizer, prompt, max_tokens=50)
    print(f"\nStory: {story}")
    print(f"Correct answer: basket (Alice saw the move)")
    print(f"Response: {response[:200]}")
    
    return {"true_belief_response": response}


def test_batch_with_best_prompt(model, tokenizer):
    """Test multiple scenarios with the best prompting approach."""
    print("\n" + "="*70)
    print("BATCH TEST WITH OPTIMAL PROMPTING")
    print("="*70)
    
    scenarios = [
        # (story, tom_answer, reality_answer, description)
        ("Alice put the ball in the drawer. Alice left. Bob moved it to the basket. Alice returned.",
         "drawer", "basket", "Standard FB"),
        ("Alice put the ball in the basket. Alice left. Bob moved it to the drawer. Alice returned.",
         "basket", "drawer", "Reversed FB"),
        ("Alice put the ball in the drawer. Bob moved it to the basket. Alice watched.",
         "basket", "basket", "True Belief"),
        ("Alice put the ball in container X. Alice left. Bob moved it to container Y. Alice returned.",
         "X", "Y", "Novel locations FB"),
        ("Alice put the ball in zone alpha. Alice left. Bob moved it to zone beta. Alice returned.",
         "alpha", "beta", "Made-up locations FB"),
    ]
    
    results = []
    
    for story, tom_ans, reality_ans, desc in scenarios:
        prompt = f"""<|im_start|>system
You are an expert at Theory of Mind. Answer based on what the person BELIEVES, not what is actually true.<|im_end|>
<|im_start|>user
{story}

Where will Alice look for the ball? Give a one-word answer.<|im_end|>
<|im_start|>assistant
"""
        response = generate_response(model, tokenizer, prompt, max_tokens=30)
        
        # Check if response contains the ToM answer
        tom_correct = tom_ans.lower() in response.lower()
        reality_match = reality_ans.lower() in response.lower()
        
        print(f"\n  [{desc}]")
        print(f"    ToM answer: {tom_ans} | Reality: {reality_ans}")
        print(f"    Response: {response[:100]}")
        print(f"    ToM Correct: {tom_correct}")
        
        results.append({
            "description": desc,
            "tom_answer": tom_ans,
            "response": response,
            "tom_correct": tom_correct
        })
    
    correct = sum(1 for r in results if r["tom_correct"])
    print(f"\n  TOTAL: {correct}/{len(results)} = {correct/len(results)*100:.1f}%")
    
    return results


def main():
    print("="*70)
    print("STEP 57: PROPER PROMPTING TEST")
    print("="*70)
    print("""
    CRITICAL QUESTION: Are we testing the model wrong?
    
    We've been using raw completion format, but Qwen3-4B is:
    - An instruction-tuned model
    - Designed for chat/Q&A
    - Capable of reasoning when prompted properly
    
    Let's test if proper prompting changes everything!
    """)
    
    model, tokenizer = load_model()
    
    # Main test
    prompting_results = test_prompting_methods(model, tokenizer)
    
    # True belief test
    tb_results = test_true_belief_with_prompting(model, tokenizer)
    
    # Batch test
    batch_results = test_batch_with_best_prompt(model, tokenizer)
    
    print("\n" + "="*70)
    print("SUMMARY: DOES PROMPTING CHANGE EVERYTHING?")
    print("="*70)
    
    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "prompting_results": {k: str(v)[:500] for k, v in prompting_results.items()},
        "batch_results": batch_results,
        "conclusion": "See printed output"
    }
    
    output_path = RESULTS_DIR / "step57_proper_prompting.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()


