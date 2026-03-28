"""
Step 59: Quick Reasoning Test with Progress

Key fix: Add progress indicators and use shorter responses first.
"""

import torch
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_model():
    print("Loading Qwen3-4B...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    model.eval()
    print("Model loaded!", flush=True)
    return model, tokenizer


def generate_with_progress(model, tokenizer, prompt, max_tokens=300):
    """Generate with visible progress."""
    print("  Generating...", end="", flush=True)
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(" Done!", flush=True)
    return response.strip()


def extract_answer(response):
    """Extract answer after </think> or from first line."""
    if "</think>" in response:
        return response.split("</think>")[-1].strip()
    return response.strip()


def main():
    print("=" * 60, flush=True)
    print("QUICK REASONING TEST", flush=True)
    print("=" * 60, flush=True)
    
    model, tokenizer = load_model()
    
    # Test 1: Simple Sally-Anne with full chat format
    print("\n[TEST 1: Sally-Anne with reasoning]", flush=True)
    
    story = "Alice put the ball in the drawer. Alice left the room. Bob moved the ball to the basket. Alice came back."
    
    prompt = f"""<|im_start|>system
You are helpful. Think step by step in <think> tags, then give a one-word answer.<|im_end|>
<|im_start|>user
{story}

Where will Alice look for the ball? (Answer: drawer or basket)<|im_end|>
<|im_start|>assistant
"""
    
    response = generate_with_progress(model, tokenizer, prompt, max_tokens=400)
    answer = extract_answer(response)
    
    print(f"\n  Story: {story}", flush=True)
    print(f"\n  Full response ({len(response)} chars):", flush=True)
    print(f"  {response[:800]}", flush=True)
    if len(response) > 800:
        print("  ...[truncated]", flush=True)
    print(f"\n  Final answer: {answer[:100]}", flush=True)
    print(f"  Contains 'drawer': {'drawer' in answer.lower()}", flush=True)
    print(f"  Contains 'basket': {'basket' in answer.lower()}", flush=True)
    
    # Test 2: True belief
    print("\n" + "-" * 60, flush=True)
    print("[TEST 2: True belief (Alice watched)]", flush=True)
    
    story2 = "Alice put the ball in the drawer. Bob moved the ball to the basket. Alice watched the whole time."
    
    prompt2 = f"""<|im_start|>system
You are helpful. Think step by step in <think> tags, then give a one-word answer.<|im_end|>
<|im_start|>user
{story2}

Where will Alice look for the ball? (Answer: drawer or basket)<|im_end|>
<|im_start|>assistant
"""
    
    response2 = generate_with_progress(model, tokenizer, prompt2, max_tokens=400)
    answer2 = extract_answer(response2)
    
    print(f"\n  Story: {story2}", flush=True)
    print(f"\n  Full response ({len(response2)} chars):", flush=True)
    print(f"  {response2[:800]}", flush=True)
    print(f"\n  Final answer: {answer2[:100]}", flush=True)
    
    # Test 3: Novel locations
    print("\n" + "-" * 60, flush=True)
    print("[TEST 3: Novel locations]", flush=True)
    
    story3 = "Alice put the ball in container X. Alice left the room. Bob moved the ball to container Y. Alice came back."
    
    prompt3 = f"""<|im_start|>system
You are helpful. Think step by step in <think> tags, then give a one-word answer.<|im_end|>
<|im_start|>user
{story3}

Where will Alice look for the ball? (Answer: X or Y)<|im_end|>
<|im_start|>assistant
"""
    
    response3 = generate_with_progress(model, tokenizer, prompt3, max_tokens=400)
    answer3 = extract_answer(response3)
    
    print(f"\n  Story: {story3}", flush=True)
    print(f"\n  Full response ({len(response3)} chars):", flush=True)
    print(f"  {response3[:800]}", flush=True)
    print(f"\n  Final answer: {answer3[:100]}", flush=True)
    
    # Summary
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    
    print(f"""
    Test 1 (Sally-Anne): {'drawer' in answer.lower()} should be drawer (ToM)
    Test 2 (True belief): {'basket' in answer2.lower()} should be basket
    Test 3 (Novel locs):  {'X' in answer3 or 'x' in answer3.lower()} should be X
    """, flush=True)


if __name__ == "__main__":
    main()


