"""
Step 61: True Belief Clarity Test

The model gets 100% on False Belief but 0% on True Belief.
Let's test if this is due to prompt ambiguity or a real limitation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    return model, tokenizer


def generate(model, tokenizer, prompt, max_tokens=600):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()


def test_scenario(model, tokenizer, name, story, expected):
    prompt = f"""<|im_start|>system
You are helpful. Think step by step in <think> tags, then give a one-word answer: drawer or basket.<|im_end|>
<|im_start|>user
{story}

Where will Alice look for the ball?<|im_end|>
<|im_start|>assistant
"""
    response = generate(model, tokenizer, prompt)
    
    # Get answer after </think>
    if "</think>" in response:
        answer_part = response.split("</think>")[-1].lower()
    else:
        answer_part = response[-200:].lower()  # Last 200 chars
    
    got = "drawer" if "drawer" in answer_part else ("basket" if "basket" in answer_part else "unclear")
    correct = got == expected
    
    print(f"\n{'='*60}", flush=True)
    print(f"[{name}]", flush=True)
    print(f"Story: {story}", flush=True)
    print(f"\nExpected: {expected}, Got: {got}, {'CORRECT' if correct else 'WRONG'}", flush=True)
    print(f"\nFull response:", flush=True)
    print(response, flush=True)
    
    return correct


def main():
    print("=" * 60, flush=True)
    print("TRUE BELIEF CLARITY TEST", flush=True)
    print("=" * 60, flush=True)
    
    model, tokenizer = load_model()
    results = []
    
    # Different phrasings for True Belief
    scenarios = [
        # Very explicit - simultaneous witnessing
        ("TB: While Watching", 
         "Alice put the ball in the drawer. While Alice was watching, Bob took the ball from the drawer and put it in the basket.",
         "basket"),
        
        # Very explicit - present the whole time  
        ("TB: Present Throughout",
         "Alice put the ball in the drawer. Alice stayed in the room watching. Bob moved the ball to the basket right in front of Alice.",
         "basket"),
        
        # Explicit - saw the action
        ("TB: Saw The Action",
         "Alice put the ball in the drawer. Bob moved the ball to the basket. Alice saw Bob move the ball.",
         "basket"),
        
        # Communication after return
        ("TOLD: Clear Communication",
         "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket. Alice came back. Bob said 'Alice, I moved your ball to the basket.'",
         "basket"),
        
        # Alice knows explicitly
        ("TB: Knows Location",
         "Alice put the ball in the drawer. Bob moved the ball to the basket. Alice knows the ball is now in the basket.",
         "basket"),
        
        # Contrast with False Belief (control)
        ("FB: Control - Left",
         "Alice put the ball in the drawer. Alice left the room. Bob moved the ball to the basket. Alice came back.",
         "drawer"),
        
        # Second order - Does Alice know Bob moved it?
        ("TB: Direct Question",
         "Alice put the ball in the drawer. Bob moved the ball to the basket while Alice watched. Alice knows that the ball is in the basket.",
         "basket"),
    ]
    
    for name, story, expected in scenarios:
        result = test_scenario(model, tokenizer, name, story, expected)
        results.append((name, result))
    
    # Summary
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    
    tb_results = [r for n, r in results if "TB" in n or "TOLD" in n]
    fb_results = [r for n, r in results if "FB" in n]
    
    print(f"\nTrue Belief scenarios: {sum(tb_results)}/{len(tb_results)}", flush=True)
    print(f"False Belief control:  {sum(fb_results)}/{len(fb_results)}", flush=True)
    
    for name, result in results:
        print(f"  {name}: {'PASS' if result else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()

