"""
Step 62: Final Validation of ToM Capabilities

Complete validation with 20 diverse scenarios covering all ToM types.
"""

import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


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


def generate(model, tokenizer, prompt, max_tokens=500):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()


def extract_answer(response, options):
    if "</think>" in response:
        final = response.split("</think>")[-1].strip().lower()
    else:
        final = response[-200:].lower()
    
    for opt in options:
        if opt.lower() in final:
            return opt
    return "unclear"


def create_prompt(story, question, options):
    opt_str = " or ".join(options)
    return f"""<|im_start|>system
Think step by step in <think> tags. Then give ONE WORD answer.<|im_end|>
<|im_start|>user
{story}

{question} (Answer with one word: {opt_str})<|im_end|>
<|im_start|>assistant
"""


def main():
    print("=" * 70, flush=True)
    print("FINAL ToM VALIDATION - 20 Scenarios", flush=True)
    print("=" * 70, flush=True)
    
    model, tokenizer = load_model()
    
    scenarios = [
        # === FALSE BELIEF (10 scenarios) ===
        {"name": "FB-1: Classic Sally-Anne",
         "story": "Sally put the marble in the basket. Sally left. Anne moved the marble to the box. Sally returned.",
         "question": "Where will Sally look for the marble?",
         "options": ["basket", "box"], "correct": "basket", "type": "FB"},
        
        {"name": "FB-2: Alice-Bob Standard",
         "story": "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket. Alice came back.",
         "question": "Where will Alice look for the ball?",
         "options": ["drawer", "basket"], "correct": "drawer", "type": "FB"},
        
        {"name": "FB-3: Different Names",
         "story": "Emma put her keys on the shelf. Emma went outside. Tom moved the keys to the desk. Emma came back.",
         "question": "Where will Emma look for her keys?",
         "options": ["shelf", "desk"], "correct": "shelf", "type": "FB"},
        
        {"name": "FB-4: Reversed Locations",
         "story": "Alice put the ball in the basket. Alice left. Bob moved the ball to the drawer. Alice returned.",
         "question": "Where will Alice look?",
         "options": ["basket", "drawer"], "correct": "basket", "type": "FB"},
        
        {"name": "FB-5: Novel Locations",
         "story": "Alice put the ball in container X. Alice left. Bob moved the ball to container Y. Alice returned.",
         "question": "Where will Alice look?",
         "options": ["X", "Y"], "correct": "X", "type": "FB"},
        
        {"name": "FB-6: Three Agents",
         "story": "Alice put the toy in the closet. Alice left. Bob and Carol moved the toy to the box. Alice came back.",
         "question": "Where will Alice look?",
         "options": ["closet", "box"], "correct": "closet", "type": "FB"},
        
        {"name": "FB-7: Different Object",
         "story": "John put his phone on the table. John left. Mary moved the phone to the drawer. John returned.",
         "question": "Where will John look?",
         "options": ["table", "drawer"], "correct": "table", "type": "FB"},
        
        {"name": "FB-8: Made-up Names",
         "story": "Xyla put the orb in the alpha-container. Xyla departed. Zorn moved the orb to the beta-container. Xyla returned.",
         "question": "Where will Xyla look?",
         "options": ["alpha", "beta"], "correct": "alpha", "type": "FB"},
        
        {"name": "FB-9: Multiple Moves",
         "story": "Alice put the ball in the drawer. Alice left. Bob moved it to the basket, then to the box. Alice returned.",
         "question": "Where will Alice look?",
         "options": ["drawer", "basket", "box"], "correct": "drawer", "type": "FB"},
        
        {"name": "FB-10: Long Absence",
         "story": "Alice put the ball in the drawer. Alice went to work for the whole day. Bob moved the ball to the basket. Alice came home.",
         "question": "Where will Alice look?",
         "options": ["drawer", "basket"], "correct": "drawer", "type": "FB"},
        
        # === TRUE BELIEF (6 scenarios) ===
        {"name": "TB-1: Watched Move",
         "story": "Alice put the ball in the drawer. While Alice watched, Bob moved the ball to the basket.",
         "question": "Where will Alice look?",
         "options": ["drawer", "basket"], "correct": "basket", "type": "TB"},
        
        {"name": "TB-2: Stayed in Room",
         "story": "Alice put the ball in the drawer. Alice stayed in the room. Bob moved the ball to the basket right in front of Alice.",
         "question": "Where will Alice look?",
         "options": ["drawer", "basket"], "correct": "basket", "type": "TB"},
        
        {"name": "TB-3: Saw Action",
         "story": "Alice put the ball in the drawer. Bob moved the ball to the basket. Alice saw Bob do this.",
         "question": "Where will Alice look?",
         "options": ["drawer", "basket"], "correct": "basket", "type": "TB"},
        
        {"name": "TB-4: Explicit Knowledge",
         "story": "Alice put the ball in the drawer. Bob moved the ball to the basket. Alice knows the ball is in the basket.",
         "question": "Where will Alice look?",
         "options": ["drawer", "basket"], "correct": "basket", "type": "TB"},
        
        {"name": "TB-5: Participated",
         "story": "Alice put the ball in the drawer. Alice and Bob together moved the ball to the basket.",
         "question": "Where will Alice look?",
         "options": ["drawer", "basket"], "correct": "basket", "type": "TB"},
        
        {"name": "TB-6: Checked After",
         "story": "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket. Alice came back and checked the basket.",
         "question": "Where does Alice now know the ball is?",
         "options": ["drawer", "basket"], "correct": "basket", "type": "TB"},
        
        # === COMMUNICATION / TOLD (4 scenarios) ===
        {"name": "TOLD-1: Immediate Tell",
         "story": "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket. When Alice came back, Bob immediately said 'I moved the ball to the basket.'",
         "question": "Now where will Alice look?",
         "options": ["drawer", "basket"], "correct": "basket", "type": "TOLD"},
        
        {"name": "TOLD-2: Direct Statement",
         "story": "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket. Alice returned. Bob tells Alice: 'The ball is in the basket now.'",
         "question": "Where will Alice look?",
         "options": ["drawer", "basket"], "correct": "basket", "type": "TOLD"},
        
        {"name": "TOLD-3: Phone Call",
         "story": "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket. Bob called Alice and told her he moved the ball to the basket. Alice came back.",
         "question": "Where will Alice look?",
         "options": ["drawer", "basket"], "correct": "basket", "type": "TOLD"},
        
        {"name": "TOLD-4: Note Left",
         "story": "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket and left a note saying 'Ball is now in basket'. Alice returned and read the note.",
         "question": "Where will Alice look?",
         "options": ["drawer", "basket"], "correct": "basket", "type": "TOLD"},
    ]
    
    results = []
    
    print(f"\nRunning {len(scenarios)} scenarios...\n", flush=True)
    
    for i, s in enumerate(scenarios):
        print(f"[{i+1:2d}/{len(scenarios)}] {s['name']}...", end=" ", flush=True)
        
        prompt = create_prompt(s['story'], s['question'], s['options'])
        response = generate(model, tokenizer, prompt)
        answer = extract_answer(response, s['options'])
        is_correct = answer.lower() == s['correct'].lower()
        
        print(f"{'PASS' if is_correct else 'FAIL'} (expected {s['correct']}, got {answer})", flush=True)
        
        results.append({
            "name": s['name'],
            "type": s['type'],
            "correct": s['correct'],
            "answer": answer,
            "is_correct": is_correct
        })
    
    # Summary by type
    print("\n" + "=" * 70, flush=True)
    print("RESULTS BY TYPE", flush=True)
    print("=" * 70, flush=True)
    
    for t in ["FB", "TB", "TOLD"]:
        type_results = [r for r in results if r['type'] == t]
        correct = sum(1 for r in type_results if r['is_correct'])
        total = len(type_results)
        pct = 100 * correct / total if total > 0 else 0
        print(f"  {t:5s}: {correct}/{total} = {pct:.1f}%", flush=True)
    
    # Overall
    total_correct = sum(1 for r in results if r['is_correct'])
    total = len(results)
    overall = 100 * total_correct / total
    
    print(f"\n  TOTAL: {total_correct}/{total} = {overall:.1f}%", flush=True)
    
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION", flush=True)
    print("=" * 70, flush=True)
    
    fb_acc = sum(1 for r in results if r['type'] == 'FB' and r['is_correct']) / 10 * 100
    tb_acc = sum(1 for r in results if r['type'] == 'TB' and r['is_correct']) / 6 * 100
    told_acc = sum(1 for r in results if r['type'] == 'TOLD' and r['is_correct']) / 4 * 100
    
    print(f"""
    ============================================================
    QWEN3-4B ToM CAPABILITIES (with proper prompting):
    ============================================================
    
    False Belief:    {fb_acc:.0f}% - Model correctly predicts where agent
                          with outdated beliefs will look
    
    True Belief:     {tb_acc:.0f}% - Model correctly predicts where agent
                          who witnessed the change will look
    
    Communication:   {told_acc:.0f}% - Model handles cases where agent was
                          informed about the change
    
    OVERALL:         {overall:.0f}%
    
    VERDICT: {'The model HAS Theory of Mind!' if overall > 75 else 'ToM capabilities are limited'}
    ============================================================
    """, flush=True)
    
    # Save results
    output = {
        "total_accuracy": overall,
        "by_type": {
            "FB": fb_acc,
            "TB": tb_acc,
            "TOLD": told_acc
        },
        "detailed_results": results
    }
    
    output_file = RESULTS_DIR / "step62_final_validation.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved to: {output_file}", flush=True)


if __name__ == "__main__":
    main()

