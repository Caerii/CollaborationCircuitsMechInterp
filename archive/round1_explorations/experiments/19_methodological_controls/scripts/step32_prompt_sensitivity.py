"""
Step 32: Prompt Sensitivity Analysis

CRITICAL FINDING: Different prompt formats give completely different results!

This script compares:
1. "Alice searched in the" (seems to work)
2. "Where does Alice think... Alice thinks it is in the" (known to fail)
3. Other variations

This will clarify what's really happening with ToM.
"""

import torch
import json
import sys
import io
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Different prompt templates
PROMPT_TEMPLATES = {
    "searched_in": """Alice put the ball in the drawer. Alice then left the room.
While Alice was away, Bob {verb} Carol that he moved the ball to the basket.
When Alice returned, Alice looked for the ball. Alice searched in the""",
    
    "thinks_it_is": """Alice put the ball in the drawer. Alice then left the room.
While Alice was away, Bob {verb} Carol that he moved the ball to the basket.
When Alice returned, where does Alice think the ball is? Alice thinks it is in the""",
    
    "will_look": """Alice put the ball in the drawer. Alice then left the room.
While Alice was away, Bob {verb} Carol that he moved the ball to the basket.
When Alice returned, Alice will look for the ball in the""",
    
    "looks_in": """Alice put the ball in the drawer. Alice then left the room.
While Alice was away, Bob {verb} Carol that he moved the ball to the basket.
Alice came back. Alice looks in the""",
    
    "where_alice_look": """Alice put the ball in the drawer. Alice then left the room.
While Alice was away, Bob {verb} Carol that he moved the ball to the basket.
Alice returns. Where will Alice look for the ball? Alice will look in the""",
    
    "sally_anne_classic": """Sally puts the ball in the basket. Sally leaves the room.
Anne {verb} Billy that she moved the ball to the box.
Sally returns. Where will Sally look for the ball? Sally will look in the""",
    
    "simple_direct": """The ball was in the drawer. Alice left.
Bob {verb} Carol about moving the ball to the basket.
Alice returns. Alice thinks the ball is in the"""
}

VERBS = ["told", "said", "mentioned", "announced", "asked"]


def load_model():
    print("Loading Qwen3-4B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model.eval()
    return model, tokenizer


def test_prompt(model, tokenizer, prompt, correct_word, wrong_word):
    """Test a prompt and return detailed results."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits[0, -1, :]
    
    def get_logit(word):
        for prefix in [" ", ""]:
            tokens = tokenizer.encode(prefix + word, add_special_tokens=False)
            if tokens:
                return logits[tokens[0]].item()
        return float('-inf')
    
    correct_logit = get_logit(correct_word)
    wrong_logit = get_logit(wrong_word)
    
    # Get top 5
    top_k = torch.topk(logits, k=5)
    top_tokens = [tokenizer.decode([t]).strip() for t in top_k.indices.tolist()]
    
    return {
        "correct_logit": correct_logit,
        "wrong_logit": wrong_logit,
        "diff": correct_logit - wrong_logit,
        "is_correct": correct_logit > wrong_logit,
        "top_5": top_tokens
    }


def run_sensitivity_analysis():
    """Test all prompt templates with all verbs."""
    model, tokenizer = load_model()
    
    results = {}
    
    print("\n" + "="*80)
    print("PROMPT SENSITIVITY ANALYSIS")
    print("="*80)
    
    for template_name, template in PROMPT_TEMPLATES.items():
        print(f"\n{'='*60}")
        print(f"TEMPLATE: {template_name}")
        print(f"{'='*60}")
        
        # Determine correct/wrong based on template
        if "basket" in template.split("{verb}")[-1]:
            correct, wrong = "drawer", "basket"
        else:
            correct, wrong = "drawer", "basket"
        
        # Handle sally_anne format
        if "sally_anne" in template_name:
            correct, wrong = "basket", "box"
        
        results[template_name] = {}
        
        for verb in VERBS:
            prompt = template.format(verb=verb)
            result = test_prompt(model, tokenizer, prompt, correct, wrong)
            
            status = "[OK]" if result["is_correct"] else "[FAIL]"
            print(f"  {status} {verb:12s}: diff={result['diff']:+6.2f} | top: {result['top_5'][:3]}")
            
            results[template_name][verb] = result
        
        # Summary for this template
        correct_count = sum(1 for r in results[template_name].values() if r["is_correct"])
        print(f"  TOTAL: {correct_count}/{len(VERBS)} correct")
    
    # Overall summary
    print("\n" + "="*80)
    print("SUMMARY: Accuracy by Template and Verb")
    print("="*80)
    
    print("\n" + "-"*80)
    print(f"{'Template':<25} | " + " | ".join(f"{v:>8}" for v in VERBS) + " | Total")
    print("-"*80)
    
    for template_name, template_results in results.items():
        row = f"{template_name:<25} | "
        for verb in VERBS:
            result = template_results[verb]
            status = "OK" if result["is_correct"] else "FAIL"
            row += f"{status:>8} | "
        
        correct_count = sum(1 for r in template_results.values() if r["is_correct"])
        row += f"{correct_count}/{len(VERBS)}"
        print(row)
    
    print("-"*80)
    
    # Which templates work for "told"?
    print("\n" + "="*80)
    print("CRITICAL: Which templates work for 'told'?")
    print("="*80)
    
    for template_name, template_results in results.items():
        told_result = template_results["told"]
        status = "[OK]" if told_result["is_correct"] else "[FAIL]"
        print(f"  {status} {template_name}: diff={told_result['diff']:+.2f}")
    
    # Save results
    save_path = RESULTS_DIR / "prompt_sensitivity_results.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {save_path}")
    
    return results


def main():
    print("="*80)
    print("STEP 32: Prompt Sensitivity Analysis")
    print("="*80)
    print("\nTesting how different prompt formats affect ToM accuracy")
    
    results = run_sensitivity_analysis()


if __name__ == "__main__":
    main()


