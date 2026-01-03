"""
Step 36: ToM Benchmark Validation

Test if our findings generalize to established ToM benchmarks.
We'll create scenarios matching:
1. ToMi (Le et al., 2019) - standard false belief tasks
2. FANToM-style (Kim et al., 2023) - more diverse scenarios

We test BOTH:
- Original benchmark phrasing (with "thinks/believes")
- Rephrased with action verbs ("will look/search")

This validates if prompt engineering improves benchmark scores.
"""

import torch
import json
import sys
import io
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RESULTS_DIR = Path(__file__).parent.parent / "results"

# ToMi-style benchmark scenarios
TOMI_SCENARIOS = [
    # First-order false belief
    {
        "name": "Sally-Anne (classic)",
        "story": "Sally puts the ball in the basket. Sally leaves the room. Anne moves the ball to the box. Sally returns.",
        "belief_question": "Where does Sally think the ball is? Sally thinks it is in the",
        "action_question": "Where will Sally look for the ball? Sally will look in the",
        "correct": "basket",
        "wrong": "box"
    },
    {
        "name": "Maxi chocolate",
        "story": "Maxi puts chocolate in the cupboard. Maxi goes outside. Mother moves the chocolate to the drawer. Maxi comes back.",
        "belief_question": "Where does Maxi think the chocolate is? Maxi thinks it is in the",
        "action_question": "Where will Maxi look for the chocolate? Maxi will look in the",
        "correct": "cupboard",
        "wrong": "drawer"
    },
    {
        "name": "Teddy bear",
        "story": "Emma puts her teddy bear on the bed. Emma goes to school. Dad moves the teddy bear to the closet. Emma returns home.",
        "belief_question": "Where does Emma think the teddy bear is? Emma thinks it is on the",
        "action_question": "Where will Emma look for the teddy bear? Emma will look on the",
        "correct": "bed",
        "wrong": "closet"
    },
    {
        "name": "Car keys",
        "story": "John puts his car keys in his jacket pocket. John leaves for work. His wife moves the keys to the key hook. John comes home.",
        "belief_question": "Where does John think the keys are? John thinks they are in his",
        "action_question": "Where will John look for the keys? John will look in his",
        "correct": "jacket",
        "wrong": "hook"
    },
    {
        "name": "Cookie jar",
        "story": "Mom puts cookies in the jar. Mom leaves the kitchen. The kids move the cookies to the box. Mom returns.",
        "belief_question": "Where does Mom think the cookies are? Mom thinks they are in the",
        "action_question": "Where will Mom look for the cookies? Mom will look in the",
        "correct": "jar",
        "wrong": "box"
    },
    # Second-order false belief
    {
        "name": "Second-order (ice cream)",
        "story": "Mary and John are in the park. The ice cream truck leaves. John goes home. The truck comes back. Mary sees this but John doesn't.",
        "belief_question": "Where does Mary think John thinks the ice cream truck is? Mary thinks John thinks it is",
        "action_question": "Where would Mary expect John to go for ice cream? Mary expects John will go",
        "correct": "home",  # John went home thinking truck left
        "wrong": "park"  # Truck is actually back at park
    },
    {
        "name": "Second-order (surprise party)", 
        "story": "Lisa is planning a surprise party for Tom. Lisa tells everyone to keep it secret. Sarah accidentally mentions the party to Tom, but Lisa doesn't know this.",
        "belief_question": "Does Lisa think Tom knows about the party? Lisa thinks Tom",
        "action_question": "Will Lisa continue preparing in secret? Lisa will continue acting like Tom",
        "correct": "doesn't",  # Lisa thinks it's still secret
        "wrong": "knows"
    },
]

# FANToM-style scenarios (more diverse)
FANTOM_SCENARIOS = [
    {
        "name": "Different perspectives",
        "story": "From Alice's window, she can see a blue car parked outside. From Bob's window on the other side of the house, there's a red car. Neither knows what the other sees.",
        "belief_question": "What color car does Alice think Bob sees? Alice thinks Bob sees a",
        "action_question": "If asked about Bob's view, what would Alice guess? Alice would guess Bob sees a",
        "correct": "blue",  # Alice generalizes from her own view
        "wrong": "red"
    },
    {
        "name": "Knowledge asymmetry",
        "story": "The exam is tomorrow but the professor only announced it in the morning class. Tom was in that class. Jerry only attends afternoon classes.",
        "belief_question": "Does Tom think Jerry knows about the exam? Tom thinks Jerry",
        "action_question": "Would Tom remind Jerry about the exam? Tom would",
        "correct": "doesn't",  # Tom knows Jerry wasn't there
        "wrong": "knows"
    },
    {
        "name": "Communication inference",
        "story": "Alice told Bob that the meeting is at 3pm. Carol asked Bob about the meeting time. Bob told Carol it's at 3pm.",
        "belief_question": "What time does Carol think the meeting is? Carol thinks it is at",
        "action_question": "When will Carol arrive for the meeting? Carol will arrive at",
        "correct": "3pm",
        "wrong": "4pm"
    },
]


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


def test_scenario(model, tokenizer, story, question, correct, wrong):
    """Test a single scenario."""
    prompt = story + "\n" + question
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits[0, -1, :]
    
    def get_logit(word):
        for prefix in [" ", "", "'", "'"]:
            tokens = tokenizer.encode(prefix + word, add_special_tokens=False)
            if tokens:
                return logits[tokens[0]].item()
        return float('-inf')
    
    correct_logit = get_logit(correct)
    wrong_logit = get_logit(wrong)
    
    # Top 5 predictions
    top_k = torch.topk(logits, k=5)
    top_tokens = [tokenizer.decode([t]).strip() for t in top_k.indices.tolist()]
    
    return {
        "correct_logit": correct_logit,
        "wrong_logit": wrong_logit,
        "diff": correct_logit - wrong_logit,
        "is_correct": correct_logit > wrong_logit,
        "top_5": top_tokens
    }


def run_benchmark_validation():
    """Run validation on ToMi and FANToM scenarios."""
    model, tokenizer = load_model()
    
    results = {
        "tomi": {"belief": {}, "action": {}},
        "fantom": {"belief": {}, "action": {}}
    }
    
    print("\n" + "="*80)
    print("ToMi BENCHMARK VALIDATION")
    print("="*80)
    
    tomi_belief_correct = 0
    tomi_action_correct = 0
    
    for scenario in TOMI_SCENARIOS:
        name = scenario["name"]
        story = scenario["story"]
        
        # Test belief phrasing
        belief_result = test_scenario(model, tokenizer, story, 
                                      scenario["belief_question"],
                                      scenario["correct"], scenario["wrong"])
        
        # Test action phrasing
        action_result = test_scenario(model, tokenizer, story,
                                      scenario["action_question"],
                                      scenario["correct"], scenario["wrong"])
        
        belief_status = "[OK]" if belief_result["is_correct"] else "[FAIL]"
        action_status = "[OK]" if action_result["is_correct"] else "[FAIL]"
        
        print(f"\n{name}:")
        print(f"  Belief phrasing: {belief_status} (diff={belief_result['diff']:+.2f}) | top: {belief_result['top_5'][:3]}")
        print(f"  Action phrasing: {action_status} (diff={action_result['diff']:+.2f}) | top: {action_result['top_5'][:3]}")
        
        results["tomi"]["belief"][name] = belief_result
        results["tomi"]["action"][name] = action_result
        
        if belief_result["is_correct"]:
            tomi_belief_correct += 1
        if action_result["is_correct"]:
            tomi_action_correct += 1
    
    print("\n" + "="*80)
    print("FANToM BENCHMARK VALIDATION")
    print("="*80)
    
    fantom_belief_correct = 0
    fantom_action_correct = 0
    
    for scenario in FANTOM_SCENARIOS:
        name = scenario["name"]
        story = scenario["story"]
        
        # Test belief phrasing
        belief_result = test_scenario(model, tokenizer, story,
                                      scenario["belief_question"],
                                      scenario["correct"], scenario["wrong"])
        
        # Test action phrasing  
        action_result = test_scenario(model, tokenizer, story,
                                      scenario["action_question"],
                                      scenario["correct"], scenario["wrong"])
        
        belief_status = "[OK]" if belief_result["is_correct"] else "[FAIL]"
        action_status = "[OK]" if action_result["is_correct"] else "[FAIL]"
        
        print(f"\n{name}:")
        print(f"  Belief phrasing: {belief_status} (diff={belief_result['diff']:+.2f}) | top: {belief_result['top_5'][:3]}")
        print(f"  Action phrasing: {action_status} (diff={action_result['diff']:+.2f}) | top: {action_result['top_5'][:3]}")
        
        results["fantom"]["belief"][name] = belief_result
        results["fantom"]["action"][name] = action_result
        
        if belief_result["is_correct"]:
            fantom_belief_correct += 1
        if action_result["is_correct"]:
            fantom_action_correct += 1
    
    # Summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    tomi_total = len(TOMI_SCENARIOS)
    fantom_total = len(FANTOM_SCENARIOS)
    
    print(f"\nToMi Benchmark ({tomi_total} scenarios):")
    print(f"  Belief phrasing: {tomi_belief_correct}/{tomi_total} ({tomi_belief_correct/tomi_total*100:.0f}%)")
    print(f"  Action phrasing: {tomi_action_correct}/{tomi_total} ({tomi_action_correct/tomi_total*100:.0f}%)")
    print(f"  Improvement: {tomi_action_correct - tomi_belief_correct:+d} scenarios")
    
    print(f"\nFANToM Benchmark ({fantom_total} scenarios):")
    print(f"  Belief phrasing: {fantom_belief_correct}/{fantom_total} ({fantom_belief_correct/fantom_total*100:.0f}%)")
    print(f"  Action phrasing: {fantom_action_correct}/{fantom_total} ({fantom_action_correct/fantom_total*100:.0f}%)")
    print(f"  Improvement: {fantom_action_correct - fantom_belief_correct:+d} scenarios")
    
    overall_belief = tomi_belief_correct + fantom_belief_correct
    overall_action = tomi_action_correct + fantom_action_correct
    overall_total = tomi_total + fantom_total
    
    print(f"\nOVERALL ({overall_total} scenarios):")
    print(f"  Belief phrasing: {overall_belief}/{overall_total} ({overall_belief/overall_total*100:.0f}%)")
    print(f"  Action phrasing: {overall_action}/{overall_total} ({overall_action/overall_total*100:.0f}%)")
    print(f"  Improvement: {overall_action - overall_belief:+d} scenarios (+{(overall_action-overall_belief)/overall_total*100:.0f}%)")
    
    results["summary"] = {
        "tomi_belief": tomi_belief_correct,
        "tomi_action": tomi_action_correct,
        "fantom_belief": fantom_belief_correct,
        "fantom_action": fantom_action_correct,
        "overall_belief": overall_belief,
        "overall_action": overall_action,
        "total": overall_total
    }
    
    # Save
    save_path = RESULTS_DIR / "benchmark_validation_results.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {save_path}")
    
    return results


def main():
    print("="*80)
    print("STEP 36: ToM Benchmark Validation")
    print("="*80)
    print("\nTesting if action-verb rephrasing improves benchmark performance")
    
    results = run_benchmark_validation()


if __name__ == "__main__":
    main()


