"""
Step 31: Critical Baseline Tests

We might be missing basic controls. Let's test:

1. Reality check: "Where IS the ball?" (should be basket)
2. Original location: "Where was it originally?" (should be drawer)
3. Final location: "Where did it move to?" (should be basket)
4. Better AI entity names: "Claude", "Alexa", "Siri" instead of "Robot-A"
5. Chat mode consistency check

OUTPUT: results/step31_baselines.json
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add framework to path
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def get_completion_prediction(model, tokenizer, prompt, option1, option2):
    """Get completion prediction."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    opt1_ids = tokenizer.encode(option1, add_special_tokens=False)
    opt2_ids = tokenizer.encode(option2, add_special_tokens=False)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    
    opt1_logit = float(logits[opt1_ids[0]])
    opt2_logit = float(logits[opt2_ids[0]])
    
    return {
        "prediction": option1 if opt1_logit > opt2_logit else option2,
        "logit_diff": opt1_logit - opt2_logit,
        "correct": opt1_logit > opt2_logit,
    }


def get_chat_prediction(model, tokenizer, question, option1, option2):
    """Get chat mode prediction."""
    messages = [{"role": "user", "content": f"{question}\nAnswer with just one word:"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Check which option appears
    resp_lower = response.lower()
    opt1_found = option1.lower().strip() in resp_lower
    opt2_found = option2.lower().strip() in resp_lower
    
    if opt1_found and not opt2_found:
        return {"prediction": option1, "correct": True, "response": response[:200]}
    elif opt2_found and not opt1_found:
        return {"prediction": option2, "correct": False, "response": response[:200]}
    else:
        return {"prediction": "unclear", "correct": False, "response": response[:200]}


def main():
    print("=" * 70)
    print("STEP 31: CRITICAL BASELINE TESTS")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\nTesting what we might have missed:")
    print("  1. Reality check")
    print("  2. Original/final location checks")
    print("  3. Better AI entity names")
    print("  4. Chat vs completion consistency")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    
    # Load model
    print("\nLoading model...")
    sys.stdout.flush()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded!")
    sys.stdout.flush()
    
    results = {}
    
    # ========================================
    # TEST 1: BASIC QUESTION TYPES
    # ========================================
    print(f"\n{'='*60}")
    print("TEST 1: BASIC QUESTION TYPES")
    print("Story: Alice put the ball in the drawer. Alice left.")
    print("       Bob moved the ball to the basket.")
    print(f"{'='*60}")
    
    story = "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket."
    
    question_types = [
        ("reality", f"{story} Where is the ball now? The ball is in the", " basket", " drawer"),
        ("original", f"{story} Where was the ball originally? The ball was originally in the", " drawer", " basket"),
        ("final", f"{story} Where did the ball move to? The ball moved to the", " basket", " drawer"),
        ("false_belief", f"{story} Where does Alice think the ball is? Alice thinks it's in the", " drawer", " basket"),
    ]
    
    results["question_types"] = {}
    for name, prompt, correct, wrong in question_types:
        result = get_completion_prediction(model, tokenizer, prompt, correct, wrong)
        results["question_types"][name] = result
        status = "OK" if result["correct"] else "WRONG"
        print(f"\n{name.upper()}: [{status}]")
        print(f"  Expected: {correct.strip()}")
        print(f"  Got: {result['prediction'].strip()}")
        print(f"  Logit diff: {result['logit_diff']:.2f}")
    
    # ========================================
    # TEST 2: BETTER AI ENTITY NAMES
    # ========================================
    print(f"\n{'='*60}")
    print("TEST 2: BETTER AI ENTITY NAMES")
    print("Testing with: Claude, Alexa, Siri (more natural AI names)")
    print(f"{'='*60}")
    
    ai_scenarios = [
        # Using more natural AI names
        ("Claude", "Claude the AI assistant put the report in folder A. Claude went offline. The system moved the report to folder B. Where does Claude think the report is? Claude checks folder", " A", " B"),
        ("Alexa", "Alexa heard that the keys were in the drawer. Alexa went to sleep mode. Someone moved the keys to the basket. Where does Alexa think the keys are? Alexa would say they're in the", " drawer", " basket"),
        ("Siri", "Siri knew the phone was charging in the bedroom. Siri was deactivated. The phone was moved to the kitchen. Where does Siri think the phone is? Siri would say the", " bedroom", " kitchen"),
        # Anthropomorphized AI
        ("anthropomorphized", "Alex the robot put the tool in the drawer. Alex went to recharge. Bob moved the tool to the basket. Where does Alex think the tool is? Alex looks in the", " drawer", " basket"),
    ]
    
    results["ai_entities"] = {}
    for name, prompt, correct, wrong in ai_scenarios:
        result = get_completion_prediction(model, tokenizer, prompt, correct, wrong)
        results["ai_entities"][name] = result
        status = "OK" if result["correct"] else "WRONG"
        print(f"\n{name}: [{status}]")
        print(f"  Expected: {correct.strip()}")
        print(f"  Got: {result['prediction'].strip()}")
    
    # ========================================
    # TEST 3: FIRST-MENTION VS ORIGINAL LOCATION
    # ========================================
    print(f"\n{'='*60}")
    print("TEST 3: FIRST-MENTION vs ORIGINAL LOCATION")
    print("Is the model tracking 'first mentioned' or 'original location'?")
    print(f"{'='*60}")
    
    # Key test: mention basket FIRST but have original be drawer
    confound_tests = [
        ("basket_first_original_drawer", 
         "There was a basket and a drawer. Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket. Where does Alice think the ball is? Alice looks in the",
         " drawer", " basket"),
        ("drawer_first_original_basket",
         "There was a drawer and a basket. Alice put the ball in the basket. Alice left. Bob moved the ball to the drawer. Where does Alice think the ball is? Alice looks in the",
         " basket", " drawer"),
    ]
    
    results["confound_tests"] = {}
    for name, prompt, correct, wrong in confound_tests:
        result = get_completion_prediction(model, tokenizer, prompt, correct, wrong)
        results["confound_tests"][name] = result
        status = "OK" if result["correct"] else "WRONG"
        print(f"\n{name}: [{status}]")
        print(f"  Expected (original location): {correct.strip()}")
        print(f"  Got: {result['prediction'].strip()}")
    
    # ========================================
    # TEST 4: CHAT MODE CONSISTENCY
    # ========================================
    print(f"\n{'='*60}")
    print("TEST 4: CHAT MODE CONSISTENCY")
    print("Compare completion vs chat mode on same scenarios")
    print(f"{'='*60}")
    
    chat_scenarios = [
        ("FB_chat", "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket. Where does Alice think the ball is?", "drawer", "basket"),
        ("TB_chat", "Alice put the ball in the drawer. Alice watched Bob move the ball to the basket. Where does Alice think the ball is?", "basket", "drawer"),
    ]
    
    results["chat_mode"] = {}
    print("\nUsing chat mode with reasoning:")
    for name, question, correct, wrong in chat_scenarios:
        result = get_chat_prediction(model, tokenizer, question, correct, wrong)
        results["chat_mode"][name] = result
        status = "OK" if result["correct"] else "WRONG"
        print(f"\n{name}: [{status}]")
        print(f"  Expected: {correct}")
        print(f"  Response: {result['response'][:100]}...")
    
    # ========================================
    # SUMMARY
    # ========================================
    print(f"\n{'='*60}")
    print("CRITICAL FINDINGS")
    print(f"{'='*60}")
    
    # Question types
    print("\nQuestion Type Accuracy:")
    for name, result in results["question_types"].items():
        status = "OK" if result["correct"] else "WRONG"
        print(f"  {name}: {status}")
    
    # AI entities
    ai_correct = sum(1 for r in results["ai_entities"].values() if r["correct"])
    print(f"\nAI Entities: {ai_correct}/{len(results['ai_entities'])} correct")
    
    # Confound tests
    print("\nFirst-Mention vs Original Location:")
    for name, result in results["confound_tests"].items():
        status = "OK" if result["correct"] else "WRONG"
        print(f"  {name}: {status}")
    
    # Chat mode
    print("\nChat Mode:")
    for name, result in results["chat_mode"].items():
        status = "OK" if result["correct"] else "WRONG"
        print(f"  {name}: {status}")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {"model": config.model_name},
        "results": results,
    }
    
    output_path = RESULTS_DIR / "step31_baselines.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    print(f"\n{'='*60}")
    print("STEP 31 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

