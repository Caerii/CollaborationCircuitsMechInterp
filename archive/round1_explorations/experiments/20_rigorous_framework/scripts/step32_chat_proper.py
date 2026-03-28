"""
Step 32: Proper Chat Mode Test

Step 31 showed chat mode "failed" but responses were truncated!
Let's test with proper token length using the library.

OUTPUT: results/step32_chat.json
"""

import sys
import json
import torch
from pathlib import Path
from datetime import datetime

# Add framework to path
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig
from core.chat_runner import ChatExperimentRunner

RESULTS_DIR = FRAMEWORK_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 70)
    print("STEP 32: PROPER CHAT MODE TEST (USING LIBRARY)")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\nStep 31 used max_new_tokens=100 which truncated responses!")
    print("Testing with proper token budget using ChatExperimentRunner")
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
    
    # Use library!
    runner = ChatExperimentRunner(model, tokenizer, config)
    
    # Create test scenarios
    scenarios = [
        {
            "story": "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket.",
            "question": "Where does Alice think the ball is?",
            "options": ["drawer", "basket"],
            "correct": "drawer",
            "type": "false_belief"
        },
        {
            "story": "Alice put the ball in the drawer. Alice watched Bob move the ball to the basket.",
            "question": "Where does Alice think the ball is?",
            "options": ["drawer", "basket"],
            "correct": "basket",
            "type": "true_belief"
        },
    ]
    
    results = {}
    
    for scenario in scenarios:
        name = scenario["type"].upper()
        print(f"\n{'='*60}")
        print(f"TESTING: {name}")
        print(f"Expected: {scenario['correct']}")
        print(f"{'='*60}")
        sys.stdout.flush()
        
        # Use library to run scenario
        result = runner.run_scenario(scenario, max_tokens=500)
        
        print(f"\nFull response:\n{result.raw_response}")
        print(f"\nParsed answer: {result.predicted_answer}")
        print(f"Result: {'CORRECT' if result.is_correct else 'WRONG'}")
        sys.stdout.flush()
        
        results[name] = {
            "correct": result.is_correct,
            "expected": scenario["correct"],
            "predicted": result.predicted_answer,
            "response": result.raw_response,
            "has_reasoning": result.parsed.has_think_tags,
            "confidence": result.parsed.confidence,
        }
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for name, r in results.items():
        status = "OK" if r["correct"] else "WRONG"
        print(f"{name}: [{status}] Expected: {r['expected']}, Got: {r['predicted']}")
    
    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": config.model_name,
            "max_tokens": 500,
            "using_library": True,
        },
        "results": results,
    }
    
    output_path = RESULTS_DIR / "step32_chat.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

