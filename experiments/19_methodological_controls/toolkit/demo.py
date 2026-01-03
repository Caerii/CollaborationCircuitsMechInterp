"""
ToM Toolkit Demo

Demonstrates how to use the toolkit to build and evaluate ToM prompts.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit import ToMPromptBuilder, ToMEvaluator, TEMPLATES


def demo_prompt_builder():
    """Demonstrate the prompt builder."""
    print("="*60)
    print("ToM PROMPT BUILDER DEMO")
    print("="*60)
    
    builder = ToMPromptBuilder()
    
    # List available templates
    print("\n1. Available Templates:")
    print("-"*40)
    for name, info in builder.list_templates().items():
        print(f"  {name}: {info['effectiveness']} - {info['note']}")
    
    # Create a false belief prompt
    print("\n2. Creating False Belief Prompt:")
    print("-"*40)
    
    prompt = builder.create_false_belief_prompt(
        agent="Sally",
        object="marble",
        original_location="basket",
        new_location="box",
        mover="Anne",
        template="action_search"
    )
    print(f"Template: action_search")
    print(f"Prompt:\n{prompt}")
    
    # Try the best template
    print("\n3. Best Template (action_remembers):")
    print("-"*40)
    
    best_prompt = builder.create_false_belief_prompt(
        agent="Sally",
        object="marble",
        original_location="basket",
        new_location="box",
        mover="Anne",
        template="action_remembers"
    )
    print(f"Prompt:\n{best_prompt}")
    
    # Verb recommendation
    print("\n4. Verb Recommendations:")
    print("-"*40)
    
    for verb in ["thinks", "believes", "searched", "looks", "expects"]:
        rec = builder.get_verb_recommendation(verb)
        status = "[OK]" if rec["recommended"] else "[AVOID]" if rec["recommended"] is False else "[?]"
        print(f"  {status} '{verb}': {rec['explanation']}")
    
    return builder


def demo_evaluator(model=None, tokenizer=None):
    """Demonstrate the evaluator (requires model)."""
    print("\n" + "="*60)
    print("ToM EVALUATOR DEMO")
    print("="*60)
    
    if model is None or tokenizer is None:
        print("\nNote: Model not provided. Loading Qwen3-4B...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-4B",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    
    evaluator = ToMEvaluator(model, tokenizer)
    builder = ToMPromptBuilder()
    
    # Evaluate a single prompt
    print("\n1. Single Prompt Evaluation:")
    print("-"*40)
    
    prompt = builder.create_false_belief_prompt(
        agent="Alice",
        object="ball",
        original_location="drawer",
        new_location="basket",
        mover="Bob"
    )
    
    result = evaluator.evaluate_false_belief(
        prompt=prompt,
        correct="drawer",
        incorrect="basket"
    )
    
    print(f"Prompt: ...{prompt[-50:]}")
    print(f"Result: {'CORRECT' if result['is_correct'] else 'INCORRECT'}")
    print(f"Confidence: {result['confidence']}")
    print(f"Logit diff: {result['logit_difference']:+.2f}")
    print(f"Top predictions: {result['top_5_predictions']}")
    
    # Compare templates
    print("\n2. Template Comparison:")
    print("-"*40)
    
    from toolkit.templates import RECOMMENDED_TEMPLATES
    
    comparison = evaluator.compare_templates(
        agent="Alice",
        object="ball",
        original_location="drawer",
        new_location="basket",
        mover="Bob",
        templates={k: v["template"] for k, v in RECOMMENDED_TEMPLATES.items()}
    )
    
    print(f"Best template: {comparison['best_template']}")
    print(f"Ranking: {comparison['ranking']}")
    
    # Diagnose a failing prompt
    print("\n3. Failure Diagnosis:")
    print("-"*40)
    
    bad_prompt = "Alice returns. Alice thinks the ball is in the"
    diagnosis = evaluator.diagnose_failure(
        prompt=bad_prompt,
        correct="drawer",
        incorrect="basket"
    )
    
    print(f"Prompt: {bad_prompt}")
    print(f"Issues found:")
    for issue in diagnosis["issues"]:
        print(f"  - {issue}")
    print(f"Suggestions:")
    for suggestion in diagnosis["suggestions"]:
        print(f"  - {suggestion}")
    
    return evaluator


def main():
    """Run the full demo."""
    print("="*60)
    print("THEORY OF MIND TOOLKIT DEMONSTRATION")
    print("="*60)
    
    # Demo prompt builder (no model needed)
    builder = demo_prompt_builder()
    
    # Demo evaluator (requires model)
    print("\n" + "-"*60)
    response = input("Run evaluator demo? (requires loading model) [y/N]: ")
    if response.lower() == 'y':
        demo_evaluator()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("""
To use the toolkit in your code:

    from toolkit import ToMPromptBuilder, ToMEvaluator

    # Build prompts
    builder = ToMPromptBuilder()
    prompt = builder.create_false_belief_prompt(
        agent="Sally",
        object="marble",
        original_location="basket",
        new_location="box",
        mover="Anne"
    )

    # Evaluate with model
    evaluator = ToMEvaluator(model, tokenizer)
    result = evaluator.evaluate_false_belief(prompt, "basket", "box")
""")


if __name__ == "__main__":
    main()


