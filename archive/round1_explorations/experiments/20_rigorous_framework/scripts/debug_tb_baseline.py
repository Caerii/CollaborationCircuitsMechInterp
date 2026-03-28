"""
Debug why TB baseline is wrong.

The model should answer "box" for True Belief (Sally saw the move),
but it's answering "basket" instead.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path

FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig
from analysis.patching import ActivationPatcher
from core.chat_runner import load_model_for_chat


def extract_answer(response: str) -> str:
    """Extract answer from response."""
    text = response.lower()
    if "</think>" in text:
        text = text.split("</think>")[-1]
    
    if "basket" in text:
        return "basket"
    elif "box" in text:
        return "box"
    return None


def main():
    print("=" * 70)
    print("DEBUGGING TB BASELINE ISSUE")
    print("=" * 70)
    
    config = ExperimentConfig()
    model, tokenizer = load_model_for_chat(config.model_name)
    
    patcher = ActivationPatcher(
        model, tokenizer,
        max_new_tokens=500,
        chat_mode=True,
    )
    
    # Test different TB prompts
    prompts = [
        # Original (too subtle?)
        "Sally put the ball in the basket. Sally stayed and watched. Anne moved the ball to the box. Where will Sally look for the ball? Answer: basket or box.",
        
        # More explicit
        "Sally put the ball in the basket. Sally stayed in the room and watched everything. Anne moved the ball to the box. Sally saw this happen. Where will Sally look for the ball? Answer: basket or box.",
        
        # Even more explicit
        "Sally put the ball in the basket. Sally stayed in the room and watched. Anne moved the ball to the box. Sally saw Anne move the ball. Where will Sally look for the ball? Answer: basket or box.",
        
        # Direct question
        "Sally put the ball in the basket. Then Sally watched as Anne moved the ball to the box. Where does Sally think the ball is? Answer: basket or box.",
        
        # Very explicit
        "Sally put the ball in the basket. Sally was present and watching when Anne moved the ball to the box. Sally knows the ball is now in the box. Where will Sally look for the ball? Answer: basket or box.",
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {prompt[:80]}...")
        print(f"{'='*60}")
        
        response = patcher.generate_baseline(prompt)
        answer = extract_answer(response)
        
        print(f"Answer: {answer}")
        print(f"Expected: box")
        print(f"Status: {'✓ CORRECT' if answer == 'box' else '✗ WRONG'}")
        print(f"\nFull response:")
        print(response[:500])
        print("...")
        
        if answer == "box":
            print(f"\n✓ Found working prompt at test {i}!")
            break
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

