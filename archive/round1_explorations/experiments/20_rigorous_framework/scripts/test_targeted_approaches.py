"""
Test targeted patching and direct logit intervention approaches.

This tests what would actually work for understanding the distributed ToM circuit.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path

FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig
from core.chat_runner import load_model_for_chat
from analysis.direct_logit_intervention import DirectLogitIntervention


def extract_answer(response: str) -> str:
    """Extract answer from response."""
    text = response.lower()
    if "</think>" in text:
        # Get part after reasoning
        parts = text.split("</think>")
        if len(parts) > 1:
            text = parts[-1]
    
    if "basket" in text:
        return "basket"
    elif "box" in text:
        return "box"
    return None


def test_direct_logit_intervention():
    """Test direct logit intervention - simpler than activation patching."""
    print("=" * 70)
    print("TESTING: Direct Logit Intervention")
    print("=" * 70)
    print("\nThis approach directly manipulates logits at answer position.")
    print("Bypasses sequence length issues of activation patching.\n")
    
    config = ExperimentConfig()
    model, tokenizer = load_model_for_chat(config.model_name)
    
    intervener = DirectLogitIntervention(
        model, tokenizer,
        max_new_tokens=500,
        chat_mode=True,
    )
    
    # Test case: FB scenario
    fb_prompt = "Sally put the ball in the basket. Sally left. Anne moved it to the box. Where will Sally look? Answer: basket or box."
    
    print(f"Testing FB scenario:")
    print(f"  Prompt: {fb_prompt[:80]}...")
    
    # Get baseline
    print("\n  Getting baseline...")
    sys.stdout.flush()
    base_result = intervener.intervene(
        prompt=fb_prompt,
        answer_tokens=["basket", "box"],
        boost_token="basket",  # Should already be correct
        strength=0.0,  # No intervention
    )
    
    base_ans = extract_answer(base_result.base_response)
    print(f"  Baseline answer: {base_ans}")
    print(f"  Baseline response: {base_result.base_response[:150]}...")
    sys.stdout.flush()
    
    # Now try to flip to "box" by boosting it
    print("\n  Intervening to boost 'box' (strength=10.0)...")
    print("  (This may take a while - generating up to 500 tokens...)")
    sys.stdout.flush()
    intervention_result = intervener.intervene(
        prompt=fb_prompt,
        answer_tokens=["basket", "box"],
        boost_token="box",
        suppress_token="basket",
        strength=10.0,  # Stronger intervention
    )
    
    intervened_ans = extract_answer(intervention_result.intervened_response)
    print(f"  Intervened answer: {intervened_ans}")
    print(f"  Intervened response: {intervention_result.intervened_response[:200]}...")
    print(f"  Flipped: {intervention_result.flipped}")
    if intervention_result.intervened_logits:
        print(f"  Intervened logits: {intervention_result.intervened_logits}")
    
    if intervention_result.flipped:
        print("\n  ✓ SUCCESS! Direct logit intervention can flip the answer!")
    elif intervened_ans != base_ans and intervened_ans is not None:
        print("\n  ✓ SUCCESS! Answer changed (even if flip detection missed it)")
    else:
        print("\n  ? Intervention didn't flip (might need higher strength or different position)")
        print(f"     Base: '{base_ans}' -> Intervened: '{intervened_ans}'")
    
    return intervention_result


def main():
    print("\n" + "=" * 70)
    print("TESTING TARGETED APPROACHES FOR DISTRIBUTED CIRCUIT")
    print("=" * 70)
    print("\nKey insight: ToM is distributed, so we need:")
    print("  1. Patch at the RIGHT TIME (answer position, not reasoning)")
    print("  2. Potentially patch MULTIPLE layers simultaneously")
    print("  3. Or use simpler direct logit manipulation\n")
    
    # Test direct logit intervention (simpler)
    result = test_direct_logit_intervention()
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print("""
For understanding the distributed ToM circuit:

1. **Direct Logit Intervention** (simplest):
   - Directly manipulate logits at answer position
   - Bypasses sequence length issues
   - Can test if answer is "flippable" at that position
   - If it works, shows decision happens at that point

2. **Targeted Residual Stream Patching** (more causal):
   - Patch residual stream at answer position only
   - Patch multiple layers simultaneously (since it's distributed)
   - More causal than logit manipulation
   - But more complex to implement correctly

3. **Logit Lens During Generation** (diagnostic):
   - Track when answer probability diverges during reasoning
   - Find exact position where decision crystallizes
   - Use this to inform where to patch/intervene

The distributed nature means:
- Single-layer interventions won't work (confirmed by Step 35)
- Need multi-layer interventions
- Or need to understand the information flow through reasoning
    """)
    
    print("\nTest complete!")


if __name__ == "__main__":
    main()

