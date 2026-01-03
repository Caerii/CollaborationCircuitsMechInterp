"""
Test logit tracking intervention - the RIGHT way to do it.

This tracks logits DURING generation to find where the decision is made,
then intervenes at that moment (or multiple moments for distributed circuits).
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path

FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig
from core.chat_runner import load_model_for_chat
from analysis.logit_tracking_intervention import LogitTrackingIntervention


def extract_answer(response: str) -> str:
    """Extract answer from response."""
    text = response.lower()
    if "</think>" in text:
        parts = text.split("</think>")
        if len(parts) > 1:
            text = parts[-1]
    
    if "basket" in text:
        return "basket"
    elif "box" in text:
        return "box"
    return None


def main():
    print("=" * 70)
    print("TESTING: Logit Tracking Intervention")
    print("=" * 70)
    print("\nKey insight: Track logits DURING generation to find where")
    print("the decision is made, not where the token appears in text.\n")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    print("Loading model...", flush=True)
    sys.stdout.flush()
    model, tokenizer = load_model_for_chat(config.model_name)
    print("Model loaded!\n", flush=True)
    sys.stdout.flush()
    
    intervener = LogitTrackingIntervention(
        model, tokenizer,
        max_new_tokens=500,
        chat_mode=True,
    )
    
    # Test case: FB scenario
    fb_prompt = "Sally put the ball in the basket. Sally left. Anne moved it to the box. Where will Sally look? Answer: basket or box."
    
    print(f"Testing FB scenario:")
    print(f"  Prompt: {fb_prompt[:80]}...")
    print(f"  Expected: 'basket' (false belief)")
    print(f"  Trying to flip to: 'box'\n")
    sys.stdout.flush()
    
    # Track and intervene
    # KEY: Intervene EARLY when logits diverge, not later!
    result = intervener.track_and_intervene(
        prompt=fb_prompt,
        answer_tokens=["basket", "box"],
        boost_token="box",
        suppress_token="basket",
        strength=20.0,  # Very strong intervention (need to overcome early decision)
        intervention_threshold=0.05,  # Intervene when prob > 5%
        max_interventions=20  # Allow many interventions (catch decision early)
    )
    
    base_ans = extract_answer(result.base_response)
    intervened_ans = extract_answer(result.intervened_response)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Baseline answer: {base_ans}")
    print(f"Intervened answer: {intervened_ans}")
    print(f"Flipped: {result.flipped}")
    
    # Show actual responses to see what changed
    print(f"\nBaseline response length: {len(result.base_response)} chars")
    print(f"Intervened response length: {len(result.intervened_response)} chars")
    print(f"\nBaseline response (last 300 chars):")
    print(f"  {result.base_response[-300:]}")
    print(f"\nIntervened response (last 300 chars):")
    print(f"  {result.intervened_response[-300:]}")
    
    # Check if responses are complete
    baseline_complete = "</think>" in result.base_response or len(result.base_response) > 200
    intervened_complete = "</think>" in result.intervened_response or len(result.intervened_response) > 200
    print(f"\nBaseline complete: {baseline_complete}, Intervened complete: {intervened_complete}")
    
    print(f"\nIntervened at {len(result.intervention_positions)} positions:")
    for pos in result.intervention_positions[:10]:  # Show first 10
        print(f"  Step {pos}")
    if len(result.intervention_positions) > 10:
        print(f"  ... and {len(result.intervention_positions) - 10} more")
    
    if result.decision_point is not None:
        print(f"\nDecision point detected at step {result.decision_point}")
    
    print(f"\nLogit trajectories (first 20 steps):")
    for tok in ["basket", "box"]:
        if tok in result.logit_trajectories:
            traj = result.logit_trajectories[tok][:20]
            print(f"  {tok}: {[f'{x:.2f}' for x in traj]}")
    
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")
    
    # Check if responses are actually different
    responses_different = result.base_response != result.intervened_response
    
    if result.flipped:
        print("✓ SUCCESS! Intervention flipped the answer!")
        print("  This shows the decision CAN be flipped with early intervention.")
        if not responses_different:
            print("  (Note: Flip detection may be based on reasoning text, not final answer)")
    elif intervened_ans != base_ans and intervened_ans is not None:
        print("✓ Answer changed (even if flip detection missed it)")
    elif responses_different:
        print("? Intervention changed the response, but answer extraction still shows same")
        print("  This suggests intervention affected reasoning but not final answer token")
        print("  OR the answer is encoded differently than we're extracting")
    else:
        print("? Intervention didn't flip")
        print(f"  Base: '{base_ans}' -> Intervened: '{intervened_ans}'")
        print("\n  Possible reasons:")
        print("  1. Decision is locked in during prompt processing (before generation)")
        print("  2. Need even stronger intervention or more positions")
        print("  3. Answer token encoding doesn't match our extraction")
        print("  4. Need to intervene at residual stream level, not just logits")
    
    print(f"\n{'='*70}")
    print("IMPLICATIONS")
    print(f"{'='*70}")
    print("""
If intervention works:
  - Decision IS flippable at answer position
  - Can use this for causal analysis
  - Shows where in generation the decision happens

If intervention doesn't work:
  - Decision is locked in during reasoning phase
  - Need to intervene DURING reasoning, not at answer
  - Or decision is too distributed (need multi-layer + multi-position)
    """)
    
    print("\nTest complete!")


if __name__ == "__main__":
    main()

