"""
Test multi-layer residual stream patching with early intervention.

This is the RIGHT approach:
- Patch residual stream (not just logits)
- Multiple layers simultaneously (distributed circuit)
- Early intervention (steps 0-50) when decision forms
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path

FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig
from core.chat_runner import load_model_for_chat
from analysis.multi_layer_residual_patching import MultiLayerResidualPatcher


def main():
    print("=" * 70)
    print("TESTING: Multi-Layer Residual Stream Patching")
    print("=" * 70)
    print("\nKey features:")
    print("  - Patch residual stream (activation-level, not logits)")
    print("  - Multiple layers simultaneously (distributed circuit)")
    print("  - Early intervention (steps 0-50) when decision forms")
    print("  - Addresses: logit manipulation was insufficient\n")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    print("Loading model...", flush=True)
    sys.stdout.flush()
    model, tokenizer = load_model_for_chat(config.model_name)
    print(f"Model loaded! {model.config.num_hidden_layers} layers\n", flush=True)
    sys.stdout.flush()
    
    patcher = MultiLayerResidualPatcher(
        model, tokenizer,
        max_new_tokens=500,
        chat_mode=True,
    )
    
    # Test case: Patch TB → FB (should flip from "basket" to "box")
    tb_prompt = "Sally put the ball in the basket. Sally watched. Anne moved it to the box. Where will Sally look? Answer: basket or box."
    fb_prompt = "Sally put the ball in the basket. Sally left. Anne moved it to the box. Where will Sally look? Answer: basket or box."
    
    print(f"Source (TB): {tb_prompt[:80]}...")
    print(f"Target (FB): {fb_prompt[:80]}...")
    print(f"\nExpected: FB baseline = 'basket', after patching TB→FB = 'box'\n")
    sys.stdout.flush()
    
    # Test with multiple layer combinations
    layer_combinations = [
        [20, 24, 28, 32],  # Late layers
        [16, 20, 24, 28],  # Mid-late layers
        [12, 16, 20, 24],  # Mid layers
        [20, 24],  # Just two layers
    ]
    
    # Early steps to patch (when decision forms)
    early_steps = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    results = {}
    
    for i, layers in enumerate(layer_combinations):
        # Filter to valid layers
        layers = [l for l in layers if l < model.config.num_hidden_layers]
        if not layers:
            continue
        
        print(f"\n{'='*70}")
        print(f"TEST {i+1}/{len(layer_combinations)}: Layers {layers}")
        print(f"{'='*70}")
        sys.stdout.flush()
        
        try:
            result = patcher.run_patching_experiment(
                source_prompt=tb_prompt,
                target_prompt=fb_prompt,
                layers=layers,
                early_steps=early_steps,
            )
            
            results[f"L{layers[0]}-L{layers[-1]}"] = result
            
            print(f"\nResults:")
            print(f"  Baseline answer: {result.base_answer}")
            print(f"  Patched answer: {result.patched_answer}")
            print(f"  Flipped: {result.flipped}")
            print(f"  Patched at {len(result.patch_positions)} positions: {result.patch_positions[:10]}...")
            print(f"\n  Baseline response (last 200 chars):")
            print(f"    {result.base_response[-200:]}")
            print(f"\n  Patched response (last 200 chars):")
            print(f"    {result.patched_response[-200:]}")
            
            if result.flipped:
                print(f"\n  ✓ SUCCESS! Multi-layer patching flipped the answer!")
            elif result.patched_answer != result.base_answer:
                print(f"\n  ✓ Answer changed!")
            else:
                print(f"\n  ? Patching didn't flip (might need different layers or more positions)")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
        
        sys.stdout.flush()
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    successful = [name for name, r in results.items() if r.flipped]
    if successful:
        print(f"\n✓ Successful layer combinations: {successful}")
    else:
        print(f"\n? No layer combinations successfully flipped the answer")
        print("  Possible reasons:")
        print("  1. Need to patch at more positions")
        print("  2. Need different layers")
        print("  3. Decision is locked in during prompt processing")
        print("  4. Need to patch ALL layers (too distributed)")
    
    print("\nTest complete!")


if __name__ == "__main__":
    main()





