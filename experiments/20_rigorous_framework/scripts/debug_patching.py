"""
Debug script to understand what's happening during activation patching.

Goal: See exactly what activations are being cached, what's being patched,
and why it's producing gibberish.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
from pathlib import Path

FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig
from analysis.patching import ActivationPatcher
from core.chat_runner import load_model_for_chat


def debug_caching():
    """Debug what activations are being cached."""
    print("=" * 70)
    print("DEBUG 1: What activations are cached?")
    print("=" * 70)
    
    config = ExperimentConfig()
    model, tokenizer = load_model_for_chat(config.model_name)
    
    patcher = ActivationPatcher(
        model, tokenizer,
        max_new_tokens=50,
        chat_mode=True,
    )
    
    # Simple prompt
    prompt = "Sally put the ball in the basket. Where is it?"
    
    print(f"\nPrompt: {prompt}")
    print(f"Formatted (chat mode): {patcher._format_prompt(prompt)[:100]}...")
    
    # Cache activations
    print("\nCaching activations at L20...")
    cached = patcher.cache_activations(prompt, layers=[20])
    
    if 20 in cached:
        act = cached[20]
        print(f"  Shape: {act.shape}")
        print(f"  Dtype: {act.dtype}")
        print(f"  Device: {act.device}")
        print(f"  Mean: {act.mean().item():.4f}")
        print(f"  Std: {act.std().item():.4f}")
        print(f"  Min: {act.min().item():.4f}")
        print(f"  Max: {act.max().item():.4f}")
        
        # Check last token
        last_token = act[0, -1, :]
        print(f"\n  Last token activation:")
        print(f"    Shape: {last_token.shape}")
        print(f"    Mean: {last_token.mean().item():.4f}")
        print(f"    Top 5 values: {last_token.topk(5).values.tolist()}")
    else:
        print("  ERROR: No activation cached!")
    
    return cached


def debug_patching_sequence_lengths():
    """Debug sequence length mismatches during patching."""
    print("\n" + "=" * 70)
    print("DEBUG 2: Sequence lengths during patching")
    print("=" * 70)
    
    config = ExperimentConfig()
    model, tokenizer = load_model_for_chat(config.model_name)
    
    patcher = ActivationPatcher(
        model, tokenizer,
        max_new_tokens=10,  # Short for debugging
        chat_mode=True,
    )
    
    source_prompt = "Sally put the ball in the basket. Where is it?"
    target_prompt = "Sally put the ball in the box. Where is it?"
    
    # Cache source
    print(f"\nSource prompt: {source_prompt}")
    source_acts = patcher.cache_activations(source_prompt, layers=[20])
    source_len = source_acts[20].shape[1]
    print(f"Source activation length: {source_len}")
    
    # Check target prompt length
    formatted_target = patcher._format_prompt(target_prompt)
    target_inputs = tokenizer(formatted_target, return_tensors="pt")
    target_len = target_inputs.input_ids.shape[1]
    print(f"Target prompt length: {target_len}")
    
    # Now trace what happens during generation
    print("\nTracing generation with patching...")
    
    hook_calls = []
    
    def make_debug_hook(layer_idx, source_act):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            current_len = hidden.shape[1]
            
            hook_calls.append({
                "layer": layer_idx,
                "seq_len": current_len,
                "source_len": source_act.shape[1],
                "hidden_shape": tuple(hidden.shape),
                "source_shape": tuple(source_act.shape),
            })
            
            # Don't actually patch, just observe
            return output
        return hook
    
    h = model.model.layers[20].register_forward_hook(
        make_debug_hook(20, source_acts[20])
    )
    
    try:
        inputs = target_inputs.to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        h.remove()
    
    print(f"\nHook was called {len(hook_calls)} times")
    print("\nFirst 5 calls:")
    for i, call in enumerate(hook_calls[:5]):
        print(f"  Call {i+1}: seq_len={call['seq_len']}, source_len={call['source_len']}")
    
    print("\nLast 5 calls:")
    for i, call in enumerate(hook_calls[-5:]):
        print(f"  Call {i+1}: seq_len={call['seq_len']}, source_len={call['source_len']}")
    
    # Check if lengths ever match
    matching = [c for c in hook_calls if c['seq_len'] == c['source_len']]
    print(f"\nTimes when seq_len == source_len: {len(matching)}")
    if matching:
        print(f"  First match at call {hook_calls.index(matching[0]) + 1}")
        print(f"    seq_len={matching[0]['seq_len']}, target_prompt_len={target_len}")


def debug_actual_patching():
    """Debug what happens when we actually patch."""
    print("\n" + "=" * 70)
    print("DEBUG 3: What happens when we actually patch?")
    print("=" * 70)
    
    config = ExperimentConfig()
    model, tokenizer = load_model_for_chat(config.model_name)
    
    patcher = ActivationPatcher(
        model, tokenizer,
        max_new_tokens=20,
        chat_mode=True,
    )
    
    source_prompt = "Sally put the ball in the basket. Where is it?"
    target_prompt = "Sally put the ball in the box. Where is it?"
    
    print(f"\nSource: {source_prompt}")
    print(f"Target: {target_prompt}")
    
    # Get baseline
    print("\nGenerating baseline...")
    baseline = patcher.generate_baseline(target_prompt)
    print(f"Baseline response: {baseline[:200]}...")
    
    # Cache source
    print("\nCaching source activations...")
    source_acts = patcher.cache_activations(source_prompt, layers=[20])
    
    # Try different patch modes
    for mode in ["prompt_end", "last"]:
        print(f"\n--- Testing patch_mode='{mode}' ---")
        try:
            patched = patcher.patch_and_generate(
                target_prompt, source_acts, [20], patch_mode=mode
            )
            print(f"Patched response: {patched[:200]}...")
            
            # Check if it's gibberish
            if patched and len(patched) > 10:
                # Check for Japanese characters (common gibberish)
                has_japanese = any('\u3040' <= c <= '\u309F' for c in patched[:50])
                if has_japanese:
                    print("  ⚠️  Contains Japanese characters (gibberish)")
                else:
                    print("  ✓ Looks like normal text")
        except Exception as e:
            print(f"  ✗ Error: {e}")


def debug_token_positions():
    """Debug what tokens are at what positions."""
    print("\n" + "=" * 70)
    print("DEBUG 4: Token positions and what we're patching")
    print("=" * 70)
    
    config = ExperimentConfig()
    model, tokenizer = load_model_for_chat(config.model_name)
    
    patcher = ActivationPatcher(
        model, tokenizer,
        max_new_tokens=5,
        chat_mode=True,
    )
    
    prompt = "Sally put the ball in the basket. Where is it?"
    formatted = patcher._format_prompt(prompt)
    
    inputs = tokenizer(formatted, return_tensors="pt")
    tokens = inputs.input_ids[0]
    
    print(f"\nPrompt: {prompt}")
    print(f"Formatted length: {len(formatted)} chars")
    print(f"Token length: {len(tokens)} tokens")
    print(f"\nFirst 10 tokens:")
    for i in range(min(10, len(tokens))):
        token_id = tokens[i].item()
        token_str = tokenizer.decode([token_id])
        print(f"  [{i}] {token_id:5d} -> '{token_str}'")
    
    print(f"\nLast 10 tokens:")
    for i in range(max(0, len(tokens) - 10), len(tokens)):
        token_id = tokens[i].item()
        token_str = tokenizer.decode([token_id])
        print(f"  [{i}] {token_id:5d} -> '{token_str}'")
    
    print(f"\nLast token (position {len(tokens)-1}): '{tokenizer.decode([tokens[-1]])}'")


if __name__ == "__main__":
    print("Starting debugging session...\n")
    
    try:
        cached = debug_caching()
        debug_patching_sequence_lengths()
        debug_actual_patching()
        debug_token_positions()
    except Exception as e:
        print(f"\n✗ Error during debugging: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Debugging complete!")
    print("=" * 70)









