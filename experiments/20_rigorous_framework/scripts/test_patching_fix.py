"""
Quick test to verify the KV cache fix works.
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
    print("Testing patching with KV cache disabled...\n")
    
    config = ExperimentConfig()
    model, tokenizer = load_model_for_chat(config.model_name)
    
    patcher = ActivationPatcher(
        model, tokenizer,
        max_new_tokens=200,
        chat_mode=True,
    )
    
    # Simple test case
    FB = "Sally put the ball in the basket. Sally left. Anne moved it to the box. Where will Sally look? Answer: basket or box."
    TB = "Sally put the ball in the basket. Sally watched. Anne moved it to the box. Where will Sally look? Answer: basket or box."
    
    print("Getting baselines...")
    fb_base = patcher.generate_baseline(FB)
    tb_base = patcher.generate_baseline(TB)
    
    fb_base_ans = extract_answer(fb_base)
    tb_base_ans = extract_answer(tb_base)
    
    print(f"FB baseline: {fb_base_ans} (should be 'basket')")
    print(f"TB baseline: {tb_base_ans} (should be 'box')")
    print(f"FB response preview: {fb_base[:150]}...")
    print(f"TB response preview: {tb_base[:150]}...")
    
    if not (fb_base_ans == "basket" and tb_base_ans == "box"):
        print("\n⚠️  Baselines not correct, but continuing test...")
    
    print("\nTesting patching FB→TB (should flip to 'box')...")
    tb_acts = patcher.cache_activations(TB, layers=[20])
    patched = patcher.patch_and_generate(FB, tb_acts, [20], patch_mode="prompt_end")
    patched_ans = extract_answer(patched)
    
    print(f"Patched answer: {patched_ans}")
    print(f"Patched response preview: {patched[:200]}...")
    
    if patched_ans == "box" and fb_base_ans == "basket":
        print("\n✓ SUCCESS! Patching flipped behavior!")
    elif "ある" in patched or "あ" in patched[:50]:
        print("\n✗ Still producing gibberish (Japanese characters)")
    else:
        print(f"\n? Patching didn't flip (base={fb_base_ans}, patched={patched_ans})")
    
    print("\nTest complete!")


if __name__ == "__main__":
    main()









