"""
Test activation extraction with a real model.
This verifies we can extract activations for probing.

NOTE: Your LM Studio model (GGUF format) is great for inference but
for mechanistic interpretability we need direct activation access.
We'll use the HuggingFace version of Qwen for this research.
"""
import torch
import gc
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_with_small_model():
    """
    Test activation extraction with a small model first.
    This uses GPT-2 small to verify the pipeline works.
    """
    print("\n=== Testing Activation Extraction (GPT-2 Small) ===")
    print("Using GPT-2 small to verify pipeline before loading larger model...")
    
    from nnsight import LanguageModel
    
    # Load small model for testing
    model = LanguageModel("gpt2", device_map="cuda")
    
    test_prompt = "User: Hello, how are you?\nAssistant: I'm doing well!"
    
    print(f"\nTest prompt: {test_prompt[:50]}...")
    
    # Extract activations using nnsight
    activations = {}
    
    with model.trace(test_prompt) as tracer:
        # GPT-2 structure: transformer.h[layer].output
        for layer_idx in [0, 5, 11]:  # First, middle, last
            hidden = model.transformer.h[layer_idx].output[0]
            activations[layer_idx] = hidden.save()
    
    print("\nExtracted activations:")
    for layer, act in activations.items():
        tensor = act.value
        print(f"  Layer {layer}: shape={tensor.shape}, dtype={tensor.dtype}")
    
    # Verify we can do operations on them
    layer_0 = activations[0].value.squeeze(0)  # [seq_len, hidden_dim]
    layer_11 = activations[11].value.squeeze(0)
    
    # Compute similarity between first and last layer representations
    cos_sim = torch.nn.functional.cosine_similarity(
        layer_0.mean(dim=0, keepdim=True),
        layer_11.mean(dim=0, keepdim=True)
    )
    print(f"\nCosine similarity (layer 0 vs 11 mean): {cos_sim.item():.4f}")
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n[OK] GPT-2 extraction test passed!")
    return True


def test_with_qwen():
    """
    Test activation extraction with Qwen model.
    Uses HuggingFace version for activation access.
    """
    print("\n=== Testing Activation Extraction (Qwen) ===")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    # Use Qwen 2.5 3B as it's similar to Qwen 3 4B but definitely available
    # We can switch to Qwen 3 when it's on HuggingFace
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    
    print(f"Loading {model_id}...")
    print("(This may take a minute on first run - downloading ~6GB)")
    
    # Check VRAM
    if torch.cuda.is_available():
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        print(f"Available VRAM: {free_mem / 1e9:.1f} GB")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with memory optimization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Use FP16 to fit in VRAM
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Model loaded! Layers: {model.config.num_hidden_layers}, Hidden: {model.config.hidden_size}")
    
    # Test prompt - multi-party dialogue
    test_prompt = """User: Can you help me understand machine learning?
You: Of course! Machine learning is a field of AI where systems learn from data.
Helper: I can add that there are three main types: supervised, unsupervised, and reinforcement learning.
User: What's the difference between them?"""
    
    print(f"\nTest prompt ({len(test_prompt)} chars)...")
    
    # Tokenize
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    print(f"Tokenized: {inputs['input_ids'].shape[1]} tokens")
    
    # Extract activations using hooks
    activations = {}
    
    def get_activation(layer_idx):
        def hook(module, input, output):
            # output is tuple, first element is hidden states
            activations[layer_idx] = output[0].detach().cpu()
        return hook
    
    # Register hooks for selected layers
    layers_to_probe = [0, 8, 16, 24, model.config.num_hidden_layers - 1]
    handles = []
    
    for layer_idx in layers_to_probe:
        if layer_idx < model.config.num_hidden_layers:
            handle = model.model.layers[layer_idx].register_forward_hook(get_activation(layer_idx))
            handles.append(handle)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    print("\nExtracted activations:")
    for layer, act in sorted(activations.items()):
        print(f"  Layer {layer}: shape={act.shape}")
    
    # Analyze: where does entity info live?
    # Get token positions for each speaker
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Find speaker tokens (simplified)
    print("\nToken analysis (first 20 tokens):")
    for i, tok in enumerate(tokens[:20]):
        print(f"  {i}: {tok}")
    
    # Compute per-layer statistics
    print("\nActivation statistics by layer:")
    for layer, act in sorted(activations.items()):
        act_squeezed = act.squeeze(0)  # [seq_len, hidden_dim]
        mean_norm = act_squeezed.norm(dim=1).mean()
        print(f"  Layer {layer}: mean L2 norm = {mean_norm:.2f}")
    
    # Clean up
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n[OK] Qwen extraction test passed!")
    print("\nYou're ready for the main experiment!")
    return True


def main():
    print("=" * 60)
    print("ACTIVATION EXTRACTION TEST")
    print("=" * 60)
    
    # Test 1: Small model
    try:
        test_with_small_model()
    except Exception as e:
        print(f"\n[FAIL] GPT-2 test failed: {e}")
        print("Try: uv pip install nnsight --upgrade")
        return
    
    # Test 2: Qwen model
    print("\n" + "-" * 60)
    response = input("\nProceed with Qwen test? This will download ~6GB. [y/N]: ")
    
    if response.lower() == 'y':
        try:
            test_with_qwen()
        except Exception as e:
            print(f"\n[FAIL] Qwen test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipped Qwen test. Run again when ready.")


if __name__ == "__main__":
    main()

