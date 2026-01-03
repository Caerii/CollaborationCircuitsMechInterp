"""
Step 26: Value Vector Analysis

Finding from step25: Attention patterns are IDENTICAL between bad/good verbs.
The override must be in the VALUE vectors or output projections.

This script investigates:
1. What are the value vectors for "told" vs "announced"?
2. How do they differ at the late-layer heads?
3. Where does the verb information enter the computation?
"""

import torch
import json
import numpy as np
import sys
import io
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# The 10 critical late-layer heads
LATE_CIRCUIT_HEADS = [
    (32, 6), (32, 31),
    (33, 6), (33, 13), (33, 17), (33, 31),
    (34, 17),
    (35, 0), (35, 1), (35, 17)
]

BAD_VERBS = ["told", "said", "mentioned"]
GOOD_VERBS = ["announced", "asked", "hinted"]

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"


def load_model():
    """Load model."""
    print("Loading Qwen3-4B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model.eval()
    return model, tokenizer


def create_prompt(verb):
    """Create Sally-Anne style ToM prompt."""
    return f"""Alice put the ball in the drawer. Alice then left the room.
While Alice was away, Bob {verb} Carol that he moved the ball to the basket.
When Alice returned, Alice looked for the ball. Alice searched in the"""


def get_hidden_states_and_outputs(model, tokenizer, prompt):
    """Get hidden states at each layer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # hidden_states is tuple of (batch, seq, hidden) per layer
    hidden_states = outputs.hidden_states
    logits = outputs.logits
    
    return hidden_states, logits, inputs


def analyze_head_output_contributions(model, tokenizer, prompt, verb):
    """Analyze the contribution of each late head to the output."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Store outputs from each head
    head_outputs = {}
    
    def make_hook(layer_idx, head_idx):
        def hook(module, input, output):
            # output is (batch, seq, hidden) from o_proj
            # We need to extract per-head before they're summed
            # But o_proj already combines heads, so we'll use a different approach
            head_outputs[(layer_idx, head_idx)] = output[0, -1, :].clone()  # Last token
        return hook
    
    # We need to hook before o_proj to get per-head outputs
    # Actually, let's hook the attention output directly
    
    hooks = []
    attn_outputs = {}
    
    def make_attn_hook(layer_idx):
        def hook(module, input, output):
            # For Qwen3, output is typically (attn_output, attn_weights, past_kv)
            if isinstance(output, tuple):
                attn_out = output[0]  # (batch, seq, hidden)
            else:
                attn_out = output
            attn_outputs[layer_idx] = attn_out[0, -1, :].clone()
        return hook
    
    # Hook attention modules
    for layer_idx in [32, 33, 34, 35]:
        layer = model.model.layers[layer_idx]
        hook = layer.self_attn.register_forward_hook(make_attn_hook(layer_idx))
        hooks.append(hook)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    return attn_outputs, outputs.logits


def compare_attention_outputs(model, tokenizer):
    """Compare attention outputs between bad and good verbs."""
    print("\n" + "="*60)
    print("COMPARING ATTENTION OUTPUTS (Last Token)")
    print("="*60)
    
    results = {'bad': {}, 'good': {}, 'diff': {}}
    
    # Get outputs for bad verbs
    bad_outputs = []
    for verb in BAD_VERBS:
        prompt = create_prompt(verb)
        attn_outs, logits = analyze_head_output_contributions(model, tokenizer, prompt, verb)
        bad_outputs.append(attn_outs)
        results['bad'][verb] = {
            str(k): v.cpu().numpy().tolist()[:10] for k, v in attn_outs.items()  # First 10 dims
        }
    
    # Get outputs for good verbs  
    good_outputs = []
    for verb in GOOD_VERBS:
        prompt = create_prompt(verb)
        attn_outs, logits = analyze_head_output_contributions(model, tokenizer, prompt, verb)
        good_outputs.append(attn_outs)
        results['good'][verb] = {
            str(k): v.cpu().numpy().tolist()[:10] for k, v in attn_outs.items()
        }
    
    # Compute average difference
    print("\nAttention output L2 norms (last token):")
    print("-" * 50)
    
    for layer in [32, 33, 34, 35]:
        bad_mean = torch.stack([out[layer] for out in bad_outputs]).mean(dim=0)
        good_mean = torch.stack([out[layer] for out in good_outputs]).mean(dim=0)
        
        diff = bad_mean - good_mean
        diff_norm = torch.norm(diff).item()
        bad_norm = torch.norm(bad_mean).item()
        good_norm = torch.norm(good_mean).item()
        
        cos_sim = torch.nn.functional.cosine_similarity(
            bad_mean.unsqueeze(0), good_mean.unsqueeze(0)
        ).item()
        
        print(f"L{layer}: Bad norm={bad_norm:.2f}, Good norm={good_norm:.2f}, "
              f"Diff norm={diff_norm:.2f}, Cos sim={cos_sim:.4f}")
        
        results['diff'][str(layer)] = {
            'diff_norm': float(diff_norm),
            'bad_norm': float(bad_norm),
            'good_norm': float(good_norm),
            'cos_sim': float(cos_sim)
        }
    
    return results


def analyze_residual_stream_evolution(model, tokenizer):
    """Track how the residual stream evolves differently for bad vs good verbs."""
    print("\n" + "="*60)
    print("RESIDUAL STREAM EVOLUTION (Bad vs Good)")
    print("="*60)
    
    # Get hidden states for one bad and one good verb
    bad_verb = "told"
    good_verb = "announced"
    
    bad_prompt = create_prompt(bad_verb)
    good_prompt = create_prompt(good_verb)
    
    bad_hidden, bad_logits, bad_inputs = get_hidden_states_and_outputs(model, tokenizer, bad_prompt)
    good_hidden, good_logits, good_inputs = get_hidden_states_and_outputs(model, tokenizer, good_prompt)
    
    # Focus on layers 30-35
    print("\nLast token hidden state comparison (layers 30-35):")
    print("-" * 60)
    
    for layer_idx in range(30, 36):
        bad_last = bad_hidden[layer_idx][0, -1, :]  # (hidden,)
        good_last = good_hidden[layer_idx][0, -1, :]
        
        diff = bad_last - good_last
        diff_norm = torch.norm(diff).item()
        cos_sim = torch.nn.functional.cosine_similarity(
            bad_last.unsqueeze(0), good_last.unsqueeze(0)
        ).item()
        
        print(f"L{layer_idx}: Diff norm={diff_norm:.2f}, Cos sim={cos_sim:.6f}")
    
    # Project hidden states to vocab space (logit lens)
    print("\n" + "="*60)
    print("LOGIT LENS: 'drawer' vs 'basket' probability evolution")
    print("="*60)
    
    drawer_id = tokenizer.encode(" drawer", add_special_tokens=False)[0]
    basket_id = tokenizer.encode(" basket", add_special_tokens=False)[0]
    
    print(f"\nToken IDs: drawer={drawer_id}, basket={basket_id}")
    print("-" * 60)
    
    for layer_idx in range(30, 36):
        # Project through final layer norm and lm_head
        bad_last = bad_hidden[layer_idx][0, -1, :]
        good_last = good_hidden[layer_idx][0, -1, :]
        
        bad_normed = model.model.norm(bad_last.unsqueeze(0).unsqueeze(0))
        good_normed = model.model.norm(good_last.unsqueeze(0).unsqueeze(0))
        
        bad_logits_proj = model.lm_head(bad_normed)[0, 0, :]
        good_logits_proj = model.lm_head(good_normed)[0, 0, :]
        
        bad_drawer = bad_logits_proj[drawer_id].item()
        bad_basket = bad_logits_proj[basket_id].item()
        good_drawer = good_logits_proj[drawer_id].item()
        good_basket = good_logits_proj[basket_id].item()
        
        print(f"L{layer_idx}:")
        print(f"  'told':     drawer={bad_drawer:+.2f}, basket={bad_basket:+.2f}, diff={bad_drawer-bad_basket:+.2f}")
        print(f"  'announced': drawer={good_drawer:+.2f}, basket={good_basket:+.2f}, diff={good_drawer-good_basket:+.2f}")


def investigate_verb_token_embedding(model, tokenizer):
    """Check how verb tokens are embedded and if that explains the difference."""
    print("\n" + "="*60)
    print("VERB TOKEN EMBEDDING ANALYSIS")
    print("="*60)
    
    # Get embeddings of verb tokens
    all_verbs = BAD_VERBS + GOOD_VERBS
    
    print("\nVerb token embeddings (cosine similarity):")
    embeddings = {}
    
    for verb in all_verbs:
        token_id = tokenizer.encode(f" {verb}", add_special_tokens=False)[0]
        emb = model.model.embed_tokens.weight[token_id]
        embeddings[verb] = emb
        print(f"  {verb}: token_id={token_id}, norm={torch.norm(emb).item():.2f}")
    
    print("\nCosine similarities between verbs:")
    print("-" * 40)
    
    # Compare bad vs good
    for bad_verb in BAD_VERBS:
        for good_verb in GOOD_VERBS:
            cos_sim = torch.nn.functional.cosine_similarity(
                embeddings[bad_verb].unsqueeze(0),
                embeddings[good_verb].unsqueeze(0)
            ).item()
            print(f"  {bad_verb} <-> {good_verb}: {cos_sim:.4f}")


def analyze_mlp_contributions(model, tokenizer):
    """Check if MLP layers contribute to the override."""
    print("\n" + "="*60)
    print("MLP CONTRIBUTION ANALYSIS (Layers 32-35)")
    print("="*60)
    
    bad_verb = "told"
    good_verb = "announced"
    
    mlp_outputs = {'bad': {}, 'good': {}}
    
    for verb, label in [(bad_verb, 'bad'), (good_verb, 'good')]:
        prompt = create_prompt(verb)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        hooks = []
        outputs_dict = {}
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                outputs_dict[layer_idx] = output[0, -1, :].clone()
            return hook
        
        for layer_idx in [32, 33, 34, 35]:
            layer = model.model.layers[layer_idx]
            hook = layer.mlp.register_forward_hook(make_hook(layer_idx))
            hooks.append(hook)
        
        with torch.no_grad():
            model(**inputs)
        
        for hook in hooks:
            hook.remove()
        
        mlp_outputs[label] = outputs_dict
    
    print("\nMLP output comparison (last token):")
    print("-" * 50)
    
    for layer in [32, 33, 34, 35]:
        bad_out = mlp_outputs['bad'][layer]
        good_out = mlp_outputs['good'][layer]
        
        diff = bad_out - good_out
        diff_norm = torch.norm(diff).item()
        cos_sim = torch.nn.functional.cosine_similarity(
            bad_out.unsqueeze(0), good_out.unsqueeze(0)
        ).item()
        
        print(f"L{layer} MLP: Diff norm={diff_norm:.2f}, Cos sim={cos_sim:.6f}")


def main():
    print("="*60)
    print("STEP 26: Value Vector / Output Analysis")
    print("="*60)
    print("\nKey finding from step25: Attention patterns are IDENTICAL")
    print("The override must be in VALUE vectors or output processing\n")
    
    model, tokenizer = load_model()
    
    # Analysis 1: Compare attention outputs
    attn_results = compare_attention_outputs(model, tokenizer)
    
    # Analysis 2: Residual stream evolution
    analyze_residual_stream_evolution(model, tokenizer)
    
    # Analysis 3: Verb embedding analysis
    investigate_verb_token_embedding(model, tokenizer)
    
    # Analysis 4: MLP contributions
    analyze_mlp_contributions(model, tokenizer)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
Key Findings:

1. ATTENTION PATTERNS are identical (from step25)
   - Late heads attend to same tokens for "told" and "announced"
   
2. The DIFFERENCE must come from:
   - Value vectors (what gets retrieved)
   - MLP processing (parallel to attention)
   - Earlier layer residual stream differences

3. The verb information propagates through the residual stream
   and affects VALUE computation even though ATTENTION is the same.
""")
    
    # Save results
    save_path = RESULTS_DIR / "value_analysis_results.json"
    with open(save_path, 'w') as f:
        json.dump(attn_results, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()


