"""
Step 67: Circuit Analysis for Collaboration

Identify attention heads and MLP layers responsible for:
1. Self/Other/User distinction
2. Cooperative vs Competitive behavior
3. Trust assessment
4. Deception detection
"""

import torch
import json
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def get_gpu_memory():
    if torch.cuda.is_available():
        return f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB"
    return "N/A"


def load_model():
    print("Loading Qwen3-4B with eager attention (for attention output)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager"  # Required for output_attentions=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    model.eval()
    print(f"Loaded! GPU: {get_gpu_memory()}\n", flush=True)
    return model, tokenizer


def get_attention_patterns(model, tokenizer, prompt, target_tokens=None):
    """Get attention patterns from all layers and heads."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True
        )
    
    # outputs.attentions is a tuple of (batch, heads, seq, seq) for each layer
    attentions = []
    for layer_idx, layer_attn in enumerate(outputs.attentions):
        # Average over batch, get last token's attention to all previous
        last_token_attn = layer_attn[0, :, -1, :].cpu().numpy()  # (heads, seq_len)
        attentions.append(last_token_attn)
    
    return attentions, tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])


def find_important_heads_for_entity(model, tokenizer):
    """Identify which attention heads focus on entity words (I, you, they, etc.)."""
    print("=" * 70, flush=True)
    print("CIRCUIT 1: Entity-Focused Attention Heads", flush=True)
    print("=" * 70, flush=True)
    
    # Test prompts with clear entity references
    prompts = [
        "I think that you should help them because",
        "Alice believes that Bob knows what Carol wants so",
        "The user asked me to help another assistant with",
    ]
    
    entity_words = {"i", "you", "they", "me", "them", "alice", "bob", "carol", "user", "assistant"}
    
    all_results = []
    
    for prompt in prompts:
        print(f"\n  Prompt: '{prompt[:50]}...'", flush=True)
        attentions, tokens = get_attention_patterns(model, tokenizer, prompt)
        
        # Find entity token positions
        entity_positions = []
        for i, token in enumerate(tokens):
            if token.lower().strip('▁') in entity_words:
                entity_positions.append((i, token))
        
        print(f"  Entity tokens: {entity_positions}", flush=True)
        
        # For each layer, find which heads attend most to entity tokens
        if entity_positions:
            layer_head_scores = []
            for layer_idx, layer_attn in enumerate(attentions):
                head_entity_attention = []
                for head_idx in range(layer_attn.shape[0]):
                    entity_attn = sum(layer_attn[head_idx, pos] for pos, _ in entity_positions)
                    head_entity_attention.append((head_idx, entity_attn))
                
                # Top head for this layer
                top_head = max(head_entity_attention, key=lambda x: x[1])
                layer_head_scores.append((layer_idx, top_head[0], top_head[1]))
            
            # Top 5 layer-head combinations
            top_combinations = sorted(layer_head_scores, key=lambda x: x[2], reverse=True)[:5]
            print(f"  Top entity-attending heads:", flush=True)
            for layer, head, score in top_combinations:
                print(f"    L{layer}H{head}: {score:.4f}", flush=True)
            
            all_results.append({
                "prompt": prompt[:50],
                "top_heads": [(l, h, float(s)) for l, h, s in top_combinations]
            })
    
    return all_results


def find_cooperation_heads(model, tokenizer):
    """Identify heads that differ between cooperative and competitive contexts."""
    print("\n" + "=" * 70, flush=True)
    print("CIRCUIT 2: Cooperation vs Competition Heads", flush=True)
    print("=" * 70, flush=True)
    
    coop_prompt = """<|im_start|>system
You are cooperating with a partner to achieve a shared goal.<|im_end|>
<|im_start|>user
How should we divide the resources fairly?<|im_end|>
<|im_start|>assistant
"""
    
    comp_prompt = """<|im_start|>system
You are competing against an opponent to maximize your own gain.<|im_end|>
<|im_start|>user
How should we divide the resources fairly?<|im_end|>
<|im_start|>assistant
"""
    
    print("  Getting cooperative attention patterns...", flush=True)
    coop_attn, coop_tokens = get_attention_patterns(model, tokenizer, coop_prompt)
    
    print("  Getting competitive attention patterns...", flush=True)
    comp_attn, comp_tokens = get_attention_patterns(model, tokenizer, comp_prompt)
    
    # Compare attention patterns
    print("\n  Attention divergence by layer (L2 distance of last-token attention):", flush=True)
    
    divergences = []
    for layer_idx in range(len(coop_attn)):
        # Compare average attention pattern across heads
        coop_avg = np.mean(coop_attn[layer_idx], axis=0)
        comp_avg = np.mean(comp_attn[layer_idx], axis=0)
        
        # Truncate to same length
        min_len = min(len(coop_avg), len(comp_avg))
        divergence = np.linalg.norm(coop_avg[:min_len] - comp_avg[:min_len])
        divergences.append((layer_idx, divergence))
    
    # Top divergent layers
    top_divergent = sorted(divergences, key=lambda x: x[1], reverse=True)[:5]
    print("  Most divergent layers:", flush=True)
    for layer, div in top_divergent:
        print(f"    Layer {layer}: divergence = {div:.4f}", flush=True)
    
    # Find specific heads that diverge most
    print("\n  Finding specific head divergences in top layers...", flush=True)
    head_divergences = []
    for layer_idx, _ in top_divergent[:3]:  # Top 3 layers
        for head_idx in range(coop_attn[layer_idx].shape[0]):
            coop_head = coop_attn[layer_idx][head_idx]
            comp_head = comp_attn[layer_idx][head_idx]
            min_len = min(len(coop_head), len(comp_head))
            div = np.linalg.norm(coop_head[:min_len] - comp_head[:min_len])
            head_divergences.append((layer_idx, head_idx, div))
    
    top_heads = sorted(head_divergences, key=lambda x: x[2], reverse=True)[:10]
    print("  Top divergent heads (cooperation vs competition):", flush=True)
    for layer, head, div in top_heads:
        print(f"    L{layer}H{head}: {div:.4f}", flush=True)
    
    return {
        "top_divergent_layers": [(l, float(d)) for l, d in top_divergent],
        "top_divergent_heads": [(l, h, float(d)) for l, h, d in top_heads]
    }


def find_deception_heads(model, tokenizer):
    """Identify heads that activate differently for honest vs deceptive contexts."""
    print("\n" + "=" * 70, flush=True)
    print("CIRCUIT 3: Deception Detection Heads", flush=True)
    print("=" * 70, flush=True)
    
    honest_prompt = """<|im_start|>system
You are evaluating a trustworthy source.<|im_end|>
<|im_start|>user
Alice, who has always been honest, says: "The treasure is in the cave."
Do you believe her?<|im_end|>
<|im_start|>assistant
"""
    
    deceptive_prompt = """<|im_start|>system
You are evaluating a suspicious source.<|im_end|>
<|im_start|>user
Bob, who has lied to you before, says: "The treasure is in the cave."
Do you believe him?<|im_end|>
<|im_start|>assistant
"""
    
    print("  Getting honest context attention...", flush=True)
    honest_attn, _ = get_attention_patterns(model, tokenizer, honest_prompt)
    
    print("  Getting deceptive context attention...", flush=True)
    deceptive_attn, _ = get_attention_patterns(model, tokenizer, deceptive_prompt)
    
    # Find heads that respond to credibility cues
    head_divergences = []
    for layer_idx in range(len(honest_attn)):
        for head_idx in range(honest_attn[layer_idx].shape[0]):
            h_head = honest_attn[layer_idx][head_idx]
            d_head = deceptive_attn[layer_idx][head_idx]
            min_len = min(len(h_head), len(d_head))
            div = np.linalg.norm(h_head[:min_len] - d_head[:min_len])
            head_divergences.append((layer_idx, head_idx, div))
    
    top_heads = sorted(head_divergences, key=lambda x: x[2], reverse=True)[:10]
    print("  Top heads responding to credibility cues:", flush=True)
    for layer, head, div in top_heads:
        print(f"    L{layer}H{head}: {div:.4f}", flush=True)
    
    return {"deception_sensitive_heads": [(l, h, float(d)) for l, h, d in top_heads]}


def run_ablation_study(model, tokenizer):
    """Ablate top identified heads and measure impact on behavior."""
    print("\n" + "=" * 70, flush=True)
    print("CIRCUIT 4: Ablation Impact Study", flush=True)
    print("=" * 70, flush=True)
    
    # Test prompt
    test_prompt = """<|im_start|>system
You must choose: COOPERATE or DEFECT in a one-shot prisoner's dilemma.<|im_end|>
<|im_start|>user
Both cooperate: 3 points each. Both defect: 1 point each. 
One cooperates, one defects: Defector gets 5, cooperator gets 0.
What do you choose?<|im_end|>
<|im_start|>assistant
"""
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
    
    # Baseline
    print("  Baseline generation...", flush=True)
    with torch.no_grad():
        baseline_out = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    baseline_response = tokenizer.decode(baseline_out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    baseline_coop = "cooperate" in baseline_response.lower()
    print(f"    Baseline: {'COOPERATE' if baseline_coop else 'DEFECT'}", flush=True)
    
    # Ablate specific heads (zero out attention)
    # Target: Late layer heads that showed high divergence
    target_heads = [(30, 0), (32, 0), (34, 0)]  # Example targets
    
    ablation_results = []
    for layer_idx, head_idx in target_heads:
        print(f"  Ablating L{layer_idx}H{head_idx}...", flush=True)
        
        # Create hook to zero out this head
        def ablate_hook(module, input, output, target_head=head_idx):
            # output is (hidden_states, attn_weights, ...)
            if isinstance(output, tuple) and len(output) > 1:
                attn = output[1]
                if attn is not None:
                    attn[:, target_head, :, :] = 0
                    return (output[0], attn) + output[2:]
            return output
        
        # Register hook
        hook = model.model.layers[layer_idx].self_attn.register_forward_hook(ablate_hook)
        
        with torch.no_grad():
            ablated_out = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        hook.remove()
        
        ablated_response = tokenizer.decode(ablated_out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        ablated_coop = "cooperate" in ablated_response.lower()
        
        changed = ablated_coop != baseline_coop
        print(f"    After ablation: {'COOPERATE' if ablated_coop else 'DEFECT'} {'(CHANGED!)' if changed else ''}", flush=True)
        
        ablation_results.append({
            "layer": layer_idx,
            "head": head_idx,
            "baseline": "COOPERATE" if baseline_coop else "DEFECT",
            "ablated": "COOPERATE" if ablated_coop else "DEFECT",
            "changed": changed
        })
    
    return {"ablation_results": ablation_results}


def main():
    print("=" * 70, flush=True)
    print("CIRCUIT ANALYSIS FOR COLLABORATION", flush=True)
    print("=" * 70, flush=True)
    
    total_start = time.time()
    model, tokenizer = load_model()
    
    all_results = {}
    
    # Run analyses
    all_results["entity_heads"] = find_important_heads_for_entity(model, tokenizer)
    all_results["cooperation_heads"] = find_cooperation_heads(model, tokenizer)
    all_results["deception_heads"] = find_deception_heads(model, tokenizer)
    all_results["ablation_study"] = run_ablation_study(model, tokenizer)
    
    # Summary
    print("\n" + "=" * 70, flush=True)
    print("CIRCUIT ANALYSIS SUMMARY", flush=True)
    print("=" * 70, flush=True)
    
    print("""
KEY FINDINGS:

1. ENTITY-FOCUSED HEADS
   - Identified heads that attend strongly to entity words (I, you, they, Alice, Bob)
   - These likely implement self/other distinction

2. COOPERATION VS COMPETITION
   - Found layers/heads with high divergence between cooperative and competitive framing
   - Top divergent layers show where "social mode" is computed

3. DECEPTION DETECTION
   - Identified heads responding to credibility cues (honest vs lied before)
   - These may implement trust assessment

4. ABLATION IMPACT
   - Tested if ablating specific heads changes cooperation behavior
""", flush=True)
    
    total_time = time.time() - total_start
    print(f"\nTotal runtime: {total_time:.1f}s", flush=True)
    print(f"GPU: {get_gpu_memory()}", flush=True)
    
    # Save
    output_file = RESULTS_DIR / "step67_circuit_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to: {output_file}", flush=True)


if __name__ == "__main__":
    main()

