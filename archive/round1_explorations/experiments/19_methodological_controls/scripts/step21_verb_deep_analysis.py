"""
Step 21: Deep Verb Analysis - What Makes Verbs Trigger Inhibition?

Key findings from step20:
- "told", "said", "mentioned", "clarified" = 0% (BAD)
- "announced", "noted", "queried", "communicated" = 100% (GOOD)

This doesn't match our earlier categories! Let's understand the PATTERN.
"""

import torch
import json
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict

# Configuration
MODEL_NAME = "Qwen/Qwen3-4B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The circuit
DECISION_HEADS = [(17, 4), (18, 11), (18, 14), (19, 30), (21, 17)]

# COMPREHENSIVE verb list - grouped by semantic category
VERBS_TO_TEST = {
    # DIRECT communication (face-to-face implication)
    "direct": ["told", "said", "mentioned", "stated", "explained", "informed", "notified"],
    
    # INDIRECT/FORMAL communication
    "formal": ["announced", "declared", "proclaimed", "reported", "communicated", "conveyed"],
    
    # QUESTIONING/BIDIRECTIONAL
    "questioning": ["asked", "queried", "inquired", "requested", "questioned"],
    
    # WRITING/RECORDING
    "written": ["wrote", "noted", "documented", "recorded", "emailed", "texted"],
    
    # SUBTLE/IMPLICIT
    "subtle": ["hinted", "implied", "suggested", "indicated", "signaled"],
    
    # LOUD/BROADCAST
    "broadcast": ["shouted", "yelled", "broadcast", "announced", "proclaimed"],
    
    # EMOTIONAL
    "emotional": ["confessed", "admitted", "revealed", "disclosed", "confided"],
    
    # PAST TENSE VARIANTS
    "past_simple": ["told", "said", "mentioned", "announced", "wrote", "asked"],
    
    # GERUND/PROGRESSIVE (different grammatical structure)
    "gerund_test": ["was telling", "had told", "has told", "will tell"]
}

# Output paths
OUTPUT_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def generate_scenario(verb: str) -> tuple[str, str, str]:
    """Generate a ToM scenario with the given verb."""
    # Handle multi-word verbs
    prompt = f"""Alice puts the ball in the drawer. Alice leaves.
Bob {verb} Carol that he moved the ball to the basket.
Alice returns. Alice will look for the ball in the"""
    
    return prompt, " drawer", " basket"


def load_model():
    """Load model with eager attention."""
    print(f"Loading {MODEL_NAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def test_verb(model, tokenizer, verb: str, n_trials: int = 10) -> dict:
    """Test a verb and return detailed results."""
    results = []
    
    for _ in range(n_trials):
        prompt, correct, wrong = generate_scenario(verb)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            logits = outputs.logits[0, -1]
        
        correct_id = tokenizer.encode(correct, add_special_tokens=False)[0]
        wrong_id = tokenizer.encode(wrong, add_special_tokens=False)[0]
        
        is_correct = logits[correct_id] > logits[wrong_id]
        
        # Get attention to key positions for decision heads
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        # Find verb position
        verb_pos = -1
        verb_first_word = verb.split()[0].lower()
        for i, tok in enumerate(tokens):
            tok_clean = tok.lower().replace("▁", "").replace("Ġ", "")
            if verb_first_word in tok_clean:
                verb_pos = i
                break
        
        # Get attention from last token to key positions for L18H11 (most important head)
        layer_attn = outputs.attentions[18]  # L18
        head_attn = layer_attn[0, 11, -1].cpu().float().numpy()  # L18H11, last token row
        
        # Find position indices
        alice_pos = [i for i, t in enumerate(tokens) if "alice" in t.lower().replace("▁", "")]
        basket_pos = [i for i, t in enumerate(tokens) if "basket" in t.lower().replace("▁", "")]
        drawer_pos = [i for i, t in enumerate(tokens) if "drawer" in t.lower().replace("▁", "")]
        
        results.append({
            "correct": bool(is_correct),
            "correct_logit": float(logits[correct_id]),
            "wrong_logit": float(logits[wrong_id]),
            "diff": float(logits[correct_id] - logits[wrong_id]),
            "attn_to_verb": float(head_attn[verb_pos]) if verb_pos >= 0 else 0.0,
            "attn_to_alice": float(sum(head_attn[i] for i in alice_pos)) if alice_pos else 0.0,
            "attn_to_basket": float(sum(head_attn[i] for i in basket_pos)) if basket_pos else 0.0,
            "attn_to_drawer": float(sum(head_attn[i] for i in drawer_pos)) if drawer_pos else 0.0,
        })
    
    accuracy = sum(r["correct"] for r in results) / len(results)
    
    return {
        "verb": verb,
        "accuracy": accuracy,
        "mean_diff": float(np.mean([r["diff"] for r in results])),
        "std_diff": float(np.std([r["diff"] for r in results])),
        "mean_attn_verb": float(np.mean([r["attn_to_verb"] for r in results])),
        "mean_attn_alice": float(np.mean([r["attn_to_alice"] for r in results])),
        "mean_attn_basket": float(np.mean([r["attn_to_basket"] for r in results])),
        "mean_attn_drawer": float(np.mean([r["attn_to_drawer"] for r in results])),
        "results": results
    }


def test_with_ablation(model, tokenizer, verb: str, n_trials: int = 10) -> dict:
    """Test with decision head ablation."""
    
    def ablation_hook(layer_idx, head_idx):
        def hook(module, input, output):
            # o_proj output is (batch, seq, hidden_size) directly, not tuple
            hidden = output if isinstance(output, torch.Tensor) else output[0]
            
            if hidden.dim() == 2:
                # Sometimes (seq, hidden_size) without batch
                seq, hidden_size = hidden.shape
                batch = 1
                hidden = hidden.unsqueeze(0)
            else:
                batch, seq, hidden_size = hidden.shape
            
            n_heads = model.config.num_attention_heads
            head_dim = hidden_size // n_heads
            
            hidden = hidden.view(batch, seq, n_heads, head_dim)
            hidden[:, :, head_idx, :] = 0
            hidden = hidden.view(batch, seq, hidden_size)
            
            if isinstance(output, torch.Tensor):
                return hidden.squeeze(0) if batch == 1 and output.dim() == 2 else hidden
            return (hidden,) + output[1:]
        return hook
    
    # Register hooks
    hooks = []
    for layer_idx, head_idx in DECISION_HEADS:
        layer = model.model.layers[layer_idx].self_attn.o_proj
        hook = layer.register_forward_hook(ablation_hook(layer_idx, head_idx))
        hooks.append(hook)
    
    results = []
    try:
        for _ in range(n_trials):
            prompt, correct, wrong = generate_scenario(verb)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1]
            
            correct_id = tokenizer.encode(correct, add_special_tokens=False)[0]
            wrong_id = tokenizer.encode(wrong, add_special_tokens=False)[0]
            
            is_correct = logits[correct_id] > logits[wrong_id]
            results.append({
                "correct": bool(is_correct),
                "diff": float(logits[correct_id] - logits[wrong_id])
            })
    finally:
        for hook in hooks:
            hook.remove()
    
    accuracy = sum(r["correct"] for r in results) / len(results)
    return {
        "accuracy_ablated": accuracy,
        "mean_diff_ablated": float(np.mean([r["diff"] for r in results]))
    }


def analyze_verb_properties(verb: str) -> dict:
    """Analyze linguistic properties of a verb."""
    # Simple heuristics
    properties = {
        "is_past_tense": verb.endswith("ed") or verb in ["told", "said", "wrote", "went"],
        "syllables": len([c for c in verb if c in "aeiou"]),  # Rough estimate
        "length": len(verb),
        "has_direct_object_implication": verb in ["told", "informed", "notified", "asked"],
        "implies_face_to_face": verb in ["told", "said", "mentioned", "whispered", "shouted"],
        "implies_recording": verb in ["wrote", "noted", "documented", "recorded", "emailed"],
        "implies_public": verb in ["announced", "broadcast", "proclaimed", "declared"],
        "implies_question": verb in ["asked", "queried", "inquired", "questioned", "requested"]
    }
    return properties


def main():
    print("=" * 70)
    print("DEEP VERB ANALYSIS: What Makes Verbs Trigger Inhibition?")
    print("=" * 70)
    print()
    
    model, tokenizer = load_model()
    
    all_results = {}
    
    # Get unique verbs
    unique_verbs = set()
    for category, verbs in VERBS_TO_TEST.items():
        for v in verbs:
            unique_verbs.add(v)
    
    unique_verbs = sorted(unique_verbs)
    print(f"\nTesting {len(unique_verbs)} unique verbs...")
    print("-" * 70)
    
    for i, verb in enumerate(unique_verbs):
        print(f"\n[{i+1}/{len(unique_verbs)}] {verb}")
        
        # Baseline test
        baseline = test_verb(model, tokenizer, verb, n_trials=10)
        
        # Ablation test (only if baseline is poor)
        if baseline["accuracy"] < 0.8:
            ablated = test_with_ablation(model, tokenizer, verb, n_trials=10)
            boost = ablated["accuracy_ablated"] - baseline["accuracy"]
        else:
            ablated = {"accuracy_ablated": baseline["accuracy"], "mean_diff_ablated": baseline["mean_diff"]}
            boost = 0.0
        
        # Linguistic properties
        properties = analyze_verb_properties(verb)
        
        result = {
            **baseline,
            **ablated,
            "boost": boost,
            "properties": properties
        }
        
        all_results[verb] = result
        
        # Print summary
        acc_str = f"{baseline['accuracy']*100:.0f}%"
        abl_str = f"{ablated['accuracy_ablated']*100:.0f}%"
        boost_str = f"+{boost*100:.0f}%" if boost > 0 else f"{boost*100:.0f}%"
        
        status = "[OK]" if baseline['accuracy'] >= 0.8 else "[FAIL]" if baseline['accuracy'] == 0 else "[MID]"
        print(f"   {status} Baseline: {acc_str}, Ablated: {abl_str}, Boost: {boost_str}")
        print(f"     L18H11 attention: alice={baseline['mean_attn_alice']:.3f}, basket={baseline['mean_attn_basket']:.3f}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("PATTERN ANALYSIS")
    print("=" * 70)
    
    # Group by accuracy
    perfect = [v for v, r in all_results.items() if r["accuracy"] == 1.0]
    zero = [v for v, r in all_results.items() if r["accuracy"] == 0.0]
    mid = [v for v, r in all_results.items() if 0 < r["accuracy"] < 1.0]
    
    print(f"\n100% accuracy (model succeeds): {perfect}")
    print(f"0% accuracy (complete failure): {zero}")
    print(f"Partial accuracy: {mid}")
    
    # Property analysis
    print("\n--- PROPERTY ANALYSIS ---")
    
    for prop in ["implies_face_to_face", "implies_public", "implies_question", "implies_recording"]:
        with_prop = [r["accuracy"] for v, r in all_results.items() if r["properties"].get(prop, False)]
        without_prop = [r["accuracy"] for v, r in all_results.items() if not r["properties"].get(prop, False)]
        
        if with_prop and without_prop:
            print(f"\n{prop}:")
            print(f"  WITH: {np.mean(with_prop)*100:.1f}% (n={len(with_prop)})")
            print(f"  WITHOUT: {np.mean(without_prop)*100:.1f}% (n={len(without_prop)})")
    
    # Save results
    print("\n" + "=" * 70)
    print("Saving results...")
    
    # Remove non-serializable 'results' list for JSON
    save_results = {
        v: {k: val for k, val in r.items() if k != "results"} 
        for v, r in all_results.items()
    }
    
    with open(OUTPUT_DIR / "verb_deep_analysis_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "verbs": save_results,
            "summary": {
                "perfect_verbs": perfect,
                "zero_verbs": zero,
                "mid_verbs": mid
            }
        }, f, indent=2)
    
    print(f"Saved to: {OUTPUT_DIR / 'verb_deep_analysis_results.json'}")
    
    # Create visualization
    create_visualization(all_results)


def create_visualization(all_results: dict):
    """Create summary visualization."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Verb Analysis: What Triggers ToM Inhibition?", fontsize=14, fontweight='bold')
    
    # Sort by accuracy
    sorted_verbs = sorted(all_results.items(), key=lambda x: x[1]["accuracy"])
    verbs = [v for v, _ in sorted_verbs]
    accuracies = [r["accuracy"] for _, r in sorted_verbs]
    ablated_accs = [r["accuracy_ablated"] for _, r in sorted_verbs]
    
    # Plot 1: Accuracy comparison
    ax = axes[0, 0]
    x = range(len(verbs))
    width = 0.35
    ax.bar([i - width/2 for i in x], accuracies, width, label='Baseline', color='#e74c3c', alpha=0.7)
    ax.bar([i + width/2 for i in x], ablated_accs, width, label='Ablated', color='#27ae60', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(verbs, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Baseline vs Ablated Accuracy by Verb")
    ax.legend()
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 2: Attention to Alice vs Basket
    ax = axes[0, 1]
    alice_attn = [r["mean_attn_alice"] for _, r in sorted_verbs]
    basket_attn = [r["mean_attn_basket"] for _, r in sorted_verbs]
    
    colors = ['#e74c3c' if acc == 0 else '#27ae60' if acc == 1 else '#f39c12' for acc in accuracies]
    
    ax.scatter(alice_attn, basket_attn, c=colors, alpha=0.7, s=100)
    for i, verb in enumerate(verbs):
        ax.annotate(verb, (alice_attn[i], basket_attn[i]), fontsize=7, alpha=0.8)
    
    ax.set_xlabel("Attention to Alice (L18H11)")
    ax.set_ylabel("Attention to Basket (L18H11)")
    ax.set_title("Attention Pattern: Alice vs Basket")
    
    # Plot 3: Boost distribution
    ax = axes[1, 0]
    boosts = [r["boost"] * 100 for _, r in sorted_verbs if r["boost"] != 0]
    if boosts:
        ax.hist(boosts, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    ax.set_xlabel("Accuracy Boost (%)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Ablation Boosts")
    ax.axvline(x=0, color='red', linestyle='--')
    
    # Plot 4: Face-to-face vs Public
    ax = axes[1, 1]
    
    face_to_face = [v for v, r in all_results.items() if r["properties"]["implies_face_to_face"]]
    public = [v for v, r in all_results.items() if r["properties"]["implies_public"]]
    other = [v for v, r in all_results.items() if not r["properties"]["implies_face_to_face"] and not r["properties"]["implies_public"]]
    
    categories = ['Face-to-Face', 'Public', 'Other']
    cat_accs = [
        np.mean([all_results[v]["accuracy"] for v in face_to_face]) if face_to_face else 0,
        np.mean([all_results[v]["accuracy"] for v in public]) if public else 0,
        np.mean([all_results[v]["accuracy"] for v in other]) if other else 0
    ]
    
    ax.bar(categories, cat_accs, color=['#e74c3c', '#27ae60', '#3498db'], alpha=0.7)
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Accuracy by Communication Type")
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "verb_deep_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'verb_deep_analysis.png'}")


if __name__ == "__main__":
    main()

