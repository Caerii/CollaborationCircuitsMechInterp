"""
DEEP CIRCUIT ANALYSIS: Understanding the ToM Neurobiology
==========================================================

Going deep on Qwen3-4B as our "model organism":

1. Create FIXED scenario set (N=100) for reproducibility
2. Retest combined ablation with CORRECT top inhibitors
3. Analyze attention patterns on L18H11 (strongest inhibitor)
4. Test AMPLIFYING enablers (L15H9, L19H2, L19H15)
5. Compare inhibitor vs enabler attention patterns
"""

import json
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import random

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Fixed seed for reproducibility
MASTER_SEED = 42


def create_fixed_scenarios(n: int = 100) -> list:
    """Create a FIXED scenario set that we'll use for all experiments."""
    random.seed(MASTER_SEED)
    np.random.seed(MASTER_SEED)
    
    scenarios = []
    
    # Diverse vocabulary to avoid lexical biases
    AGENTS = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry"]
    INFORMERS = ["Iris", "Jack", "Kate", "Leo", "Maya", "Noah", "Olivia", "Paul"]
    OBJECTS = ["ball", "key", "book", "toy", "pen", "hat", "cup", "bag"]
    LOCATIONS = ["drawer", "basket", "cupboard", "shelf", "box", "chest", "cabinet", "trunk"]
    
    for i in range(n):
        agent = AGENTS[i % len(AGENTS)]
        informer = INFORMERS[i % len(INFORMERS)]
        obj = OBJECTS[i % len(OBJECTS)]
        
        # Ensure different locations
        loc_idx = i % len(LOCATIONS)
        loc1 = LOCATIONS[loc_idx]
        loc2 = LOCATIONS[(loc_idx + 4) % len(LOCATIONS)]  # Offset by 4 to ensure different
        
        # Vary the communication verb
        verbs = ["tells", "informs", "says to", "lets know"]
        verb = verbs[i % len(verbs)]
        
        if verb == "lets know":
            prompt = (
                f"{agent} put the {obj} in the {loc1}. "
                f"{informer} lets {agent} know: 'I moved the {obj} to the {loc2}.' "
                f"Where will {agent} look for the {obj}? {agent} will look in the"
            )
        else:
            prompt = (
                f"{agent} put the {obj} in the {loc1}. "
                f"{informer} {verb} {agent}: 'I moved the {obj} to the {loc2}.' "
                f"Where will {agent} look for the {obj}? {agent} will look in the"
            )
        
        scenarios.append({
            "id": i,
            "prompt": prompt,
            "correct": f" {loc2}",
            "wrong": f" {loc1}",
            "agent": agent,
            "informer": informer,
            "object": obj,
            "loc1": loc1,
            "loc2": loc2,
            "verb": verb,
        })
    
    return scenarios


class DeepCircuitAnalyzer:
    """Comprehensive circuit analysis tools."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = []
        self.n_heads = model.config.num_attention_heads
        self.n_layers = model.config.num_hidden_layers
        
    def _create_ablation_hook(self, head_indices: list):
        n_heads = self.n_heads
        def hook(module, args):
            hidden_states = args[0]
            batch, seq_len, hidden = hidden_states.shape
            head_dim = hidden // n_heads
            reshaped = hidden_states.view(batch, seq_len, n_heads, head_dim)
            for head_idx in head_indices:
                if head_idx < n_heads:
                    reshaped[:, :, head_idx, :] = 0
            return (reshaped.view(batch, seq_len, hidden),)
        return hook
    
    def _create_amplification_hook(self, head_idx: int, scale: float):
        n_heads = self.n_heads
        def hook(module, args):
            hidden_states = args[0]
            batch, seq_len, hidden = hidden_states.shape
            head_dim = hidden // n_heads
            reshaped = hidden_states.view(batch, seq_len, n_heads, head_dim)
            reshaped[:, :, head_idx, :] = reshaped[:, :, head_idx, :] * scale
            return (reshaped.view(batch, seq_len, hidden),)
        return hook
    
    def install_multi_ablation(self, layer_head_pairs: list):
        self.clear_hooks()
        layer_to_heads = {}
        for layer, head in layer_head_pairs:
            if layer not in layer_to_heads:
                layer_to_heads[layer] = []
            layer_to_heads[layer].append(head)
        
        for layer_idx, head_indices in layer_to_heads.items():
            o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
            hook = o_proj.register_forward_pre_hook(self._create_ablation_hook(head_indices))
            self.hooks.append(hook)
    
    def install_amplification(self, layer_idx: int, head_idx: int, scale: float):
        self.clear_hooks()
        o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
        hook = o_proj.register_forward_pre_hook(self._create_amplification_hook(head_idx, scale))
        self.hooks.append(hook)
    
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def test_scenario(self, scenario: dict) -> dict:
        inputs = self.tokenizer(scenario["prompt"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1, :]
        
        correct_id = self.tokenizer.encode(scenario["correct"], add_special_tokens=False)[0]
        wrong_id = self.tokenizer.encode(scenario["wrong"], add_special_tokens=False)[0]
        
        correct_logit = logits[correct_id].item()
        wrong_logit = logits[wrong_id].item()
        
        return {
            "correct": correct_logit > wrong_logit,
            "margin": correct_logit - wrong_logit,
        }
    
    def test_batch(self, scenarios: list) -> tuple:
        correct = 0
        margins = []
        for s in scenarios:
            result = self.test_scenario(s)
            if result["correct"]:
                correct += 1
            margins.append(result["margin"])
        return correct / len(scenarios), np.mean(margins), np.std(margins)
    
    def get_attention_pattern(self, prompt: str, layer_idx: int, head_idx: int) -> tuple:
        """Get attention weights for a specific head."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        # Qwen uses GQA, so attention shape is different
        # attentions[layer] shape: (batch, n_kv_heads, seq, seq) or similar
        attn = outputs.attentions[layer_idx]
        
        # Handle GQA - may have fewer KV heads
        n_kv_heads = attn.shape[1]
        if n_kv_heads < self.n_heads:
            # Map head_idx to KV head
            heads_per_kv = self.n_heads // n_kv_heads
            kv_head_idx = head_idx // heads_per_kv
        else:
            kv_head_idx = head_idx
        
        attn_pattern = attn[0, min(kv_head_idx, n_kv_heads-1)].cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        return attn_pattern, tokens


def analyze_attention_patterns(analyzer, scenarios, layer_head_pairs, n_examples=10):
    """Analyze what specific heads attend to."""
    
    results = {}
    
    for layer, head in layer_head_pairs:
        key = f"L{layer}H{head}"
        head_results = {
            "loc1_attention": [],
            "loc2_attention": [],
            "agent_attention": [],
            "verb_attention": [],
            "ratio_loc1_loc2": [],
        }
        
        for scenario in scenarios[:n_examples]:
            try:
                attn, tokens = analyzer.get_attention_pattern(
                    scenario["prompt"], layer, head
                )
                
                # Last token's attention (what the model looks at when predicting)
                last_attn = attn[-1]
                
                # Find token indices
                tokens_lower = [t.lower() if isinstance(t, str) else str(t).lower() for t in tokens]
                prompt_lower = scenario["prompt"].lower()
                
                loc1_indices = [i for i, t in enumerate(tokens_lower) if scenario["loc1"].lower() in t]
                loc2_indices = [i for i, t in enumerate(tokens_lower) if scenario["loc2"].lower() in t]
                agent_indices = [i for i, t in enumerate(tokens_lower) if scenario["agent"].lower() in t]
                verb_indices = [i for i, t in enumerate(tokens_lower) if scenario["verb"].split()[0].lower() in t]
                
                # Sum attention to each type
                loc1_attn = sum(last_attn[i] for i in loc1_indices) if loc1_indices else 0
                loc2_attn = sum(last_attn[i] for i in loc2_indices) if loc2_indices else 0
                agent_attn = sum(last_attn[i] for i in agent_indices) if agent_indices else 0
                verb_attn = sum(last_attn[i] for i in verb_indices) if verb_indices else 0
                
                head_results["loc1_attention"].append(float(loc1_attn))
                head_results["loc2_attention"].append(float(loc2_attn))
                head_results["agent_attention"].append(float(agent_attn))
                head_results["verb_attention"].append(float(verb_attn))
                
                if loc2_attn > 0.001:
                    head_results["ratio_loc1_loc2"].append(float(loc1_attn / loc2_attn))
                    
            except Exception as e:
                print(f"    Warning: Error analyzing {key}: {e}")
                continue
        
        # Compute averages
        results[key] = {
            "avg_loc1_attention": np.mean(head_results["loc1_attention"]) if head_results["loc1_attention"] else 0,
            "avg_loc2_attention": np.mean(head_results["loc2_attention"]) if head_results["loc2_attention"] else 0,
            "avg_agent_attention": np.mean(head_results["agent_attention"]) if head_results["agent_attention"] else 0,
            "avg_verb_attention": np.mean(head_results["verb_attention"]) if head_results["verb_attention"] else 0,
            "avg_ratio": np.mean(head_results["ratio_loc1_loc2"]) if head_results["ratio_loc1_loc2"] else 0,
        }
    
    return results


def main():
    start_time = time.time()
    
    print("=" * 70)
    print("DEEP CIRCUIT ANALYSIS: ToM Neurobiology")
    print("=" * 70)
    print("Model Organism: Qwen3-4B")
    print()
    
    # Create fixed scenario set
    print("[1/7] Creating FIXED scenario set (N=100)...", flush=True)
    scenarios = create_fixed_scenarios(100)
    
    # Save scenarios for future reproducibility
    scenarios_file = RESULTS_DIR / "fixed_scenarios.json"
    with open(scenarios_file, "w") as f:
        json.dump(scenarios, f, indent=2)
    print(f"  Saved to {scenarios_file}", flush=True)
    
    # Load model
    print("\n[2/7] Loading model...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CFG.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CFG.model_name,
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    print("  [OK]", flush=True)
    
    analyzer = DeepCircuitAnalyzer(model, tokenizer)
    
    # Define heads from complete search
    TOP_INHIBITORS = [
        (18, 11),  # +43% - STRONGEST
        (17, 4),   # +43%
        (18, 14),  # +43%
        (21, 17),  # +43%
        (19, 30),  # +40%
    ]
    
    CRITICAL_ENABLERS = [
        (15, 9),   # 0% when ablated - ESSENTIAL
        (19, 2),   # 0% when ablated - ESSENTIAL
        (19, 15),  # 0% when ablated - ESSENTIAL
    ]
    
    results = {}
    
    # ========================================
    # TEST 1: Baseline on fixed scenarios
    # ========================================
    print("\n[3/7] Testing baseline on fixed scenarios...", flush=True)
    analyzer.clear_hooks()
    baseline_acc, baseline_margin, baseline_std = analyzer.test_batch(scenarios)
    results["baseline"] = {
        "accuracy": baseline_acc,
        "mean_margin": baseline_margin,
        "std_margin": baseline_std,
    }
    print(f"  Baseline: {baseline_acc:.1%} (margin: {baseline_margin:.2f} +/- {baseline_std:.2f})")
    
    # ========================================
    # TEST 2: Single inhibitor ablations
    # ========================================
    print("\n[4/7] Testing single inhibitor ablations...", flush=True)
    print("  " + "-" * 50)
    
    single_results = {}
    for layer, head in TOP_INHIBITORS:
        analyzer.install_multi_ablation([(layer, head)])
        acc, margin, std = analyzer.test_batch(scenarios)
        analyzer.clear_hooks()
        
        boost = acc - baseline_acc
        single_results[f"L{layer}H{head}"] = {
            "accuracy": acc,
            "boost": boost,
            "margin": margin,
        }
        print(f"  L{layer}H{head}: {acc:.1%} (boost: {boost:+.1%}, margin: {margin:.2f})")
    
    results["single_ablations"] = single_results
    
    # ========================================
    # TEST 3: Combined inhibitor ablations
    # ========================================
    print("\n[5/7] Testing combined inhibitor ablations...", flush=True)
    print("  " + "-" * 50)
    
    combined_results = {}
    
    # Top 2
    analyzer.install_multi_ablation(TOP_INHIBITORS[:2])
    acc, margin, std = analyzer.test_batch(scenarios)
    combined_results["top_2"] = {"heads": TOP_INHIBITORS[:2], "accuracy": acc, "boost": acc - baseline_acc}
    print(f"  Top 2: {acc:.1%} (boost: {acc - baseline_acc:+.1%})")
    
    # Top 3
    analyzer.install_multi_ablation(TOP_INHIBITORS[:3])
    acc, margin, std = analyzer.test_batch(scenarios)
    combined_results["top_3"] = {"heads": TOP_INHIBITORS[:3], "accuracy": acc, "boost": acc - baseline_acc}
    print(f"  Top 3: {acc:.1%} (boost: {acc - baseline_acc:+.1%})")
    
    # Top 5
    analyzer.install_multi_ablation(TOP_INHIBITORS[:5])
    acc, margin, std = analyzer.test_batch(scenarios)
    combined_results["top_5"] = {"heads": TOP_INHIBITORS[:5], "accuracy": acc, "boost": acc - baseline_acc}
    print(f"  Top 5: {acc:.1%} (boost: {acc - baseline_acc:+.1%})")
    
    analyzer.clear_hooks()
    results["combined_ablations"] = combined_results
    
    # ========================================
    # TEST 4: Enabler amplification
    # ========================================
    print("\n[6/7] Testing enabler AMPLIFICATION...", flush=True)
    print("  Can we boost ToM by amplifying enablers?")
    print("  " + "-" * 50)
    
    amplification_results = {}
    
    for layer, head in CRITICAL_ENABLERS:
        head_results = {}
        for scale in [1.0, 1.5, 2.0, 3.0]:
            if scale == 1.0:
                analyzer.clear_hooks()
            else:
                analyzer.install_amplification(layer, head, scale)
            
            acc, margin, std = analyzer.test_batch(scenarios)
            head_results[f"{scale}x"] = acc
            
            if scale == 1.0:
                print(f"  L{layer}H{head} @ {scale}x: {acc:.1%} (baseline)")
            else:
                boost = acc - baseline_acc
                print(f"  L{layer}H{head} @ {scale}x: {acc:.1%} ({boost:+.1%})")
        
        amplification_results[f"L{layer}H{head}"] = head_results
        analyzer.clear_hooks()
    
    results["enabler_amplification"] = amplification_results
    
    # ========================================
    # TEST 5: Attention pattern analysis
    # ========================================
    print("\n[7/7] Analyzing attention patterns...", flush=True)
    print("  What do inhibitors vs enablers attend to?")
    print("  " + "-" * 50)
    
    # Analyze inhibitors
    print("\n  INHIBITORS:")
    inhibitor_attention = analyze_attention_patterns(
        analyzer, scenarios, TOP_INHIBITORS[:3], n_examples=20
    )
    
    for key, data in inhibitor_attention.items():
        ratio = data["avg_ratio"]
        interpretation = ""
        if ratio > 1.5:
            interpretation = "<-- ANCHORS to original location!"
        elif ratio < 0.7:
            interpretation = "<-- Attends to new location"
        print(f"    {key}: loc1/loc2 ratio = {ratio:.2f} {interpretation}")
    
    # Analyze enablers
    print("\n  ENABLERS:")
    enabler_attention = analyze_attention_patterns(
        analyzer, scenarios, CRITICAL_ENABLERS, n_examples=20
    )
    
    for key, data in enabler_attention.items():
        ratio = data["avg_ratio"]
        interpretation = ""
        if ratio > 1.5:
            interpretation = "<-- Attends to original location"
        elif ratio < 0.7:
            interpretation = "<-- FOCUSES on new location!"
        print(f"    {key}: loc1/loc2 ratio = {ratio:.2f} {interpretation}")
    
    results["attention_analysis"] = {
        "inhibitors": inhibitor_attention,
        "enablers": enabler_attention,
    }
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("DEEP CIRCUIT ANALYSIS: SUMMARY")
    print("=" * 70)
    
    print(f"\n  Fixed scenario set: N={len(scenarios)}")
    print(f"  Baseline accuracy: {baseline_acc:.1%}")
    
    print("\n  INHIBITOR ABLATION EFFECTS:")
    best_single = max(single_results.items(), key=lambda x: x[1]["accuracy"])
    print(f"    Best single: {best_single[0]} -> {best_single[1]['accuracy']:.1%}")
    print(f"    Top 3 combined: {combined_results['top_3']['accuracy']:.1%}")
    print(f"    Top 5 combined: {combined_results['top_5']['accuracy']:.1%}")
    
    print("\n  ENABLER AMPLIFICATION EFFECTS:")
    for key, data in amplification_results.items():
        effect = data["2.0x"] - data["1.0x"]
        print(f"    {key} @ 2x: {data['2.0x']:.1%} (effect: {effect:+.1%})")
    
    print("\n  ATTENTION PATTERN INSIGHTS:")
    for key, data in inhibitor_attention.items():
        if data["avg_ratio"] > 1.2:
            print(f"    {key} ANCHORS to original location (ratio {data['avg_ratio']:.2f})")
    for key, data in enabler_attention.items():
        if data["avg_ratio"] < 0.8:
            print(f"    {key} FOCUSES on new location (ratio {data['avg_ratio']:.2f})")
    
    # Save results
    output_file = RESULTS_DIR / "deep_circuit_analysis.json"
    
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        return obj
    
    def convert_dict(d):
        if isinstance(d, dict):
            return {k: convert_dict(v) for k, v in d.items()}
        if isinstance(d, list):
            return [convert_dict(i) for i in d]
        return convert(d)
    
    with open(output_file, "w") as f:
        json.dump(convert_dict(results), f, indent=2)
    
    total_time = (time.time() - start_time) / 60
    print(f"\n  Total time: {total_time:.1f} minutes")
    print(f"\n[OK] Saved to {output_file}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

