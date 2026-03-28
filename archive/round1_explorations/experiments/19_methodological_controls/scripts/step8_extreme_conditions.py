"""
EXTREME CONDITIONS: Push the Circuit to Its Limits
===================================================

Going deeper:
1. AMPLIFY L24H29 (multiply by 2x, 3x) - does ToM get WORSE?
2. Ablate L24H29 on BASELINE (no bridge) - how high can we go?
3. What does L24H29 ATTEND to? Does it look at original location?
4. Ablate update circuit + inhibitor together - who wins?
5. Search for OTHER inhibitory heads systematically
6. Triple ablation: inhibitor + enable update circuit

This will tell us the TRUE mechanistic story.
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


class ExtremeConditionTester:
    """Test extreme circuit manipulations."""
    
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
            return (reshaped.view(batch, seq_len, hidden),) + args[1:] if len(args) > 1 else (reshaped.view(batch, seq_len, hidden),)
        return hook
    
    def _create_amplification_hook(self, head_idx: int, scale: float):
        """AMPLIFY a head instead of ablating it."""
        n_heads = self.n_heads
        def hook(module, args):
            hidden_states = args[0]
            batch, seq_len, hidden = hidden_states.shape
            head_dim = hidden // n_heads
            reshaped = hidden_states.view(batch, seq_len, n_heads, head_dim)
            reshaped[:, :, head_idx, :] = reshaped[:, :, head_idx, :] * scale
            return (reshaped.view(batch, seq_len, hidden),) + args[1:] if len(args) > 1 else (reshaped.view(batch, seq_len, hidden),)
        return hook
    
    def install_multi_layer_ablation(self, layer_head_pairs: list):
        """Ablate heads across multiple layers."""
        self.clear_hooks()
        layer_to_heads = {}
        for layer_idx, head_idx in layer_head_pairs:
            if layer_idx not in layer_to_heads:
                layer_to_heads[layer_idx] = []
            layer_to_heads[layer_idx].append(head_idx)
        
        for layer_idx, head_indices in layer_to_heads.items():
            o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
            hook = o_proj.register_forward_pre_hook(self._create_ablation_hook(head_indices))
            self.hooks.append(hook)
    
    def install_amplification(self, layer_idx: int, head_idx: int, scale: float):
        """Amplify a single head."""
        self.clear_hooks()
        o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
        hook = o_proj.register_forward_pre_hook(self._create_amplification_hook(head_idx, scale))
        self.hooks.append(hook)
    
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def test_prompt(self, prompt: str, correct: str, wrong: str) -> dict:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]
        
        correct_id = self.tokenizer.encode(correct, add_special_tokens=False)[0]
        wrong_id = self.tokenizer.encode(wrong, add_special_tokens=False)[0]
        
        correct_logit = logits[correct_id].item()
        wrong_logit = logits[wrong_id].item()
        
        max_logit = max(correct_logit, wrong_logit)
        correct_prob = np.exp(correct_logit - max_logit) / (np.exp(correct_logit - max_logit) + np.exp(wrong_logit - max_logit))
        
        return {
            "chose_correct": correct_logit > wrong_logit,
            "margin": correct_logit - wrong_logit,
            "prob": float(correct_prob),
        }
    
    def get_attention_pattern(self, prompt: str, layer_idx: int, head_idx: int) -> np.ndarray:
        """Get attention pattern for a specific head."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        # Get attention from specified layer and head
        attn = outputs.attentions[layer_idx][0, head_idx].cpu().numpy()  # (seq, seq)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        return attn, tokens


def generate_test_scenarios(n: int = 50) -> list:
    random.seed(42)
    
    AGENTS = ["Alice", "Bob", "Carol", "David"]
    INFORMERS = ["Eve", "Frank", "Grace", "Henry"]
    OBJECTS = ["ball", "key", "book", "toy"]
    LOCATIONS = ["drawer", "basket", "cupboard", "shelf"]
    
    scenarios = []
    for i in range(n):
        agent = random.choice(AGENTS)
        informer = random.choice([x for x in INFORMERS if x != agent])
        obj = random.choice(OBJECTS)
        loc1, loc2 = random.sample(LOCATIONS, 2)
        
        baseline = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2}.' "
            f"Where will {agent} look for the {obj}? {agent} will look in the"
        )
        
        bridged = (
            f"{agent} put the {obj} in the {loc1}. "
            f"{informer} tells {agent}: 'I moved the {obj} to the {loc2},' "
            f"so {agent} updated their belief. "
            f"Where will {agent} look for the {obj}? {agent} will look in the"
        )
        
        scenarios.append({
            "baseline": baseline,
            "bridged": bridged,
            "correct": f" {loc2}",
            "wrong": f" {loc1}",
            "agent": agent,
            "loc1": loc1,
            "loc2": loc2,
        })
    
    return scenarios


def run_condition(tester, scenarios, prompt_type="baseline"):
    correct = 0
    probs = []
    for s in scenarios:
        result = tester.test_prompt(s[prompt_type], s["correct"], s["wrong"])
        if result["chose_correct"]:
            correct += 1
        probs.append(result["prob"])
    return correct / len(scenarios), np.mean(probs)


def main():
    timer_start = time.perf_counter()
    
    print("=" * 70)
    print("EXTREME CONDITIONS: Pushing the Circuit to Its Limits")
    print("=" * 70)
    
    # Load model
    print("\n[1/6] Loading model...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CFG.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CFG.model_name,
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    print("  [OK] Model loaded", flush=True)
    
    tester = ExtremeConditionTester(model, tokenizer)
    
    # Generate scenarios
    print("\n[2/6] Generating scenarios...", flush=True)
    scenarios = generate_test_scenarios(50)
    print(f"  Generated {len(scenarios)} scenarios")
    
    results = {}
    
    # ========================================
    # EXTREME TEST 1: Amplify L24H29
    # ========================================
    print("\n[3/6] EXTREME TEST 1: Amplifying L24H29...", flush=True)
    print("  If L24H29 inhibits ToM, amplifying it should make ToM WORSE")
    
    amplification_results = {}
    
    for scale in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        if scale == 0.0:
            # This is ablation
            tester.install_multi_layer_ablation([(24, 29)])
        elif scale == 1.0:
            # This is no change
            tester.clear_hooks()
        else:
            tester.install_amplification(24, 29, scale)
        
        base_acc, base_prob = run_condition(tester, scenarios, "baseline")
        bridge_acc, bridge_prob = run_condition(tester, scenarios, "bridged")
        
        amplification_results[scale] = {
            "baseline_acc": base_acc,
            "bridged_acc": bridge_acc,
        }
        print(f"    Scale {scale}x: baseline={base_acc:.1%}, bridged={bridge_acc:.1%}")
    
    tester.clear_hooks()
    results["amplification"] = amplification_results
    
    # ========================================
    # EXTREME TEST 2: Ablate inhibitor + update circuit
    # ========================================
    print("\n[4/6] EXTREME TEST 2: Ablate inhibitor AND update circuit...", flush=True)
    print("  Who wins - the enabler or the disabler?")
    
    INHIBITOR = [(24, 29)]
    UPDATE_CIRCUIT = [(23, 4), (28, 0), (28, 23), (23, 31), (26, 25)]
    
    combo_results = {}
    
    # Just inhibitor
    tester.install_multi_layer_ablation(INHIBITOR)
    acc, _ = run_condition(tester, scenarios, "baseline")
    combo_results["inhibitor_only"] = acc
    print(f"    Ablate inhibitor only: {acc:.1%}")
    
    # Just update circuit
    tester.install_multi_layer_ablation(UPDATE_CIRCUIT)
    acc, _ = run_condition(tester, scenarios, "baseline")
    combo_results["update_only"] = acc
    print(f"    Ablate update circuit only: {acc:.1%}")
    
    # Both
    tester.install_multi_layer_ablation(INHIBITOR + UPDATE_CIRCUIT)
    acc, _ = run_condition(tester, scenarios, "baseline")
    combo_results["both"] = acc
    print(f"    Ablate BOTH: {acc:.1%}")
    
    # Neither (baseline)
    tester.clear_hooks()
    acc, _ = run_condition(tester, scenarios, "baseline")
    combo_results["neither"] = acc
    print(f"    Neither (baseline): {acc:.1%}")
    
    results["combo_ablation"] = combo_results
    
    # ========================================
    # EXTREME TEST 3: SKIPPED - Run step8b_inhibitory_search.py separately
    # ========================================
    print("\n[5/6] SKIPPING inhibitory head search (run step8b separately)", flush=True)
    print("  See: step8b_inhibitory_search.py for thorough inhibitory head search")
    results["inhibitory_search"] = "run_step8b_separately"
    
    # ========================================
    # EXTREME TEST 4: What does L24H29 attend to?
    # ========================================
    print("\n[6/6] EXTREME TEST 4: Analyzing L24H29 attention pattern...", flush=True)
    
    # Get attention for a few examples
    attention_analysis = []
    
    for i, scenario in enumerate(scenarios[:5]):
        attn, tokens = tester.get_attention_pattern(scenario["baseline"], 24, 29)
        
        # Focus on attention FROM the last token
        last_token_attn = attn[-1]  # (seq,)
        
        # Find indices of key tokens
        tokens_lower = [t.lower() for t in tokens]
        
        loc1_indices = [j for j, t in enumerate(tokens_lower) if scenario["loc1"].lower() in t]
        loc2_indices = [j for j, t in enumerate(tokens_lower) if scenario["loc2"].lower() in t]
        agent_indices = [j for j, t in enumerate(tokens_lower) if scenario["agent"].lower() in t]
        
        # Average attention to each type
        loc1_attn = np.mean([last_token_attn[j] for j in loc1_indices]) if loc1_indices else 0
        loc2_attn = np.mean([last_token_attn[j] for j in loc2_indices]) if loc2_indices else 0
        agent_attn = np.mean([last_token_attn[j] for j in agent_indices]) if agent_indices else 0
        
        attention_analysis.append({
            "loc1_attention": float(loc1_attn),
            "loc2_attention": float(loc2_attn),
            "agent_attention": float(agent_attn),
            "loc1": scenario["loc1"],
            "loc2": scenario["loc2"],
        })
    
    results["attention_analysis"] = attention_analysis
    
    # Summary
    avg_loc1_attn = np.mean([a["loc1_attention"] for a in attention_analysis])
    avg_loc2_attn = np.mean([a["loc2_attention"] for a in attention_analysis])
    print(f"    Avg attention to ORIGINAL location (loc1): {avg_loc1_attn:.4f}")
    print(f"    Avg attention to NEW location (loc2): {avg_loc2_attn:.4f}")
    print(f"    Ratio loc1/loc2: {avg_loc1_attn/max(avg_loc2_attn, 0.0001):.2f}x")
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("EXTREME CONDITIONS: FINAL SUMMARY")
    print("=" * 70)
    
    print("\n[1] AMPLIFICATION TEST:")
    ref = amplification_results[1.0]["baseline_acc"]
    for scale, data in sorted(amplification_results.items()):
        delta = data["baseline_acc"] - ref
        marker = ""
        if scale == 0.0:
            marker = " <-- ABLATED"
        elif scale == 1.0:
            marker = " <-- NORMAL"
        elif delta < -0.10:
            marker = " ** INHIBITION CONFIRMED **"
        print(f"    {scale}x: {data['baseline_acc']:.1%} ({delta:+.1%}){marker}")
    
    print("\n[2] COMBO ABLATION:")
    print(f"    Neither:              {combo_results['neither']:.1%}")
    print(f"    Ablate inhibitor:     {combo_results['inhibitor_only']:.1%}")
    print(f"    Ablate update circuit:{combo_results['update_only']:.1%}")
    print(f"    Ablate BOTH:          {combo_results['both']:.1%}")
    
    print("\n[3] OTHER INHIBITORY HEADS (top 5):")
    for r in results["inhibitory_search"][:5]:
        print(f"    L{r['layer']}H{r['head']}: boost {r['boost']:+.1%}")
    
    print("\n[4] L24H29 ATTENTION PATTERN:")
    print(f"    Attends to original location {avg_loc1_attn/max(avg_loc2_attn, 0.0001):.1f}x more than new location")
    if avg_loc1_attn > avg_loc2_attn * 1.5:
        print("    ** L24H29 is ANCHORING to the original location! **")
    
    # Save
    with open(RESULTS_DIR / "extreme_conditions_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    total_time = time.perf_counter() - timer_start
    print(f"\n[OK] Completed in {total_time:.1f}s")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


