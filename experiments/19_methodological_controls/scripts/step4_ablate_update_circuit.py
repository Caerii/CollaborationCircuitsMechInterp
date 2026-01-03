"""
Ablate the Belief Update Circuit: Confirm L23H4, L28H0 Role
============================================================

We identified L23H4 and L28H0 as showing the BIGGEST attention changes
when we add the "belief update bridge" phrase.

Hypothesis: These heads ARE the belief update circuit.

Test: If we ablate these heads on BRIDGED prompts (which normally get 98%),
the accuracy should DROP significantly (back toward baseline ~18%).

If ablation DOESN'T hurt bridged prompts → these heads aren't the circuit.
If ablation DOES hurt bridged prompts → CONFIRMED: these ARE the circuit.
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


class BridgeCircuitAblator:
    """Ablate specific attention heads using proper o_proj pre-hook."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = []
        self.n_heads = model.config.num_attention_heads
        self.hidden_size = model.config.hidden_size
        # For GQA models, head_dim is hidden_size // n_heads
        self.head_dim = self.hidden_size // self.n_heads
        
        print(f"    Ablator config: n_heads={self.n_heads}, head_dim={self.head_dim}, hidden={self.hidden_size}")
        
    def _create_ablation_hook(self, head_indices: list):
        """Create hook that zeros out multiple heads."""
        n_heads = self.n_heads
        head_dim = self.head_dim
        
        def hook(module, args):
            hidden_states = args[0]
            batch, seq_len, hidden = hidden_states.shape
            
            # Calculate actual head_dim from input shape
            actual_head_dim = hidden // n_heads
            
            # Reshape to expose heads
            reshaped = hidden_states.view(batch, seq_len, n_heads, actual_head_dim)
            
            # Zero out specified heads
            for head_idx in head_indices:
                if head_idx < n_heads:
                    reshaped[:, :, head_idx, :] = 0
            
            modified = reshaped.view(batch, seq_len, hidden)
            return (modified,) + args[1:] if len(args) > 1 else (modified,)
        return hook
    
    def install_ablation(self, layer_head_pairs: list):
        """Install ablation hooks for given (layer, head) pairs."""
        self.clear_hooks()
        
        # Group by layer
        layer_to_heads = {}
        for layer_idx, head_idx in layer_head_pairs:
            if layer_idx not in layer_to_heads:
                layer_to_heads[layer_idx] = []
            layer_to_heads[layer_idx].append(head_idx)
        
        # Install hooks
        for layer_idx, head_indices in layer_to_heads.items():
            o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
            hook = o_proj.register_forward_pre_hook(self._create_ablation_hook(head_indices))
            self.hooks.append(hook)
    
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def test_prompt(self, prompt: str, correct: str, wrong: str) -> dict:
        """Test if model prefers correct over wrong completion."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]
        
        correct_id = self.tokenizer.encode(correct, add_special_tokens=False)[0]
        wrong_id = self.tokenizer.encode(wrong, add_special_tokens=False)[0]
        
        correct_logit = logits[correct_id].item()
        wrong_logit = logits[wrong_id].item()
        
        return {
            "chose_correct": correct_logit > wrong_logit,
            "margin": correct_logit - wrong_logit,
        }


def generate_test_scenarios(n: int = 50) -> list:
    """Generate scenarios with baseline and bridged versions."""
    random.seed(42)
    
    AGENTS = ["Alice", "Bob", "Carol", "David"]
    INFORMERS = ["Eve", "Frank", "Grace", "Henry"]
    OBJECTS = ["ball", "key", "book", "toy"]
    LOCATIONS = ["drawer", "basket", "cupboard", "shelf"]
    
    scenarios = []
    for i in range(n):
        agent = random.choice(AGENTS)
        informer = random.choice(INFORMERS)
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
        })
    
    return scenarios


def main():
    timer_start = time.perf_counter()
    
    print("=" * 70)
    print("ABLATION TEST: Confirm Belief Update Circuit")
    print("=" * 70)
    print("""
    Hypothesis: L23H4 and L28H0 are the "belief update bridge" circuit.
    
    Test: Ablate these heads and see if BRIDGED prompts (normally 98%)
    drop significantly in accuracy.
    
    Expected results if hypothesis is correct:
    - Baseline: ~18% (with or without ablation - update circuit not used)
    - Bridged without ablation: ~98%
    - Bridged WITH ablation: Should DROP (circuit disabled)
    """)
    
    # Load model
    print("[1/5] Loading model...", flush=True)
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
    
    ablator = BridgeCircuitAblator(model, tokenizer)
    
    # Generate scenarios
    print("\n[2/5] Generating test scenarios...", flush=True)
    scenarios = generate_test_scenarios(50)
    print(f"  Generated {len(scenarios)} scenarios")
    
    # Define heads to ablate
    UPDATE_CIRCUIT_HEADS = [
        (23, 4),   # L23H4 - biggest attention change (+0.54)
        (28, 0),   # L28H0 - second biggest (+0.50)
    ]
    
    # Also test with more heads from our analysis
    EXTENDED_UPDATE_CIRCUIT = [
        (23, 4), (28, 0), (24, 29), (26, 26), (23, 30)
    ]
    
    # Control: ablate DIFFERENT heads (shouldn't affect bridged prompts)
    CONTROL_HEADS = [
        (5, 5), (10, 10)  # Random early/mid heads
    ]
    
    # Also test our original ToM heads
    EXPLICIT_PARSER_HEADS = [
        (12, 0), (23, 0)  # Should NOT affect bridged prompts
    ]
    
    results = {
        "conditions": {},
        "summary": {},
    }
    
    conditions = [
        ("no_ablation", []),
        ("update_circuit_core", UPDATE_CIRCUIT_HEADS),
        ("update_circuit_extended", EXTENDED_UPDATE_CIRCUIT),
        ("control_random", CONTROL_HEADS),
        ("explicit_parser", EXPLICIT_PARSER_HEADS),
    ]
    
    # Run tests
    for cond_name, heads_to_ablate in conditions:
        print(f"\n[3/5] Testing condition: {cond_name}...", flush=True)
        
        if heads_to_ablate:
            ablator.install_ablation(heads_to_ablate)
        else:
            ablator.clear_hooks()
        
        baseline_correct = 0
        bridged_correct = 0
        
        for i, scenario in enumerate(scenarios):
            if i % 20 == 0:
                print(f"  [{i}/{len(scenarios)}]", flush=True)
            
            # Test baseline
            result_base = ablator.test_prompt(
                scenario["baseline"], scenario["correct"], scenario["wrong"]
            )
            if result_base["chose_correct"]:
                baseline_correct += 1
            
            # Test bridged
            result_bridge = ablator.test_prompt(
                scenario["bridged"], scenario["correct"], scenario["wrong"]
            )
            if result_bridge["chose_correct"]:
                bridged_correct += 1
        
        ablator.clear_hooks()
        
        baseline_acc = baseline_correct / len(scenarios)
        bridged_acc = bridged_correct / len(scenarios)
        
        results["conditions"][cond_name] = {
            "heads_ablated": heads_to_ablate,
            "baseline_accuracy": baseline_acc,
            "bridged_accuracy": bridged_acc,
        }
        
        print(f"  {cond_name}: baseline={baseline_acc:.1%}, bridged={bridged_acc:.1%}")
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS: Update Circuit Ablation")
    print("=" * 70)
    
    print("\n  CONDITION               BASELINE    BRIDGED     DELTA")
    print("  " + "-" * 60)
    
    no_abl_bridge = results["conditions"]["no_ablation"]["bridged_accuracy"]
    
    for cond_name, data in results["conditions"].items():
        base_acc = data["baseline_accuracy"]
        bridge_acc = data["bridged_accuracy"]
        delta = bridge_acc - no_abl_bridge
        
        marker = ""
        if cond_name == "update_circuit_core" and delta < -0.20:
            marker = " <-- CIRCUIT CONFIRMED!"
        elif cond_name == "no_ablation":
            marker = " (baseline comparison)"
        
        print(f"  {cond_name:22s}: {base_acc:6.1%}     {bridge_acc:6.1%}     {delta:+6.1%}{marker}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    core_bridged = results["conditions"]["update_circuit_core"]["bridged_accuracy"]
    extended_bridged = results["conditions"]["update_circuit_extended"]["bridged_accuracy"]
    control_bridged = results["conditions"]["control_random"]["bridged_accuracy"]
    explicit_bridged = results["conditions"]["explicit_parser"]["bridged_accuracy"]
    
    print(f"\n  Without ablation (bridged):      {no_abl_bridge:.1%}")
    print(f"  With update circuit ablation:    {core_bridged:.1%} ({core_bridged - no_abl_bridge:+.1%})")
    print(f"  With extended circuit ablation:  {extended_bridged:.1%} ({extended_bridged - no_abl_bridge:+.1%})")
    print(f"  With control ablation:           {control_bridged:.1%} ({control_bridged - no_abl_bridge:+.1%})")
    print(f"  With explicit parser ablation:   {explicit_bridged:.1%} ({explicit_bridged - no_abl_bridge:+.1%})")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    core_drop = no_abl_bridge - core_bridged
    control_drop = no_abl_bridge - control_bridged
    
    if core_drop > 0.20 and control_drop < 0.10:
        print(f"""
    [HYPOTHESIS CONFIRMED] L23H4 and L28H0 ARE the belief update circuit!
    
    Evidence:
    - Ablating update circuit (L23H4, L28H0) drops bridged accuracy by {core_drop:.1%}
    - Ablating control heads has minimal effect ({control_drop:+.1%})
    - This is a SELECTIVE effect on the update circuit
    
    The "belief update bridge" works by activating L23H4 and L28H0.
    These heads are responsible for updating the belief representation.
        """)
    elif core_drop > 0.10:
        print(f"""
    [PARTIAL CONFIRMATION] 
    
    Update circuit ablation causes {core_drop:.1%} drop (moderate effect).
    Control ablation causes {control_drop:.1%} drop.
    
    L23H4 and L28H0 are PART of the update circuit, but not the whole story.
        """)
    else:
        print(f"""
    [HYPOTHESIS NOT CONFIRMED]
    
    Update circuit ablation only causes {core_drop:.1%} drop.
    The attention pattern analysis may have identified correlation, not causation.
    
    The belief update bridge may work through a different mechanism.
        """)
    
    # Save results
    with open(RESULTS_DIR / "update_circuit_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    total_time = time.perf_counter() - timer_start
    print(f"\n[OK] Completed in {total_time:.1f}s")
    print(f"     Saved to {RESULTS_DIR / 'update_circuit_ablation_results.json'}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

