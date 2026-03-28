"""
Step 14: Maximize Implicit Multi-Agent ToM

Current best: 48% on implicit_comm (from 24% baseline)
Goal: Can we push higher by combining interventions?

Strategies to test:
1. Ablate MORE inhibitors (L17H4, L18H11, L18H14, L19H30)
2. Amplify enablers (L19H2, L19H15)
3. COMBINE ablation + amplification
4. Find the optimal intervention for max implicit ToM
"""

import torch
import json
import random
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# All known circuit components
INHIBITORS = [(17, 4), (18, 11), (18, 14), (19, 30), (21, 17)]
ENABLERS = [(15, 9), (19, 2), (19, 15)]


class MaxToMTester:
    """Test combined interventions for maximum ToM."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_heads = model.config.num_attention_heads
        self.hooks = []
        
    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def install_ablation(self, heads: list):
        """Zero out specified heads."""
        n_heads = self.n_heads
        layers_to_heads = defaultdict(list)
        for l, h in heads:
            layers_to_heads[l].append(h)
        
        def make_hook(head_list):
            def hook(module, args):
                hidden = args[0]
                batch, seq, dim = hidden.shape
                head_dim = dim // n_heads
                reshaped = hidden.view(batch, seq, n_heads, head_dim)
                for h in head_list:
                    reshaped[:, :, h, :] = 0
                return (reshaped.view(batch, seq, dim),)
            return hook
        
        for layer, head_list in layers_to_heads.items():
            o_proj = self.model.model.layers[layer].self_attn.o_proj
            h = o_proj.register_forward_pre_hook(make_hook(head_list))
            self.hooks.append(h)
    
    def install_amplification(self, heads_scales: list):
        """Amplify specified heads by given scales. heads_scales = [(layer, head, scale), ...]"""
        n_heads = self.n_heads
        
        # Group by layer
        layers_to_heads = defaultdict(list)
        for l, h, s in heads_scales:
            layers_to_heads[l].append((h, s))
        
        def make_hook(head_scale_pairs):
            def hook(module, args):
                hidden = args[0]
                batch, seq, dim = hidden.shape
                head_dim = dim // n_heads
                reshaped = hidden.view(batch, seq, n_heads, head_dim)
                for h, s in head_scale_pairs:
                    reshaped[:, :, h, :] = reshaped[:, :, h, :] * s
                return (reshaped.view(batch, seq, dim),)
            return hook
        
        for layer, head_scale_pairs in layers_to_heads.items():
            o_proj = self.model.model.layers[layer].self_attn.o_proj
            h = o_proj.register_forward_pre_hook(make_hook(head_scale_pairs))
            self.hooks.append(h)
    
    def install_combined(self, ablate_heads: list, amplify_heads_scales: list):
        """Combined ablation and amplification."""
        n_heads = self.n_heads
        
        # Merge all layers
        all_layers = set()
        ablate_by_layer = defaultdict(list)
        amplify_by_layer = defaultdict(list)
        
        for l, h in ablate_heads:
            ablate_by_layer[l].append(h)
            all_layers.add(l)
        
        for l, h, s in amplify_heads_scales:
            amplify_by_layer[l].append((h, s))
            all_layers.add(l)
        
        def make_combined_hook(layer):
            abl = ablate_by_layer.get(layer, [])
            amp = amplify_by_layer.get(layer, [])
            
            def hook(module, args):
                hidden = args[0]
                batch, seq, dim = hidden.shape
                head_dim = dim // n_heads
                reshaped = hidden.view(batch, seq, n_heads, head_dim)
                
                # Ablate
                for h in abl:
                    reshaped[:, :, h, :] = 0
                
                # Amplify
                for h, s in amp:
                    reshaped[:, :, h, :] = reshaped[:, :, h, :] * s
                
                return (reshaped.view(batch, seq, dim),)
            return hook
        
        for layer in all_layers:
            o_proj = self.model.model.layers[layer].self_attn.o_proj
            h = o_proj.register_forward_pre_hook(make_combined_hook(layer))
            self.hooks.append(h)
    
    def get_probs(self, prompt: str, correct: str, wrong: str) -> dict:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        c_id = self.tokenizer.encode(correct, add_special_tokens=False)[0]
        w_id = self.tokenizer.encode(wrong, add_special_tokens=False)[0]
        return {
            'correct_prob': probs[c_id].item(),
            'wrong_prob': probs[w_id].item(),
            'predicts_correct': probs[c_id] > probs[w_id],
        }
    
    def test_intervention(self, scenarios, name="test"):
        """Test a single intervention configuration."""
        correct = 0
        for s in scenarios:
            result = self.get_probs(s['prompt'], s['correct'], s['wrong'])
            if result['predicts_correct']:
                correct += 1
        return correct / len(scenarios)


def generate_implicit_scenarios(n: int = 50):
    """Generate implicit multi-agent scenarios."""
    scenarios = []
    random.seed(42)
    
    agents = ["Alice", "Bob", "Carol", "David"]
    objects = ["ball", "book", "key", "toy"]
    locations = ["basket", "box", "drawer", "shelf"]
    
    for i in range(n):
        a, b = random.sample(agents, 2)
        obj = random.choice(objects)
        loc1, loc2 = random.sample(locations, 2)
        
        prompt = (
            f"{a} put the {obj} in the {loc1}. "
            f"{a} left the room. "
            f"{b} moved the {obj} to the {loc2}. "
            f"{b} called {a} and said: 'Hey, I moved the {obj} to the {loc2}.' "
            f"When {a} returns, {a} will look for the {obj} in the"
        )
        
        scenarios.append({
            'prompt': prompt,
            'correct': f" {loc2}",
            'wrong': f" {loc1}",
        })
    
    return scenarios


def run_optimization(model, tokenizer):
    """Test various intervention combinations."""
    tester = MaxToMTester(model, tokenizer)
    scenarios = generate_implicit_scenarios(50)
    
    results = {}
    
    print(f"\n{'='*70}")
    print("MAXIMIZING IMPLICIT MULTI-AGENT ToM")
    print(f"{'='*70}")
    print(f"Testing {len(scenarios)} scenarios")
    print()
    
    # 1. Baseline
    tester.clear_hooks()
    baseline = tester.test_intervention(scenarios, "baseline")
    results['baseline'] = baseline
    print(f"Baseline:                               {baseline*100:5.1f}%")
    
    # 2. Previous best: L17H4 + L18H14
    tester.clear_hooks()
    tester.install_ablation([(17, 4), (18, 14)])
    prev_best = tester.test_intervention(scenarios, "prev_best")
    results['ablate_2_inhibitors'] = prev_best
    print(f"Ablate L17H4+L18H14 (prev best):        {prev_best*100:5.1f}%")
    tester.clear_hooks()
    
    # 3. Add L18H11
    tester.install_ablation([(17, 4), (18, 11), (18, 14)])
    three_inh = tester.test_intervention(scenarios, "3_inhibitors")
    results['ablate_3_inhibitors'] = three_inh
    print(f"Ablate L17H4+L18H11+L18H14:             {three_inh*100:5.1f}%")
    tester.clear_hooks()
    
    # 4. All 5 inhibitors
    tester.install_ablation(INHIBITORS)
    all_inh = tester.test_intervention(scenarios, "all_inhibitors")
    results['ablate_5_inhibitors'] = all_inh
    print(f"Ablate ALL 5 inhibitors:                {all_inh*100:5.1f}%")
    tester.clear_hooks()
    
    # 5. Amplify L19H2 only
    tester.install_amplification([(19, 2, 2.0)])
    amp_19h2 = tester.test_intervention(scenarios, "amp_l19h2")
    results['amplify_L19H2_2x'] = amp_19h2
    print(f"Amplify L19H2 (2x):                     {amp_19h2*100:5.1f}%")
    tester.clear_hooks()
    
    # 6. Amplify L19H2 more
    tester.install_amplification([(19, 2, 3.0)])
    amp_19h2_3 = tester.test_intervention(scenarios, "amp_l19h2_3x")
    results['amplify_L19H2_3x'] = amp_19h2_3
    print(f"Amplify L19H2 (3x):                     {amp_19h2_3*100:5.1f}%")
    tester.clear_hooks()
    
    # 7. COMBINED: Ablate inhibitors + Amplify enabler
    tester.install_combined(
        ablate_heads=[(17, 4), (18, 14)],
        amplify_heads_scales=[(19, 2, 2.0)]
    )
    combined_1 = tester.test_intervention(scenarios, "combined_1")
    results['ablate_2_plus_amp_L19H2'] = combined_1
    print(f"Ablate 2 + Amplify L19H2 (2x):          {combined_1*100:5.1f}%")
    tester.clear_hooks()
    
    # 8. Stronger combined
    tester.install_combined(
        ablate_heads=[(17, 4), (18, 11), (18, 14)],
        amplify_heads_scales=[(19, 2, 2.0), (19, 15, 1.5)]
    )
    combined_2 = tester.test_intervention(scenarios, "combined_2")
    results['ablate_3_plus_amp_2_enablers'] = combined_2
    print(f"Ablate 3 + Amplify L19H2,H15:           {combined_2*100:5.1f}%")
    tester.clear_hooks()
    
    # 9. Maximum intervention
    tester.install_combined(
        ablate_heads=INHIBITORS,
        amplify_heads_scales=[(19, 2, 2.0), (19, 15, 2.0)]
    )
    maximum = tester.test_intervention(scenarios, "maximum")
    results['maximum_intervention'] = maximum
    print(f"Ablate ALL + Amplify 2 enablers (2x):   {maximum*100:5.1f}%")
    tester.clear_hooks()
    
    # 10. Try different amplification scales
    print(f"\nScale sweep for combined (ablate 3 + amp L19H2):")
    for scale in [1.5, 2.0, 2.5, 3.0, 4.0]:
        tester.install_combined(
            ablate_heads=[(17, 4), (18, 11), (18, 14)],
            amplify_heads_scales=[(19, 2, scale)]
        )
        acc = tester.test_intervention(scenarios, f"scale_{scale}")
        results[f'ablate_3_amp_scale_{scale}'] = acc
        print(f"  Scale {scale}x: {acc*100:5.1f}%")
        tester.clear_hooks()
    
    return results


def main():
    print("="*70)
    print("STEP 14: MAXIMIZE IMPLICIT MULTI-AGENT ToM")
    print("="*70)
    print()
    print("Current best: 48% (ablate L17H4+L18H14)")
    print("Goal: Find the optimal intervention to push higher")
    print()
    
    # Load model
    print("Loading model...")
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    results = run_optimization(model, tokenizer)
    
    # Find best
    best_name = max(results.items(), key=lambda x: x[1])
    
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\nBaseline:     {results['baseline']*100:.1f}%")
    print(f"Best result:  {best_name[1]*100:.1f}% ({best_name[0]})")
    print(f"Improvement:  +{(best_name[1] - results['baseline'])*100:.1f}%")
    
    if best_name[1] > 0.6:
        print(f"\n*** EXCELLENT: Achieved >{60}% on implicit multi-agent! ***")
    elif best_name[1] > 0.5:
        print(f"\n*** GOOD: Achieved >{50}% on implicit multi-agent ***")
    
    # Save
    output_file = RESULTS_DIR / "maximize_implicit_tom_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()

