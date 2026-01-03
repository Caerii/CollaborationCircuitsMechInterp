"""
Step 12: Signal Injection Test

Now that we know L18H11 CARRIES the belief-update signal,
can we directly INJECT an "update" signal without needing the bridge phrase?

Method:
1. Compute the DIFFERENCE between clean and corrupted activations at L18H11
2. This difference IS the "belief update" signal
3. Inject this signal into corrupted prompts
4. Measure if it restores correct belief predictions

This would prove we've isolated the actual causal mechanism.
"""

import torch
import json
import random
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import numpy as np

# Setup
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Key decision heads from path patching
DECISION_HEADS = [
    (18, 11),  # 36% restoration - strongest
    (17, 4),   # 22% restoration
    (18, 14),  # 4% restoration
]

BRIDGES = [
    "so {agent} updated their belief about the {obj}'s location",
    "therefore {agent} now knows the {obj} is in the {loc2}",
]


def generate_scenario(seed: int):
    """Generate a scenario for signal extraction/injection."""
    random.seed(seed)
    
    agents = ["Alice", "Bob", "Carol", "David", "Eve", "Frank"]
    informers = ["Iris", "Jack", "Kate", "Leo", "Mia", "Nick"]
    objects = ["ball", "book", "key", "toy", "phone", "wallet"]
    locations = ["basket", "box", "drawer", "shelf", "cabinet", "chest"]
    verbs = ["tells", "informs", "says to"]
    
    agent = random.choice(agents)
    informer = random.choice(informers)
    obj = random.choice(objects)
    loc1, loc2 = random.sample(locations, 2)
    verb = random.choice(verbs)
    
    corrupted_prompt = (
        f"{agent} put the {obj} in the {loc1}. "
        f"{informer} {verb} {agent}: 'I moved the {obj} to the {loc2}.' "
        f"Where will {agent} look for the {obj}? {agent} will look in the"
    )
    
    bridge = random.choice(BRIDGES).format(agent=agent, obj=obj, loc2=loc2)
    clean_prompt = (
        f"{agent} put the {obj} in the {loc1}. "
        f"{informer} {verb} {agent}: 'I moved the {obj} to the {loc2}.' "
        f"{bridge}. "
        f"Where will {agent} look for the {obj}? {agent} will look in the"
    )
    
    return {
        'corrupted': corrupted_prompt,
        'clean': clean_prompt,
        'agent': agent,
        'obj': obj,
        'loc1': loc1,
        'loc2': loc2,
    }


class SignalInjector:
    """Extract and inject the belief-update signal."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_heads = model.config.num_attention_heads
        self.hooks = []
        self.captured_activations = {}
        
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def capture_activations(self, prompt: str, layers: list) -> dict:
        """Run prompt and capture activations at specified layers."""
        self.clear_hooks()
        self.captured_activations = {}
        
        def make_capture_hook(layer_idx):
            def hook(module, args):
                hidden_states = args[0].clone().detach()
                self.captured_activations[layer_idx] = hidden_states
                return args
            return hook
        
        for layer_idx in layers:
            o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
            hook = o_proj.register_forward_pre_hook(make_capture_hook(layer_idx))
            self.hooks.append(hook)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            self.model(**inputs)
        
        self.clear_hooks()
        return self.captured_activations.copy()
    
    def extract_update_signal(self, clean_acts: dict, corrupted_acts: dict, 
                               layer: int, head: int) -> torch.Tensor:
        """
        Extract the "belief update" signal as the difference between clean and corrupted.
        
        Returns the signal vector for the last token position.
        """
        clean = clean_acts[layer]
        corrupted = corrupted_acts[layer]
        
        # Get dimensions
        batch, seq_len_clean, hidden = clean.shape
        _, seq_len_corrupt, _ = corrupted.shape
        head_dim = hidden // self.n_heads
        
        # Reshape to access individual heads
        clean_reshaped = clean.view(batch, seq_len_clean, self.n_heads, head_dim)
        corrupt_reshaped = corrupted.view(batch, seq_len_corrupt, self.n_heads, head_dim)
        
        # Get the head output at the LAST token (where prediction happens)
        clean_head = clean_reshaped[0, -1, head, :]  # [head_dim]
        corrupt_head = corrupt_reshaped[0, -1, head, :]  # [head_dim]
        
        # The "update signal" is the difference
        update_signal = clean_head - corrupt_head
        
        return update_signal
    
    def run_with_injection(self, corrupted_prompt: str, layer: int, head: int,
                           signal: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """
        Run corrupted prompt with injected signal at the specified head.
        """
        self.clear_hooks()
        
        n_heads = self.n_heads
        
        def make_injection_hook(target_layer, target_head, inject_signal, inject_scale):
            def hook(module, args):
                hidden_states = args[0]
                batch, seq_len, hidden = hidden_states.shape
                head_dim = hidden // n_heads
                
                reshaped = hidden_states.view(batch, seq_len, n_heads, head_dim)
                
                # Inject signal at the LAST token position
                reshaped[:, -1, target_head, :] = (
                    reshaped[:, -1, target_head, :] + inject_signal.to(hidden_states.device) * inject_scale
                )
                
                new_hidden = reshaped.view(batch, seq_len, hidden)
                
                if len(args) > 1:
                    return (new_hidden,) + args[1:]
                return (new_hidden,)
            return hook
        
        o_proj = self.model.model.layers[layer].self_attn.o_proj
        hook = o_proj.register_forward_pre_hook(
            make_injection_hook(layer, head, signal, scale)
        )
        self.hooks.append(hook)
        
        inputs = self.tokenizer(corrupted_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        self.clear_hooks()
        return outputs.logits
    
    def run_baseline(self, prompt: str) -> torch.Tensor:
        """Run prompt without any modification."""
        self.clear_hooks()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits
    
    def get_location_probs(self, logits: torch.Tensor, loc1: str, loc2: str) -> dict:
        """Get probabilities for the two location tokens."""
        last_logits = logits[0, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        
        loc1_id = self.tokenizer.encode(" " + loc1, add_special_tokens=False)[0]
        loc2_id = self.tokenizer.encode(" " + loc2, add_special_tokens=False)[0]
        
        return {
            'loc1_prob': probs[loc1_id].item(),
            'loc2_prob': probs[loc2_id].item(),
            'predicts_correct': probs[loc2_id] > probs[loc1_id],
        }


def run_signal_injection_experiment(model, tokenizer, n_scenarios: int = 50):
    """
    Main experiment: Extract update signal and inject it.
    """
    injector = SignalInjector(model, tokenizer)
    
    # Layers we need
    layers = list(set(layer for layer, head in DECISION_HEADS))
    
    print(f"\n{'='*60}")
    print("SIGNAL INJECTION EXPERIMENT")
    print(f"{'='*60}")
    print(f"Testing {n_scenarios} scenarios")
    print(f"Target heads: {DECISION_HEADS}")
    print()
    
    results = {
        'baseline_corrupted': [],
        'baseline_clean': [],
        'injection_results': {f"L{l}H{h}": [] for l, h in DECISION_HEADS},
        'combined_injection': [],
        'scaled_injection': defaultdict(list),
    }
    
    # First, extract signals from multiple scenarios to get a robust estimate
    print("Phase 1: Extracting update signals...")
    signals = {(l, h): [] for l, h in DECISION_HEADS}
    
    for i in range(min(20, n_scenarios)):  # Use 20 scenarios to compute average signal
        scenario = generate_scenario(seed=42 + i)
        
        clean_acts = injector.capture_activations(scenario['clean'], layers)
        corrupted_acts = injector.capture_activations(scenario['corrupted'], layers)
        
        for layer, head in DECISION_HEADS:
            signal = injector.extract_update_signal(clean_acts, corrupted_acts, layer, head)
            signals[(layer, head)].append(signal)
    
    # Compute mean signal for each head
    mean_signals = {}
    for (layer, head), signal_list in signals.items():
        stacked = torch.stack(signal_list)
        mean_signal = stacked.mean(dim=0)
        mean_signals[(layer, head)] = mean_signal
        print(f"L{layer}H{head} signal norm: {mean_signal.norm().item():.4f}")
    
    # Phase 2: Test injection
    print(f"\nPhase 2: Testing injection on {n_scenarios} scenarios...")
    
    for i in range(n_scenarios):
        scenario = generate_scenario(seed=100 + i)
        
        if i % 10 == 0:
            print(f"Processing scenario {i+1}/{n_scenarios}...")
        
        # Baseline
        clean_logits = injector.run_baseline(scenario['clean'])
        corrupted_logits = injector.run_baseline(scenario['corrupted'])
        
        clean_result = injector.get_location_probs(clean_logits, scenario['loc1'], scenario['loc2'])
        corrupted_result = injector.get_location_probs(corrupted_logits, scenario['loc1'], scenario['loc2'])
        
        results['baseline_clean'].append(clean_result['predicts_correct'])
        results['baseline_corrupted'].append(corrupted_result['predicts_correct'])
        
        # Test single-head injection
        for layer, head in DECISION_HEADS:
            signal = mean_signals[(layer, head)]
            injected_logits = injector.run_with_injection(
                scenario['corrupted'], layer, head, signal, scale=1.0
            )
            injected_result = injector.get_location_probs(
                injected_logits, scenario['loc1'], scenario['loc2']
            )
            results['injection_results'][f"L{layer}H{head}"].append(
                injected_result['predicts_correct']
            )
        
        # Test combined injection (all decision heads)
        combined_logits = run_combined_injection(
            injector, scenario['corrupted'], DECISION_HEADS, mean_signals
        )
        combined_result = injector.get_location_probs(
            combined_logits, scenario['loc1'], scenario['loc2']
        )
        results['combined_injection'].append(combined_result['predicts_correct'])
        
        # Test different scales for strongest head (L18H11)
        for scale in [0.5, 1.0, 1.5, 2.0]:
            scaled_logits = injector.run_with_injection(
                scenario['corrupted'], 18, 11, mean_signals[(18, 11)], scale=scale
            )
            scaled_result = injector.get_location_probs(
                scaled_logits, scenario['loc1'], scenario['loc2']
            )
            results['scaled_injection'][scale].append(scaled_result['predicts_correct'])
    
    # Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    
    clean_acc = sum(results['baseline_clean']) / len(results['baseline_clean'])
    corrupt_acc = sum(results['baseline_corrupted']) / len(results['baseline_corrupted'])
    
    print(f"\nBaselines:")
    print(f"  Clean (with bridge): {clean_acc*100:.1f}%")
    print(f"  Corrupted (no bridge): {corrupt_acc*100:.1f}%")
    
    print(f"\nSingle-Head Signal Injection:")
    for head_name, correct_list in results['injection_results'].items():
        acc = sum(correct_list) / len(correct_list)
        boost = acc - corrupt_acc
        print(f"  {head_name}: {acc*100:.1f}% (boost: {boost*100:+.1f}%)")
    
    combined_acc = sum(results['combined_injection']) / len(results['combined_injection'])
    print(f"\nCombined Injection (all 3 heads):")
    print(f"  Accuracy: {combined_acc*100:.1f}% (boost: {(combined_acc-corrupt_acc)*100:+.1f}%)")
    
    print(f"\nScaled Injection (L18H11 only):")
    for scale in [0.5, 1.0, 1.5, 2.0]:
        acc = sum(results['scaled_injection'][scale]) / len(results['scaled_injection'][scale])
        print(f"  Scale {scale}x: {acc*100:.1f}% (boost: {(acc-corrupt_acc)*100:+.1f}%)")
    
    return results


def run_combined_injection(injector, prompt, heads, signals):
    """Inject signals into multiple heads simultaneously."""
    injector.clear_hooks()
    
    n_heads = injector.n_heads
    
    def make_multi_injection_hook(layer, head_signal_pairs):
        def hook(module, args):
            hidden_states = args[0]
            batch, seq_len, hidden = hidden_states.shape
            head_dim = hidden // n_heads
            
            reshaped = hidden_states.view(batch, seq_len, n_heads, head_dim)
            
            for head, signal in head_signal_pairs:
                reshaped[:, -1, head, :] = (
                    reshaped[:, -1, head, :] + signal.to(hidden_states.device)
                )
            
            new_hidden = reshaped.view(batch, seq_len, hidden)
            
            if len(args) > 1:
                return (new_hidden,) + args[1:]
            return (new_hidden,)
        return hook
    
    # Group by layer
    layer_to_heads = defaultdict(list)
    for (layer, head) in heads:
        layer_to_heads[layer].append((head, signals[(layer, head)]))
    
    for layer, head_signal_pairs in layer_to_heads.items():
        o_proj = injector.model.model.layers[layer].self_attn.o_proj
        hook = o_proj.register_forward_pre_hook(
            make_multi_injection_hook(layer, head_signal_pairs)
        )
        injector.hooks.append(hook)
    
    inputs = injector.tokenizer(prompt, return_tensors="pt").to(injector.model.device)
    with torch.no_grad():
        outputs = injector.model(**inputs)
    
    injector.clear_hooks()
    return outputs.logits


def main():
    print("="*60)
    print("STEP 12: SIGNAL INJECTION TEST")
    print("="*60)
    print()
    print("Can we directly inject the 'belief update' signal?")
    print("This would prove we've isolated the causal mechanism.")
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
    print(f"Model loaded: {model_name}")
    
    # Run experiment
    results = run_signal_injection_experiment(model, tokenizer, n_scenarios=50)
    
    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    
    corrupt_acc = sum(results['baseline_corrupted']) / len(results['baseline_corrupted'])
    best_injection = max(
        results['injection_results'].items(),
        key=lambda x: sum(x[1]) / len(x[1])
    )
    best_head, best_results = best_injection
    best_acc = sum(best_results) / len(best_results)
    
    if best_acc > corrupt_acc + 0.1:
        print(f"\nSUCCESS: Signal injection works!")
        print(f"Best head: {best_head} achieves {best_acc*100:.1f}% (+{(best_acc-corrupt_acc)*100:.1f}%)")
        print("\nThis PROVES we've isolated the causal mechanism.")
        print("The 'belief update' signal is a learnable, transferable vector.")
    else:
        print(f"\nMixed results: Injection provides limited boost ({(best_acc-corrupt_acc)*100:.1f}%)")
        print("The signal may be more complex than a simple additive vector.")
    
    # Save results
    output_file = RESULTS_DIR / "signal_injection_results.json"
    
    # Convert to serializable format
    serializable_results = {
        'baseline_clean': [bool(x) for x in results['baseline_clean']],
        'baseline_corrupted': [bool(x) for x in results['baseline_corrupted']],
        'injection_results': {k: [bool(x) for x in v] for k, v in results['injection_results'].items()},
        'combined_injection': [bool(x) for x in results['combined_injection']],
        'scaled_injection': {str(k): [bool(x) for x in v] for k, v in results['scaled_injection'].items()},
    }
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    print(f"\n{'='*60}")
    print("SIGNAL INJECTION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

