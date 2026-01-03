"""
Step 11: Path Patching Through ToM Circuit

This is the gold standard in mechanistic interpretability.
We trace HOW information flows through the discovered ToM circuit.

Method:
1. CLEAN run: Prompt with bridging phrase -> model gets ToM correct
2. CORRUPTED run: Same prompt WITHOUT bridge -> model gets ToM wrong  
3. PATCH: At specific components, replace corrupted activations with clean
4. MEASURE: Does patching at component X restore correct ToM?

Key heads to test:
- L15H9  (enabler - essential for ToM)
- L17H4  (inhibitor - ablating helps)
- L18H11 (paradoxical inhibitor)
- L18H14 (inhibitor - ablating helps)
- L19H2  (enabler - essential)
- L19H15 (enabler - essential)

Expected insights:
- Which heads carry the "belief update" signal?
- Where does the inhibitory veto occur?
- What's the causal pathway for ToM?
"""

import torch
import json
import random
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

# Setup
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Key heads from our discovery
KEY_HEADS = {
    'enablers': [(15, 9), (19, 2), (19, 15)],
    'inhibitors': [(17, 4), (18, 11), (18, 14)],
}

# Bridging phrases that work
BRIDGES = [
    "so {agent} updated their belief about the {obj}'s location",
    "therefore {agent} now knows the {obj} is in the {loc2}",
]


def generate_scenario_pair(seed: int):
    """Generate a clean/corrupted prompt pair for path patching.
    
    Uses EXACT same structure as our working fixed_scenarios.
    """
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
    
    # Base prompt (corrupted - matches our working baseline)
    # This is the IMPLICIT version where belief update must be inferred
    corrupted_prompt = (
        f"{agent} put the {obj} in the {loc1}. "
        f"{informer} {verb} {agent}: 'I moved the {obj} to the {loc2}.' "
        f"Where will {agent} look for the {obj}? {agent} will look in the"
    )
    
    # Clean prompt (with bridging phrase that makes belief update explicit)
    bridge = random.choice(BRIDGES).format(agent=agent, obj=obj, loc2=loc2)
    clean_prompt = (
        f"{agent} put the {obj} in the {loc1}. "
        f"{informer} {verb} {agent}: 'I moved the {obj} to the {loc2}.' "
        f"{bridge}. "  # Explicit bridge
        f"Where will {agent} look for the {obj}? {agent} will look in the"
    )
    
    return {
        'corrupted': corrupted_prompt,
        'clean': clean_prompt,
        'agent': agent,
        'obj': obj,
        'loc1': loc1,  # Original (wrong answer)
        'loc2': loc2,  # Updated (correct answer)
    }


class PathPatcher:
    """
    Path patching for mechanistic interpretability.
    
    Method:
    1. Run model on clean input, cache activations
    2. Run model on corrupted input, patch in clean activations at target
    3. Measure if patching restores correct behavior
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads
        self.hidden_size = model.config.hidden_size
        self.head_dim = self.hidden_size // self.n_heads
        self.cached_activations = {}
        self.hooks = []
        
        print(f"PathPatcher initialized: hidden_size={self.hidden_size}, n_heads={self.n_heads}, head_dim={self.head_dim}")
        
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.cached_activations = {}
    
    def cache_clean_activations(self, clean_prompt: str, layers_to_cache: list):
        """
        Run clean prompt and cache activations at specified layers.
        """
        self.clear_hooks()
        self.cached_activations = {}
        
        def make_cache_hook(layer_idx):
            def hook(module, args):
                # Cache the input to o_proj (attention output before projection)
                hidden_states = args[0].clone().detach()
                self.cached_activations[layer_idx] = hidden_states
                return args
            return hook
        
        # Install caching hooks
        for layer_idx in layers_to_cache:
            o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
            hook = o_proj.register_forward_pre_hook(make_cache_hook(layer_idx))
            self.hooks.append(hook)
        
        # Run clean forward pass
        inputs = self.tokenizer(clean_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            self.model(**inputs)
        
        # Keep hooks cleared but keep cached activations
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        return self.cached_activations.copy()
    
    def run_with_patch(self, corrupted_prompt: str, layer_idx: int, head_idx: int, 
                       clean_activations: dict) -> dict:
        """
        Run corrupted prompt, but patch in clean activation for a specific head.
        
        Returns log probs for answer tokens.
        """
        self.clear_hooks()
        
        # Capture instance variables for closure
        n_heads = self.n_heads
        
        def make_patch_hook(layer, head):
            def hook(module, args):
                if layer not in clean_activations:
                    return args
                    
                hidden_states = args[0]
                batch, seq_len, hidden = hidden_states.shape
                
                # Calculate head_dim from actual tensor (may differ from config due to GQA)
                head_dim = hidden // n_heads
                
                clean = clean_activations[layer]
                
                # Handle sequence length mismatch (corrupted may be shorter)
                min_seq = min(seq_len, clean.shape[1])
                
                # Reshape to access individual heads
                reshaped = hidden_states.view(batch, seq_len, n_heads, head_dim)
                clean_reshaped = clean.view(clean.shape[0], clean.shape[1], n_heads, head_dim)
                
                # Patch only the target head
                reshaped[:, :min_seq, head, :] = clean_reshaped[:, :min_seq, head, :]
                
                new_hidden = reshaped.view(batch, seq_len, hidden)
                
                if len(args) > 1:
                    return (new_hidden,) + args[1:]
                return (new_hidden,)
            return hook
        
        # Install patching hook
        o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
        hook = o_proj.register_forward_pre_hook(make_patch_hook(layer_idx, head_idx))
        self.hooks.append(hook)
        
        # Run corrupted forward pass with patching
        inputs = self.tokenizer(corrupted_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        self.clear_hooks()
        
        return outputs.logits
    
    def run_without_patch(self, prompt: str) -> torch.Tensor:
        """Run prompt without any patching."""
        self.clear_hooks()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits
    
    def get_location_probs(self, logits: torch.Tensor, loc1: str, loc2: str) -> dict:
        """Get probabilities for the two location tokens."""
        last_logits = logits[0, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        
        # Get token IDs (try with and without space prefix)
        loc1_id = self.tokenizer.encode(" " + loc1, add_special_tokens=False)[0]
        loc2_id = self.tokenizer.encode(" " + loc2, add_special_tokens=False)[0]
        
        return {
            'loc1_prob': probs[loc1_id].item(),
            'loc2_prob': probs[loc2_id].item(),
            'predicts_correct': probs[loc2_id] > probs[loc1_id],
        }


def run_path_patching_experiment(model, tokenizer, n_scenarios: int = 50):
    """
    Main path patching experiment.
    
    For each scenario:
    1. Run clean (bridged) - should be correct
    2. Run corrupted (no bridge) - should be wrong
    3. Patch each key head from clean -> corrupted
    4. Measure which patches restore correctness
    """
    patcher = PathPatcher(model, tokenizer)
    
    # All layers we need to cache
    all_layers = set()
    for heads in KEY_HEADS.values():
        for layer, head in heads:
            all_layers.add(layer)
    all_layers = sorted(list(all_layers))
    
    results = {
        'scenarios': [],
        'head_patch_effects': defaultdict(list),
        'summary': {}
    }
    
    print(f"\n{'='*60}")
    print("PATH PATCHING EXPERIMENT")
    print(f"{'='*60}")
    print(f"Testing {n_scenarios} scenarios")
    print(f"Layers to patch: {all_layers}")
    print(f"Key heads: {KEY_HEADS}")
    print()
    
    for i in range(n_scenarios):
        scenario = generate_scenario_pair(seed=42 + i)
        
        if i % 10 == 0:
            print(f"Processing scenario {i+1}/{n_scenarios}...")
        
        # Step 1: Cache clean activations
        clean_acts = patcher.cache_clean_activations(scenario['clean'], all_layers)
        
        # Step 2: Get baseline (no patch) results
        clean_logits = patcher.run_without_patch(scenario['clean'])
        corrupted_logits = patcher.run_without_patch(scenario['corrupted'])
        
        clean_result = patcher.get_location_probs(clean_logits, scenario['loc1'], scenario['loc2'])
        corrupted_result = patcher.get_location_probs(corrupted_logits, scenario['loc1'], scenario['loc2'])
        
        scenario_result = {
            'idx': i,
            'clean_correct': clean_result['predicts_correct'],
            'corrupted_correct': corrupted_result['predicts_correct'],
            'patch_results': {}
        }
        
        # Step 3: Test patching each key head
        for head_type, heads in KEY_HEADS.items():
            for layer, head in heads:
                head_name = f"L{layer}H{head}"
                
                patched_logits = patcher.run_with_patch(
                    scenario['corrupted'], 
                    layer, head, 
                    clean_acts
                )
                patched_result = patcher.get_location_probs(
                    patched_logits, 
                    scenario['loc1'], 
                    scenario['loc2']
                )
                
                # Did patching restore correctness?
                restored = patched_result['predicts_correct'] and not corrupted_result['predicts_correct']
                
                scenario_result['patch_results'][head_name] = {
                    'type': head_type,
                    'patched_correct': patched_result['predicts_correct'],
                    'restored': restored,
                    'loc2_prob_change': patched_result['loc2_prob'] - corrupted_result['loc2_prob'],
                }
                
                results['head_patch_effects'][head_name].append({
                    'restored': restored,
                    'prob_change': patched_result['loc2_prob'] - corrupted_result['loc2_prob'],
                })
        
        results['scenarios'].append(scenario_result)
    
    # Compute summary statistics
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    
    # Baseline accuracies
    clean_acc = sum(s['clean_correct'] for s in results['scenarios']) / len(results['scenarios'])
    corrupted_acc = sum(s['corrupted_correct'] for s in results['scenarios']) / len(results['scenarios'])
    
    print(f"\nBaseline Accuracies:")
    print(f"  Clean (with bridge): {clean_acc*100:.1f}%")
    print(f"  Corrupted (no bridge): {corrupted_acc*100:.1f}%")
    print(f"  Gap: {(clean_acc - corrupted_acc)*100:.1f}%")
    
    # Head patch effects
    print(f"\n{'='*60}")
    print("HEAD PATCH EFFECTS")
    print("(Patching clean -> corrupted at each head)")
    print(f"{'='*60}")
    
    head_summaries = {}
    
    print(f"\n{'Head':<10} {'Type':<12} {'Restoration %':<15} {'Avg Prob Change':<15}")
    print("-" * 55)
    
    for head_name, effects in sorted(results['head_patch_effects'].items()):
        restoration_rate = sum(e['restored'] for e in effects) / len(effects)
        avg_prob_change = sum(e['prob_change'] for e in effects) / len(effects)
        
        # Determine type
        layer = int(head_name.split('H')[0][1:])
        head = int(head_name.split('H')[1])
        head_type = 'inhibitor' if (layer, head) in KEY_HEADS['inhibitors'] else 'enabler'
        
        print(f"{head_name:<10} {head_type:<12} {restoration_rate*100:>10.1f}%     {avg_prob_change:>+.4f}")
        
        head_summaries[head_name] = {
            'type': head_type,
            'restoration_rate': restoration_rate,
            'avg_prob_change': avg_prob_change,
        }
    
    results['summary'] = {
        'clean_accuracy': clean_acc,
        'corrupted_accuracy': corrupted_acc,
        'gap': clean_acc - corrupted_acc,
        'head_summaries': head_summaries,
    }
    
    return results


def run_layer_patching(model, tokenizer, n_scenarios: int = 30):
    """
    Patch entire layers (all heads) to find which layers carry ToM signal.
    """
    patcher = PathPatcher(model, tokenizer)
    
    # Test layers around our zone of interest
    test_layers = list(range(12, 25))
    
    print(f"\n{'='*60}")
    print("LAYER-LEVEL PATH PATCHING")
    print(f"{'='*60}")
    print(f"Testing layers {test_layers[0]}-{test_layers[-1]}")
    print()
    
    results = defaultdict(list)
    
    for i in range(n_scenarios):
        scenario = generate_scenario_pair(seed=100 + i)
        
        if i % 10 == 0:
            print(f"Processing scenario {i+1}/{n_scenarios}...")
        
        # Cache all test layers
        clean_acts = patcher.cache_clean_activations(scenario['clean'], test_layers)
        
        # Baseline corrupted
        corrupted_logits = patcher.run_without_patch(scenario['corrupted'])
        corrupted_result = patcher.get_location_probs(corrupted_logits, scenario['loc1'], scenario['loc2'])
        
        # Patch each layer (all heads at once)
        for layer in test_layers:
            # Patch ALL heads in this layer
            patched_logits = patch_full_layer(
                patcher, scenario['corrupted'], layer, clean_acts
            )
            patched_result = patcher.get_location_probs(patched_logits, scenario['loc1'], scenario['loc2'])
            
            restored = patched_result['predicts_correct'] and not corrupted_result['predicts_correct']
            
            results[layer].append({
                'restored': restored,
                'prob_change': patched_result['loc2_prob'] - corrupted_result['loc2_prob'],
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("LAYER PATCH EFFECTS")
    print(f"{'='*60}")
    
    print(f"\n{'Layer':<8} {'Restoration %':<15} {'Avg Prob Change':<15}")
    print("-" * 40)
    
    for layer in test_layers:
        effects = results[layer]
        restoration_rate = sum(e['restored'] for e in effects) / len(effects)
        avg_prob_change = sum(e['prob_change'] for e in effects) / len(effects)
        
        bar = "*" * int(restoration_rate * 20)
        print(f"L{layer:<6} {restoration_rate*100:>10.1f}%     {avg_prob_change:>+.4f}  {bar}")
    
    return dict(results)


def patch_full_layer(patcher, corrupted_prompt, layer_idx, clean_activations):
    """Patch ALL heads in a layer (entire layer activation)."""
    patcher.clear_hooks()
    
    def make_full_layer_patch_hook(layer):
        def hook(module, args):
            if layer not in clean_activations:
                return args
            
            hidden_states = args[0]
            clean = clean_activations[layer]
            
            # Handle sequence length mismatch
            min_seq = min(hidden_states.shape[1], clean.shape[1])
            
            # Replace entire activation
            new_hidden = hidden_states.clone()
            new_hidden[:, :min_seq, :] = clean[:, :min_seq, :]
            
            if len(args) > 1:
                return (new_hidden,) + args[1:]
            return (new_hidden,)
        return hook
    
    # Install patch hook
    o_proj = patcher.model.model.layers[layer_idx].self_attn.o_proj
    hook = o_proj.register_forward_pre_hook(make_full_layer_patch_hook(layer_idx))
    patcher.hooks.append(hook)
    
    # Run
    inputs = patcher.tokenizer(corrupted_prompt, return_tensors="pt").to(patcher.model.device)
    with torch.no_grad():
        outputs = patcher.model(**inputs)
    
    patcher.clear_hooks()
    
    return outputs.logits


def run_causal_path_tracing(model, tokenizer, n_scenarios: int = 30):
    """
    Advanced: Trace the causal path by patching combinations.
    
    Key question: Does L15H9 -> L17H4 -> L19H2 form a serial path?
    """
    patcher = PathPatcher(model, tokenizer)
    
    print(f"\n{'='*60}")
    print("CAUSAL PATH TRACING")
    print(f"{'='*60}")
    print("Testing if information flows: L15H9 -> L17/18 -> L19")
    print()
    
    # Test specific pathways
    pathways = [
        # Single heads
        [('enabler_early', [(15, 9)])],
        [('inhibitor_17', [(17, 4)])],
        [('inhibitor_18', [(18, 11), (18, 14)])],
        [('enabler_late', [(19, 2), (19, 15)])],
        
        # Combined paths
        [('early_to_mid', [(15, 9), (17, 4)])],
        [('mid_to_late', [(17, 4), (19, 2)])],
        [('full_enabler_path', [(15, 9), (19, 2), (19, 15)])],
        [('skip_inhibitors', [(15, 9), (19, 2)])],  # Skip L17-18
    ]
    
    results = {}
    
    for pathway in pathways:
        name, heads = pathway[0]
        layers_needed = sorted(set(h[0] for h in heads))
        
        correct_count = 0
        total_count = 0
        
        for i in range(n_scenarios):
            scenario = generate_scenario_pair(seed=200 + i)
            
            # Cache clean
            clean_acts = patcher.cache_clean_activations(scenario['clean'], layers_needed)
            
            # Baseline corrupted
            corrupted_logits = patcher.run_without_patch(scenario['corrupted'])
            corrupted_result = patcher.get_location_probs(corrupted_logits, scenario['loc1'], scenario['loc2'])
            
            # Patch all heads in pathway
            patched_logits = patch_multiple_heads(patcher, scenario['corrupted'], heads, clean_acts)
            patched_result = patcher.get_location_probs(patched_logits, scenario['loc1'], scenario['loc2'])
            
            if patched_result['predicts_correct']:
                correct_count += 1
            total_count += 1
        
        accuracy = correct_count / total_count
        results[name] = accuracy
        print(f"{name:<25} {accuracy*100:>6.1f}%")
    
    return results


def patch_multiple_heads(patcher, corrupted_prompt, heads, clean_activations):
    """Patch multiple specific heads."""
    patcher.clear_hooks()
    
    # Capture patcher instance variables
    n_heads = patcher.n_heads
    
    # Group heads by layer
    heads_by_layer = defaultdict(list)
    for layer, head in heads:
        heads_by_layer[layer].append(head)
    
    def make_multi_head_patch_hook(layer, head_list):
        def hook(module, args):
            if layer not in clean_activations:
                return args
            
            hidden_states = args[0]
            batch, seq_len, hidden = hidden_states.shape
            
            # Calculate head_dim from actual tensor (may differ from config due to GQA)
            head_dim = hidden // n_heads
            
            clean = clean_activations[layer]
            min_seq = min(seq_len, clean.shape[1])
            
            reshaped = hidden_states.view(batch, seq_len, n_heads, head_dim)
            clean_reshaped = clean.view(clean.shape[0], clean.shape[1], n_heads, head_dim)
            
            # Patch all specified heads
            for head in head_list:
                reshaped[:, :min_seq, head, :] = clean_reshaped[:, :min_seq, head, :]
            
            new_hidden = reshaped.view(batch, seq_len, hidden)
            
            if len(args) > 1:
                return (new_hidden,) + args[1:]
            return (new_hidden,)
        return hook
    
    # Install hooks for each layer that has heads to patch
    for layer, head_list in heads_by_layer.items():
        o_proj = patcher.model.model.layers[layer].self_attn.o_proj
        hook = o_proj.register_forward_pre_hook(make_multi_head_patch_hook(layer, head_list))
        patcher.hooks.append(hook)
    
    # Run
    inputs = patcher.tokenizer(corrupted_prompt, return_tensors="pt").to(patcher.model.device)
    with torch.no_grad():
        outputs = patcher.model(**inputs)
    
    patcher.clear_hooks()
    
    return outputs.logits


def main():
    print("="*60)
    print("STEP 11: PATH PATCHING THROUGH ToM CIRCUIT")
    print("="*60)
    print()
    print("This experiment traces information flow through the")
    print("discovered Theory of Mind circuit using path patching.")
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
    print(f"Layers: {model.config.num_hidden_layers}, Heads: {model.config.num_attention_heads}")
    
    # Run experiments
    all_results = {}
    
    # 1. Individual head patching
    print("\n" + "="*60)
    print("EXPERIMENT 1: INDIVIDUAL HEAD PATCHING")
    print("="*60)
    head_results = run_path_patching_experiment(model, tokenizer, n_scenarios=50)
    all_results['head_patching'] = head_results['summary']
    
    # 2. Layer-level patching
    print("\n" + "="*60)
    print("EXPERIMENT 2: LAYER-LEVEL PATCHING")
    print("="*60)
    layer_results = run_layer_patching(model, tokenizer, n_scenarios=30)
    all_results['layer_patching'] = {
        str(k): {
            'restoration_rate': sum(e['restored'] for e in v) / len(v),
            'avg_prob_change': sum(e['prob_change'] for e in v) / len(v),
        }
        for k, v in layer_results.items()
    }
    
    # 3. Causal path tracing
    print("\n" + "="*60)
    print("EXPERIMENT 3: CAUSAL PATH TRACING")
    print("="*60)
    path_results = run_causal_path_tracing(model, tokenizer, n_scenarios=30)
    all_results['path_tracing'] = path_results
    
    # Final interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    print("\n1. HEAD PATCH EFFECTS:")
    print("   Which heads carry the belief-update signal?")
    for head, data in sorted(all_results['head_patching']['head_summaries'].items(), 
                              key=lambda x: -x[1]['restoration_rate']):
        print(f"   {head} ({data['type']}): {data['restoration_rate']*100:.1f}% restoration")
    
    print("\n2. LAYER IMPORTANCE:")
    print("   Which layers are critical for ToM?")
    for layer, data in sorted(all_results['layer_patching'].items(), 
                               key=lambda x: -x[1]['restoration_rate']):
        if data['restoration_rate'] > 0.1:
            print(f"   Layer {layer}: {data['restoration_rate']*100:.1f}% restoration")
    
    print("\n3. CAUSAL PATHS:")
    print("   Which paths restore ToM when patched?")
    for path, acc in sorted(all_results['path_tracing'].items(), key=lambda x: -x[1]):
        bar = "*" * int(acc * 10)
        print(f"   {path:<25} {acc*100:>5.1f}%  {bar}")
    
    # Save results (convert any remaining tensors to floats)
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, (bool,)):
            return bool(obj)
        elif hasattr(obj, 'item'):  # numpy types
            return obj.item()
        return obj
    
    output_file = RESULTS_DIR / "path_patching_results.json"
    with open(output_file, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    print("\n" + "="*60)
    print("PATH PATCHING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

