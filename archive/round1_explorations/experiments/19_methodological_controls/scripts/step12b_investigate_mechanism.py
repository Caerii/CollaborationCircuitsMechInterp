"""
Step 12b: Investigate Why Path Patching Works But Injection Doesn't

Path patching: REPLACE corrupted with clean → works (36% restoration)
Signal injection: ADD (clean - corrupted) → doesn't work (0% boost)

Mathematically: corrupted + (clean - corrupted) = clean
So why the difference?

Hypotheses:
1. Sequence length mismatch - patching patches all positions, injection only last
2. Mean signal doesn't generalize - need scenario-specific signal
3. The mechanism isn't linear - nonlinear interactions matter

Let's test each hypothesis.
"""

import torch
import json
import random
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BRIDGES = [
    "so {agent} updated their belief about the {obj}'s location",
    "therefore {agent} now knows the {obj} is in the {loc2}",
]


def generate_scenario(seed: int):
    """Generate a scenario."""
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
    
    corrupted = (
        f"{agent} put the {obj} in the {loc1}. "
        f"{informer} {verb} {agent}: 'I moved the {obj} to the {loc2}.' "
        f"Where will {agent} look for the {obj}? {agent} will look in the"
    )
    
    bridge = random.choice(BRIDGES).format(agent=agent, obj=obj, loc2=loc2)
    clean = (
        f"{agent} put the {obj} in the {loc1}. "
        f"{informer} {verb} {agent}: 'I moved the {obj} to the {loc2}.' "
        f"{bridge}. "
        f"Where will {agent} look for the {obj}? {agent} will look in the"
    )
    
    return {'corrupted': corrupted, 'clean': clean, 'loc1': loc1, 'loc2': loc2}


class MechanismInvestigator:
    """Investigate the ToM mechanism."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_heads = model.config.num_attention_heads
        self.hooks = []
        self.cached = {}
        
    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        
    def capture(self, prompt: str, layers: list) -> dict:
        """Capture activations."""
        self.clear_hooks()
        self.cached = {}
        
        def make_hook(layer):
            def hook(module, args):
                self.cached[layer] = args[0].clone().detach()
                return args
            return hook
        
        for l in layers:
            h = self.model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(make_hook(l))
            self.hooks.append(h)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            self.model(**inputs)
        
        self.clear_hooks()
        return self.cached.copy()
    
    def run_baseline(self, prompt: str) -> torch.Tensor:
        """Run without hooks."""
        self.clear_hooks()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            return self.model(**inputs).logits
    
    def get_probs(self, logits, loc1, loc2):
        """Get location probs."""
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        id1 = self.tokenizer.encode(" " + loc1, add_special_tokens=False)[0]
        id2 = self.tokenizer.encode(" " + loc2, add_special_tokens=False)[0]
        return probs[id2] > probs[id1]
    
    def test_same_scenario_injection(self, n: int = 30):
        """
        Hypothesis 1: Mean signal doesn't work, but scenario-specific signal might.
        
        For each scenario, extract signal from ITS OWN clean/corrupted pair
        and inject immediately.
        """
        print("\nTest 1: Same-scenario injection (not mean)")
        
        layer, head = 18, 11
        correct = 0
        baseline_correct = 0
        
        for i in range(n):
            scenario = generate_scenario(seed=100 + i)
            
            # Baseline
            baseline_logits = self.run_baseline(scenario['corrupted'])
            if self.get_probs(baseline_logits, scenario['loc1'], scenario['loc2']):
                baseline_correct += 1
            
            # Get THIS scenario's clean/corrupted activations
            clean_acts = self.capture(scenario['clean'], [layer])
            corrupted_acts = self.capture(scenario['corrupted'], [layer])
            
            # Compute difference for THIS scenario
            clean = clean_acts[layer]
            corrupted = corrupted_acts[layer]
            
            # Inject at last position only (like our failed approach)
            head_dim = clean.shape[-1] // self.n_heads
            
            # Extract head-specific signal at last position
            clean_head = clean[0, -1, head*head_dim:(head+1)*head_dim]
            corrupt_head = corrupted[0, -1, head*head_dim:(head+1)*head_dim]
            signal = clean_head - corrupt_head
            
            # Now inject and run
            self.clear_hooks()
            n_heads = self.n_heads
            
            def make_inject_hook():
                def hook(module, args):
                    h = args[0]
                    batch, seq, hidden = h.shape
                    hd = hidden // n_heads
                    reshaped = h.view(batch, seq, n_heads, hd)
                    reshaped[:, -1, head, :] = reshaped[:, -1, head, :] + signal.to(h.device)
                    return (reshaped.view(batch, seq, hidden),)
                return hook
            
            hook = self.model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(make_inject_hook())
            self.hooks.append(hook)
            
            inputs = self.tokenizer(scenario['corrupted'], return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            self.clear_hooks()
            
            if self.get_probs(logits, scenario['loc1'], scenario['loc2']):
                correct += 1
        
        print(f"  Baseline: {baseline_correct/n*100:.1f}%")
        print(f"  Same-scenario injection: {correct/n*100:.1f}%")
        return correct / n, baseline_correct / n
    
    def test_replace_vs_add(self, n: int = 30):
        """
        Hypothesis 2: REPLACE works but ADD doesn't because of nonlinearity.
        
        Compare:
        - Add signal: corrupted + (clean - corrupted)
        - Replace: set corrupted = clean (should be identical mathematically)
        """
        print("\nTest 2: Replace vs Add (should be equivalent)")
        
        layer, head = 18, 11
        add_correct = 0
        replace_correct = 0
        baseline_correct = 0
        
        for i in range(n):
            scenario = generate_scenario(seed=200 + i)
            
            # Baseline
            baseline_logits = self.run_baseline(scenario['corrupted'])
            if self.get_probs(baseline_logits, scenario['loc1'], scenario['loc2']):
                baseline_correct += 1
            
            # Get activations
            clean_acts = self.capture(scenario['clean'], [layer])
            corrupted_acts = self.capture(scenario['corrupted'], [layer])
            
            clean = clean_acts[layer]
            corrupted = corrupted_acts[layer]
            head_dim = clean.shape[-1] // self.n_heads
            
            # Test ADD
            signal = clean[0, -1, head*head_dim:(head+1)*head_dim] - corrupted[0, -1, head*head_dim:(head+1)*head_dim]
            
            self.clear_hooks()
            n_heads = self.n_heads
            
            def make_add_hook():
                def hook(module, args):
                    h = args[0]
                    batch, seq, hidden = h.shape
                    hd = hidden // n_heads
                    reshaped = h.view(batch, seq, n_heads, hd)
                    reshaped[:, -1, head, :] = reshaped[:, -1, head, :] + signal.to(h.device)
                    return (reshaped.view(batch, seq, hidden),)
                return hook
            
            hook = self.model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(make_add_hook())
            self.hooks.append(hook)
            
            inputs = self.tokenizer(scenario['corrupted'], return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                add_logits = self.model(**inputs).logits
            
            self.clear_hooks()
            
            if self.get_probs(add_logits, scenario['loc1'], scenario['loc2']):
                add_correct += 1
            
            # Test REPLACE
            clean_head_value = clean[0, -1, head*head_dim:(head+1)*head_dim]
            
            def make_replace_hook():
                def hook(module, args):
                    h = args[0]
                    batch, seq, hidden = h.shape
                    hd = hidden // n_heads
                    reshaped = h.view(batch, seq, n_heads, hd)
                    reshaped[:, -1, head, :] = clean_head_value.to(h.device)
                    return (reshaped.view(batch, seq, hidden),)
                return hook
            
            hook = self.model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(make_replace_hook())
            self.hooks.append(hook)
            
            inputs = self.tokenizer(scenario['corrupted'], return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                replace_logits = self.model(**inputs).logits
            
            self.clear_hooks()
            
            if self.get_probs(replace_logits, scenario['loc1'], scenario['loc2']):
                replace_correct += 1
        
        print(f"  Baseline: {baseline_correct/n*100:.1f}%")
        print(f"  Add (corrupted + signal): {add_correct/n*100:.1f}%")
        print(f"  Replace (set to clean): {replace_correct/n*100:.1f}%")
        return add_correct/n, replace_correct/n, baseline_correct/n
    
    def test_all_positions_vs_last(self, n: int = 30):
        """
        Hypothesis 3: Patching all positions matters, not just last.
        
        Compare:
        - Patch last position only
        - Patch ALL positions (like real path patching)
        """
        print("\nTest 3: All positions vs last position only")
        
        layer, head = 18, 11
        last_only_correct = 0
        all_pos_correct = 0
        baseline_correct = 0
        
        for i in range(n):
            scenario = generate_scenario(seed=300 + i)
            
            # Baseline
            baseline_logits = self.run_baseline(scenario['corrupted'])
            if self.get_probs(baseline_logits, scenario['loc1'], scenario['loc2']):
                baseline_correct += 1
            
            # Get activations
            clean_acts = self.capture(scenario['clean'], [layer])
            corrupted_acts = self.capture(scenario['corrupted'], [layer])
            
            clean = clean_acts[layer]  # [1, seq_clean, hidden]
            head_dim = clean.shape[-1] // self.n_heads
            
            # Test LAST ONLY
            clean_last = clean[0, -1, head*head_dim:(head+1)*head_dim]
            
            self.clear_hooks()
            n_heads = self.n_heads
            
            def make_last_hook():
                def hook(module, args):
                    h = args[0]
                    batch, seq, hidden = h.shape
                    hd = hidden // n_heads
                    reshaped = h.view(batch, seq, n_heads, hd)
                    reshaped[:, -1, head, :] = clean_last.to(h.device)
                    return (reshaped.view(batch, seq, hidden),)
                return hook
            
            hook = self.model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(make_last_hook())
            self.hooks.append(hook)
            
            inputs = self.tokenizer(scenario['corrupted'], return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                last_logits = self.model(**inputs).logits
            
            self.clear_hooks()
            
            if self.get_probs(last_logits, scenario['loc1'], scenario['loc2']):
                last_only_correct += 1
            
            # Test ALL POSITIONS
            seq_corrupt = self.tokenizer(scenario['corrupted'], return_tensors="pt")['input_ids'].shape[1]
            min_seq = min(clean.shape[1], seq_corrupt)
            
            # Extract clean values for all positions (for head only)
            clean_head_all = clean[0, :min_seq, head*head_dim:(head+1)*head_dim]  # [min_seq, head_dim]
            
            def make_all_hook():
                def hook(module, args):
                    h = args[0]
                    batch, seq, hidden = h.shape
                    hd = hidden // n_heads
                    reshaped = h.view(batch, seq, n_heads, hd)
                    # Replace at all positions up to min_seq
                    actual_min = min(min_seq, seq)
                    reshaped[:, :actual_min, head, :] = clean_head_all[:actual_min, :].to(h.device)
                    return (reshaped.view(batch, seq, hidden),)
                return hook
            
            hook = self.model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(make_all_hook())
            self.hooks.append(hook)
            
            inputs = self.tokenizer(scenario['corrupted'], return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                all_logits = self.model(**inputs).logits
            
            self.clear_hooks()
            
            if self.get_probs(all_logits, scenario['loc1'], scenario['loc2']):
                all_pos_correct += 1
        
        print(f"  Baseline: {baseline_correct/n*100:.1f}%")
        print(f"  Last position only: {last_only_correct/n*100:.1f}%")
        print(f"  All positions: {all_pos_correct/n*100:.1f}%")
        return last_only_correct/n, all_pos_correct/n, baseline_correct/n


def main():
    print("="*60)
    print("INVESTIGATING: Why Path Patching Works But Injection Doesn't")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    investigator = MechanismInvestigator(model, tokenizer)
    
    # Run tests
    results = {}
    
    # Test 1: Same scenario injection
    inject_acc, baseline1 = investigator.test_same_scenario_injection(n=30)
    results['same_scenario'] = {'inject': inject_acc, 'baseline': baseline1}
    
    # Test 2: Replace vs Add
    add_acc, replace_acc, baseline2 = investigator.test_replace_vs_add(n=30)
    results['replace_vs_add'] = {'add': add_acc, 'replace': replace_acc, 'baseline': baseline2}
    
    # Test 3: All positions vs last
    last_acc, all_acc, baseline3 = investigator.test_all_positions_vs_last(n=30)
    results['positions'] = {'last': last_acc, 'all': all_acc, 'baseline': baseline3}
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    print("\nHypothesis 1: Mean signal doesn't generalize")
    print(f"  Same-scenario injection: {results['same_scenario']['inject']*100:.1f}%")
    if results['same_scenario']['inject'] > results['same_scenario']['baseline'] + 0.05:
        print("  SUPPORTED: Same-scenario works better than mean")
    else:
        print("  NOT SUPPORTED: Even same-scenario doesn't help")
    
    print("\nHypothesis 2: Nonlinearity (ADD vs REPLACE)")
    print(f"  Add: {results['replace_vs_add']['add']*100:.1f}%")
    print(f"  Replace: {results['replace_vs_add']['replace']*100:.1f}%")
    if abs(results['replace_vs_add']['add'] - results['replace_vs_add']['replace']) > 0.1:
        print("  SUPPORTED: Replace != Add (nonlinear effects)")
    else:
        print("  NOT SUPPORTED: Replace ~ Add (linear)")
    
    print("\nHypothesis 3: Position matters")
    print(f"  Last only: {results['positions']['last']*100:.1f}%")
    print(f"  All positions: {results['positions']['all']*100:.1f}%")
    if results['positions']['all'] > results['positions']['last'] + 0.1:
        print("  SUPPORTED: All positions matters more than last only")
    else:
        print("  NOT SUPPORTED: Position doesn't matter much")
    
    # Save
    output_file = RESULTS_DIR / "mechanism_investigation.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()

