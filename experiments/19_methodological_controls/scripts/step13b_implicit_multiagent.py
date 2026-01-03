"""
Step 13b: IMPLICIT Multi-Agent ToM Test

Key finding from step13: Model gets 100% when multi-agent scenarios are
EXPLICITLY framed. But our Sally-Anne tests (~30%) used IMPLICIT framing.

The circuit we discovered suppresses ToM only when:
- Belief updates must be INFERRED
- No explicit "Agent X believes..." markers

For multi-agent systems, this matters when:
- Agents communicate naturally (not with explicit belief markers)
- Belief state must be inferred from actions/dialogue

This test creates IMPLICIT multi-agent scenarios - closer to real collaboration.
"""

import torch
import json
import random
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Decision heads from our discovery
DECISION_HEADS = [(17, 4), (18, 14)]


class ImplicitMultiAgentTester:
    """Test on implicit multi-agent scenarios."""
    
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
        self.clear_hooks()
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
    
    def test_baseline(self, prompt, correct, wrong):
        self.clear_hooks()
        return self.get_probs(prompt, correct, wrong)
    
    def test_ablated(self, prompt, correct, wrong):
        self.install_ablation(DECISION_HEADS)
        result = self.get_probs(prompt, correct, wrong)
        self.clear_hooks()
        return result


def generate_implicit_agent_comm(n: int = 50):
    """
    IMPLICIT agent-to-agent communication.
    No explicit belief markers. Must infer from narrative.
    """
    scenarios = []
    random.seed(42)
    
    agents = ["Alice", "Bob", "Carol", "David"]
    objects = ["ball", "book", "key", "toy"]
    locations = ["basket", "box", "drawer", "shelf"]
    
    for i in range(n):
        a, b = random.sample(agents, 2)
        obj = random.choice(objects)
        loc1, loc2 = random.sample(locations, 2)
        
        # IMPLICIT: No "Agent X believes" - just narrative
        prompt = (
            f"{a} put the {obj} in the {loc1}. "
            f"{a} left the room. "
            f"{b} moved the {obj} to the {loc2}. "
            f"{b} called {a} and said: 'Hey, I moved the {obj} to the {loc2}.' "
            f"When {a} returns, {a} will look for the {obj} in the"
        )
        
        scenarios.append({
            'type': 'implicit_comm',
            'prompt': prompt,
            'correct': f" {loc2}",  # A was told!
            'wrong': f" {loc1}",
        })
    
    return scenarios


def generate_implicit_second_order(n: int = 50):
    """
    IMPLICIT second-order beliefs.
    "What does A think B thinks?" without explicit markers.
    """
    scenarios = []
    random.seed(123)
    
    agents = ["Alice", "Bob", "Carol", "David"]
    objects = ["ball", "book", "key", "toy"]
    locations = ["basket", "box", "drawer", "shelf"]
    
    for i in range(n):
        a, b = random.sample(agents, 2)
        obj = random.choice(objects)
        loc1, loc2 = random.sample(locations, 2)
        
        # IMPLICIT second-order: A knows, B doesn't, A knows B doesn't
        prompt = (
            f"{a} and {b} saw the {obj} in the {loc1}. "
            f"Then {b} left the room. "
            f"While {b} was away, the {obj} was moved to the {loc2}. "
            f"{a} watched this happen but {b} didn't see it. "
            f"If asked where {b} would look for the {obj}, {a} would say the"
        )
        
        scenarios.append({
            'type': 'implicit_second_order',
            'prompt': prompt,
            'correct': f" {loc1}",  # A knows B still thinks loc1
            'wrong': f" {loc2}",
        })
    
    return scenarios


def generate_implicit_dialogue(n: int = 50):
    """
    IMPLICIT multi-turn dialogue.
    Track beliefs through conversation without explicit markers.
    """
    scenarios = []
    random.seed(456)
    
    agents = ["Alice", "Bob"]
    objects = ["report", "document", "file", "package"]
    locations = ["desk", "cabinet", "inbox", "folder"]
    
    for i in range(n):
        a, b = agents if random.random() > 0.5 else agents[::-1]
        obj = random.choice(objects)
        loc1, loc2 = random.sample(locations, 2)
        
        # Natural dialogue without explicit belief markers
        prompt = (
            f'{a}: "I put the {obj} in the {loc1}."\n'
            f'{b}: "Actually, I need to move it. I\'m putting it in the {loc2}."\n'
            f'{a}: "Got it, thanks for letting me know."\n'
            f"Later, {a} needs the {obj}. {a} will check the"
        )
        
        scenarios.append({
            'type': 'implicit_dialogue',
            'prompt': prompt,
            'correct': f" {loc2}",  # A was informed
            'wrong': f" {loc1}",
        })
    
    return scenarios


def generate_partial_information(n: int = 50):
    """
    HARDEST: Partial information scenarios.
    One agent knows something the other doesn't - no explicit communication.
    """
    scenarios = []
    random.seed(789)
    
    agents = ["Alice", "Bob", "Carol", "David"]
    objects = ["ball", "book", "key", "toy"]
    locations = ["basket", "box", "drawer", "shelf"]
    
    for i in range(n):
        a, b, c = random.sample(agents, 3)
        obj = random.choice(objects)
        loc1, loc2 = random.sample(locations, 2)
        
        # A knows something B doesn't, and we ask from A's perspective about B
        prompt = (
            f"{a}, {b}, and {c} were in the room. "
            f"The {obj} was in the {loc1}. "
            f"{b} left to run an errand. "
            f"After {b} left, {c} moved the {obj} to the {loc2}. "
            f"{a} saw this happen. "
            f"{a} knows that {b} would search for the {obj} in the"
        )
        
        scenarios.append({
            'type': 'partial_info',
            'prompt': prompt,
            'correct': f" {loc1}",  # B doesn't know about move
            'wrong': f" {loc2}",
        })
    
    return scenarios


def run_experiment(model, tokenizer):
    """Run all implicit multi-agent tests."""
    tester = ImplicitMultiAgentTester(model, tokenizer)
    
    scenario_generators = {
        'implicit_comm': generate_implicit_agent_comm,
        'implicit_second_order': generate_implicit_second_order,
        'implicit_dialogue': generate_implicit_dialogue,
        'partial_info': generate_partial_information,
    }
    
    results = {}
    
    print(f"\n{'='*70}")
    print("IMPLICIT MULTI-AGENT ToM TEST")
    print(f"{'='*70}")
    print("Testing scenarios WITHOUT explicit belief markers")
    print("This is where our circuit should matter!")
    print()
    
    for name, generator in scenario_generators.items():
        scenarios = generator(50)
        
        print(f"\n{'='*50}")
        print(f"SCENARIO: {name.upper()}")
        print(f"{'='*50}")
        
        baseline_correct = 0
        ablated_correct = 0
        
        for i, s in enumerate(scenarios):
            if i % 10 == 0:
                print(f"  Processing {i+1}/{len(scenarios)}...")
            
            base = tester.test_baseline(s['prompt'], s['correct'], s['wrong'])
            if base['predicts_correct']:
                baseline_correct += 1
            
            abl = tester.test_ablated(s['prompt'], s['correct'], s['wrong'])
            if abl['predicts_correct']:
                ablated_correct += 1
        
        base_acc = baseline_correct / len(scenarios)
        abl_acc = ablated_correct / len(scenarios)
        boost = abl_acc - base_acc
        
        print(f"\n  Results:")
        print(f"    Baseline:      {base_acc*100:5.1f}%")
        print(f"    With ablation: {abl_acc*100:5.1f}%")
        print(f"    Boost:         {boost*100:+5.1f}%")
        
        results[name] = {
            'baseline': base_acc,
            'ablated': abl_acc,
            'boost': boost,
        }
        
        if boost > 0.1:
            print(f"    --> CIRCUIT ACTIVATES: Ablation helps!")
        elif base_acc < 0.5:
            print(f"    --> MODEL STRUGGLES: This is where we need intervention")
    
    return results


def main():
    print("="*70)
    print("STEP 13b: IMPLICIT MULTI-AGENT ToM")
    print("="*70)
    print()
    print("Previous finding: Model gets 100% on EXPLICIT multi-agent scenarios")
    print("This test: IMPLICIT scenarios (no belief markers)")
    print("Hypothesis: Circuit matters for IMPLICIT inference")
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
    
    results = run_experiment(model, tokenizer)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: IMPLICIT vs EXPLICIT MULTI-AGENT ToM")
    print(f"{'='*70}")
    
    print(f"\n{'Scenario':<25} {'Baseline':>10} {'Ablated':>10} {'Boost':>10}")
    print("-" * 57)
    
    for name, data in results.items():
        print(f"{name:<25} {data['baseline']*100:>9.1f}% {data['ablated']*100:>9.1f}% {data['boost']*100:>+9.1f}%")
    
    print(f"\n{'='*70}")
    print("KEY INSIGHT")
    print(f"{'='*70}")
    
    avg_baseline = sum(d['baseline'] for d in results.values()) / len(results)
    avg_ablated = sum(d['ablated'] for d in results.values()) / len(results)
    
    if avg_baseline < 0.7:
        print("\nModel STRUGGLES with IMPLICIT multi-agent ToM")
        print("This is where the circuit intervention matters!")
        if avg_ablated > avg_baseline + 0.05:
            print("Ablating decision heads HELPS - validates our finding!")
    else:
        print("\nModel handles implicit scenarios well")
        print("Circuit may be less relevant for these specific framings")
    
    # Save
    output_file = RESULTS_DIR / "implicit_multiagent_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()

