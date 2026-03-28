"""
Step 13: Multi-Agent Circuit Validation

CRITICAL EXPERIMENT: Does our discovered ToM circuit (L17H4, L18H11, L18H14)
actually matter for REAL multi-agent collaboration scenarios?

This experiment tests:
1. Agent-to-Agent communication (Agent A tells Agent B)
2. Second-order beliefs (A thinks B thinks...)
3. Multi-turn dialogue (beliefs updating across turns)
4. Role-based perspective taking (acting AS an agent)

For each scenario type, we test:
- Baseline accuracy (model's natural performance)
- Performance with decision heads ablated (L17H4 + L18H14)
- Whether the circuit matters for multi-agent ToM
"""

import torch
import json
import random
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

# Setup
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Key decision heads from our discovery
DECISION_HEADS = [(17, 4), (18, 14)]  # The true inhibitors that achieve 100% when ablated


class MultiAgentValidator:
    """Test ToM circuit on multi-agent scenarios."""
    
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
        """Ablate specified heads."""
        self.clear_hooks()
        
        n_heads = self.n_heads
        layers_to_heads = defaultdict(list)
        for layer, head in heads:
            layers_to_heads[layer].append(head)
        
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
    
    def get_completion_prob(self, prompt: str, correct: str, wrong: str) -> dict:
        """Get probability of correct vs wrong completion."""
        self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        
        correct_id = self.tokenizer.encode(correct, add_special_tokens=False)[0]
        wrong_id = self.tokenizer.encode(wrong, add_special_tokens=False)[0]
        
        return {
            'correct_prob': probs[correct_id].item(),
            'wrong_prob': probs[wrong_id].item(),
            'predicts_correct': probs[correct_id] > probs[wrong_id],
        }
    
    def run_baseline(self, prompt: str, correct: str, wrong: str) -> dict:
        """Run without ablation."""
        self.clear_hooks()
        return self.get_completion_prob(prompt, correct, wrong)
    
    def run_ablated(self, prompt: str, correct: str, wrong: str) -> dict:
        """Run with decision heads ablated."""
        self.install_ablation(DECISION_HEADS)
        result = self.get_completion_prob(prompt, correct, wrong)
        self.clear_hooks()
        return result


def generate_agent_to_agent_scenarios(n: int = 50):
    """
    Scenario Type 1: Agent-to-Agent Communication
    
    Format: Agent A has information, Agent B needs to infer A's belief
    This is the core multi-agent scenario.
    """
    scenarios = []
    random.seed(42)
    
    agents = ["Alice", "Bob", "Carol", "David", "Eve", "Frank"]
    objects = ["ball", "book", "key", "toy", "phone", "document"]
    locations = ["basket", "box", "drawer", "shelf", "cabinet", "desk"]
    
    for i in range(n):
        agent_a = random.choice(agents)
        agent_b = random.choice([a for a in agents if a != agent_a])
        obj = random.choice(objects)
        loc1, loc2 = random.sample(locations, 2)
        
        # Scenario: A has outdated info, B tells A the update
        prompt = (
            f"[Multi-Agent Scenario]\n"
            f"Agent {agent_a} believes the {obj} is in the {loc1}.\n"
            f"Agent {agent_b} tells Agent {agent_a}: 'The {obj} has been moved to the {loc2}.'\n"
            f"After this communication, where does Agent {agent_a} believe the {obj} is?\n"
            f"Agent {agent_a} now believes the {obj} is in the"
        )
        
        scenarios.append({
            'type': 'agent_to_agent',
            'prompt': prompt,
            'correct': f" {loc2}",
            'wrong': f" {loc1}",
            'agent_a': agent_a,
            'agent_b': agent_b,
        })
    
    return scenarios


def generate_second_order_scenarios(n: int = 50):
    """
    Scenario Type 2: Second-Order Beliefs (A thinks B thinks...)
    
    This tests recursive ToM - crucial for multi-agent coordination.
    """
    scenarios = []
    random.seed(123)
    
    agents = ["Alice", "Bob", "Carol", "David", "Eve", "Frank"]
    objects = ["ball", "book", "key", "toy", "phone", "document"]
    locations = ["basket", "box", "drawer", "shelf", "cabinet", "desk"]
    
    for i in range(n):
        agent_a = random.choice(agents)
        agent_b = random.choice([a for a in agents if a != agent_a])
        obj = random.choice(objects)
        loc1, loc2 = random.sample(locations, 2)
        
        # A knows reality is loc2, but A knows B still thinks loc1
        prompt = (
            f"[Second-Order Belief Test]\n"
            f"The {obj} is actually in the {loc2}.\n"
            f"{agent_a} knows the {obj} was moved to the {loc2}.\n"
            f"{agent_b} hasn't been told about the move and still thinks it's in the {loc1}.\n"
            f"{agent_a} is aware that {agent_b} hasn't been updated.\n"
            f"What does {agent_a} think {agent_b} believes about where the {obj} is?\n"
            f"{agent_a} thinks {agent_b} believes the {obj} is in the"
        )
        
        scenarios.append({
            'type': 'second_order',
            'prompt': prompt,
            'correct': f" {loc1}",  # B still thinks old location
            'wrong': f" {loc2}",    # Reality (but not what B thinks)
            'agent_a': agent_a,
            'agent_b': agent_b,
        })
    
    return scenarios


def generate_dialogue_scenarios(n: int = 50):
    """
    Scenario Type 3: Multi-Turn Dialogue
    
    Beliefs update across multiple conversation turns.
    """
    scenarios = []
    random.seed(456)
    
    agents = ["Alice", "Bob", "Carol", "David"]
    objects = ["ball", "book", "key", "document"]
    locations = ["basket", "box", "drawer", "shelf", "cabinet", "desk"]
    
    for i in range(n):
        a, b = random.sample(agents, 2)
        obj = random.choice(objects)
        loc1, loc2, loc3 = random.sample(locations, 3)
        
        # Multi-turn: object moves twice, second agent only knows first move
        prompt = (
            f"[Dialogue Tracking]\n"
            f"Turn 1: {a} puts the {obj} in the {loc1}.\n"
            f"Turn 2: {b} moves the {obj} to the {loc2} and tells {a}.\n"
            f"Turn 3: {a} acknowledges: 'Got it, it's now in the {loc2}.'\n"
            f"Turn 4: Later, {a} secretly moves it to the {loc3} without telling {b}.\n"
            f"Where does {b} currently believe the {obj} is?\n"
            f"{b} believes the {obj} is in the"
        )
        
        scenarios.append({
            'type': 'dialogue',
            'prompt': prompt,
            'correct': f" {loc2}",  # B's last known location
            'wrong': f" {loc3}",    # Reality (but B doesn't know)
            'agent_a': a,
            'agent_b': b,
        })
    
    return scenarios


def generate_role_based_scenarios(n: int = 50):
    """
    Scenario Type 4: Role-Based Perspective Taking
    
    The model must ACT AS an agent and track what it knows vs what others know.
    This is critical for actual multi-agent systems.
    """
    scenarios = []
    random.seed(789)
    
    agents = ["Alice", "Bob", "Carol", "David"]
    objects = ["ball", "book", "key", "file"]
    locations = ["basket", "box", "drawer", "folder"]
    
    for i in range(n):
        other = random.choice(agents)
        obj = random.choice(objects)
        loc1, loc2 = random.sample(locations, 2)
        
        # Model is prompted to act AS an agent
        prompt = (
            f"[You are Agent X in a collaborative system]\n"
            f"Your knowledge state:\n"
            f"- You initially knew the {obj} was in the {loc1}\n"
            f"- Agent {other} just told you: 'I moved the {obj} to the {loc2}'\n"
            f"Based on this communication, where do YOU believe the {obj} is?\n"
            f"As Agent X, I believe the {obj} is in the"
        )
        
        scenarios.append({
            'type': 'role_based',
            'prompt': prompt,
            'correct': f" {loc2}",
            'wrong': f" {loc1}",
            'other_agent': other,
        })
    
    return scenarios


def run_validation_experiment(model, tokenizer):
    """Run the full multi-agent validation experiment."""
    validator = MultiAgentValidator(model, tokenizer)
    
    # Generate all scenario types
    scenario_types = {
        'agent_to_agent': generate_agent_to_agent_scenarios(50),
        'second_order': generate_second_order_scenarios(50),
        'dialogue': generate_dialogue_scenarios(50),
        'role_based': generate_role_based_scenarios(50),
    }
    
    results = {}
    
    print(f"\n{'='*70}")
    print("MULTI-AGENT ToM CIRCUIT VALIDATION")
    print(f"{'='*70}")
    print(f"Testing if decision heads (L17H4, L18H14) matter for multi-agent ToM")
    print(f"Scenario types: {list(scenario_types.keys())}")
    print()
    
    for scenario_type, scenarios in scenario_types.items():
        print(f"\n{'='*50}")
        print(f"SCENARIO TYPE: {scenario_type.upper()}")
        print(f"{'='*50}")
        
        baseline_correct = 0
        ablated_correct = 0
        
        for i, s in enumerate(scenarios):
            if i % 10 == 0:
                print(f"  Processing {i+1}/{len(scenarios)}...")
            
            # Baseline
            baseline_result = validator.run_baseline(s['prompt'], s['correct'], s['wrong'])
            if baseline_result['predicts_correct']:
                baseline_correct += 1
            
            # Ablated
            ablated_result = validator.run_ablated(s['prompt'], s['correct'], s['wrong'])
            if ablated_result['predicts_correct']:
                ablated_correct += 1
        
        baseline_acc = baseline_correct / len(scenarios)
        ablated_acc = ablated_correct / len(scenarios)
        boost = ablated_acc - baseline_acc
        
        print(f"\n  Results:")
        print(f"    Baseline:      {baseline_acc*100:5.1f}%")
        print(f"    With ablation: {ablated_acc*100:5.1f}%")
        print(f"    Boost:         {boost*100:+5.1f}%")
        
        results[scenario_type] = {
            'baseline': baseline_acc,
            'ablated': ablated_acc,
            'boost': boost,
            'n': len(scenarios),
        }
        
        # Interpretation
        if boost > 0.1:
            print(f"    --> CIRCUIT MATTERS: Ablation helps multi-agent ToM!")
        elif boost < -0.1:
            print(f"    --> CIRCUIT MATTERS: But ablation hurts this scenario type")
        else:
            print(f"    --> Minimal effect on this scenario type")
    
    return results


def main():
    print("="*70)
    print("STEP 13: MULTI-AGENT ToM CIRCUIT VALIDATION")
    print("="*70)
    print()
    print("CRITICAL QUESTION: Does our discovered ToM circuit matter for")
    print("actual multi-agent collaboration scenarios?")
    print()
    print("We test 4 scenario types:")
    print("  1. Agent-to-Agent: A tells B new information")
    print("  2. Second-Order: A thinks B thinks...")
    print("  3. Dialogue: Multi-turn belief tracking")
    print("  4. Role-Based: Model acts AS an agent")
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
    results = run_validation_experiment(model, tokenizer)
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY: MULTI-AGENT CIRCUIT VALIDATION")
    print(f"{'='*70}")
    
    print(f"\n{'Scenario Type':<20} {'Baseline':>10} {'Ablated':>10} {'Boost':>10}")
    print("-" * 52)
    
    total_baseline = 0
    total_ablated = 0
    total_n = 0
    
    for scenario_type, data in results.items():
        print(f"{scenario_type:<20} {data['baseline']*100:>9.1f}% {data['ablated']*100:>9.1f}% {data['boost']*100:>+9.1f}%")
        total_baseline += data['baseline'] * data['n']
        total_ablated += data['ablated'] * data['n']
        total_n += data['n']
    
    avg_baseline = total_baseline / total_n
    avg_ablated = total_ablated / total_n
    avg_boost = avg_ablated - avg_baseline
    
    print("-" * 52)
    print(f"{'AVERAGE':<20} {avg_baseline*100:>9.1f}% {avg_ablated*100:>9.1f}% {avg_boost*100:>+9.1f}%")
    
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    
    if avg_boost > 0.1:
        print("\nSUCCESS: The ToM circuit (L17H4, L18H14) IS critical for multi-agent ToM!")
        print("Ablating these decision heads improves multi-agent belief tracking.")
        print("This validates our mechanistic findings for real agent collaboration.")
    elif avg_boost > 0:
        print("\nPARTIAL: Circuit has some effect on multi-agent scenarios.")
        print("More work needed to understand context-dependent effects.")
    else:
        print("\nWARNING: Circuit doesn't help multi-agent scenarios.")
        print("May need different intervention strategy for multi-agent context.")
    
    # Save results
    output_file = RESULTS_DIR / "multiagent_validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    print(f"\n{'='*70}")
    print("MULTI-AGENT VALIDATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

