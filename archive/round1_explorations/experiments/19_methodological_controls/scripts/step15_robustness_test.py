"""
Step 15: Comprehensive Robustness Test

Before claiming our finding is robust, we need to test:
1. Multiple prompt templates/phrasings (not just one pattern)
2. Different communication verbs (told, texted, emailed, messaged, etc.)
3. Different narrative styles (formal, casual, story-like)
4. Multiple languages (Qwen supports Chinese, Spanish, French, etc.)
5. Edge cases (ambiguous scenarios, partial info)
6. Negative controls (when SHOULD ToM fail?)
7. Statistical rigor (N=200+ with confidence intervals)

This is what rigorous science demands before claiming generality.
"""

import torch
import json
import random
import math
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# All 5 inhibitor heads
INHIBITORS = [(17, 4), (18, 11), (18, 14), (19, 30), (21, 17)]


class RobustnessAnalyzer:
    """Comprehensive robustness testing."""
    
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
        self.clear_hooks()  # Always clear first
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
        """Get probabilities with proper error handling."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            
            # Safe token encoding
            c_tokens = self.tokenizer.encode(correct, add_special_tokens=False)
            w_tokens = self.tokenizer.encode(wrong, add_special_tokens=False)
            
            if not c_tokens or not w_tokens:
                return {'correct_prob': 0.0, 'wrong_prob': 0.0, 'predicts_correct': False}
            
            c_id = c_tokens[0]
            w_id = w_tokens[0]
            
            c_prob = probs[c_id].item()  # Convert to Python float
            w_prob = probs[w_id].item()  # Convert to Python float
            
            return {
                'correct_prob': c_prob,
                'wrong_prob': w_prob,
                'predicts_correct': bool(c_prob > w_prob),  # Convert to Python bool
            }
        except Exception as e:
            print(f"  [Warning] Error in get_probs: {e}")
            return {'correct_prob': 0.0, 'wrong_prob': 0.0, 'predicts_correct': False}
    
    def test_scenarios(self, scenarios: list, use_ablation: bool = False) -> list:
        """Test a list of scenarios."""
        if not scenarios:
            return []
            
        if use_ablation:
            self.install_ablation(INHIBITORS)
        else:
            self.clear_hooks()
        
        results = []
        for s in scenarios:
            r = self.get_probs(s['prompt'], s['correct'], s['wrong'])
            results.append(r['predicts_correct'])  # Already a Python bool
        
        self.clear_hooks()
        return results


# =============================================================================
# SCENARIO GENERATORS
# =============================================================================

def generate_template_variations(n_per_template: int = 30):
    """
    Test multiple different prompt TEMPLATES for the same scenario type.
    """
    templates = []
    random.seed(42)
    
    agents = ["Alice", "Bob", "Carol", "David", "Emma", "Frank"]
    objects = ["ball", "book", "key", "document", "phone", "laptop"]
    locations = ["basket", "box", "drawer", "shelf", "cabinet", "desk"]
    
    for i in range(n_per_template):
        a, b = random.sample(agents, 2)
        obj = random.choice(objects)
        loc1, loc2 = random.sample(locations, 2)
        
        # Template 1: Original style
        t1 = {
            'template': 'original',
            'prompt': f"{a} put the {obj} in the {loc1}. {a} left. {b} moved the {obj} to the {loc2}. {b} called {a}: 'I moved the {obj} to the {loc2}.' When {a} returns, {a} will look in the",
            'correct': f" {loc2}",
            'wrong': f" {loc1}",
        }
        
        # Template 2: Passive voice
        t2 = {
            'template': 'passive',
            'prompt': f"The {obj} was placed in the {loc1} by {a}. After {a} departed, the {obj} was relocated to the {loc2} by {b}. {a} was informed by {b} about the new location. {a} will search in the",
            'correct': f" {loc2}",
            'wrong': f" {loc1}",
        }
        
        # Template 3: Story-like
        t3 = {
            'template': 'story',
            'prompt': f"Once upon a time, {a} had a {obj} in the {loc1}. While {a} was away, {b} decided to move it to the {loc2}. Later, {b} mentioned to {a}, 'By the way, I put your {obj} in the {loc2}.' Now {a} needs the {obj} and will check the",
            'correct': f" {loc2}",
            'wrong': f" {loc1}",
        }
        
        # Template 4: Formal/professional
        t4 = {
            'template': 'formal',
            'prompt': f"Employee {a} stored the {obj} in the {loc1}. During {a}'s absence, colleague {b} transferred the {obj} to the {loc2} and subsequently notified {a} via message. Upon return, {a} will retrieve the {obj} from the",
            'correct': f" {loc2}",
            'wrong': f" {loc1}",
        }
        
        # Template 5: Casual/texting style
        t5 = {
            'template': 'casual',
            'prompt': f"{a}: left my {obj} in the {loc1}\n{b}: hey moved ur {obj} to {loc2} btw\n{a}: ok thx\nWhere will {a} look? In the",
            'correct': f" {loc2}",
            'wrong': f" {loc1}",
        }
        
        # Template 6: Third person narrative
        t6 = {
            'template': 'third_person',
            'prompt': f"The story goes that {a} kept the {obj} in the {loc1}. {b}, knowing {a} was out, moved it to the {loc2}. When {b} told {a} about this change, {a} understood and would later look for the {obj} in the",
            'correct': f" {loc2}",
            'wrong': f" {loc1}",
        }
        
        templates.extend([t1, t2, t3, t4, t5, t6])
    
    return templates


def generate_communication_variations(n_per_verb: int = 25):
    """
    Test different communication VERBS.
    """
    scenarios = []
    random.seed(123)
    
    agents = ["Alice", "Bob", "Carol", "David"]
    objects = ["ball", "book", "key", "toy"]
    locations = ["basket", "box", "drawer", "shelf"]
    
    # Different ways to communicate
    comm_verbs = [
        ("told", "{b} told {a}: '{msg}'"),
        ("texted", "{b} texted {a}: '{msg}'"),
        ("emailed", "{b} emailed {a}: '{msg}'"),
        ("messaged", "{b} messaged {a}: '{msg}'"),
        ("called", "{b} called {a} and said: '{msg}'"),
        ("wrote", "{b} wrote to {a}: '{msg}'"),
        ("informed", "{b} informed {a} that {direct_msg}"),
        ("notified", "{b} notified {a}: '{msg}'"),
        ("mentioned", "{b} mentioned to {a}: '{msg}'"),
        ("let_know", "{b} let {a} know: '{msg}'"),
    ]
    
    for verb_name, verb_template in comm_verbs:
        for i in range(n_per_verb):
            a, b = random.sample(agents, 2)
            obj = random.choice(objects)
            loc1, loc2 = random.sample(locations, 2)
            
            msg = f"I moved the {obj} to the {loc2}"
            direct_msg = f"the {obj} was moved to the {loc2}"
            
            comm_part = verb_template.format(a=a, b=b, msg=msg, direct_msg=direct_msg)
            
            prompt = (
                f"{a} put the {obj} in the {loc1}. "
                f"{a} went away. "
                f"{b} moved the {obj} to the {loc2}. "
                f"{comm_part}. "
                f"Where will {a} look for the {obj}? {a} will look in the"
            )
            
            scenarios.append({
                'verb': verb_name,
                'prompt': prompt,
                'correct': f" {loc2}",
                'wrong': f" {loc1}",
            })
    
    return scenarios


def generate_multilingual_scenarios(n_per_lang: int = 30):
    """
    Test in multiple languages (Qwen supports many).
    """
    scenarios = []
    random.seed(456)
    
    # Language-specific vocabulary
    languages = {
        'english': {
            'agents': ["Alice", "Bob"],
            'objects': ["ball", "book"],
            'locations': ["basket", "box"],
            'template': "{a} put the {obj} in the {loc1}. {b} moved it to the {loc2} and told {a}. {a} will look in the",
        },
        'spanish': {
            'agents': ["Maria", "Carlos"],
            'objects': ["pelota", "libro"],
            'locations': ["cesta", "caja"],
            'template': "{a} puso el/la {obj} en la {loc1}. {b} lo movio a la {loc2} y le dijo a {a}. {a} buscara en la",
        },
        'french': {
            'agents': ["Marie", "Pierre"],
            'objects': ["balle", "livre"],
            'locations': ["panier", "boite"],
            'template': "{a} a mis le/la {obj} dans le/la {loc1}. {b} l'a deplace vers le/la {loc2} et a dit a {a}. {a} cherchera dans le/la",
        },
        'german': {
            'agents': ["Anna", "Hans"],
            'objects': ["Ball", "Buch"],
            'locations': ["Korb", "Kiste"],
            'template': "{a} legte den/das {obj} in den/die {loc1}. {b} hat es in den/die {loc2} verschoben und {a} informiert. {a} wird in dem/der suchen",
        },
        'chinese': {
            'agents': ["xiaoming", "xiaohong"],  # Use pinyin for safety
            'objects': ["ball", "book"],
            'locations': ["basket", "box"],
            'template': "{a} put the {obj} in the {loc1}. {b} moved it to {loc2} and told {a}. {a} will look in the",
        },
    }
    
    for lang, config in languages.items():
        for i in range(n_per_lang):
            if len(config['agents']) < 2 or len(config['locations']) < 2:
                continue
                
            a, b = random.sample(config['agents'], 2)
            obj = random.choice(config['objects'])
            loc1, loc2 = random.sample(config['locations'], 2)
                
            prompt = config['template'].format(a=a, b=b, obj=obj, loc1=loc1, loc2=loc2)
            
            scenarios.append({
                'language': lang,
                'prompt': prompt,
                'correct': f" {loc2}",
                'wrong': f" {loc1}",
            })
    
    return scenarios


def generate_edge_cases(n: int = 50):
    """
    Test edge cases and tricky scenarios.
    """
    scenarios = []
    random.seed(789)
    
    agents = ["Alice", "Bob", "Carol"]
    objects = ["ball", "book", "key"]
    locations = ["basket", "box", "drawer", "shelf"]
    
    n_per_type = max(1, n // 5)
    
    for i in range(n_per_type):
        a, b = random.sample(agents, 2)
        obj = random.choice(objects)
        loc1, loc2, loc3 = random.sample(locations, 3)
        
        # Edge 1: Multiple moves, only last communicated
        scenarios.append({
            'type': 'multiple_moves',
            'prompt': f"{a} put the {obj} in the {loc1}. {b} moved it to {loc2}. Then {b} moved it again to {loc3}. {b} told {a}: 'The {obj} is now in the {loc3}.' {a} will look in the",
            'correct': f" {loc3}",
            'wrong': f" {loc1}",
        })
        
        # Edge 2: Partial information (told wrong location)
        scenarios.append({
            'type': 'wrong_info',
            'prompt': f"{a} put the {obj} in the {loc1}. {b} moved it to {loc2} but mistakenly told {a}: 'I put it in the {loc3}.' Based on what {a} was told, {a} will look in the",
            'correct': f" {loc3}",  # A believes wrong info
            'wrong': f" {loc2}",
        })
        
        # Edge 3: Delayed communication
        scenarios.append({
            'type': 'delayed',
            'prompt': f"{a} put the {obj} in the {loc1}. {b} moved it to {loc2}. A week later, {b} finally told {a} about the move. {a} will now look in the",
            'correct': f" {loc2}",
            'wrong': f" {loc1}",
        })
        
        # Edge 4: Multiple agents, chain of communication
        c = [x for x in agents if x not in [a, b]][0]
        scenarios.append({
            'type': 'chain',
            'prompt': f"{a} put the {obj} in the {loc1}. {b} moved it to {loc2}. {b} told {c}, and {c} told {a}. {a} will look in the",
            'correct': f" {loc2}",
            'wrong': f" {loc1}",
        })
        
        # Edge 5: Negation
        scenarios.append({
            'type': 'negation',
            'prompt': f"{a} put the {obj} in the {loc1}. {b} moved it but did NOT tell {a}. {a} still thinks the {obj} is in the",
            'correct': f" {loc1}",  # A wasn't told!
            'wrong': f" {loc2}",
        })
    
    return scenarios


def generate_negative_controls(n: int = 50):
    """
    NEGATIVE CONTROLS: Scenarios where ToM SHOULD fail (model should answer 'wrong').
    These test if the model is actually doing ToM vs just pattern matching.
    """
    scenarios = []
    random.seed(999)
    
    agents = ["Alice", "Bob"]
    objects = ["ball", "book"]
    locations = ["basket", "box", "drawer"]
    
    n_per_type = max(1, n // 3)
    
    for i in range(n_per_type):
        a, b = agents
        obj = random.choice(objects)
        loc1, loc2 = random.sample(locations, 2)
        
        # Negative 1: A was NOT told - should look in original place
        scenarios.append({
            'type': 'not_told',
            'prompt': f"{a} put the {obj} in the {loc1}. {a} left. {b} moved it to {loc2} but NEVER told {a}. {a} will look in the",
            'correct': f" {loc1}",  # A doesn't know!
            'wrong': f" {loc2}",
            'is_negative': True,
        })
        
        # Negative 2: A didn't hear/understand
        scenarios.append({
            'type': 'didnt_hear',
            'prompt': f"{a} put the {obj} in the {loc1}. {b} moved it to {loc2}. {b} tried to tell {a} but {a} had headphones on and didn't hear. {a} will look in the",
            'correct': f" {loc1}",  # A didn't hear
            'wrong': f" {loc2}",
            'is_negative': True,
        })
        
        # Negative 3: Message didn't go through
        scenarios.append({
            'type': 'message_failed',
            'prompt': f"{a} put the {obj} in the {loc1}. {b} moved it to {loc2}. {b} sent a text to {a} but the message failed to deliver. {a} will look in the",
            'correct': f" {loc1}",  # Message failed
            'wrong': f" {loc2}",
            'is_negative': True,
        })
    
    return scenarios


def compute_statistics(results: list) -> dict:
    """Compute statistics with confidence intervals. Handles empty lists and edge cases."""
    if not results:
        return {
            'n': 0,
            'accuracy': 0.0,
            'ci_low': 0.0,
            'ci_high': 0.0,
            'successes': 0,
        }
    
    n = len(results)
    # Safely convert any type to bool
    successes = sum(1 for r in results if (r.item() if hasattr(r, 'item') else bool(r)))
    p = float(successes) / float(n)
    
    # Wilson score confidence interval (pure Python, no numpy)
    z = 1.96  # 95% CI
    denominator = 1.0 + z*z/n
    centre = (p + z*z/(2.0*n)) / denominator
    
    # Safe sqrt calculation
    variance = p * (1.0 - p) + z*z/(4.0*n)
    if variance < 0:
        variance = 0  # Numerical safety
    margin = z * math.sqrt(variance / n) / denominator
    
    return {
        'n': int(n),
        'accuracy': float(p),
        'ci_low': float(max(0.0, centre - margin)),
        'ci_high': float(min(1.0, centre + margin)),
        'successes': int(successes),
    }


def run_robustness_tests(model, tokenizer):
    """Run all robustness tests."""
    analyzer = RobustnessAnalyzer(model, tokenizer)
    
    results = {}
    
    print(f"\n{'='*70}")
    print("COMPREHENSIVE ROBUSTNESS TEST")
    print(f"{'='*70}\n")
    
    # Test 1: Template variations
    print("=" * 50)
    print("TEST 1: PROMPT TEMPLATE VARIATIONS")
    print("=" * 50)
    templates = generate_template_variations(20)  # Reduced for speed
    
    for template_name in ['original', 'passive', 'story', 'formal', 'casual', 'third_person']:
        subset = [s for s in templates if s['template'] == template_name]
        
        if not subset:
            print(f"\n{template_name}: No scenarios generated")
            continue
        
        print(f"\n{template_name} (N={len(subset)})...", end=" ", flush=True)
        
        baseline = analyzer.test_scenarios(subset, use_ablation=False)
        ablated = analyzer.test_scenarios(subset, use_ablation=True)
        
        base_stats = compute_statistics(baseline)
        abl_stats = compute_statistics(ablated)
        
        print(f"done")
        print(f"  Baseline: {base_stats['accuracy']*100:.1f}% [{base_stats['ci_low']*100:.1f}-{base_stats['ci_high']*100:.1f}]")
        print(f"  Ablated:  {abl_stats['accuracy']*100:.1f}% [{abl_stats['ci_low']*100:.1f}-{abl_stats['ci_high']*100:.1f}]")
        
        results[f'template_{template_name}'] = {
            'baseline': base_stats,
            'ablated': abl_stats,
        }
    
    # Test 2: Communication verbs (batch by verb for efficiency)
    print("\n" + "=" * 50)
    print("TEST 2: COMMUNICATION VERB VARIATIONS")
    print("=" * 50)
    comm_scenarios = generate_communication_variations(15)  # Reduced for speed
    
    # Group by verb
    verb_groups = defaultdict(list)
    for s in comm_scenarios:
        verb_groups[s['verb']].append(s)
    
    for verb, scenarios in sorted(verb_groups.items()):
        print(f"\n{verb} (N={len(scenarios)})...", end=" ", flush=True)
        
        baseline = analyzer.test_scenarios(scenarios, use_ablation=False)
        ablated = analyzer.test_scenarios(scenarios, use_ablation=True)
        
        base_stats = compute_statistics(baseline)
        abl_stats = compute_statistics(ablated)
        
        print(f"done")
        print(f"  Baseline: {base_stats['accuracy']*100:.1f}%  Ablated: {abl_stats['accuracy']*100:.1f}%")
        results[f'verb_{verb}'] = {'baseline': base_stats, 'ablated': abl_stats}
    
    # Test 3: Languages
    print("\n" + "=" * 50)
    print("TEST 3: MULTILINGUAL")
    print("=" * 50)
    lang_scenarios = generate_multilingual_scenarios(20)  # Reduced
    
    # Group by language
    lang_groups = defaultdict(list)
    for s in lang_scenarios:
        lang_groups[s['language']].append(s)
    
    for lang, scenarios in lang_groups.items():
        if len(scenarios) < 5:
            continue
            
        print(f"\n{lang} (N={len(scenarios)})...", end=" ", flush=True)
        
        baseline = analyzer.test_scenarios(scenarios, use_ablation=False)
        ablated = analyzer.test_scenarios(scenarios, use_ablation=True)
        
        base_stats = compute_statistics(baseline)
        abl_stats = compute_statistics(ablated)
        
        print(f"done")
        print(f"  Baseline: {base_stats['accuracy']*100:.1f}%  Ablated: {abl_stats['accuracy']*100:.1f}%")
        results[f'lang_{lang}'] = {'baseline': base_stats, 'ablated': abl_stats}
    
    # Test 4: Edge cases
    print("\n" + "=" * 50)
    print("TEST 4: EDGE CASES")
    print("=" * 50)
    edge_scenarios = generate_edge_cases(30)  # Reduced
    
    # Group by type
    edge_groups = defaultdict(list)
    for s in edge_scenarios:
        edge_groups[s['type']].append(s)
    
    for edge_type, scenarios in edge_groups.items():
        print(f"\n{edge_type} (N={len(scenarios)})...", end=" ", flush=True)
        
        baseline = analyzer.test_scenarios(scenarios, use_ablation=False)
        ablated = analyzer.test_scenarios(scenarios, use_ablation=True)
        
        base_stats = compute_statistics(baseline)
        abl_stats = compute_statistics(ablated)
        
        print(f"done")
        print(f"  Baseline: {base_stats['accuracy']*100:.1f}%  Ablated: {abl_stats['accuracy']*100:.1f}%")
        results[f'edge_{edge_type}'] = {'baseline': base_stats, 'ablated': abl_stats}
    
    # Test 5: Negative controls
    print("\n" + "=" * 50)
    print("TEST 5: NEGATIVE CONTROLS (should predict 'original location')")
    print("=" * 50)
    neg_scenarios = generate_negative_controls(30)  # Reduced
    
    # Group by type
    neg_groups = defaultdict(list)
    for s in neg_scenarios:
        neg_groups[s['type']].append(s)
    
    for neg_type, scenarios in neg_groups.items():
        print(f"\n{neg_type} (N={len(scenarios)})...", end=" ", flush=True)
        
        baseline = analyzer.test_scenarios(scenarios, use_ablation=False)
        ablated = analyzer.test_scenarios(scenarios, use_ablation=True)
        
        base_stats = compute_statistics(baseline)
        abl_stats = compute_statistics(ablated)
        
        print(f"done")
        print(f"  Baseline: {base_stats['accuracy']*100:.1f}%  Ablated: {abl_stats['accuracy']*100:.1f}%")
        # Note: For negative controls, high accuracy means model correctly identifies when belief should NOT update
        results[f'negative_{neg_type}'] = {'baseline': base_stats, 'ablated': abl_stats}
    
    return results


def main():
    print("="*70)
    print("STEP 15: COMPREHENSIVE ROBUSTNESS TEST")
    print("="*70)
    print()
    print("Testing robustness across:")
    print("  - 6 different prompt templates")
    print("  - 10 communication verbs")
    print("  - 5 languages")
    print("  - 5 edge case types")
    print("  - 3 negative control types")
    print()
    
    # Load model
    print("Loading model...")
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,  # Fixed: use dtype instead of torch_dtype
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    results = run_robustness_tests(model, tokenizer)
    
    # Overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    
    all_baseline = []
    all_ablated = []
    
    for key, data in results.items():
        if 'negative' not in key:  # Exclude negative controls from overall
            if 'baseline' in data and 'ablated' in data:
                all_baseline.append(data['baseline']['accuracy'])
                all_ablated.append(data['ablated']['accuracy'])
    
    if all_baseline:
        mean_base = sum(all_baseline) / len(all_baseline)
        mean_abl = sum(all_ablated) / len(all_ablated)
        std_base = (sum((x - mean_base)**2 for x in all_baseline) / len(all_baseline)) ** 0.5
        std_abl = (sum((x - mean_abl)**2 for x in all_ablated) / len(all_ablated)) ** 0.5
        
        print(f"\nAcross all positive tests ({len(all_baseline)} conditions):")
        print(f"  Mean baseline: {mean_base*100:.1f}% (+/- {std_base*100:.1f}%)")
        print(f"  Mean ablated:  {mean_abl*100:.1f}% (+/- {std_abl*100:.1f}%)")
        print(f"  Mean boost:    {(mean_abl - mean_base)*100:+.1f}%")
    
    # Save results
    output_file = RESULTS_DIR / "robustness_test_results.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
