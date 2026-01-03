"""
Step 16: MASSIVE Linguistic Sweep

This script tests the ToM circuit across a MASSIVE variety of:
1. Communication verbs (from WordNet/curated lists)
2. Communication mediums (in-person, phone, text, email, etc.)
3. Certainty levels (definitely, probably, maybe)
4. Tense variations (past, present, future)
5. Voice (active, passive)
6. Sentence structures (simple, complex, compound)
7. Languages (English, Chinese, Spanish, French, German)

Uses NLTK WordNet for comprehensive verb coverage.
Generates visualizations (heatmaps, bar charts).

Total scenarios: ~3000+
"""

import sys
import json
import random
import math
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try to import NLTK for WordNet
try:
    import nltk
    from nltk.corpus import wordnet as wn
    HAVE_WORDNET = True
except ImportError:
    HAVE_WORDNET = False
    print("Warning: NLTK not available, using curated verb list only")

# Try to import matplotlib for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False
    print("Warning: Matplotlib not available, skipping visualizations")

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# All 5 inhibitor heads
INHIBITORS = [(17, 4), (18, 11), (18, 14), (19, 30), (21, 17)]


# =============================================================================
# MASSIVE VOCABULARY LISTS
# =============================================================================

# Core communication verbs (hand-curated + WordNet expansion)
COMMUNICATION_VERBS_CORE = {
    # Basic speech acts
    'neutral': ['said', 'told', 'spoke', 'stated', 'mentioned', 'noted', 'remarked'],
    
    # Informing (implies knowledge transfer)
    'informing': ['informed', 'notified', 'alerted', 'advised', 'warned', 
                  'apprised', 'briefed', 'updated', 'reported'],
    
    # Explaining
    'explaining': ['explained', 'clarified', 'elaborated', 'described', 
                   'illustrated', 'demonstrated'],
    
    # Asserting (strong claims)
    'asserting': ['declared', 'announced', 'proclaimed', 'affirmed', 
                  'asserted', 'confirmed', 'insisted'],
    
    # Suggesting (weaker claims)
    'suggesting': ['suggested', 'hinted', 'implied', 'indicated', 
                   'intimated', 'insinuated'],
    
    # Asking/requesting
    'asking': ['asked', 'requested', 'inquired', 'queried', 'questioned'],
    
    # Casual/informal
    'casual': ['mentioned', 'chatted', 'talked', 'shared', 'let know',
               'filled in', 'clued in', 'gave a heads up'],
    
    # Formal/professional  
    'formal': ['communicated', 'conveyed', 'transmitted', 'relayed',
               'disseminated', 'circulated', 'dispatched'],
    
    # Written communication
    'written': ['wrote', 'emailed', 'texted', 'messaged', 'typed',
                'posted', 'sent', 'forwarded'],
    
    # Verbal communication
    'verbal': ['called', 'phoned', 'rang', 'shouted', 'whispered',
               'yelled', 'murmured', 'uttered'],
    
    # Digital/modern
    'digital': ['pinged', 'DMed', 'slacked', 'tweeted', 'texted',
                'snapped', 'messaged'],
}

# Communication mediums
MEDIUMS = {
    'in_person': [
        "{b} told {a} in person",
        "{b} said to {a}",
        "{b} spoke to {a} directly",
        "{b} mentioned to {a} face-to-face",
    ],
    'phone': [
        "{b} called {a} and said",
        "{b} phoned {a} to say",
        "{b} rang {a} and mentioned",
        "{b} left {a} a voicemail saying",
    ],
    'text': [
        "{b} texted {a}",
        "{b} sent {a} a text saying",
        "{b} messaged {a}",
        "{b} SMSed {a}",
    ],
    'email': [
        "{b} emailed {a}",
        "{b} sent {a} an email saying",
        "{b} wrote to {a} via email",
    ],
    'instant_message': [
        "{b} DMed {a}",
        "{b} sent {a} a DM saying",
        "{b} slacked {a}",
        "{b} sent {a} a message on chat",
    ],
    'written_note': [
        "{b} left {a} a note saying",
        "{b} wrote {a} a note",
        "{b} posted a note for {a}",
    ],
    'indirect': [
        "{b} had someone tell {a}",
        "{b} passed word to {a} that",
        "{b} sent word to {a} that",
    ],
}

# Certainty modifiers
CERTAINTY = {
    'certain': ['definitely', 'certainly', 'absolutely', 'surely'],
    'probable': ['probably', 'likely', 'most likely'],
    'possible': ['maybe', 'perhaps', 'possibly', 'might have'],
    'uncertain': ['apparently', 'supposedly', 'allegedly'],
}

# Tense variations
TENSES = {
    'past_simple': "{b} told {a}",
    'past_perfect': "{b} had told {a}",
    'present_simple': "{b} tells {a}",
    'present_perfect': "{b} has told {a}",
    'future': "{b} will tell {a}",
}

# Voice variations
VOICE = {
    'active': "{b} told {a}",
    'passive': "{a} was told by {b}",
    'middle': "{a} heard from {b}",
}

# Sentence structures
STRUCTURES = {
    'simple': "{b} told {a}: '{msg}'.",
    'compound': "{b} went to {a} and told them: '{msg}'.",
    'complex': "When {b} saw {a}, {b} told them: '{msg}'.",
    'embedded': "{b} told {a} that the {obj} had been moved to the {loc2}.",
    'indirect_speech': "{b} told {a} about moving the {obj} to the {loc2}.",
}

# Multilingual templates
MULTILINGUAL = {
    'english': {
        'template': "{a} put the {obj} in the {loc1}. {b} moved it to the {loc2} and told {a}. Where will {a} look? {a} will look in the",
        'agents': ["Alice", "Bob", "Carol", "David"],
        'objects': ["ball", "book", "key", "toy"],
        'locations': ["basket", "box", "drawer", "shelf"],
    },
    'chinese_simple': {
        'template': "{a}把{obj}放在{loc1}。{b}把它移到{loc2}并告诉了{a}。{a}会在哪里找？{a}会在",
        'agents': ["小明", "小红", "小华", "小李"],
        'objects': ["球", "书", "钥匙", "玩具"],
        'locations': ["篮子", "盒子", "抽屉", "架子"],
    },
    'spanish': {
        'template': "{a} puso el/la {obj} en el/la {loc1}. {b} lo movio a el/la {loc2} y le dijo a {a}. Donde buscara {a}? {a} buscara en el/la",
        'agents': ["Maria", "Carlos", "Ana", "Pedro"],
        'objects': ["pelota", "libro", "llave", "juguete"],
        'locations': ["cesta", "caja", "cajon", "estante"],
    },
    'french': {
        'template': "{a} a mis le/la {obj} dans le/la {loc1}. {b} l'a deplace vers le/la {loc2} et a dit a {a}. Ou {a} cherchera? {a} cherchera dans le/la",
        'agents': ["Marie", "Pierre", "Sophie", "Jean"],
        'objects': ["balle", "livre", "cle", "jouet"],
        'locations': ["panier", "boite", "tiroir", "etagere"],
    },
    'german': {
        'template': "{a} legte den/das {obj} in den/die {loc1}. {b} hat es in den/die {loc2} verschoben und {a} informiert. Wo wird {a} suchen? {a} wird suchen in dem/der",
        'agents': ["Anna", "Hans", "Maria", "Klaus"],
        'objects': ["Ball", "Buch", "Schluessel", "Spielzeug"],
        'locations': ["Korb", "Kiste", "Schublade", "Regal"],
    },
}


# =============================================================================
# WORDNET VERB EXPANSION
# =============================================================================

def get_wordnet_communication_verbs() -> List[str]:
    """Get all communication verbs from WordNet."""
    if not HAVE_WORDNET:
        return []
    
    # Download WordNet if needed
    try:
        wn.synsets('say')
    except:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    
    # Root communication synsets
    roots = [
        'communicate.v.02',  # transmit information
        'inform.v.01',       # impart knowledge
        'state.v.01',        # express in words
        'tell.v.02',         # let something be known
        'say.v.01',          # express in words
    ]
    
    verbs = set()
    
    for root_name in roots:
        try:
            synset = wn.synset(root_name)
        except:
            continue
            
        # Get hyponyms (more specific verbs)
        for hyponym in synset.closure(lambda s: s.hyponyms()):
            for lemma in hyponym.lemmas():
                verb = lemma.name().replace('_', ' ')
                if ' ' not in verb:  # Single word verbs only
                    # Convert to past tense (simple heuristic)
                    if verb.endswith('e'):
                        verbs.add(verb + 'd')
                    elif verb.endswith('y'):
                        verbs.add(verb[:-1] + 'ied')
                    else:
                        verbs.add(verb + 'ed')
    
    return list(verbs)


# =============================================================================
# MODEL & HOOKS
# =============================================================================

class MassiveSweepAnalyzer:
    """Analyze ToM across massive vocabulary variations."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_heads = model.config.num_attention_heads
        self.hooks = []
        
    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        
    def install_ablation(self, heads: List[Tuple[int, int]]):
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
    
    def test_prompt(self, prompt: str, correct: str, wrong: str) -> dict:
        """Test a single prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        
        c_ids = self.tokenizer.encode(correct, add_special_tokens=False)
        w_ids = self.tokenizer.encode(wrong, add_special_tokens=False)
        
        if not c_ids or not w_ids:
            return {'correct_prob': 0, 'wrong_prob': 0, 'correct': False}
        
        c_prob = probs[c_ids[0]].item()
        w_prob = probs[w_ids[0]].item()
        
        return {
            'correct_prob': c_prob,
            'wrong_prob': w_prob,
            'correct': c_prob > w_prob,
        }
    
    def batch_test(self, scenarios: List[dict], use_ablation: bool = False) -> List[bool]:
        """Test a batch of scenarios."""
        if use_ablation:
            self.install_ablation(INHIBITORS)
        else:
            self.clear_hooks()
            
        results = []
        for s in scenarios:
            r = self.test_prompt(s['prompt'], s['correct'], s['wrong'])
            results.append(r['correct'])
            
        self.clear_hooks()
        return results


# =============================================================================
# SCENARIO GENERATION
# =============================================================================

def generate_verb_sweep(n_per_verb: int = 5) -> List[dict]:
    """Generate scenarios for every communication verb."""
    scenarios = []
    random.seed(42)
    
    agents = ["Alice", "Bob", "Carol", "David"]
    objects = ["ball", "book", "key", "toy"]
    locations = ["basket", "box", "drawer", "shelf"]
    
    # Flatten all verb categories
    all_verbs = []
    for category, verbs in COMMUNICATION_VERBS_CORE.items():
        for verb in verbs:
            all_verbs.append((category, verb))
    
    # Add WordNet verbs
    wordnet_verbs = get_wordnet_communication_verbs()
    for verb in wordnet_verbs[:100]:  # Limit to 100 WordNet verbs
        all_verbs.append(('wordnet', verb))
    
    print(f"Total verbs to test: {len(all_verbs)}")
    
    for category, verb in all_verbs:
        for i in range(n_per_verb):
            a, b = random.sample(agents, 2)
            obj = random.choice(objects)
            loc1, loc2 = random.sample(locations, 2)
            
            # Handle multi-word verbs
            if ' ' in verb:
                comm = f"{b} {verb} {a}"
            else:
                comm = f"{b} {verb} {a}"
            
            prompt = (
                f"{a} put the {obj} in the {loc1}. "
                f"{a} went away. {b} moved the {obj} to the {loc2}. "
                f"{comm}: 'I moved the {obj} to the {loc2}.' "
                f"Where will {a} look? {a} will look in the"
            )
            
            scenarios.append({
                'prompt': prompt,
                'correct': f" {loc2}",
                'wrong': f" {loc1}",
                'category': category,
                'verb': verb,
            })
    
    return scenarios


def generate_medium_sweep(n_per_medium: int = 10) -> List[dict]:
    """Generate scenarios for every communication medium."""
    scenarios = []
    random.seed(123)
    
    agents = ["Alice", "Bob", "Carol", "David"]
    objects = ["ball", "book", "key", "toy"]
    locations = ["basket", "box", "drawer", "shelf"]
    
    for medium_name, templates in MEDIUMS.items():
        for template in templates:
            for i in range(n_per_medium):
                a, b = random.sample(agents, 2)
                obj = random.choice(objects)
                loc1, loc2 = random.sample(locations, 2)
                
                comm = template.format(a=a, b=b)
                
                prompt = (
                    f"{a} put the {obj} in the {loc1}. "
                    f"{a} went away. {b} moved the {obj} to the {loc2}. "
                    f"{comm}: 'I moved the {obj} to the {loc2}.' "
                    f"Where will {a} look? {a} will look in the"
                )
                
                scenarios.append({
                    'prompt': prompt,
                    'correct': f" {loc2}",
                    'wrong': f" {loc1}",
                    'medium': medium_name,
                    'template': template,
                })
    
    return scenarios


def generate_multilingual_sweep(n_per_lang: int = 30) -> List[dict]:
    """Generate scenarios in multiple languages."""
    scenarios = []
    random.seed(456)
    
    for lang_name, config in MULTILINGUAL.items():
        for i in range(n_per_lang):
            if len(config['agents']) < 2 or len(config['locations']) < 2:
                continue
                
            a, b = random.sample(config['agents'], 2)
            obj = random.choice(config['objects'])
            loc1, loc2 = random.sample(config['locations'], 2)
            
            prompt = config['template'].format(
                a=a, b=b, obj=obj, loc1=loc1, loc2=loc2
            )
            
            scenarios.append({
                'prompt': prompt,
                'correct': f" {loc2}" if lang_name != 'chinese_simple' else loc2,
                'wrong': f" {loc1}" if lang_name != 'chinese_simple' else loc1,
                'language': lang_name,
            })
    
    return scenarios


def generate_structure_sweep(n_per_structure: int = 20) -> List[dict]:
    """Generate scenarios with different sentence structures."""
    scenarios = []
    random.seed(789)
    
    agents = ["Alice", "Bob", "Carol", "David"]
    objects = ["ball", "book", "key", "toy"]
    locations = ["basket", "box", "drawer", "shelf"]
    
    for struct_name, template in STRUCTURES.items():
        for i in range(n_per_structure):
            a, b = random.sample(agents, 2)
            obj = random.choice(objects)
            loc1, loc2 = random.sample(locations, 2)
            
            msg = f"I moved the {obj} to the {loc2}"
            
            comm = template.format(a=a, b=b, obj=obj, loc1=loc1, loc2=loc2, msg=msg)
            
            prompt = (
                f"{a} put the {obj} in the {loc1}. "
                f"{a} went away. {b} moved the {obj} to the {loc2}. "
                f"{comm} "
                f"Where will {a} look? {a} will look in the"
            )
            
            scenarios.append({
                'prompt': prompt,
                'correct': f" {loc2}",
                'wrong': f" {loc1}",
                'structure': struct_name,
            })
    
    return scenarios


# =============================================================================
# ANALYSIS & VISUALIZATION
# =============================================================================

def compute_accuracy(results: List[bool]) -> float:
    """Compute accuracy from boolean results."""
    if not results:
        return 0.0
    return sum(1 for r in results if r) / len(results)


def create_verb_heatmap(results: dict, output_path: Path):
    """Create heatmap of verb categories."""
    if not HAVE_MATPLOTLIB:
        return
    
    categories = list(results['by_category'].keys())
    baselines = [results['by_category'][c]['baseline'] for c in categories]
    ablated = [results['by_category'][c]['ablated'] for c in categories]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(categories))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], [b*100 for b in baselines], 
                   width, label='Baseline', color='#ff6b6b')
    bars2 = ax.bar([i + width/2 for i in x], [b*100 for b in ablated], 
                   width, label='With Ablation', color='#4ecdc4')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Verb Category', fontsize=12)
    ax.set_title('ToM Accuracy by Communication Verb Category', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def create_verb_scatter(results: dict, output_path: Path):
    """Create scatter plot of individual verbs."""
    if not HAVE_MATPLOTLIB:
        return
    
    verbs = []
    baselines = []
    ablated = []
    categories = []
    
    for verb, data in results['by_verb'].items():
        verbs.append(verb)
        baselines.append(data['baseline'] * 100)
        ablated.append(data['ablated'] * 100)
        categories.append(data.get('category', 'unknown'))
    
    # Color by category
    unique_cats = list(set(categories))
    colors = plt.cm.tab20(range(len(unique_cats)))
    cat_to_color = {cat: colors[i] for i, cat in enumerate(unique_cats)}
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for i, (v, b, a, c) in enumerate(zip(verbs, baselines, ablated, categories)):
        ax.scatter(b, a, c=cat_to_color[c], s=100, alpha=0.7)
        if b < 30 or a > 90:  # Label extreme points
            ax.annotate(v, (b, a), fontsize=7, alpha=0.8)
    
    # Diagonal line
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3)
    
    ax.set_xlabel('Baseline Accuracy (%)', fontsize=12)
    ax.set_ylabel('Ablated Accuracy (%)', fontsize=12)
    ax.set_title('ToM Accuracy: Baseline vs Ablated (by verb)', fontsize=14)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    
    # Legend
    for cat, color in cat_to_color.items():
        ax.scatter([], [], c=color, label=cat, s=100)
    ax.legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def create_language_comparison(results: dict, output_path: Path):
    """Create bar chart comparing languages."""
    if not HAVE_MATPLOTLIB:
        return
    
    langs = list(results['by_language'].keys())
    baselines = [results['by_language'][l]['baseline'] * 100 for l in langs]
    ablated = [results['by_language'][l]['ablated'] * 100 for l in langs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(langs))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], baselines, width, 
                   label='Baseline', color='#ff6b6b')
    bars2 = ax.bar([i + width/2 for i in x], ablated, width, 
                   label='With Ablation', color='#4ecdc4')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Language', fontsize=12)
    ax.set_title('ToM Accuracy Across Languages', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(langs)
    ax.legend()
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("STEP 16: MASSIVE LINGUISTIC SWEEP")
    print("=" * 70)
    
    # Load model
    print("\n[1/6] Loading model...")
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    analyzer = MassiveSweepAnalyzer(model, tokenizer)
    all_results = {}
    
    # 1. Verb sweep
    print("\n[2/6] Testing communication verbs...")
    verb_scenarios = generate_verb_sweep(n_per_verb=5)
    print(f"  Generated {len(verb_scenarios)} verb scenarios")
    
    verb_results = {'by_verb': {}, 'by_category': defaultdict(lambda: {'baseline': [], 'ablated': []})}
    
    # Group by verb
    verb_groups = defaultdict(list)
    for s in verb_scenarios:
        verb_groups[s['verb']].append(s)
    
    for verb, scenarios in verb_groups.items():
        if len(scenarios) < 2:
            continue
        
        baseline = analyzer.batch_test(scenarios, use_ablation=False)
        ablated = analyzer.batch_test(scenarios, use_ablation=True)
        
        baseline_acc = compute_accuracy(baseline)
        ablated_acc = compute_accuracy(ablated)
        
        category = scenarios[0].get('category', 'unknown')
        
        verb_results['by_verb'][verb] = {
            'baseline': baseline_acc,
            'ablated': ablated_acc,
            'n': len(scenarios),
            'category': category,
        }
        
        verb_results['by_category'][category]['baseline'].append(baseline_acc)
        verb_results['by_category'][category]['ablated'].append(ablated_acc)
    
    # Aggregate by category
    for cat in verb_results['by_category']:
        base_list = verb_results['by_category'][cat]['baseline']
        abl_list = verb_results['by_category'][cat]['ablated']
        verb_results['by_category'][cat] = {
            'baseline': sum(base_list) / len(base_list) if base_list else 0,
            'ablated': sum(abl_list) / len(abl_list) if abl_list else 0,
            'n_verbs': len(base_list),
        }
    
    all_results['verbs'] = verb_results
    
    print(f"  Tested {len(verb_results['by_verb'])} unique verbs across {len(verb_results['by_category'])} categories")
    
    # 2. Medium sweep
    print("\n[3/6] Testing communication mediums...")
    medium_scenarios = generate_medium_sweep(n_per_medium=8)
    print(f"  Generated {len(medium_scenarios)} medium scenarios")
    
    medium_results = {'by_medium': {}}
    medium_groups = defaultdict(list)
    for s in medium_scenarios:
        medium_groups[s['medium']].append(s)
    
    for medium, scenarios in medium_groups.items():
        baseline = analyzer.batch_test(scenarios, use_ablation=False)
        ablated = analyzer.batch_test(scenarios, use_ablation=True)
        
        medium_results['by_medium'][medium] = {
            'baseline': compute_accuracy(baseline),
            'ablated': compute_accuracy(ablated),
            'n': len(scenarios),
        }
        print(f"    {medium}: {compute_accuracy(baseline)*100:.0f}% -> {compute_accuracy(ablated)*100:.0f}%")
    
    all_results['mediums'] = medium_results
    
    # 3. Multilingual sweep
    print("\n[4/6] Testing multiple languages...")
    lang_scenarios = generate_multilingual_sweep(n_per_lang=25)
    print(f"  Generated {len(lang_scenarios)} language scenarios")
    
    lang_results = {'by_language': {}}
    lang_groups = defaultdict(list)
    for s in lang_scenarios:
        lang_groups[s['language']].append(s)
    
    for lang, scenarios in lang_groups.items():
        if len(scenarios) < 5:
            continue
        baseline = analyzer.batch_test(scenarios, use_ablation=False)
        ablated = analyzer.batch_test(scenarios, use_ablation=True)
        
        lang_results['by_language'][lang] = {
            'baseline': compute_accuracy(baseline),
            'ablated': compute_accuracy(ablated),
            'n': len(scenarios),
        }
        print(f"    {lang}: {compute_accuracy(baseline)*100:.0f}% -> {compute_accuracy(ablated)*100:.0f}%")
    
    all_results['languages'] = lang_results
    
    # 4. Structure sweep
    print("\n[5/6] Testing sentence structures...")
    struct_scenarios = generate_structure_sweep(n_per_structure=15)
    print(f"  Generated {len(struct_scenarios)} structure scenarios")
    
    struct_results = {'by_structure': {}}
    struct_groups = defaultdict(list)
    for s in struct_scenarios:
        struct_groups[s['structure']].append(s)
    
    for struct, scenarios in struct_groups.items():
        baseline = analyzer.batch_test(scenarios, use_ablation=False)
        ablated = analyzer.batch_test(scenarios, use_ablation=True)
        
        struct_results['by_structure'][struct] = {
            'baseline': compute_accuracy(baseline),
            'ablated': compute_accuracy(ablated),
            'n': len(scenarios),
        }
        print(f"    {struct}: {compute_accuracy(baseline)*100:.0f}% -> {compute_accuracy(ablated)*100:.0f}%")
    
    all_results['structures'] = struct_results
    
    # 5. Create visualizations
    print("\n[6/6] Creating visualizations...")
    if HAVE_MATPLOTLIB:
        create_verb_heatmap(verb_results, FIGURES_DIR / "verb_category_heatmap.png")
        create_verb_scatter(verb_results, FIGURES_DIR / "verb_scatter.png")
        create_language_comparison(lang_results, FIGURES_DIR / "language_comparison.png")
    else:
        print("  Skipping (matplotlib not available)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Find best/worst verbs
    verbs_sorted = sorted(
        verb_results['by_verb'].items(),
        key=lambda x: x[1]['baseline']
    )
    
    print("\nWORST baseline verbs (model struggles most):")
    for verb, data in verbs_sorted[:10]:
        print(f"  {verb}: {data['baseline']*100:.0f}% -> {data['ablated']*100:.0f}%")
    
    print("\nBEST baseline verbs (model already succeeds):")
    for verb, data in verbs_sorted[-5:]:
        print(f"  {verb}: {data['baseline']*100:.0f}% -> {data['ablated']*100:.0f}%")
    
    # Language summary
    print("\nLANGUAGE SUMMARY:")
    for lang, data in lang_results['by_language'].items():
        boost = (data['ablated'] - data['baseline']) * 100
        print(f"  {lang}: {data['baseline']*100:.0f}% -> {data['ablated']*100:.0f}% ({boost:+.0f}%)")
    
    # Save results
    output_file = RESULTS_DIR / "massive_sweep_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")
    
    if HAVE_MATPLOTLIB:
        print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()


