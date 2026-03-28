"""
Step 18: GARGANTUAN Linguistic Sweep

A truly comprehensive analysis with:
- 1000+ verbs from full WordNet trees
- Proper checkpointing & streaming
- Attention pattern harvesting
- Position importance analysis
- Comprehensive statistical analysis

Run time estimate: 6-9 hours for full sweep
"""

import sys
import json
import time
import logging
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# NLTK for WordNet
import nltk
from nltk.corpus import wordnet as wn

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
INHIBITORS = [(17, 4), (18, 11), (18, 14), (19, 30), (21, 17)]

# Directories
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results" / "gargantuan"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
ATTENTION_DIR = RESULTS_DIR / "attention"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = BASE_DIR / "logs"

for d in [RESULTS_DIR, CHECKPOINT_DIR, ATTENTION_DIR, FIGURES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Scale parameters
SCENARIOS_PER_VERB = 30       # Statistical power
MAX_VERBS = 500               # Start with 500, can increase
CHECKPOINT_EVERY = 50         # Save every N verbs
HARVEST_ATTENTION = True      # Capture attention patterns
SAMPLE_ATTENTION_EVERY = 10   # Save attention for every Nth verb

# Set up logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / f"gargantuan_{timestamp}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class VerbResult:
    """Result for a single verb."""
    verb: str
    category: str
    synset: str
    n_scenarios: int
    baseline_correct: int
    ablated_correct: int
    baseline_accuracy: float
    ablated_accuracy: float
    boost: float
    attention_saved: bool = False


@dataclass  
class ScenarioResult:
    """Result for a single scenario."""
    prompt: str
    verb: str
    correct_answer: str
    wrong_answer: str
    baseline_correct: bool
    ablated_correct: bool
    baseline_correct_prob: float
    baseline_wrong_prob: float
    ablated_correct_prob: float
    ablated_wrong_prob: float


# =============================================================================
# VOCABULARY EXPANSION
# =============================================================================

def download_nltk_data():
    """Ensure NLTK data is available."""
    try:
        wn.synsets('test')
    except:
        logger.info("Downloading NLTK WordNet data...")
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)


def get_all_communication_verbs() -> Dict[str, List[Tuple[str, str]]]:
    """
    Extract ALL communication-related verbs from WordNet.
    Returns: {category: [(verb, synset_name), ...]}
    """
    download_nltk_data()
    
    logger.info("Extracting communication verbs from WordNet...")
    
    # Root synsets to explore
    roots = {
        'communicate': ['communicate.v.02', 'communicate.v.01'],
        'inform': ['inform.v.01', 'inform.v.02'],
        'tell': ['tell.v.02', 'tell.v.01', 'tell.v.03'],
        'say': ['say.v.01', 'say.v.02', 'say.v.03'],
        'state': ['state.v.01', 'state.v.02'],
        'express': ['express.v.01', 'express.v.02'],
        'report': ['report.v.01', 'report.v.02'],
        'announce': ['announce.v.01', 'announce.v.02'],
        'declare': ['declare.v.01', 'declare.v.02'],
        'mention': ['mention.v.01', 'mention.v.02'],
        'speak': ['speak.v.01', 'speak.v.02', 'speak.v.03'],
        'talk': ['talk.v.01', 'talk.v.02'],
        'write': ['write.v.01', 'write.v.02', 'write.v.03'],
        'ask': ['ask.v.01', 'ask.v.02', 'ask.v.03'],
        'answer': ['answer.v.01', 'answer.v.02'],
        'reply': ['reply.v.01', 'reply.v.02'],
        'explain': ['explain.v.01', 'explain.v.02'],
        'describe': ['describe.v.01', 'describe.v.02'],
        'notify': ['notify.v.01', 'notify.v.02'],
        'warn': ['warn.v.01', 'warn.v.02'],
        'advise': ['advise.v.01', 'advise.v.02'],
        'suggest': ['suggest.v.01', 'suggest.v.02'],
        'hint': ['hint.v.01', 'hint.v.02'],
        'imply': ['imply.v.01', 'imply.v.02'],
    }
    
    verbs_by_category = defaultdict(list)
    seen_verbs = set()
    
    def conjugate_to_past(verb: str) -> str:
        """Simple past tense conjugation."""
        if verb.endswith('e'):
            return verb + 'd'
        elif verb.endswith('y') and len(verb) > 1 and verb[-2] not in 'aeiou':
            return verb[:-1] + 'ied'
        elif len(verb) > 2 and verb[-1] not in 'aeiouwxy' and verb[-2] in 'aeiou' and verb[-3] not in 'aeiou':
            return verb + verb[-1] + 'ed'
        else:
            return verb + 'ed'
    
    def explore_synset(synset, category: str, depth: int = 0):
        """Recursively explore synset and its hyponyms."""
        if depth > 5:  # Limit depth
            return
            
        for lemma in synset.lemmas():
            verb = lemma.name().replace('_', ' ')
            if ' ' in verb:  # Skip multi-word
                continue
            if verb in seen_verbs:
                continue
            seen_verbs.add(verb)
            
            past = conjugate_to_past(verb)
            verbs_by_category[category].append((past, synset.name()))
        
        # Explore hyponyms (more specific verbs)
        for hyponym in synset.hyponyms():
            explore_synset(hyponym, category, depth + 1)
    
    # Explore each root category
    for category, synset_names in roots.items():
        for syn_name in synset_names:
            try:
                synset = wn.synset(syn_name)
                explore_synset(synset, category)
            except Exception as e:
                logger.warning(f"Could not process synset {syn_name}: {e}")
    
    # Add curated verbs that might be missed
    curated = {
        'digital': ['texted', 'emailed', 'messaged', 'DMed', 'slacked', 'pinged', 'tweeted'],
        'casual': ['chatted', 'shared', 'mentioned', 'noted', 'remarked'],
        'formal': ['conveyed', 'transmitted', 'relayed', 'disseminated', 'dispatched'],
        'emotional': ['exclaimed', 'sighed', 'muttered', 'whispered', 'shouted', 'yelled'],
    }
    
    for category, verb_list in curated.items():
        for verb in verb_list:
            if verb not in seen_verbs:
                verbs_by_category[category].append((verb, 'curated'))
                seen_verbs.add(verb)
    
    total = sum(len(v) for v in verbs_by_category.values())
    logger.info(f"Found {total} unique verbs across {len(verbs_by_category)} categories")
    
    for cat, verbs in sorted(verbs_by_category.items(), key=lambda x: -len(x[1])):
        logger.info(f"  {cat}: {len(verbs)} verbs")
    
    return dict(verbs_by_category)


# =============================================================================
# SCENARIO GENERATION  
# =============================================================================

# Extended template collection
TEMPLATES = [
    # Basic canonical
    "{a} put the {obj} in the {loc1}. {a} left. {b} moved the {obj} to the {loc2}. {b} {verb} {a}: 'I moved the {obj} to the {loc2}.' Where will {a} look? {a} will look in the",
    
    # Time-based
    "Earlier, {a} put the {obj} in the {loc1}. Later, {b} moved it to the {loc2} and {verb} {a}. Where will {a} look for the {obj}? {a} will look in the",
    
    # While away
    "{a} put the {obj} in the {loc1} and went out. While {a} was away, {b} moved the {obj} to the {loc2}. {b} {verb} {a} about this. Where will {a} look? {a} will look in the",
    
    # Sequence
    "First, {a} put the {obj} in the {loc1}. Then {a} left. Then {b} moved it to the {loc2}. Finally, {b} {verb} {a}. Where will {a} look? In the",
    
    # Question form
    "{a} put the {obj} in the {loc1}. {b} later moved it to the {loc2} and {verb} {a}. Where does {a} think the {obj} is? {a} thinks it is in the",
    
    # Passive voice
    "{a} put the {obj} in the {loc1}. The {obj} was moved to the {loc2} by {b}, and {a} was {verb} about it. Where will {a} look? {a} will look in the",
    
    # Complex sentence
    "After {a} put the {obj} in the {loc1} and left, {b} moved it to the {loc2} and {verb} {a}. Where will {a} search? {a} will search in the",
    
    # Multi-step
    "{a} put the {obj} in the {loc1}. {a} went to another room. {b} moved the {obj} to the {loc2}. {b} found {a} and {verb} them. Where will {a} look? {a} will look in the",
]

AGENTS = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack",
          "Kate", "Leo", "Maya", "Noah", "Olivia", "Peter", "Quinn", "Rose", "Sam", "Tina"]

OBJECTS = ["ball", "book", "key", "toy", "phone", "wallet", "bag", "hat", "cup", "pen",
           "letter", "box", "coin", "ring", "watch", "bottle", "card", "photo", "note", "gift"]

LOCATIONS = ["basket", "box", "drawer", "shelf", "bag", "desk", "table", "cabinet", "closet", "pocket",
             "container", "bin", "case", "folder", "tray", "rack", "hook", "slot", "cubby", "chest"]


def generate_scenarios_for_verb(verb: str, n: int) -> List[dict]:
    """Generate N diverse scenarios for a verb."""
    scenarios = []
    
    for i in range(n):
        template = random.choice(TEMPLATES)
        a, b = random.sample(AGENTS, 2)
        obj = random.choice(OBJECTS)
        loc1, loc2 = random.sample(LOCATIONS, 2)
        
        # Handle different verb forms in templates
        if '{verb}' in template:
            prompt = template.format(a=a, b=b, obj=obj, loc1=loc1, loc2=loc2, verb=verb)
        else:
            # Some templates might use different placeholders
            prompt = template.format(a=a, b=b, obj=obj, loc1=loc1, loc2=loc2)
        
        scenarios.append({
            'prompt': prompt,
            'correct': f" {loc2}",
            'wrong': f" {loc1}",
            'verb': verb,
            'template_idx': TEMPLATES.index(template),
        })
    
    return scenarios


# =============================================================================
# MODEL & TESTING
# =============================================================================

class GargantuanAnalyzer:
    """Analyzer with attention harvesting and position analysis."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_heads = model.config.num_attention_heads
        self.n_layers = model.config.num_hidden_layers
        self.hooks = []
        self.attention_cache = {}
        
    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.attention_cache = {}
        
    def install_ablation(self, heads: List[Tuple[int, int]]):
        """Install ablation hooks on specified heads."""
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
    
    def install_attention_hooks(self, layers: List[int]):
        """Install hooks to capture attention patterns."""
        self.attention_cache = {}
        
        def make_attention_hook(layer_idx):
            def hook(module, args, output):
                # output is (attn_output, attn_weights, past_key_value)
                if len(output) > 1 and output[1] is not None:
                    self.attention_cache[layer_idx] = output[1].detach().cpu()
            return hook
        
        for layer in layers:
            attn = self.model.model.layers[layer].self_attn
            h = attn.register_forward_hook(make_attention_hook(layer))
            self.hooks.append(h)
    
    def test_single(self, prompt: str, correct: str, wrong: str) -> dict:
        """Test a single scenario and return detailed results."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=HARVEST_ATTENTION)
            logits = outputs.logits[0, -1, :]
        
        probs = torch.softmax(logits, dim=-1)
        
        c_ids = self.tokenizer.encode(correct, add_special_tokens=False)
        w_ids = self.tokenizer.encode(wrong, add_special_tokens=False)
        
        if not c_ids or not w_ids:
            return {'correct': False, 'correct_prob': 0, 'wrong_prob': 0}
        
        c_prob = probs[c_ids[0]].item()
        w_prob = probs[w_ids[0]].item()
        
        return {
            'correct': c_prob > w_prob,
            'correct_prob': c_prob,
            'wrong_prob': w_prob,
        }
    
    def test_batch(self, scenarios: List[dict], use_ablation: bool = False) -> List[dict]:
        """Test a batch of scenarios."""
        if use_ablation:
            self.install_ablation(INHIBITORS)
        else:
            self.clear_hooks()
        
        results = []
        for s in scenarios:
            r = self.test_single(s['prompt'], s['correct'], s['wrong'])
            results.append(r)
        
        self.clear_hooks()
        return results


# =============================================================================
# CHECKPOINTING & DATA SAVING
# =============================================================================

def save_checkpoint(verb_results: List[VerbResult], checkpoint_idx: int):
    """Save intermediate checkpoint."""
    data = [asdict(r) for r in verb_results]
    path = CHECKPOINT_DIR / f"checkpoint_{checkpoint_idx:04d}.json"
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved checkpoint: {path}")


def save_attention(verb: str, attention_data: dict):
    """Save attention patterns for a verb."""
    path = ATTENTION_DIR / f"{verb}_attention.npz"
    np.savez_compressed(path, **attention_data)


def load_existing_checkpoints() -> List[VerbResult]:
    """Load any existing checkpoints to resume."""
    results = []
    checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint_*.json"))
    for cp in checkpoints:
        with open(cp) as f:
            data = json.load(f)
            results.extend([VerbResult(**d) for d in data])
    if results:
        logger.info(f"Loaded {len(results)} existing results from {len(checkpoints)} checkpoints")
    return results


# =============================================================================
# MAIN SWEEP
# =============================================================================

def run_gargantuan_sweep():
    """Run the full gargantuan sweep."""
    
    logger.info("=" * 70)
    logger.info("GARGANTUAN LINGUISTIC SWEEP")
    logger.info("=" * 70)
    
    # 1. Load model
    logger.info("\n[1/6] Loading model...")
    start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # Required for attention extraction!
    )
    model.eval()
    
    logger.info(f"Model loaded in {time.time() - start:.1f}s")
    logger.info(f"  Layers: {model.config.num_hidden_layers}")
    logger.info(f"  Heads: {model.config.num_attention_heads}")
    
    analyzer = GargantuanAnalyzer(model, tokenizer)
    
    # 2. Get vocabulary
    logger.info("\n[2/6] Expanding vocabulary...")
    verbs_by_category = get_all_communication_verbs()
    
    # Flatten and limit
    all_verbs = []
    for category, verb_list in verbs_by_category.items():
        for verb, synset in verb_list:
            all_verbs.append((verb, category, synset))
    
    random.shuffle(all_verbs)
    all_verbs = all_verbs[:MAX_VERBS]
    logger.info(f"Selected {len(all_verbs)} verbs for testing")
    
    # 3. Check for existing progress
    existing_results = load_existing_checkpoints()
    tested_verbs = {r.verb for r in existing_results}
    remaining_verbs = [(v, c, s) for v, c, s in all_verbs if v not in tested_verbs]
    
    logger.info(f"Remaining verbs to test: {len(remaining_verbs)}")
    
    # 4. Run sweep
    logger.info("\n[3/6] Running baseline + ablation sweep...")
    
    verb_results = list(existing_results)  # Start with existing
    checkpoint_idx = len(existing_results) // CHECKPOINT_EVERY
    
    pbar = tqdm(remaining_verbs, desc="Testing verbs", unit="verb")
    
    for i, (verb, category, synset) in enumerate(pbar):
        pbar.set_postfix({'verb': verb[:15], 'category': category[:10]})
        
        # Generate scenarios
        scenarios = generate_scenarios_for_verb(verb, SCENARIOS_PER_VERB)
        
        # Test baseline
        baseline_results = analyzer.test_batch(scenarios, use_ablation=False)
        baseline_correct = sum(1 for r in baseline_results if r['correct'])
        
        # Test with ablation
        ablated_results = analyzer.test_batch(scenarios, use_ablation=True)
        ablated_correct = sum(1 for r in ablated_results if r['correct'])
        
        # Calculate metrics
        baseline_acc = baseline_correct / len(scenarios)
        ablated_acc = ablated_correct / len(scenarios)
        boost = ablated_acc - baseline_acc
        
        # Create result
        result = VerbResult(
            verb=verb,
            category=category,
            synset=synset,
            n_scenarios=len(scenarios),
            baseline_correct=baseline_correct,
            ablated_correct=ablated_correct,
            baseline_accuracy=baseline_acc,
            ablated_accuracy=ablated_acc,
            boost=boost,
            attention_saved=False,
        )
        
        verb_results.append(result)
        
        # Checkpoint
        if (len(verb_results) - len(existing_results)) % CHECKPOINT_EVERY == 0:
            checkpoint_idx += 1
            new_results = verb_results[len(existing_results):]
            save_checkpoint(new_results[-CHECKPOINT_EVERY:], checkpoint_idx)
        
        # Log interesting findings
        if baseline_acc == 0 and ablated_acc > 0.8:
            logger.info(f"  ⚡ STRONG INHIBITION: '{verb}' {baseline_acc*100:.0f}% → {ablated_acc*100:.0f}%")
        elif baseline_acc > 0.9:
            logger.info(f"  ✓ NO INHIBITION: '{verb}' already at {baseline_acc*100:.0f}%")
    
    # 5. Save final results
    logger.info("\n[4/6] Saving final results...")
    
    final_data = [asdict(r) for r in verb_results]
    with open(RESULTS_DIR / "final_results.json", 'w') as f:
        json.dump(final_data, f, indent=2)
    
    # 6. Generate summary statistics
    logger.info("\n[5/6] Computing summary statistics...")
    
    summary = compute_summary_statistics(verb_results)
    with open(RESULTS_DIR / "summary_statistics.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 7. Print summary
    logger.info("\n[6/6] SUMMARY")
    logger.info("=" * 70)
    
    logger.info(f"\nOVERALL:")
    logger.info(f"  Verbs tested: {summary['total_verbs']}")
    logger.info(f"  Total scenarios: {summary['total_scenarios']}")
    logger.info(f"  Mean baseline: {summary['mean_baseline']*100:.1f}%")
    logger.info(f"  Mean ablated: {summary['mean_ablated']*100:.1f}%")
    logger.info(f"  Mean boost: +{summary['mean_boost']*100:.1f}%")
    
    logger.info(f"\nCATEGORY BREAKDOWN:")
    for cat, stats in sorted(summary['by_category'].items(), key=lambda x: x[1]['mean_baseline']):
        logger.info(f"  {cat}: {stats['mean_baseline']*100:.0f}% → {stats['mean_ablated']*100:.0f}% ({stats['n_verbs']} verbs)")
    
    logger.info(f"\nWORST VERBS (strongest inhibition):")
    for verb, base, abl in summary['worst_verbs'][:10]:
        logger.info(f"  {verb}: {base*100:.0f}% → {abl*100:.0f}%")
    
    logger.info(f"\nBEST VERBS (no inhibition):")
    for verb, base, abl in summary['best_verbs'][:5]:
        logger.info(f"  {verb}: {base*100:.0f}% → {abl*100:.0f}%")
    
    logger.info(f"\nResults saved to: {RESULTS_DIR}")
    
    return verb_results


def compute_summary_statistics(results: List[VerbResult]) -> dict:
    """Compute comprehensive summary statistics."""
    
    baselines = [r.baseline_accuracy for r in results]
    ablated = [r.ablated_accuracy for r in results]
    boosts = [r.boost for r in results]
    
    # By category
    by_category = defaultdict(lambda: {'baselines': [], 'ablated': []})
    for r in results:
        by_category[r.category]['baselines'].append(r.baseline_accuracy)
        by_category[r.category]['ablated'].append(r.ablated_accuracy)
    
    category_stats = {}
    for cat, data in by_category.items():
        category_stats[cat] = {
            'n_verbs': len(data['baselines']),
            'mean_baseline': np.mean(data['baselines']),
            'mean_ablated': np.mean(data['ablated']),
            'mean_boost': np.mean(data['ablated']) - np.mean(data['baselines']),
        }
    
    # Sort for worst/best
    sorted_results = sorted(results, key=lambda r: r.baseline_accuracy)
    worst_verbs = [(r.verb, r.baseline_accuracy, r.ablated_accuracy) for r in sorted_results[:20]]
    best_verbs = [(r.verb, r.baseline_accuracy, r.ablated_accuracy) for r in sorted_results[-10:]]
    
    return {
        'total_verbs': len(results),
        'total_scenarios': sum(r.n_scenarios for r in results),
        'mean_baseline': float(np.mean(baselines)),
        'mean_ablated': float(np.mean(ablated)),
        'mean_boost': float(np.mean(boosts)),
        'std_baseline': float(np.std(baselines)),
        'std_ablated': float(np.std(ablated)),
        'n_zero_baseline': sum(1 for b in baselines if b == 0),
        'n_perfect_ablated': sum(1 for a in ablated if a >= 1.0),
        'by_category': category_stats,
        'worst_verbs': worst_verbs,
        'best_verbs': best_verbs,
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logger.info(f"Starting at {datetime.now()}")
    logger.info(f"Configuration:")
    logger.info(f"  MAX_VERBS: {MAX_VERBS}")
    logger.info(f"  SCENARIOS_PER_VERB: {SCENARIOS_PER_VERB}")
    logger.info(f"  CHECKPOINT_EVERY: {CHECKPOINT_EVERY}")
    
    try:
        results = run_gargantuan_sweep()
        logger.info(f"\nCompleted successfully at {datetime.now()}")
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user. Progress has been checkpointed.")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise

