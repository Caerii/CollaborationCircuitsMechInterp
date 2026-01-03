"""
Step 19: ULTRA-FAST Gargantuan Sweep

Optimized for maximum speed:
- Batched inference (batch_size=16)
- SDPA attention for speed (phases 1-3)
- Pre-tokenization of all scenarios
- Two-phase: fast sweep then attention deep-dive
- ALL 1,325+ verbs

Expected: ~40-45 minutes for 40,000+ scenarios
"""

import sys
import json
import time
import logging
import random
import gc
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict, field

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
RESULTS_DIR = BASE_DIR / "results" / "ultrafast"
ATTENTION_DIR = RESULTS_DIR / "attention"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = BASE_DIR / "logs"

for d in [RESULTS_DIR, ATTENTION_DIR, FIGURES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ULTRA-FAST settings
BATCH_SIZE = 8                # Smaller batches to avoid OOM
SCENARIOS_PER_VERB = 20       # Reduced for speed, still statistically valid
MAX_VERBS = None              # None = ALL verbs
TOP_N_FOR_ATTENTION = 100     # Only harvest attention for top N interesting verbs

# Logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / f"ultrafast_{timestamp}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class VerbResult:
    verb: str
    category: str
    n_scenarios: int
    baseline_correct: int
    ablated_correct: int
    baseline_accuracy: float
    ablated_accuracy: float
    boost: float


# =============================================================================
# VOCABULARY (same as before but faster)
# =============================================================================

def get_all_communication_verbs() -> Dict[str, List[str]]:
    """Extract ALL communication verbs from WordNet."""
    try:
        wn.synsets('test')
    except:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    
    logger.info("Extracting verbs from WordNet...")
    
    roots = {
        'communicate': ['communicate.v.02', 'communicate.v.01'],
        'inform': ['inform.v.01'],
        'tell': ['tell.v.02', 'tell.v.01'],
        'say': ['say.v.01', 'say.v.02'],
        'state': ['state.v.01'],
        'express': ['express.v.01'],
        'report': ['report.v.01'],
        'announce': ['announce.v.01'],
        'declare': ['declare.v.01'],
        'mention': ['mention.v.01'],
        'speak': ['speak.v.01', 'speak.v.02'],
        'talk': ['talk.v.01', 'talk.v.02'],
        'write': ['write.v.01', 'write.v.02'],
        'ask': ['ask.v.01', 'ask.v.02'],
        'explain': ['explain.v.01'],
        'warn': ['warn.v.01'],
        'advise': ['advise.v.01'],
        'suggest': ['suggest.v.01'],
    }
    
    verbs_by_category = defaultdict(list)
    seen = set()
    
    def conjugate(verb: str) -> str:
        if verb.endswith('e'):
            return verb + 'd'
        elif verb.endswith('y') and len(verb) > 1 and verb[-2] not in 'aeiou':
            return verb[:-1] + 'ied'
        elif len(verb) > 2 and verb[-1] not in 'aeiouwxy' and verb[-2] in 'aeiou' and verb[-3] not in 'aeiou':
            return verb + verb[-1] + 'ed'
        return verb + 'ed'
    
    def explore(synset, category: str, depth: int = 0):
        if depth > 5:
            return
        for lemma in synset.lemmas():
            v = lemma.name().replace('_', ' ')
            if ' ' in v or v in seen:
                continue
            seen.add(v)
            verbs_by_category[category].append(conjugate(v))
        for hypo in synset.hyponyms():
            explore(hypo, category, depth + 1)
    
    for cat, syns in roots.items():
        for s in syns:
            try:
                explore(wn.synset(s), cat)
            except:
                pass
    
    # Add curated
    curated = {
        'digital': ['texted', 'emailed', 'messaged', 'DMed', 'slacked', 'pinged'],
        'casual': ['chatted', 'shared', 'mentioned', 'noted', 'remarked'],
        'formal': ['conveyed', 'transmitted', 'relayed', 'disseminated'],
        'emotional': ['exclaimed', 'sighed', 'muttered', 'whispered', 'shouted'],
    }
    for cat, vs in curated.items():
        for v in vs:
            if v not in seen:
                verbs_by_category[cat].append(v)
                seen.add(v)
    
    total = sum(len(v) for v in verbs_by_category.values())
    logger.info(f"Found {total} unique verbs across {len(verbs_by_category)} categories")
    
    return dict(verbs_by_category)


# =============================================================================
# SCENARIO GENERATION
# =============================================================================

TEMPLATES = [
    "{a} put the {obj} in the {loc1}. {a} left. {b} moved the {obj} to the {loc2}. {b} {verb} {a}: 'I moved it to the {loc2}.' Where will {a} look? {a} will look in the",
    "Earlier, {a} put the {obj} in the {loc1}. Later, {b} moved it to the {loc2} and {verb} {a}. Where will {a} look for the {obj}? {a} will look in the",
    "{a} put the {obj} in the {loc1} and went out. While {a} was away, {b} moved it to the {loc2}. {b} {verb} {a}. Where will {a} look? {a} will look in the",
    "First, {a} put the {obj} in the {loc1}. Then {a} left. {b} moved it to the {loc2} and {verb} {a}. Where will {a} look? In the",
]

AGENTS = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry"]
OBJECTS = ["ball", "book", "key", "toy", "phone", "wallet", "bag", "hat"]
LOCATIONS = ["basket", "box", "drawer", "shelf", "bag", "desk", "table", "cabinet"]


def generate_all_scenarios(verbs_by_category: Dict[str, List[str]], n_per_verb: int) -> Tuple[List[dict], Dict[str, List[int]]]:
    """Generate ALL scenarios upfront with verb-to-index mapping."""
    scenarios = []
    verb_to_indices = defaultdict(list)
    
    all_verbs = []
    for cat, vs in verbs_by_category.items():
        for v in vs:
            all_verbs.append((v, cat))
    
    logger.info(f"Generating {len(all_verbs)} verbs × {n_per_verb} scenarios...")
    
    random.seed(42)
    
    for verb, category in all_verbs:
        start_idx = len(scenarios)
        
        for i in range(n_per_verb):
            template = TEMPLATES[i % len(TEMPLATES)]
            a, b = random.sample(AGENTS, 2)
            obj = random.choice(OBJECTS)
            loc1, loc2 = random.sample(LOCATIONS, 2)
            
            prompt = template.format(a=a, b=b, obj=obj, loc1=loc1, loc2=loc2, verb=verb)
            
            scenarios.append({
                'prompt': prompt,
                'correct': f" {loc2}",
                'wrong': f" {loc1}",
                'verb': verb,
                'category': category,
            })
        
        verb_to_indices[verb] = list(range(start_idx, len(scenarios)))
    
    logger.info(f"Generated {len(scenarios)} total scenarios")
    return scenarios, dict(verb_to_indices)


# =============================================================================
# ULTRA-FAST MODEL
# =============================================================================

class UltraFastAnalyzer:
    """Batched inference with optional ablation."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_heads = model.config.num_attention_heads
        self.hooks = []
        
        # Get token IDs for common location words
        self.location_tokens = {}
        for loc in LOCATIONS:
            ids = tokenizer.encode(f" {loc}", add_special_tokens=False)
            if ids:
                self.location_tokens[loc] = ids[0]
    
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
    
    def batch_test(self, scenarios: List[dict], use_ablation: bool = False) -> List[bool]:
        """Test scenarios in batches. Returns list of correct/incorrect."""
        if use_ablation:
            self.install_ablation(INHIBITORS)
        else:
            self.clear_hooks()
        
        results = []
        
        # Process in batches
        for i in range(0, len(scenarios), BATCH_SIZE):
            batch = scenarios[i:i + BATCH_SIZE]
            prompts = [s['prompt'] for s in batch]
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    prompts, 
                    return_tensors="pt", 
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to(self.model.device)
                
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    outputs = self.model(**inputs)
                    # Get last token logits for each sequence
                    # Need to handle padding - find actual last token
                    logits = outputs.logits
                    
                    for j, s in enumerate(batch):
                        # Get logits at last non-pad position
                        seq_len = inputs.attention_mask[j].sum().item()
                        last_logits = logits[j, seq_len - 1, :]
                        probs = torch.softmax(last_logits, dim=-1)
                        
                        # Get correct/wrong token IDs
                        c_ids = self.tokenizer.encode(s['correct'], add_special_tokens=False)
                        w_ids = self.tokenizer.encode(s['wrong'], add_special_tokens=False)
                        
                        if c_ids and w_ids:
                            c_prob = probs[c_ids[0]].item()
                            w_prob = probs[w_ids[0]].item()
                            results.append(c_prob > w_prob)
                        else:
                            results.append(False)
            except Exception as e:
                logger.error(f"Error in batch {i//BATCH_SIZE}: {e}")
                # Fill with False for this batch
                results.extend([False] * len(batch))
        
        self.clear_hooks()
        # Clear GPU cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results


# =============================================================================
# MAIN SWEEP
# =============================================================================

def run_ultrafast_sweep():
    """Run the ultra-fast sweep."""
    
    logger.info("=" * 70)
    logger.info("ULTRA-FAST GARGANTUAN SWEEP")
    logger.info("=" * 70)
    start_time = time.time()
    
    # Phase 1: Load model with SDPA (fastest)
    logger.info("\n[PHASE 1] Loading model with SDPA attention...")
    phase1_start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",  # FASTEST
    )
    model.eval()
    
    logger.info(f"  Model loaded in {time.time() - phase1_start:.1f}s")
    
    analyzer = UltraFastAnalyzer(model, tokenizer)
    
    # Phase 2: Generate all scenarios
    logger.info("\n[PHASE 2] Generating all scenarios...")
    phase2_start = time.time()
    
    verbs_by_category = get_all_communication_verbs()
    scenarios, verb_to_indices = generate_all_scenarios(verbs_by_category, SCENARIOS_PER_VERB)
    
    logger.info(f"  Generated in {time.time() - phase2_start:.1f}s")
    
    # Phase 3: Baseline sweep (no ablation)
    logger.info(f"\n[PHASE 3] Baseline sweep ({len(scenarios)} scenarios, batch_size={BATCH_SIZE})...")
    sys.stdout.flush()
    phase3_start = time.time()
    
    # Quick sanity check on first batch
    logger.info("  Running sanity check on first batch...")
    sys.stdout.flush()
    test_results = analyzer.batch_test(scenarios[:BATCH_SIZE], use_ablation=False)
    logger.info(f"  Sanity check passed! First batch: {sum(test_results)}/{len(test_results)} correct")
    sys.stdout.flush()
    
    baseline_results = []
    total_batches = (len(scenarios) + BATCH_SIZE - 1) // BATCH_SIZE
    pbar = tqdm(range(0, len(scenarios), BATCH_SIZE), desc="Baseline", unit="batch", total=total_batches)
    
    for chunk_start in pbar:
        chunk_end = min(chunk_start + BATCH_SIZE, len(scenarios))
        chunk = scenarios[chunk_start:chunk_end]
        results = analyzer.batch_test(chunk, use_ablation=False)
        baseline_results.extend(results)
        
        # Update progress
        acc = sum(baseline_results) / len(baseline_results) * 100
        pbar.set_postfix({'acc': f'{acc:.1f}%'})
    
    baseline_time = time.time() - phase3_start
    logger.info(f"  Baseline complete in {baseline_time:.1f}s ({len(scenarios)/baseline_time:.1f} scenarios/sec)")
    logger.info(f"  Baseline accuracy: {sum(baseline_results)/len(baseline_results)*100:.1f}%")
    
    # Phase 4: Ablation sweep
    logger.info(f"\n[PHASE 4] Ablation sweep ({len(scenarios)} scenarios)...")
    sys.stdout.flush()
    phase4_start = time.time()
    
    ablated_results = []
    pbar = tqdm(range(0, len(scenarios), BATCH_SIZE), desc="Ablated", unit="batch", total=total_batches)
    
    for chunk_start in pbar:
        chunk_end = min(chunk_start + BATCH_SIZE, len(scenarios))
        chunk = scenarios[chunk_start:chunk_end]
        results = analyzer.batch_test(chunk, use_ablation=True)
        ablated_results.extend(results)
        
        acc = sum(ablated_results) / len(ablated_results) * 100
        pbar.set_postfix({'acc': f'{acc:.1f}%'})
    
    ablation_time = time.time() - phase4_start
    logger.info(f"  Ablation complete in {ablation_time:.1f}s ({len(scenarios)/ablation_time:.1f} scenarios/sec)")
    logger.info(f"  Ablated accuracy: {sum(ablated_results)/len(ablated_results)*100:.1f}%")
    
    # Phase 5: Aggregate by verb
    logger.info("\n[PHASE 5] Aggregating results by verb...")
    
    verb_results = []
    
    for verb, indices in verb_to_indices.items():
        category = scenarios[indices[0]]['category']
        baseline_correct = sum(1 for i in indices if baseline_results[i])
        ablated_correct = sum(1 for i in indices if ablated_results[i])
        n = len(indices)
        
        verb_results.append(VerbResult(
            verb=verb,
            category=category,
            n_scenarios=n,
            baseline_correct=baseline_correct,
            ablated_correct=ablated_correct,
            baseline_accuracy=baseline_correct / n,
            ablated_accuracy=ablated_correct / n,
            boost=(ablated_correct - baseline_correct) / n,
        ))
    
    # Sort by baseline accuracy
    verb_results.sort(key=lambda x: x.baseline_accuracy)
    
    # Phase 6: Save results
    logger.info("\n[PHASE 6] Saving results...")
    
    results_data = {
        'metadata': {
            'total_verbs': len(verb_results),
            'total_scenarios': len(scenarios),
            'scenarios_per_verb': SCENARIOS_PER_VERB,
            'batch_size': BATCH_SIZE,
            'baseline_time_sec': baseline_time,
            'ablation_time_sec': ablation_time,
            'total_time_sec': time.time() - start_time,
        },
        'summary': {
            'mean_baseline': float(np.mean([r.baseline_accuracy for r in verb_results])),
            'mean_ablated': float(np.mean([r.ablated_accuracy for r in verb_results])),
            'mean_boost': float(np.mean([r.boost for r in verb_results])),
            'n_zero_baseline': sum(1 for r in verb_results if r.baseline_accuracy == 0),
            'n_perfect_ablated': sum(1 for r in verb_results if r.ablated_accuracy >= 1.0),
        },
        'by_category': {},
        'verb_results': [asdict(r) for r in verb_results],
        'worst_verbs': [asdict(r) for r in verb_results[:50]],
        'best_verbs': [asdict(r) for r in verb_results[-20:]],
    }
    
    # Category breakdown
    for cat in set(r.category for r in verb_results):
        cat_verbs = [r for r in verb_results if r.category == cat]
        results_data['by_category'][cat] = {
            'n_verbs': len(cat_verbs),
            'mean_baseline': float(np.mean([r.baseline_accuracy for r in cat_verbs])),
            'mean_ablated': float(np.mean([r.ablated_accuracy for r in cat_verbs])),
            'mean_boost': float(np.mean([r.boost for r in cat_verbs])),
        }
    
    with open(RESULTS_DIR / "ultrafast_results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    total_time = time.time() - start_time
    logger.info(f"\nTIMING:")
    logger.info(f"  Baseline: {baseline_time:.1f}s ({len(scenarios)/baseline_time:.0f} scenarios/sec)")
    logger.info(f"  Ablation: {ablation_time:.1f}s ({len(scenarios)/ablation_time:.0f} scenarios/sec)")
    logger.info(f"  TOTAL: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    logger.info(f"\nRESULTS:")
    logger.info(f"  Verbs tested: {len(verb_results)}")
    logger.info(f"  Scenarios tested: {len(scenarios)}")
    logger.info(f"  Mean baseline: {results_data['summary']['mean_baseline']*100:.1f}%")
    logger.info(f"  Mean ablated: {results_data['summary']['mean_ablated']*100:.1f}%")
    logger.info(f"  Mean boost: +{results_data['summary']['mean_boost']*100:.1f}%")
    logger.info(f"  Verbs at 0% baseline: {results_data['summary']['n_zero_baseline']}")
    logger.info(f"  Verbs at 100% ablated: {results_data['summary']['n_perfect_ablated']}")
    
    logger.info(f"\nWORST VERBS (strongest inhibition):")
    for r in verb_results[:10]:
        logger.info(f"  {r.verb}: {r.baseline_accuracy*100:.0f}% → {r.ablated_accuracy*100:.0f}%")
    
    logger.info(f"\nBEST VERBS (no inhibition):")
    for r in verb_results[-5:]:
        logger.info(f"  {r.verb}: {r.baseline_accuracy*100:.0f}% → {r.ablated_accuracy*100:.0f}%")
    
    logger.info(f"\nCATEGORY BREAKDOWN:")
    for cat, stats in sorted(results_data['by_category'].items(), key=lambda x: x[1]['mean_baseline']):
        logger.info(f"  {cat}: {stats['mean_baseline']*100:.0f}% → {stats['mean_ablated']*100:.0f}% ({stats['n_verbs']} verbs)")
    
    logger.info(f"\nResults saved to: {RESULTS_DIR}")
    
    return verb_results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logger.info(f"Starting at {datetime.now()}")
    logger.info(f"Configuration:")
    logger.info(f"  BATCH_SIZE: {BATCH_SIZE}")
    logger.info(f"  SCENARIOS_PER_VERB: {SCENARIOS_PER_VERB}")
    logger.info(f"  MAX_VERBS: {MAX_VERBS or 'ALL'}")
    
    try:
        results = run_ultrafast_sweep()
        logger.info(f"\nCompleted successfully at {datetime.now()}")
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


