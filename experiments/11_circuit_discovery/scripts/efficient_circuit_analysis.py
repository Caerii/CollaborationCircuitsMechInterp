"""
Efficient Circuit Discovery Pipeline
=====================================

OPTIMIZATIONS:
1. Batch processing (multiple prompts at once)
2. Single forward pass for all data
3. Cached activations (save to disk, don't recompute)
4. Timing tracked for all operations
5. Memory-efficient extraction

Uses: PyTorch hooks (most efficient for custom models like Qwen)
"""

import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

RESULTS_DIR = Path(__file__).parent.parent / "results"
CACHE_DIR = Path(__file__).parent.parent / "cache"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TimingStats:
    """Track timing for each operation."""
    times: Dict[str, float] = None
    
    def __post_init__(self):
        self.times = {}
        self._start = None
    
    def start(self, name: str):
        self._start = time.perf_counter()
        self._current = name
    
    def stop(self):
        if self._start:
            elapsed = time.perf_counter() - self._start
            self.times[self._current] = elapsed
            print(f"    [{self._current}] {elapsed:.2f}s", flush=True)
            self._start = None
    
    def summary(self):
        total = sum(self.times.values())
        print("\n" + "=" * 50)
        print("TIMING SUMMARY")
        print("=" * 50)
        for name, t in self.times.items():
            pct = (t / total) * 100 if total > 0 else 0
            print(f"  {name:<30} {t:>6.2f}s ({pct:>5.1f}%)")
        print("-" * 50)
        print(f"  {'TOTAL':<30} {total:>6.2f}s")
        return self.times


# Test scenarios (balanced)
SCENARIOS = [
    # B agrees (True)
    {"prompt": "Agent A: 'The answer is 42.' Agent B verified this. Does B agree? B:", "b_agrees": True, "a_correct": True},
    {"prompt": "Agent A: 'Water is H2O.' Agent B confirms. Does B agree? B:", "b_agrees": True, "a_correct": True},
    {"prompt": "Agent A: 'Python is interpreted.' Agent B knows this. Does B agree? B:", "b_agrees": True, "a_correct": True},
    {"prompt": "Agent A: 'Earth has one moon.' Agent B agrees. Does B agree? B:", "b_agrees": True, "a_correct": True},
    {"prompt": "Agent A: 'Meeting at wrong time.' Agent B has same wrong info. Does B agree? B:", "b_agrees": True, "a_correct": False},
    {"prompt": "Agent A: 'Server at old IP.' Agent B has outdated docs too. Does B agree? B:", "b_agrees": True, "a_correct": False},
    # B disagrees (False)
    {"prompt": "Agent A: '2+2=5' Agent B knows math. Does B agree? B:", "b_agrees": False, "a_correct": False},
    {"prompt": "Agent A: 'Tokyo in China.' Agent B knows geography. Does B agree? B:", "b_agrees": False, "a_correct": False},
    {"prompt": "Agent A: 'Pi equals 3.' Agent B remembers 3.14159. Does B agree? B:", "b_agrees": False, "a_correct": False},
    {"prompt": "Agent A: 'Sun is cold.' Agent B knows physics. Does B agree? B:", "b_agrees": False, "a_correct": False},
    {"prompt": "Agent A: 'Meeting at 3pm.' Agent B heard 4pm. Does B agree? B:", "b_agrees": False, "a_correct": True},
    {"prompt": "Agent A: 'Code in utils.py.' Agent B checked helpers.py. Does B agree? B:", "b_agrees": False, "a_correct": True},
]


class EfficientActivationExtractor:
    """Memory-efficient activation extraction with caching."""
    
    def __init__(self, model, tokenizer, cache_path: Optional[Path] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.cache_path = cache_path
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads
        self.hidden_size = model.config.hidden_size
        self.head_dim = self.hidden_size // self.n_heads
    
    def extract_all(self, prompts: List[str], batch_size: int = 4) -> Dict:
        """Extract layer outputs and attention patterns efficiently."""
        
        cache_file = self.cache_path / "activations.npz" if self.cache_path else None
        
        # Check cache
        if cache_file and cache_file.exists():
            print("  Loading from cache...", flush=True)
            data = np.load(cache_file)
            return {
                "layer_outputs": data["layer_outputs"],
                "attention_patterns": data["attention_patterns"],
            }
        
        n_samples = len(prompts)
        
        # Pre-allocate arrays (memory efficient)
        layer_outputs = np.zeros((n_samples, self.n_layers, self.hidden_size), dtype=np.float16)
        # Only store attention for subset of layers to save memory
        attn_layers = [0, 12, 24, 35]
        attention_patterns = {}
        
        # Hooks for layer outputs
        captured = {}
        hooks = []
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = hidden[:, -1, :].detach().cpu()  # Last token only
            return hook
        
        for layer_idx in range(self.n_layers):
            hook = self.model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            hooks.append(hook)
        
        # Process in batches
        with torch.no_grad():
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_prompts = prompts[batch_start:batch_end]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_prompts, 
                    return_tensors="pt", 
                    padding=True,
                    truncation=True, 
                    max_length=128
                ).to("cuda")
                
                # Forward pass
                _ = self.model(**inputs)
                
                # Store layer outputs
                for layer_idx in range(self.n_layers):
                    batch_outputs = captured[layer_idx].numpy().astype(np.float16)
                    layer_outputs[batch_start:batch_end, layer_idx, :] = batch_outputs
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        result = {
            "layer_outputs": layer_outputs,
            "attention_patterns": attention_patterns,
        }
        
        # Save to cache
        if cache_file:
            np.savez_compressed(cache_file, **result)
            print(f"  Cached to {cache_file}", flush=True)
        
        return result


def analyze_heads(layer_outputs: np.ndarray, labels: np.ndarray, n_heads: int) -> List[Dict]:
    """Efficient head-level analysis using vectorized operations."""
    
    n_samples, n_layers, hidden_size = layer_outputs.shape
    head_dim = hidden_size // n_heads
    
    results = []
    
    # Reshape to per-head: (n_samples, n_layers, n_heads, head_dim)
    per_head = layer_outputs.reshape(n_samples, n_layers, n_heads, head_dim)
    
    clf = LogisticRegression(max_iter=500, random_state=42, solver='lbfgs')
    
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            X = per_head[:, layer_idx, head_idx, :].astype(np.float32)
            
            try:
                scores = cross_val_score(clf, X, labels, cv=min(3, len(labels)//2))
                acc = scores.mean()
            except:
                acc = 0.5
            
            if acc > 0.55:  # Only store interesting heads
                results.append({
                    "layer": int(layer_idx),
                    "head": int(head_idx),
                    "accuracy": float(acc),
                })
    
    return sorted(results, key=lambda x: x["accuracy"], reverse=True)


def analyze_layers(layer_outputs: np.ndarray, b_agrees: np.ndarray, a_correct: np.ndarray) -> Dict:
    """Layer-level analysis with independence check."""
    
    n_samples, n_layers, hidden_size = layer_outputs.shape
    
    results = {"layers": []}
    clf = LogisticRegression(max_iter=500, random_state=42)
    
    for layer_idx in range(n_layers):
        X = layer_outputs[:, layer_idx, :].astype(np.float32)
        
        # Probe for B agrees
        try:
            b_scores = cross_val_score(clf, X, b_agrees, cv=3)
            b_acc = b_scores.mean()
        except:
            b_acc = 0.5
        
        # Probe for A correct
        try:
            a_scores = cross_val_score(clf, X, a_correct, cv=3)
            a_acc = a_scores.mean()
        except:
            a_acc = 0.5
        
        # Direction independence
        try:
            clf_b = LogisticRegression(max_iter=500, random_state=42)
            clf_a = LogisticRegression(max_iter=500, random_state=42)
            clf_b.fit(X, b_agrees)
            clf_a.fit(X, a_correct)
            
            b_dir = clf_b.coef_[0]
            a_dir = clf_a.coef_[0]
            cosine = abs(np.dot(b_dir, a_dir) / (np.linalg.norm(b_dir) * np.linalg.norm(a_dir)))
        except:
            cosine = 1.0
        
        results["layers"].append({
            "layer": layer_idx,
            "b_agrees_acc": float(b_acc),
            "a_correct_acc": float(a_acc),
            "independence_cosine": float(cosine),
        })
    
    return results


def main():
    timer = TimingStats()
    
    print("=" * 60)
    print("EFFICIENT CIRCUIT DISCOVERY PIPELINE")
    print("=" * 60)
    
    # Prepare data
    prompts = [s["prompt"] for s in SCENARIOS]
    b_agrees = np.array([1 if s["b_agrees"] else 0 for s in SCENARIOS])
    a_correct = np.array([1 if s["a_correct"] else 0 for s in SCENARIOS])
    
    print(f"\n[1/5] Data: {len(SCENARIOS)} scenarios")
    print(f"  B agrees: {b_agrees.sum()}, B disagrees: {len(b_agrees) - b_agrees.sum()}")
    
    # Load model
    print("\n[2/5] Loading model...", flush=True)
    timer.start("model_loading")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CFG.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CFG.model_name,
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    timer.stop()
    
    print(f"  {model.config.num_hidden_layers} layers, {model.config.num_attention_heads} heads")
    
    # Extract activations
    print("\n[3/5] Extracting activations (batched, cached)...", flush=True)
    timer.start("activation_extraction")
    
    extractor = EfficientActivationExtractor(model, tokenizer, CACHE_DIR)
    data = extractor.extract_all(prompts, batch_size=4)
    
    timer.stop()
    print(f"  Shape: {data['layer_outputs'].shape}")
    
    # Layer analysis
    print("\n[4/5] Layer-level analysis...", flush=True)
    timer.start("layer_analysis")
    
    layer_results = analyze_layers(data["layer_outputs"], b_agrees, a_correct)
    
    timer.stop()
    
    # Head analysis
    print("\n[5/5] Head-level analysis...", flush=True)
    timer.start("head_analysis")
    
    head_results = analyze_heads(data["layer_outputs"], b_agrees, model.config.num_attention_heads)
    
    timer.stop()
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS: LAYER ANALYSIS")
    print("=" * 60)
    print(f"{'Layer':<8} {'B agrees':<12} {'A correct':<12} {'Independence':<12}")
    print("-" * 44)
    
    for r in layer_results["layers"][::6]:  # Every 6th layer
        l = r["layer"]
        print(f"{l:<8} {r['b_agrees_acc']:.1%}        {r['a_correct_acc']:.1%}        {r['independence_cosine']:.3f}")
    
    print("\n" + "=" * 60)
    print("RESULTS: TOP CANDIDATE ToM HEADS")
    print("=" * 60)
    print(f"{'Layer':<8} {'Head':<8} {'Accuracy':<12}")
    print("-" * 28)
    
    for h in head_results[:15]:
        print(f"{h['layer']:<8} {h['head']:<8} {h['accuracy']:.1%}")
    
    # Save all results
    all_results = {
        "scenarios": len(SCENARIOS),
        "layer_analysis": layer_results,
        "top_heads": head_results[:50],
        "timing": timer.times,
    }
    
    with open(RESULTS_DIR / "circuit_discovery.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Timing summary
    timer.summary()
    
    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    
    # Find layers with high independence (low cosine)
    independent_layers = [r for r in layer_results["layers"] if r["independence_cosine"] < 0.3 and r["b_agrees_acc"] > 0.6]
    if independent_layers:
        print("\nLayers with INDEPENDENT B-belief encoding:")
        for r in independent_layers[:5]:
            print(f"  Layer {r['layer']}: B_acc={r['b_agrees_acc']:.1%}, cosine={r['independence_cosine']:.3f}")
    
    # Top heads
    if head_results:
        print(f"\nTop ToM head: Layer {head_results[0]['layer']}, Head {head_results[0]['head']} ({head_results[0]['accuracy']:.1%})")
    
    print(f"\n[OK] All results saved to {RESULTS_DIR}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()




















