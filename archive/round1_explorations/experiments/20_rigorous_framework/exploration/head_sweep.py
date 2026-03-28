"""
Systematic Head Discovery Sweep

From experiment 11 circuit discovery methodology.
Systematically identifies candidate circuit heads.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import defaultdict
import time


@dataclass
class HeadCandidate:
    """A candidate attention head for further investigation."""
    layer: int
    head: int
    score: float
    method: str
    rank: int
    details: Dict
    
    def to_dict(self) -> Dict:
        return {
            "layer": self.layer,
            "head": self.head,
            "score": float(self.score),
            "method": self.method,
            "rank": self.rank,
            "details": self.details,
        }


class HeadDiscoverySweep:
    """
    Systematically identify candidate circuit heads.
    
    Methods:
    1. Attention pattern analysis: Which heads attend to relevant tokens?
    2. Ablation effects: Which heads change behavior when removed?
    3. Information content: Which heads have high MI with target variables?
    
    Example:
        sweep = HeadDiscoverySweep(model, tokenizer)
        
        # Find heads important for ToM
        candidates = sweep.full_sweep(
            scenarios,
            evaluator_fn,
            target_tokens=["believes", "thinks", "knows"]
        )
        
        # Get top candidates
        top = candidates[:10]
        print(f"Top head: L{top[0].layer}H{top[0].head}")
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        n_layers: Optional[int] = None,
        n_heads: Optional[int] = None
    ):
        """
        Initialize sweep.
        
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            n_layers: Number of layers (auto-detected)
            n_heads: Heads per layer (auto-detected)
        """
        self.model = model
        self.tokenizer = tokenizer
        
        self.n_layers = n_layers or model.config.num_hidden_layers
        self.n_heads = n_heads or model.config.num_attention_heads
        self.hidden_size = model.config.hidden_size
        self.head_dim = self.hidden_size // self.n_heads
    
    def sweep_attention_patterns(
        self,
        texts: List[str],
        target_tokens: List[str],
        layers: Optional[List[int]] = None,
        verbose: bool = False
    ) -> List[HeadCandidate]:
        """
        Find heads that attend to target tokens.
        
        Args:
            texts: Sample texts
            target_tokens: Tokens to check attention to
            layers: Layers to sweep (default: all)
            verbose: Print progress
            
        Returns:
            List of HeadCandidate sorted by attention score
        """
        layers = layers or list(range(self.n_layers))
        head_scores = defaultdict(list)
        
        if verbose:
            print("Sweeping attention patterns...")
        
        for i, text in enumerate(texts):
            if verbose and (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(texts)}]")
            
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
            
            # Find target positions
            target_positions = []
            for j, tok in enumerate(tokens):
                tok_clean = tok.replace("▁", "").replace("Ġ", "").lower()
                for target in target_tokens:
                    if target.lower() in tok_clean:
                        target_positions.append(j)
                        break
            
            if not target_positions:
                continue
            
            # Get attention
            with torch.no_grad():
                outputs = self.model.model(**inputs, output_attentions=True)
            
            if outputs.attentions is None:
                raise ValueError("Model not returning attentions")
            
            # Score heads by attention to targets
            for layer in layers:
                attn = outputs.attentions[layer][0].cpu().numpy()  # [n_heads, seq, seq]
                
                for head in range(self.n_heads):
                    # Average attention TO target positions FROM all positions
                    attn_to_targets = attn[head, :, target_positions].mean()
                    head_scores[(layer, head)].append(attn_to_targets)
        
        # Aggregate and rank
        candidates = []
        for (layer, head), scores in head_scores.items():
            candidates.append(HeadCandidate(
                layer=layer,
                head=head,
                score=np.mean(scores),
                method="attention_pattern",
                rank=-1,
                details={
                    "n_samples": len(scores),
                    "std": np.std(scores),
                    "target_tokens": target_tokens,
                }
            ))
        
        candidates.sort(key=lambda x: x.score, reverse=True)
        for i, c in enumerate(candidates):
            c.rank = i + 1
        
        return candidates
    
    def sweep_ablation_effects(
        self,
        scenarios: List[Dict],
        evaluator_fn: Callable,
        layers: Optional[List[int]] = None,
        sample_per_head: int = 30,
        verbose: bool = False
    ) -> List[HeadCandidate]:
        """
        Find heads where ablation changes behavior.
        
        Args:
            scenarios: Test scenarios
            evaluator_fn: Function(model, scenarios) -> accuracy
            layers: Layers to sweep
            sample_per_head: Scenarios to test per head
            verbose: Print progress
            
        Returns:
            List of HeadCandidate sorted by ablation effect
        """
        layers = layers or list(range(0, self.n_layers, 4))  # Sample layers
        
        # Limit scenarios for speed
        test_scenarios = scenarios[:sample_per_head]
        
        # Get baseline
        baseline = evaluator_fn(self.model, test_scenarios)
        if verbose:
            print(f"Baseline accuracy: {baseline:.1%}")
        
        candidates = []
        hooks = []
        
        for layer in layers:
            if verbose:
                print(f"  Layer {layer}...")
            
            for head in range(self.n_heads):
                # Create ablation hook
                o_proj = self.model.model.layers[layer].self_attn.o_proj
                
                def make_hook(h):
                    def hook(module, args):
                        x = args[0]
                        bs, seq, _ = x.shape
                        x = x.view(bs, seq, self.n_heads, self.head_dim)
                        x[:, :, h, :] = 0
                        x = x.view(bs, seq, self.hidden_size)
                        return (x,)
                    return hook
                
                handle = o_proj.register_forward_pre_hook(make_hook(head))
                
                try:
                    ablated_acc = evaluator_fn(self.model, test_scenarios)
                finally:
                    handle.remove()
                
                effect = baseline - ablated_acc
                
                candidates.append(HeadCandidate(
                    layer=layer,
                    head=head,
                    score=abs(effect),
                    method="ablation",
                    rank=-1,
                    details={
                        "baseline": baseline,
                        "ablated": ablated_acc,
                        "effect": effect,
                        "direction": "harmful" if effect > 0 else "helpful",
                    }
                ))
        
        candidates.sort(key=lambda x: x.score, reverse=True)
        for i, c in enumerate(candidates):
            c.rank = i + 1
        
        return candidates
    
    def sweep_information_content(
        self,
        activations: Dict[int, np.ndarray],
        labels: np.ndarray,
        verbose: bool = False
    ) -> List[HeadCandidate]:
        """
        Find heads with high mutual information with labels.
        
        Uses probe accuracy as proxy for information content.
        
        Args:
            activations: Dict mapping layer to activation matrix
            labels: Classification labels
            verbose: Print progress
            
        Returns:
            List of HeadCandidate sorted by information content
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        candidates = []
        
        for layer, acts in activations.items():
            if verbose:
                print(f"  Layer {layer}...")
            
            # Reshape to get per-head activations
            n_samples = acts.shape[0]
            
            # If activations are full hidden state, we can't split by head
            # So use full layer probe accuracy as proxy
            clf = LogisticRegression(max_iter=500, random_state=42)
            try:
                scores = cross_val_score(clf, acts, labels, cv=3)
                accuracy = np.mean(scores)
            except:
                accuracy = 0.0
            
            candidates.append(HeadCandidate(
                layer=layer,
                head=-1,  # Full layer
                score=accuracy,
                method="information_content",
                rank=-1,
                details={
                    "probe_accuracy": accuracy,
                    "n_samples": n_samples,
                }
            ))
        
        candidates.sort(key=lambda x: x.score, reverse=True)
        for i, c in enumerate(candidates):
            c.rank = i + 1
        
        return candidates
    
    def full_sweep(
        self,
        scenarios: List[Dict],
        evaluator_fn: Callable,
        target_tokens: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, List[HeadCandidate]]:
        """
        Run all sweep methods.
        
        Args:
            scenarios: Test scenarios
            evaluator_fn: Evaluation function
            target_tokens: Tokens for attention sweep
            verbose: Print progress
            
        Returns:
            Dict mapping method name to candidates
        """
        results = {}
        
        if target_tokens:
            if verbose:
                print("\n=== Attention Pattern Sweep ===")
            texts = [s.get("story", s.get("prompt", "")) for s in scenarios[:50]]
            results["attention"] = self.sweep_attention_patterns(
                texts, target_tokens, verbose=verbose
            )
        
        if verbose:
            print("\n=== Ablation Sweep ===")
        results["ablation"] = self.sweep_ablation_effects(
            scenarios, evaluator_fn, verbose=verbose
        )
        
        return results
    
    def get_consensus_heads(
        self,
        results: Dict[str, List[HeadCandidate]],
        top_k: int = 20
    ) -> List[Tuple[int, int, float]]:
        """
        Find heads that appear important across multiple methods.
        
        Args:
            results: Results from full_sweep
            top_k: Consider top K from each method
            
        Returns:
            List of (layer, head, consensus_score) tuples
        """
        head_votes = defaultdict(float)
        
        for method, candidates in results.items():
            for c in candidates[:top_k]:
                if c.head >= 0:  # Skip layer-level results
                    # Weight by inverse rank
                    weight = 1.0 / c.rank
                    head_votes[(c.layer, c.head)] += weight
        
        # Sort by consensus score
        sorted_heads = sorted(head_votes.items(), key=lambda x: x[1], reverse=True)
        
        return [(layer, head, score) for (layer, head), score in sorted_heads]

