"""
Circuit Analysis Tools

Provides attention head ablation and analysis tools extending
experiments/19_methodological_controls/lib/hooks.py.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import sys

# Import from existing lib if available
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class HeadImportance:
    """Importance score for an attention head."""
    layer: int
    head: int
    importance: float
    method: str  # "ablation", "attention", "probing"
    details: Dict
    
    def to_dict(self) -> Dict:
        return {
            "layer": self.layer,
            "head": self.head,
            "importance": float(self.importance),
            "method": self.method,
            "details": self.details,
        }


class CircuitAnalysis:
    """
    Analyze attention head circuits for ToM and collaboration.
    
    Extends the HookManager from experiments/19_methodological_controls/lib/hooks.py
    with additional analysis capabilities.
    
    Example:
        analyzer = CircuitAnalysis(model, tokenizer)
        
        # Find important heads via ablation
        head_scores = analyzer.ablation_sweep(
            scenarios,
            n_per_head=30  # Minimum for statistical validity
        )
        
        # Get top heads
        top_heads = analyzer.get_top_heads(head_scores, n=10)
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        n_layers: Optional[int] = None,
        n_heads: Optional[int] = None
    ):
        """
        Initialize analyzer.
        
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            n_layers: Number of layers (auto-detected if None)
            n_heads: Number of heads per layer (auto-detected if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        
        self.n_layers = n_layers or model.config.num_hidden_layers
        self.n_heads = n_heads or model.config.num_attention_heads
        self.hidden_size = model.config.hidden_size
        
        # Handle models with different head_dim (e.g., GQA models)
        if hasattr(model.config, 'head_dim'):
            self.head_dim = model.config.head_dim
        else:
            self.head_dim = self.hidden_size // self.n_heads
        
        print(f"CircuitAnalysis: hidden={self.hidden_size}, n_heads={self.n_heads}, head_dim={self.head_dim}")
        
        self.hooks = []
    
    def _get_o_proj(self, layer: int):
        """Get output projection module for a layer."""
        return self.model.model.layers[layer].self_attn.o_proj
    
    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def ablate_heads(
        self,
        heads: List[Tuple[int, int]]
    ):
        """
        Install ablation hooks for specified heads.
        
        Args:
            heads: List of (layer, head) tuples to ablate
        """
        self._clear_hooks()
        
        layers_to_heads = defaultdict(list)
        for layer, head in heads:
            layers_to_heads[layer].append(head)
        
        for layer, head_list in layers_to_heads.items():
            o_proj = self._get_o_proj(layer)
            
            def make_hook(heads_to_zero, n_heads, head_dim):
                def hook(module, args):
                    x = args[0]
                    batch_size, seq_len, hidden_dim = x.shape
                    
                    # Dynamically compute head_dim from actual tensor
                    actual_head_dim = hidden_dim // n_heads
                    
                    try:
                        x = x.view(batch_size, seq_len, n_heads, actual_head_dim)
                        for h in heads_to_zero:
                            x[:, :, h, :] = 0
                        x = x.view(batch_size, seq_len, hidden_dim)
                    except RuntimeError as e:
                        # If reshape fails, skip ablation for this layer
                        print(f"Warning: ablation reshape failed ({hidden_dim} / {n_heads} = {hidden_dim / n_heads})")
                        pass
                    
                    return (x,) + args[1:] if len(args) > 1 else (x,)
                return hook
            
            handle = o_proj.register_forward_pre_hook(make_hook(head_list, self.n_heads, self.head_dim))
            self.hooks.append(handle)
    
    def evaluate_with_ablation(
        self,
        scenarios: List[Dict],
        heads: List[Tuple[int, int]],
        evaluator_fn: Callable
    ) -> float:
        """
        Evaluate scenarios with specific heads ablated.
        
        Args:
            scenarios: List of scenarios
            heads: Heads to ablate
            evaluator_fn: Function(model, scenarios) -> accuracy
            
        Returns:
            Accuracy with ablation
        """
        self.ablate_heads(heads)
        try:
            accuracy = evaluator_fn(self.model, scenarios)
        finally:
            self._clear_hooks()
        return accuracy
    
    def ablation_sweep(
        self,
        scenarios: List[Dict],
        evaluator_fn: Callable,
        layers_to_sweep: Optional[List[int]] = None,
        heads_to_sweep: Optional[List[int]] = None,
        verbose: bool = False
    ) -> List[HeadImportance]:
        """
        Sweep ablation across heads to find important ones.
        
        Args:
            scenarios: List of scenarios for evaluation
            evaluator_fn: Function(model, scenarios) -> accuracy
            layers_to_sweep: Layers to check (default: all)
            heads_to_sweep: Heads to check (default: all)
            verbose: Print progress
            
        Returns:
            List of HeadImportance sorted by importance
        """
        layers = layers_to_sweep or list(range(self.n_layers))
        heads = heads_to_sweep or list(range(self.n_heads))
        
        # Get baseline accuracy
        baseline = evaluator_fn(self.model, scenarios)
        if verbose:
            print(f"Baseline accuracy: {baseline:.1%}")
        
        results = []
        
        for layer in layers:
            for head in heads:
                ablated_acc = self.evaluate_with_ablation(
                    scenarios, [(layer, head)], evaluator_fn
                )
                
                # Importance = how much accuracy drops when ablated
                importance = baseline - ablated_acc
                
                results.append(HeadImportance(
                    layer=layer,
                    head=head,
                    importance=importance,
                    method="ablation",
                    details={
                        "baseline_accuracy": baseline,
                        "ablated_accuracy": ablated_acc,
                    }
                ))
                
                if verbose:
                    direction = "+" if importance > 0 else ""
                    print(f"  L{layer}H{head}: {direction}{importance:.1%}")
        
        # Sort by importance (descending)
        results.sort(key=lambda x: abs(x.importance), reverse=True)
        
        return results
    
    def get_attention_patterns(
        self,
        text: str,
        layers: Optional[List[int]] = None
    ) -> Dict[int, np.ndarray]:
        """
        Get attention patterns for given text.
        
        Args:
            text: Input text
            layers: Layers to extract (default: all)
            
        Returns:
            Dict mapping layer to attention array [n_heads, seq_len, seq_len]
        """
        layers = layers or list(range(self.n_layers))
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.model(**inputs, output_attentions=True)
        
        if outputs.attentions is None:
            raise ValueError("Model not returning attentions. Load with attn_implementation='eager'")
        
        patterns = {}
        for layer in layers:
            if layer < len(outputs.attentions):
                patterns[layer] = outputs.attentions[layer][0].cpu().numpy()
        
        return patterns
    
    def find_entity_focused_heads(
        self,
        texts: List[str],
        entity_tokens: List[str],
        top_k: int = 10
    ) -> List[HeadImportance]:
        """
        Find heads that attend strongly to entity tokens.
        
        Args:
            texts: List of texts containing entities
            entity_tokens: Tokens representing entities (e.g., ["Alice", "Bob"])
            top_k: Number of top heads to return
            
        Returns:
            Top heads by entity attention
        """
        head_scores = defaultdict(list)
        
        for text in texts:
            patterns = self.get_attention_patterns(text)
            
            # Tokenize to find entity positions
            inputs = self.tokenizer(text, return_tensors="pt")
            tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
            
            entity_positions = []
            for i, tok in enumerate(tokens):
                tok_clean = tok.replace("▁", "").replace("Ġ", "").lower()
                if any(e.lower() in tok_clean for e in entity_tokens):
                    entity_positions.append(i)
            
            if not entity_positions:
                continue
            
            # Score each head by attention to entities
            for layer, attn in patterns.items():
                for head in range(attn.shape[0]):
                    # Average attention TO entity positions FROM all positions
                    entity_attn = attn[head, :, entity_positions].mean()
                    head_scores[(layer, head)].append(entity_attn)
        
        # Aggregate scores
        results = []
        for (layer, head), scores in head_scores.items():
            avg_score = np.mean(scores)
            results.append(HeadImportance(
                layer=layer,
                head=head,
                importance=avg_score,
                method="attention",
                details={"n_samples": len(scores)}
            ))
        
        results.sort(key=lambda x: x.importance, reverse=True)
        return results[:top_k]
    
    def get_top_heads(
        self,
        results: List[HeadImportance],
        n: int = 10,
        min_importance: float = 0.0
    ) -> List[Tuple[int, int]]:
        """
        Get top N heads from importance results.
        
        Args:
            results: List of HeadImportance
            n: Number of heads to return
            min_importance: Minimum importance threshold
            
        Returns:
            List of (layer, head) tuples
        """
        filtered = [r for r in results if abs(r.importance) >= min_importance]
        return [(r.layer, r.head) for r in filtered[:n]]
    
    def summarize_by_layer(
        self,
        results: List[HeadImportance]
    ) -> Dict[int, Dict]:
        """
        Summarize head importance by layer.
        
        Args:
            results: List of HeadImportance
            
        Returns:
            Dict mapping layer to summary stats
        """
        by_layer = defaultdict(list)
        for r in results:
            by_layer[r.layer].append(r.importance)
        
        summary = {}
        for layer, importances in sorted(by_layer.items()):
            summary[layer] = {
                "mean": np.mean(importances),
                "max": np.max(importances),
                "min": np.min(importances),
                "n_heads": len(importances),
            }
        
        return summary

