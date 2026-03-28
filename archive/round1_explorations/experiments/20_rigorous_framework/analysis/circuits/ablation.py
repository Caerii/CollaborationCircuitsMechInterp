"""
Head Ablation Utilities

Correct implementation of attention head ablation using pre-hooks on o_proj input.
"""

import torch
from typing import List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class AblationResult:
    """Result from ablating a head."""
    layer: int
    head: int
    baseline_accuracy: float
    ablated_accuracy: float
    effect: float  # baseline - ablated (positive = head was helpful)
    individual_results: List[bool]  # Per-scenario results
    
    @property
    def is_helpful(self) -> bool:
        """True if ablation hurts performance (head is helpful)."""
        return self.effect > 0
    
    @property
    def is_inhibitory(self) -> bool:
        """True if ablation helps performance (head is inhibitory)."""
        return self.effect < 0


class HeadAblator:
    """
    Correctly ablate attention heads using pre-hooks on o_proj.
    
    The key insight: o_proj INPUT (before projection) has heads still separate.
    o_proj OUTPUT (after projection) has heads already combined - can't separate!
    
    OPTIMIZED: Hooks are registered once and toggled on/off to reduce overhead.
    
    Example:
        ablator = HeadAblator(model)
        
        # Ablate a single head
        ablator.ablate_head(32, 0)
        result = model.generate(...)
        ablator.clear()
        
        # Ablate multiple heads
        ablator.ablate_heads([(32, 0), (33, 4)])
        result = model.generate(...)
        ablator.clear()
    """
    
    def __init__(self, model, optimize_hooks: bool = True):
        """
        Initialize ablator.
        
        Args:
            model: HuggingFace model
            optimize_hooks: If True, keep hooks registered and toggle on/off (faster)
        """
        self.model = model
        self.n_heads = model.config.num_attention_heads
        self.hidden_size = model.config.hidden_size
        
        # Handle models with different head_dim (e.g., GQA models)
        if hasattr(model.config, 'head_dim'):
            self.head_dim = model.config.head_dim
        else:
            self.head_dim = self.hidden_size // self.n_heads
        
        self.optimize_hooks = optimize_hooks
        self.hooks = []
        self.active_heads = set()  # Set of (layer, head) tuples currently ablated
        self.hook_registry = {}  # Map (layer, head) -> hook handle (for optimization)
    
    def _get_o_proj(self, layer: int):
        """Get output projection module for a layer."""
        return self.model.model.layers[layer].self_attn.o_proj
    
    def ablate_head(self, layer: int, head: int):
        """
        Ablate a single attention head.
        
        Args:
            layer: Layer index
            head: Head index within layer
        """
        self.ablate_heads([(layer, head)])
    
    def ablate_heads(self, heads: List[Tuple[int, int]]):
        """
        Ablate multiple attention heads.
        
        Args:
            heads: List of (layer, head) tuples to ablate
        """
        if not self.optimize_hooks:
            # Original behavior: clear and re-register
            self.clear()
        
        # Update active heads set
        self.active_heads = set(heads)
        
        # Group by layer for efficiency
        layers_to_heads = defaultdict(list)
        for layer, head in heads:
            layers_to_heads[layer].append(head)
        
        for layer, head_list in layers_to_heads.items():
            if self.optimize_hooks and (layer, -1) in self.hook_registry:
                # Hook already registered for this layer, just update active heads
                continue
            
            o_proj = self._get_o_proj(layer)
            
            # Create closure that captures layer and ablator reference
            def make_hook_closure(layer_idx, n_heads):
                def hook(module, args):
                    # Check if this layer has any active ablations
                    # Access active_heads from the ablator instance
                    layer_heads = [h for l, h in self.active_heads if l == layer_idx]
                    if not layer_heads:
                        # No active ablations for this layer, return unchanged
                        return args
                    
                    # CRITICAL: args[0] is the INPUT to o_proj (heads still separate!)
                    x = args[0].clone()  # Clone to avoid in-place modification issues
                    batch_size, seq_len, hidden_dim = x.shape
                    
                    # Verify dimensions
                    if hidden_dim % n_heads != 0:
                        # Skip if dimensions don't match (shouldn't happen, but be safe)
                        return args
                    
                    actual_head_dim = hidden_dim // n_heads
                    
                    # Reshape to separate heads
                    x = x.view(batch_size, seq_len, n_heads, actual_head_dim)
                    
                    # Zero out specified heads for this layer
                    for h in layer_heads:
                        x[:, :, h, :] = 0
                    
                    # Reshape back
                    x = x.view(batch_size, seq_len, hidden_dim)
                    
                    # Return modified input
                    return (x,) + args[1:] if len(args) > 1 else (x,)
                
                return hook
            
            # CRITICAL: Use pre_hook (on INPUT) not post_hook (on OUTPUT)
            if self.optimize_hooks:
                # Register hook once per layer, keep it active
                if (layer, -1) not in self.hook_registry:
                    handle = o_proj.register_forward_pre_hook(
                        make_hook_closure(layer, self.n_heads)
                    )
                    self.hooks.append(handle)
                    self.hook_registry[(layer, -1)] = handle
            else:
                # Original behavior: register new hook
                handle = o_proj.register_forward_pre_hook(
                    make_hook_closure(layer, self.n_heads)
                )
                self.hooks.append(handle)
    
    def clear(self):
        """Remove all ablation hooks or deactivate them."""
        if self.optimize_hooks:
            # Just clear active heads, keep hooks registered (faster for repeated use)
            self.active_heads = set()
        else:
            # Original behavior: remove all hooks
            for hook in self.hooks:
                hook.remove()
            self.hooks = []
            self.hook_registry = {}
    
    def cleanup(self):
        """
        Fully remove all hooks and clean up resources.
        Use this when done with all ablations to free memory.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.hook_registry = {}
        self.active_heads = set()
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support - auto-clear hooks."""
        self.clear()

