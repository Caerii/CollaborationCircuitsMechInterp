"""
Hook management for mechanistic interpretability interventions.

Provides clean interfaces for:
- Ablation (zeroing heads)
- Amplification (scaling heads)
- Activation caching (for path patching)
- Activation patching (replacing activations)

Note: These hooks work at the o_proj input (after attention, before projection),
which is the correct intervention point for head-level modifications in GQA models.
"""

import torch
from typing import List, Tuple, Dict, Optional, Callable
from collections import defaultdict


class ActivationCache:
    """
    Store and retrieve cached activations for path patching.
    
    Activations are stored per (layer, head) with position information.
    """
    
    def __init__(self):
        self.cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self._hooks = []
        
    def store(self, layer: int, head: int, activation: torch.Tensor):
        """Store activation for a head."""
        self.cache[(layer, head)] = activation.detach().clone()
        
    def retrieve(self, layer: int, head: int) -> Optional[torch.Tensor]:
        """Retrieve cached activation."""
        return self.cache.get((layer, head))
    
    def store_layer(self, layer: int, activations: torch.Tensor, n_heads: int):
        """Store all heads for a layer from reshaped tensor."""
        # activations: (batch, seq, n_heads, head_dim)
        for h in range(n_heads):
            self.store(layer, h, activations[:, :, h, :])
    
    def clear(self):
        """Clear all cached activations."""
        self.cache.clear()
        
    def has(self, layer: int, head: int) -> bool:
        """Check if activation is cached."""
        return (layer, head) in self.cache
    
    def keys(self) -> List[Tuple[int, int]]:
        """Get all cached (layer, head) pairs."""
        return list(self.cache.keys())


class HookManager:
    """
    Manage PyTorch hooks for mechanistic interpretability interventions.
    
    This class provides a unified interface for:
    - Ablating (zeroing) specific attention heads
    - Amplifying (scaling) specific heads
    - Caching activations for later patching
    - Patching activations from a cache
    - Combined interventions
    
    All interventions target the o_proj input (attention output before projection),
    which is the correct point for head-level interventions in GQA architectures.
    
    Example:
        manager = HookManager(model)
        manager.ablate_heads([(17, 4), (18, 11)])  # Ablate two heads
        # ... run inference ...
        manager.clear()  # Remove hooks
    """
    
    def __init__(self, model):
        """
        Initialize hook manager.
        
        Args:
            model: QwenModel instance or raw model with .model.layers structure
        """
        # Handle both QwenModel wrapper and raw model
        if hasattr(model, 'model') and hasattr(model.model, 'model'):
            # QwenModel wrapper
            self._model = model.model
            self.n_heads = model.n_heads
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # Raw HF model
            self._model = model
            self.n_heads = model.config.num_attention_heads
        else:
            raise ValueError("Unsupported model structure")
            
        self.hooks: List = []
        self._intervention_config = {}
        
    def clear(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self._intervention_config = {}
        
    def _get_o_proj(self, layer_idx: int):
        """Get output projection module for a layer."""
        return self._model.model.layers[layer_idx].self_attn.o_proj
    
    def _create_ablation_hook(self, heads_to_ablate: List[int]) -> Callable:
        """Create hook that zeros specified heads."""
        n_heads = self.n_heads
        
        def hook(module, args):
            hidden = args[0]
            batch, seq, dim = hidden.shape
            head_dim = dim // n_heads
            
            reshaped = hidden.view(batch, seq, n_heads, head_dim)
            for h in heads_to_ablate:
                reshaped[:, :, h, :] = 0
                
            return (reshaped.view(batch, seq, dim),) + args[1:] if len(args) > 1 else (reshaped.view(batch, seq, dim),)
        
        return hook
    
    def _create_amplification_hook(self, heads_scales: List[Tuple[int, float]]) -> Callable:
        """Create hook that scales specified heads."""
        n_heads = self.n_heads
        
        def hook(module, args):
            hidden = args[0]
            batch, seq, dim = hidden.shape
            head_dim = dim // n_heads
            
            reshaped = hidden.view(batch, seq, n_heads, head_dim)
            for h, scale in heads_scales:
                reshaped[:, :, h, :] = reshaped[:, :, h, :] * scale
                
            return (reshaped.view(batch, seq, dim),) + args[1:] if len(args) > 1 else (reshaped.view(batch, seq, dim),)
        
        return hook
    
    def _create_combined_hook(
        self,
        ablate_heads: List[int],
        amplify_heads: List[Tuple[int, float]]
    ) -> Callable:
        """Create hook that both ablates and amplifies."""
        n_heads = self.n_heads
        
        def hook(module, args):
            hidden = args[0]
            batch, seq, dim = hidden.shape
            head_dim = dim // n_heads
            
            reshaped = hidden.view(batch, seq, n_heads, head_dim)
            
            # Ablate first
            for h in ablate_heads:
                reshaped[:, :, h, :] = 0
                
            # Then amplify
            for h, scale in amplify_heads:
                reshaped[:, :, h, :] = reshaped[:, :, h, :] * scale
                
            return (reshaped.view(batch, seq, dim),) + args[1:] if len(args) > 1 else (reshaped.view(batch, seq, dim),)
        
        return hook
    
    def _create_cache_hook(self, layer: int, cache: ActivationCache) -> Callable:
        """Create hook that caches activations."""
        n_heads = self.n_heads
        
        def hook(module, args):
            hidden = args[0]
            batch, seq, dim = hidden.shape
            head_dim = dim // n_heads
            
            reshaped = hidden.view(batch, seq, n_heads, head_dim)
            cache.store_layer(layer, reshaped, n_heads)
            
            # Don't modify - just cache
            return args
        
        return hook
    
    def _create_patch_hook(
        self,
        heads_to_patch: List[int],
        cache: ActivationCache,
        layer: int
    ) -> Callable:
        """Create hook that patches specified heads from cache."""
        n_heads = self.n_heads
        
        def hook(module, args):
            hidden = args[0]
            batch, seq, dim = hidden.shape
            head_dim = dim // n_heads
            
            reshaped = hidden.view(batch, seq, n_heads, head_dim)
            
            for h in heads_to_patch:
                cached = cache.retrieve(layer, h)
                if cached is not None:
                    # Handle sequence length mismatch
                    min_seq = min(seq, cached.shape[1])
                    reshaped[:, :min_seq, h, :] = cached[:, :min_seq, :]
                    
            return (reshaped.view(batch, seq, dim),) + args[1:] if len(args) > 1 else (reshaped.view(batch, seq, dim),)
        
        return hook
    
    def ablate_heads(self, heads: List[Tuple[int, int]]):
        """
        Zero out specified attention heads.
        
        Args:
            heads: List of (layer, head) tuples to ablate
        """
        self.clear()
        
        # Group by layer
        layers_to_heads = defaultdict(list)
        for layer, head in heads:
            layers_to_heads[layer].append(head)
        
        for layer, head_list in layers_to_heads.items():
            o_proj = self._get_o_proj(layer)
            hook = o_proj.register_forward_pre_hook(
                self._create_ablation_hook(head_list)
            )
            self.hooks.append(hook)
            
        self._intervention_config = {'type': 'ablation', 'heads': heads}
    
    def amplify_heads(self, heads_scales: List[Tuple[int, int, float]]):
        """
        Scale specified attention heads.
        
        Args:
            heads_scales: List of (layer, head, scale) tuples
        """
        self.clear()
        
        # Group by layer
        layers_to_heads = defaultdict(list)
        for layer, head, scale in heads_scales:
            layers_to_heads[layer].append((head, scale))
        
        for layer, head_scale_list in layers_to_heads.items():
            o_proj = self._get_o_proj(layer)
            hook = o_proj.register_forward_pre_hook(
                self._create_amplification_hook(head_scale_list)
            )
            self.hooks.append(hook)
            
        self._intervention_config = {'type': 'amplification', 'heads_scales': heads_scales}
    
    def combined_intervention(
        self,
        ablate_heads: List[Tuple[int, int]],
        amplify_heads_scales: List[Tuple[int, int, float]]
    ):
        """
        Apply both ablation and amplification.
        
        Args:
            ablate_heads: List of (layer, head) tuples to zero
            amplify_heads_scales: List of (layer, head, scale) tuples to scale
        """
        self.clear()
        
        # Collect all layers involved
        all_layers = set()
        ablate_by_layer = defaultdict(list)
        amplify_by_layer = defaultdict(list)
        
        for layer, head in ablate_heads:
            all_layers.add(layer)
            ablate_by_layer[layer].append(head)
            
        for layer, head, scale in amplify_heads_scales:
            all_layers.add(layer)
            amplify_by_layer[layer].append((head, scale))
        
        for layer in all_layers:
            o_proj = self._get_o_proj(layer)
            hook = o_proj.register_forward_pre_hook(
                self._create_combined_hook(
                    ablate_by_layer[layer],
                    amplify_by_layer[layer]
                )
            )
            self.hooks.append(hook)
            
        self._intervention_config = {
            'type': 'combined',
            'ablate_heads': ablate_heads,
            'amplify_heads_scales': amplify_heads_scales
        }
    
    def install_cache_hooks(self, layers: List[int], cache: ActivationCache):
        """
        Install hooks to cache activations for specified layers.
        
        Args:
            layers: Layer indices to cache
            cache: ActivationCache instance to store to
        """
        self.clear()
        cache.clear()
        
        for layer in layers:
            o_proj = self._get_o_proj(layer)
            hook = o_proj.register_forward_pre_hook(
                self._create_cache_hook(layer, cache)
            )
            self.hooks.append(hook)
            
        self._intervention_config = {'type': 'caching', 'layers': layers}
    
    def install_patch_hooks(
        self,
        heads: List[Tuple[int, int]],
        cache: ActivationCache
    ):
        """
        Install hooks to patch specified heads from cache.
        
        Args:
            heads: List of (layer, head) tuples to patch
            cache: ActivationCache with stored activations
        """
        self.clear()
        
        # Group by layer
        layers_to_heads = defaultdict(list)
        for layer, head in heads:
            layers_to_heads[layer].append(head)
        
        for layer, head_list in layers_to_heads.items():
            o_proj = self._get_o_proj(layer)
            hook = o_proj.register_forward_pre_hook(
                self._create_patch_hook(head_list, cache, layer)
            )
            self.hooks.append(hook)
            
        self._intervention_config = {'type': 'patching', 'heads': heads}
    
    @property
    def active(self) -> bool:
        """Check if any hooks are installed."""
        return len(self.hooks) > 0
    
    @property
    def config(self) -> dict:
        """Get current intervention configuration."""
        return self._intervention_config.copy()


# Convenience context manager
class InterventionContext:
    """
    Context manager for temporary interventions.
    
    Example:
        with InterventionContext(manager, 'ablate', [(17, 4)]):
            result = model(...)
        # Hooks automatically removed
    """
    
    def __init__(
        self,
        manager: HookManager,
        intervention_type: str,
        config: any
    ):
        self.manager = manager
        self.type = intervention_type
        self.config = config
        
    def __enter__(self):
        if self.type == 'ablate':
            self.manager.ablate_heads(self.config)
        elif self.type == 'amplify':
            self.manager.amplify_heads(self.config)
        elif self.type == 'combined':
            self.manager.combined_intervention(*self.config)
        return self
        
    def __exit__(self, *args):
        self.manager.clear()


