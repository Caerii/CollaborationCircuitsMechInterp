"""
Unified Activation Extractor

Consolidates activation extraction patterns from experiments 7, 8, 9, 11, 15, 17.
Provides efficient, batched extraction with optional caching.

Key features:
- Hook-based extraction (faster than nnsight)
- Batched processing for efficiency  
- Automatic caching to disk
- Support for layer outputs and attention patterns
- Memory-efficient with float16 storage
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from collections import defaultdict
import hashlib
import json

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FRAMEWORK_ROOT))

try:
    from ..config import ExperimentConfig
except ImportError:
    from config import ExperimentConfig


class ActivationCache:
    """
    Cache for storing and retrieving activations.
    
    Uses content-addressable storage based on prompt hashes.
    """
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / "index.json"
        self.index = self._load_index()
    
    def _load_index(self) -> Dict:
        if self.index_path.exists():
            with open(self.index_path, "r") as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, indent=2)
    
    def _hash_prompts(self, prompts: List[str], layers: List[int]) -> str:
        """Create unique hash for prompts + layers combination."""
        content = json.dumps({"prompts": prompts, "layers": sorted(layers)}, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def get(self, prompts: List[str], layers: List[int]) -> Optional[Dict]:
        """Retrieve cached activations if available."""
        key = self._hash_prompts(prompts, layers)
        if key in self.index:
            cache_file = self.cache_dir / f"{key}.npz"
            if cache_file.exists():
                data = np.load(cache_file)
                return {int(k): data[k] for k in data.files}
        return None
    
    def put(self, prompts: List[str], layers: List[int], activations: Dict[int, np.ndarray]):
        """Store activations in cache."""
        key = self._hash_prompts(prompts, layers)
        cache_file = self.cache_dir / f"{key}.npz"
        
        # Convert dict keys to strings for npz
        np.savez_compressed(cache_file, **{str(k): v for k, v in activations.items()})
        
        self.index[key] = {
            "n_prompts": len(prompts),
            "layers": layers,
            "file": str(cache_file.name),
        }
        self._save_index()


class ActivationExtractor:
    """
    Unified activation extraction with caching and batching.
    
    Consolidates best practices from experiments 7, 8, 9, 11, 15, 17:
    - Hook-based extraction for efficiency
    - Batched processing to reduce memory usage
    - Optional caching to avoid re-extraction
    - Support for both layer outputs and attention patterns
    
    Example:
        extractor = ActivationExtractor(model, tokenizer, config)
        
        # Extract layer outputs
        activations = extractor.extract_layer_outputs(texts)
        
        # Extract with labels for probing
        X, y = extractor.extract_with_labels(scenarios, label_key="belief_type")
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        config: ExperimentConfig,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize extractor.
        
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            config: Experiment configuration
            cache_dir: Optional directory for caching activations
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.layers = config.probe_layers
        self.batch_size = config.batch_size
        
        # Model properties
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads
        self.hidden_size = model.config.hidden_size
        self.head_dim = self.hidden_size // self.n_heads
        
        # Cache
        self.cache = ActivationCache(cache_dir) if cache_dir else None
    
    def extract_layer_outputs(
        self,
        texts: List[str],
        layers: Optional[List[int]] = None,
        position: str = "last",
        use_cache: bool = True
    ) -> Dict[int, np.ndarray]:
        """
        Extract hidden state activations from specified layers.
        
        Args:
            texts: List of input texts
            layers: Layers to extract from (default: config.probe_layers)
            position: Token position - "last", "first", or "all"
            use_cache: Whether to use/update cache
            
        Returns:
            Dict mapping layer index to activation array [n_samples, hidden_size]
        """
        layers = layers or self.layers
        
        # Check cache
        if use_cache and self.cache:
            cached = self.cache.get(texts, layers)
            if cached is not None:
                return cached
        
        # Storage
        activations = {layer: [] for layer in layers}
        captured = {}
        hooks = []
        
        # Create hooks
        def make_hook(layer_idx):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = hidden.detach()
            return hook
        
        # Register hooks
        for layer_idx in layers:
            hook = self.model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            hooks.append(hook)
        
        # Process in batches
        with torch.no_grad():
            for batch_start in range(0, len(texts), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(texts))
                batch_texts = texts[batch_start:batch_end]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.model.device)
                
                # Forward pass
                _ = self.model(**inputs)
                
                # Extract activations
                for layer_idx in layers:
                    hidden = captured[layer_idx]
                    
                    if position == "last":
                        # Get last non-padding token
                        seq_lens = inputs.attention_mask.sum(dim=1)
                        batch_acts = []
                        for i, seq_len in enumerate(seq_lens):
                            batch_acts.append(hidden[i, seq_len - 1, :].cpu().numpy())
                        activations[layer_idx].extend(batch_acts)
                    elif position == "first":
                        activations[layer_idx].extend(hidden[:, 0, :].cpu().numpy())
                    else:  # "all"
                        activations[layer_idx].extend(hidden.cpu().numpy())
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Convert to arrays
        result = {layer: np.array(acts, dtype=np.float16) for layer, acts in activations.items()}
        
        # Update cache
        if use_cache and self.cache:
            self.cache.put(texts, layers, result)
        
        return result
    
    def extract_attention_patterns(
        self,
        texts: List[str],
        layers: Optional[List[int]] = None
    ) -> Dict[int, np.ndarray]:
        """
        Extract attention patterns from specified layers.
        
        Note: Model must be loaded with attn_implementation='eager'.
        
        Args:
            texts: List of input texts
            layers: Layers to extract from
            
        Returns:
            Dict mapping layer index to attention array [n_samples, n_heads, seq_len, seq_len]
        """
        layers = layers or self.layers
        attention_patterns = {layer: [] for layer in layers}
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.model.device)
                
                outputs = self.model.model(**inputs, output_attentions=True)
                
                if outputs.attentions is None:
                    raise ValueError(
                        "Model not returning attention weights. "
                        "Load with attn_implementation='eager'"
                    )
                
                for layer_idx in layers:
                    if layer_idx < len(outputs.attentions):
                        attn = outputs.attentions[layer_idx][0].cpu().numpy()  # Remove batch dim
                        attention_patterns[layer_idx].append(attn)
        
        return {layer: np.array(attn) for layer, attn in attention_patterns.items()}
    
    def extract_with_labels(
        self,
        scenarios: List[Dict],
        text_key: str = "prompt",
        label_key: str = "type",
        layers: Optional[List[int]] = None
    ) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        """
        Extract activations with scenario labels for probing.
        
        Args:
            scenarios: List of scenario dictionaries
            text_key: Key for text in scenario dict
            label_key: Key for label in scenario dict
            layers: Layers to extract from
            
        Returns:
            Tuple of (activations dict, labels array)
        """
        texts = [s[text_key] for s in scenarios]
        labels = np.array([s.get(label_key, "unknown") for s in scenarios])
        
        activations = self.extract_layer_outputs(texts, layers=layers)
        
        return activations, labels
    
    def extract_head_outputs(
        self,
        texts: List[str],
        layer: int,
        head: int
    ) -> np.ndarray:
        """
        Extract output from a specific attention head.
        
        Args:
            texts: List of input texts
            layer: Layer index
            head: Head index
            
        Returns:
            Array of head outputs [n_samples, head_dim]
        """
        outputs = []
        captured = {}
        
        # Hook the attention output projection
        def hook(module, input, output):
            captured["input"] = input[0].detach()
        
        # Get o_proj module
        attn = self.model.model.layers[layer].self_attn
        o_proj = attn.o_proj
        handle = o_proj.register_forward_pre_hook(hook)
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.model.device)
                
                _ = self.model(**inputs)
                
                # Extract specific head
                hidden = captured["input"]
                seq_len = hidden.shape[1]
                hidden = hidden.view(1, seq_len, self.n_heads, self.head_dim)
                head_output = hidden[0, -1, head, :].cpu().numpy()
                outputs.append(head_output)
        
        handle.remove()
        
        return np.array(outputs, dtype=np.float16)
    
    def get_token_positions(self, text: str, target_tokens: List[str]) -> Dict[str, List[int]]:
        """
        Find positions of target tokens in tokenized text.
        
        Useful for identifying which positions to extract activations from.
        
        Args:
            text: Input text
            target_tokens: List of tokens/words to find
            
        Returns:
            Dict mapping target to list of positions
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        positions = {target: [] for target in target_tokens}
        
        for i, token in enumerate(tokens):
            # Clean token (remove special chars like Ġ for GPT-style)
            clean_token = token.replace("Ġ", "").replace("▁", "").lower()
            
            for target in target_tokens:
                if target.lower() in clean_token or clean_token in target.lower():
                    positions[target].append(i)
        
        return positions

