"""
Model loading and activation extraction.

Uses HuggingFace transformers with forward hooks for clean activation access.
This approach works reliably with Qwen models on your RTX 3080.
"""
import torch
import gc
from typing import Dict, List, Optional, Tuple, Callable
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

from .config import MODEL_CFG, CACHE_DIR


class ModelWrapper:
    """
    Wrapper for Qwen model with activation extraction capabilities.
    Uses forward hooks for reliable activation capture.
    """
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or MODEL_CFG.model_name
        self.device = device or MODEL_CFG.device
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._hooks = []
        
    def load(self):
        """Load model and tokenizer."""
        if self._loaded:
            return self
        
        print(f"Loading {self.model_name}...")
        print(f"Using device: {self.device}, dtype: {MODEL_CFG.dtype}")
        
        # Check VRAM
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            free_mem = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
            print(f"GPU Memory: {free_mem:.1f} GB free / {total_mem:.1f} GB total")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with memory optimization
        dtype = getattr(torch, MODEL_CFG.dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto",  # Automatic device placement
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        self.model.eval()
        
        self._loaded = True
        print(f"[OK] Model loaded!")
        print(f"  Layers: {self.get_num_layers()}")
        print(f"  Hidden size: {self.get_hidden_size()}")
        
        return self
    
    def get_num_layers(self) -> int:
        """Get number of transformer layers."""
        if not self._loaded:
            self.load()
        return self.model.config.num_hidden_layers
    
    def get_hidden_size(self) -> int:
        """Get hidden dimension size."""
        if not self._loaded:
            self.load()
        return self.model.config.hidden_size
    
    def _get_layer(self, layer_idx: int):
        """Get a specific transformer layer module."""
        # Qwen structure: model.model.layers[i]
        return self.model.model.layers[layer_idx]
    
    @torch.no_grad()
    def extract_activations(
        self,
        text: str,
        layers: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract residual stream activations at specified layers.
        
        Args:
            text: Input text to process
            layers: Which layers to extract (default: probe_layers from config)
            
        Returns:
            Dict mapping layer index to activation tensor [seq_len, hidden_dim]
        """
        if not self._loaded:
            self.load()
        
        layers = layers or list(MODEL_CFG.probe_layers)
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MODEL_CFG.max_length
        ).to(self.model.device)
        
        # Storage for activations
        activations = {}
        
        # Create hooks
        def make_hook(layer_idx):
            def hook(module, input, output):
                # output is tuple, first element is hidden states
                hidden_states = output[0] if isinstance(output, tuple) else output
                activations[layer_idx] = hidden_states.detach().cpu()
            return hook
        
        # Register hooks
        handles = []
        for layer_idx in layers:
            if layer_idx < self.get_num_layers():
                handle = self._get_layer(layer_idx).register_forward_hook(make_hook(layer_idx))
                handles.append(handle)
        
        # Forward pass
        try:
            _ = self.model(**inputs)
        finally:
            # Always remove hooks
            for handle in handles:
                handle.remove()
        
        # Process activations - squeeze batch dimension
        result = {}
        for layer_idx, act in activations.items():
            result[layer_idx] = act.squeeze(0)  # [seq_len, hidden_dim]
        
        return result
    
    @torch.no_grad()
    def extract_activations_batch(
        self,
        texts: List[str],
        layers: Optional[List[int]] = None,
        last_token_only: bool = True,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract activations for a batch of texts.
        
        Args:
            texts: List of input texts
            layers: Which layers to extract
            last_token_only: If True, only extract last token's activation
            
        Returns:
            Dict mapping layer index to activation tensor [batch, hidden_dim] or [batch, seq_len, hidden_dim]
        """
        if not self._loaded:
            self.load()
        
        layers = layers or list(MODEL_CFG.probe_layers)
        
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MODEL_CFG.max_length
        ).to(self.model.device)
        
        attention_mask = inputs["attention_mask"]
        
        # Storage
        activations = {}
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                hidden_states = output[0] if isinstance(output, tuple) else output
                activations[layer_idx] = hidden_states.detach().cpu()
            return hook
        
        # Register hooks
        handles = []
        for layer_idx in layers:
            if layer_idx < self.get_num_layers():
                handle = self._get_layer(layer_idx).register_forward_hook(make_hook(layer_idx))
                handles.append(handle)
        
        # Forward
        try:
            _ = self.model(**inputs)
        finally:
            for handle in handles:
                handle.remove()
        
        # Process
        result = {}
        for layer_idx, act in activations.items():
            if last_token_only:
                # Get last non-padding token for each sequence
                seq_lens = attention_mask.sum(dim=1).cpu() - 1
                batch_indices = torch.arange(act.size(0))
                result[layer_idx] = act[batch_indices, seq_lens]  # [batch, hidden_dim]
            else:
                result[layer_idx] = act  # [batch, seq_len, hidden_dim]
        
        return result
    
    def tokenize(self, text: str) -> Dict:
        """Tokenize text and return token info."""
        if not self._loaded:
            self.load()
        
        tokens = self.tokenizer(text, return_tensors="pt")
        token_ids = tokens["input_ids"][0].tolist()
        token_strs = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        return {
            "input_ids": token_ids,
            "tokens": token_strs,
            "length": len(token_ids)
        }
    
    def get_token_positions(self, text: str, substrings: List[str]) -> Dict[str, List[int]]:
        """
        Find token positions for given substrings.
        Useful for identifying speaker turns.
        """
        if not self._loaded:
            self.load()
        
        encoded = self.tokenizer(text, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        
        results = {}
        for substr in substrings:
            start_idx = text.find(substr)
            if start_idx == -1:
                results[substr] = []
                continue
            
            end_idx = start_idx + len(substr)
            token_positions = []
            
            for tok_idx, (tok_start, tok_end) in enumerate(offsets):
                if tok_start >= start_idx and tok_end <= end_idx:
                    token_positions.append(tok_idx)
                elif tok_start < end_idx and tok_end > start_idx:
                    token_positions.append(tok_idx)
            
            results[substr] = token_positions
        
        return results
    
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 100, **kwargs) -> str:
        """Generate text completion."""
        if not self._loaded:
            self.load()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def unload(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False
        gc.collect()
        torch.cuda.empty_cache()
        print("Model unloaded, GPU memory freed")


# Singleton instance
_model_wrapper: Optional[ModelWrapper] = None


def get_model() -> ModelWrapper:
    """Get or create model wrapper singleton."""
    global _model_wrapper
    if _model_wrapper is None:
        _model_wrapper = ModelWrapper()
    return _model_wrapper


def clear_model():
    """Clear the model singleton and free memory."""
    global _model_wrapper
    if _model_wrapper is not None:
        _model_wrapper.unload()
        _model_wrapper = None
