"""
Model utilities for loading and managing transformer models.

Provides a standardized interface for:
- Loading Qwen and other models
- Getting logits and probabilities
- Model configuration access
"""

import torch
from typing import Optional, Dict, List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer


# Default model configurations
DEFAULT_MODELS = {
    'qwen3-4b': 'Qwen/Qwen3-4B-Instruct-2507',
    'qwen3-8b': 'Qwen/Qwen3-8B-Instruct-2507',
}


class QwenModel:
    """
    Wrapper for Qwen model with mechanistic interpretability utilities.
    
    Provides a clean interface for:
    - Model loading with proper settings
    - Token probability extraction
    - Configuration access
    
    Example:
        model = QwenModel()
        model.load()
        probs = model.get_token_probs("Hello", [" world", " there"])
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Instruct-2507"):
        """
        Initialize model wrapper.
        
        Args:
            model_name: HuggingFace model identifier or path
        """
        # Resolve aliases
        if model_name in DEFAULT_MODELS:
            model_name = DEFAULT_MODELS[model_name]
            
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._config = None
        
    def load(
        self,
        dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        attn_implementation: Optional[str] = None,
        trust_remote_code: bool = True,
    ) -> 'QwenModel':
        """
        Load model and tokenizer.
        
        Args:
            dtype: Model dtype (float16 recommended for speed)
            device_map: Device placement strategy
            attn_implementation: "eager" for attention weights access, None for default
            trust_remote_code: Allow custom model code
            
        Returns:
            self for chaining
        """
        print(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote_code
        )
        
        load_kwargs = {
            'dtype': dtype,
            'device_map': device_map,
            'trust_remote_code': trust_remote_code,
        }
        
        if attn_implementation:
            load_kwargs['attn_implementation'] = attn_implementation
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs
        )
        self.model.eval()
        
        self._config = self.model.config
        print(f"Model loaded: {self.n_layers} layers, {self.n_heads} heads")
        
        return self
    
    @property
    def device(self) -> torch.device:
        """Get model device."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return next(self.model.parameters()).device
    
    @property
    def n_heads(self) -> int:
        """Number of attention heads."""
        return self._config.num_attention_heads
    
    @property
    def n_layers(self) -> int:
        """Number of transformer layers."""
        return self._config.num_hidden_layers
    
    @property
    def hidden_size(self) -> int:
        """Hidden dimension size."""
        return self._config.hidden_size
    
    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_size // self.n_heads
    
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text and move to model device."""
        inputs = self.tokenizer(text, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def encode_token(self, text: str) -> List[int]:
        """Get token IDs for text (no special tokens)."""
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def get_logits(self, text: str) -> torch.Tensor:
        """
        Get logits for text.
        
        Args:
            text: Input text
            
        Returns:
            Logits tensor of shape (vocab_size,) for last position
        """
        inputs = self.tokenize(text)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits[0, -1, :]  # Last token logits
    
    def get_token_probs(
        self,
        text: str,
        target_tokens: List[str]
    ) -> Dict[str, float]:
        """
        Get probabilities for specific tokens.
        
        Args:
            text: Input prompt
            target_tokens: List of token strings to get probs for
            
        Returns:
            Dict mapping token -> probability
        """
        logits = self.get_logits(text)
        probs = torch.softmax(logits, dim=-1)
        
        result = {}
        for token in target_tokens:
            token_ids = self.encode_token(token)
            if token_ids:
                result[token] = probs[token_ids[0]].item()
            else:
                result[token] = 0.0
                
        return result
    
    def compare_tokens(
        self,
        text: str,
        token_a: str,
        token_b: str
    ) -> Dict[str, Union[float, bool, str]]:
        """
        Compare probabilities of two tokens.
        
        Args:
            text: Input prompt
            token_a: First token
            token_b: Second token
            
        Returns:
            Dict with probs, comparison, and winner
        """
        probs = self.get_token_probs(text, [token_a, token_b])
        prob_a = probs.get(token_a, 0.0)
        prob_b = probs.get(token_b, 0.0)
        
        return {
            'prob_a': prob_a,
            'prob_b': prob_b,
            'a_wins': prob_a > prob_b,
            'winner': token_a if prob_a > prob_b else token_b,
            'margin': abs(prob_a - prob_b),
        }
    
    def get_layer(self, layer_idx: int):
        """Get a specific transformer layer module."""
        return self.model.model.layers[layer_idx]
    
    def get_attention(self, layer_idx: int):
        """Get attention module for a layer."""
        return self.get_layer(layer_idx).self_attn
    
    def get_o_proj(self, layer_idx: int):
        """Get output projection for a layer (hook point for interventions)."""
        return self.get_attention(layer_idx).o_proj


# Convenience function for quick loading
def load_model(
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    **kwargs
) -> QwenModel:
    """
    Quick model loading.
    
    Args:
        model_name: Model identifier
        **kwargs: Passed to QwenModel.load()
        
    Returns:
        Loaded QwenModel instance
    """
    return QwenModel(model_name).load(**kwargs)


