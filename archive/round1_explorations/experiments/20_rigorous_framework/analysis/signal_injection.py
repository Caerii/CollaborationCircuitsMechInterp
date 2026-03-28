"""
Signal Injection / Extraction

Extract the "update signal" by computing the difference between
clean and corrupted activations, then inject it to test causality.

This is powerful causal evidence - if injecting a signal restores
behavior, you've isolated the actual mechanism.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class InjectionResult:
    """Result of signal injection."""
    original_correct: bool
    injected_correct: bool
    flipped: bool
    original_prob: float
    injected_prob: float


class SignalExtractor:
    """
    Extract and inject activation signals.
    
    Method:
    1. Run CLEAN prompt (model gets it right), cache activations
    2. Run CORRUPTED prompt (model gets it wrong), cache activations
    3. Compute DIFFERENCE = clean - corrupted (this IS the signal)
    4. Inject signal into new corrupted prompts
    5. Measure if injection restores correct behavior
    
    Example:
        extractor = SignalExtractor(model, tokenizer)
        
        # Extract signal from one example
        signal = extractor.extract_signal(
            clean_prompt="Alice put ball in drawer. Bob told Alice he moved it to basket. Alice now knows. Alice looks in",
            corrupted_prompt="Alice put ball in drawer. Bob told Alice he moved it to basket. Alice looks in",
            layers=[17, 18, 19]
        )
        
        # Test injection on new examples
        result = extractor.inject_and_test(
            prompt="Carol put toy in box. Dan told Carol he moved it to shelf. Carol looks in",
            signal=signal,
            target_token=" shelf",
            contrast_token=" box"
        )
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_heads = model.config.num_attention_heads
        self.hooks = []
        self.cached = {}
    
    def _clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def _capture_activations(self, prompt: str, layers: List[int]) -> Dict[int, torch.Tensor]:
        """Run prompt and capture activations."""
        self._clear_hooks()
        captured = {}
        
        def make_hook(layer_idx):
            def hook(module, args):
                captured[layer_idx] = args[0].clone().detach()
                return args
            return hook
        
        for layer_idx in layers:
            o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
            h = o_proj.register_forward_pre_hook(make_hook(layer_idx))
            self.hooks.append(h)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            self.model(**inputs)
        
        self._clear_hooks()
        return captured
    
    def extract_signal(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        layers: List[int],
    ) -> Dict[int, torch.Tensor]:
        """
        Extract the update signal as clean - corrupted.
        
        Args:
            clean_prompt: Prompt where model succeeds
            corrupted_prompt: Prompt where model fails
            layers: Layers to extract from
            
        Returns:
            Dict mapping layer -> signal tensor
        """
        clean_acts = self._capture_activations(clean_prompt, layers)
        corrupt_acts = self._capture_activations(corrupted_prompt, layers)
        
        signals = {}
        for layer_idx in layers:
            clean = clean_acts[layer_idx]
            corrupt = corrupt_acts[layer_idx]
            
            # Align sequence lengths (use last position)
            signal = clean[0, -1, :] - corrupt[0, -1, :]
            signals[layer_idx] = signal
        
        return signals
    
    def inject_and_test(
        self,
        prompt: str,
        signal: Dict[int, torch.Tensor],
        target_token: str,
        contrast_token: str,
        strength: float = 1.0,
    ) -> InjectionResult:
        """
        Inject signal and test effect.
        
        Args:
            prompt: Test prompt
            signal: Signal to inject
            target_token: Expected correct token
            contrast_token: Wrong token
            strength: Injection strength multiplier
            
        Returns:
            InjectionResult
        """
        self._clear_hooks()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        target_id = self.tokenizer.encode(target_token, add_special_tokens=False)[0]
        contrast_id = self.tokenizer.encode(contrast_token, add_special_tokens=False)[0]
        
        # Test without injection
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1]
        
        orig_target = float(logits[target_id])
        orig_contrast = float(logits[contrast_id])
        orig_correct = orig_target > orig_contrast
        orig_prob = float(torch.softmax(logits, dim=-1)[target_id])
        
        # Setup injection hooks
        def make_inject_hook(layer_idx):
            sig = signal[layer_idx]
            def hook(module, args):
                hidden = args[0].clone()
                # Add signal to last position
                hidden[0, -1, :] += strength * sig.to(hidden.device).to(hidden.dtype)
                return (hidden,) + args[1:] if len(args) > 1 else (hidden,)
            return hook
        
        for layer_idx in signal.keys():
            o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
            h = o_proj.register_forward_pre_hook(make_inject_hook(layer_idx))
            self.hooks.append(h)
        
        # Test with injection
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1]
            
            inj_target = float(logits[target_id])
            inj_contrast = float(logits[contrast_id])
            inj_correct = inj_target > inj_contrast
            inj_prob = float(torch.softmax(logits, dim=-1)[target_id])
        finally:
            self._clear_hooks()
        
        return InjectionResult(
            original_correct=orig_correct,
            injected_correct=inj_correct,
            flipped=orig_correct != inj_correct,
            original_prob=orig_prob,
            injected_prob=inj_prob,
        )


class HeadAmplifier:
    """
    Amplify (or attenuate) specific attention heads.
    
    Amplification is complementary to ablation:
    - Ablation: "Is this head necessary?"
    - Amplification: "What happens if we boost this head?"
    
    If amplifying an inhibitory head makes behavior WORSE,
    that's strong evidence it's actively suppressing something.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_heads = model.config.num_attention_heads
        self.hooks = []
    
    def _clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def amplify_heads(
        self,
        heads: List[Tuple[int, int]],
        scale: float = 2.0,
    ):
        """
        Install amplification hooks.
        
        Args:
            heads: List of (layer, head) tuples
            scale: Amplification factor (2.0 = double)
        """
        self._clear_hooks()
        
        layer_to_heads = {}
        for layer, head in heads:
            if layer not in layer_to_heads:
                layer_to_heads[layer] = []
            layer_to_heads[layer].append(head)
        
        n_heads = self.n_heads
        
        def make_hook(head_indices, scale_factor):
            def hook(module, args):
                hidden = args[0].clone()
                batch, seq, hidden_dim = hidden.shape
                head_dim = hidden_dim // n_heads
                
                reshaped = hidden.view(batch, seq, n_heads, head_dim)
                for head_idx in head_indices:
                    reshaped[:, :, head_idx, :] *= scale_factor
                
                modified = reshaped.view(batch, seq, hidden_dim)
                return (modified,) + args[1:] if len(args) > 1 else (modified,)
            return hook
        
        for layer_idx, head_indices in layer_to_heads.items():
            o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
            h = o_proj.register_forward_pre_hook(make_hook(head_indices, scale))
            self.hooks.append(h)
    
    def test_with_amplification(
        self,
        prompt: str,
        heads: List[Tuple[int, int]],
        scales: List[float],
        target_token: str,
        contrast_token: str,
    ) -> Dict[float, Dict]:
        """
        Test behavior across amplification scales.
        
        Args:
            prompt: Test prompt
            heads: Heads to amplify
            scales: List of scale factors to test
            target_token: Expected token
            contrast_token: Alternative token
            
        Returns:
            Dict mapping scale -> result
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        target_id = self.tokenizer.encode(target_token, add_special_tokens=False)[0]
        contrast_id = self.tokenizer.encode(contrast_token, add_special_tokens=False)[0]
        
        results = {}
        
        for scale in scales:
            self.amplify_heads(heads, scale)
            
            try:
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits[0, -1]
                
                target_logit = float(logits[target_id])
                contrast_logit = float(logits[contrast_id])
                
                results[scale] = {
                    "correct": target_logit > contrast_logit,
                    "diff": target_logit - contrast_logit,
                    "target_prob": float(torch.softmax(logits, dim=-1)[target_id]),
                }
            finally:
                self._clear_hooks()
        
        return results

