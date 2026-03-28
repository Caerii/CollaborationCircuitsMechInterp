"""
Activation Patching for Causal Analysis

Ported from experiment 13's activation_patching.py.
Key technique for proving causal relationships.

The idea:
1. Run model on source context (e.g., "agent agrees"), cache activations
2. Run model on target context (e.g., "agent disagrees")  
3. Patch source activations into target at specific layers
4. If behavior flips, those activations CAUSE the behavior

IMPORTANT: For reasoning models like Qwen3-4B, use chat_mode=True!
Completion mode produces unreliable results for ToM tasks.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class PatchingResult:
    """Result from an activation patching experiment."""
    source_context: str
    target_context: str
    layers_patched: List[int]
    base_response: str
    patched_response: str
    flipped: bool
    
    def to_dict(self) -> Dict:
        return {
            "source": self.source_context[:50],
            "target": self.target_context[:50],
            "layers": self.layers_patched,
            "base_response": self.base_response[:50],
            "patched_response": self.patched_response[:50],
            "flipped": self.flipped,
        }


class ActivationPatcher:
    """
    Perform activation patching experiments.
    
    This is the GOLD STANDARD for causal claims in MI.
    If patching activations from context A into context B
    makes B behave like A, then those activations CAUSE the behavior.
    
    IMPORTANT: Use chat_mode=True for reasoning models like Qwen3-4B!
    
    Example (chat mode - recommended):
        patcher = ActivationPatcher(model, tokenizer, chat_mode=True)
        
        result = patcher.run_patching_experiment(
            source_prompt="Sally saw the ball move to the box. Where does Sally think it is?",
            target_prompt="Sally didn't see the ball move. Where does Sally think it is?",
            layers=[20],
            check_flip_fn=lambda b, p: "box" in p.lower() and "basket" in b.lower()
        )
    
    Example (completion mode - legacy):
        patcher = ActivationPatcher(model, tokenizer, chat_mode=False)
        # ... same API
    """
    
    # Chat template for Qwen-style models
    CHAT_TEMPLATE = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""
    
    DEFAULT_SYSTEM = "Think step by step in <think> tags. Then give ONE WORD answer."
    
    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens: int = 30,
        chat_mode: bool = False,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize patcher.
        
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            max_new_tokens: Max tokens to generate
            chat_mode: If True, wrap prompts in chat format (RECOMMENDED for reasoning models)
            system_prompt: Custom system prompt for chat mode
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.n_layers = model.config.num_hidden_layers
        self.chat_mode = chat_mode
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM
        
        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt for chat mode if enabled."""
        if not self.chat_mode:
            return prompt
        return self.CHAT_TEMPLATE.format(
            system_prompt=self.system_prompt,
            user_prompt=prompt
        )
    
    def _decode_response(self, output_ids: torch.Tensor, input_len: int) -> str:
        """Decode only the generated tokens."""
        new_tokens = output_ids[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    def cache_activations(
        self,
        prompt: str,
        layers: List[int]
    ) -> Dict[int, torch.Tensor]:
        """
        Run forward pass and cache layer activations.
        
        Args:
            prompt: Source context (will be chat-formatted if chat_mode=True)
            layers: Layers to cache
            
        Returns:
            Dict mapping layer -> activation tensor
        """
        formatted = self._format_prompt(prompt)
        
        cached = {}
        hooks = []
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                cached[layer_idx] = hidden.detach().clone()
            return hook
        
        for layer in layers:
            h = self.model.model.layers[layer].register_forward_hook(make_hook(layer))
            hooks.append(h)
        
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            _ = self.model(**inputs)
        
        for h in hooks:
            h.remove()
        
        return cached
    
    def patch_and_generate(
        self,
        target_prompt: str,
        source_activations: Dict[int, torch.Tensor],
        layers: List[int],
        patch_mode: str = "prompt_end"  # "all", "last", "first", "prompt_end"
    ) -> str:
        """
        Generate with source activations patched in.
        
        Args:
            target_prompt: Target context (will be chat-formatted if chat_mode=True)
            source_activations: Cached activations from source
            layers: Layers to patch
            patch_mode: Which positions to patch
                - "all": patch all overlapping positions (risky for chat mode)
                - "last": only patch last token of current sequence (during generation)
                - "first": only patch first token
                - "prompt_end": patch only at the end of prompt (before generation starts)
            
        Returns:
            Generated text (response only, not prompt)
        """
        formatted = self._format_prompt(target_prompt)
        hooks = []
        
        # Get target prompt length for "prompt_end" mode
        target_inputs = self.tokenizer(formatted, return_tensors="pt")
        target_prompt_len = target_inputs.input_ids.shape[1]
        
        # Get source prompt length
        source_prompt_len = None
        if source_activations:
            first_layer = list(source_activations.keys())[0]
            source_prompt_len = source_activations[first_layer].shape[1]
        
        def make_patch_hook(layer_idx):
            call_count = [0]  # Track generation step
            
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                source = source_activations[layer_idx]
                current_seq_len = hidden.shape[1]
                call_count[0] += 1
                
                if patch_mode == "prompt_end":
                    # Only patch at the exact position where prompt ends
                    # With use_cache=False, this happens on the first forward pass
                    if current_seq_len == target_prompt_len:
                        # We're at the end of the prompt, patch the last token
                        if source.shape[1] >= 1:
                            hidden[:, -1, :] = source[:, -1, :].clone()
                elif patch_mode == "last_token_only":
                    # Patch only the last token, but only during prompt processing
                    # During generation, we want to patch the prompt's last token position
                    if current_seq_len == target_prompt_len:
                        # At end of prompt: patch the last token
                        if source.shape[1] >= 1:
                            hidden[:, -1, :] = source[:, -1, :].clone()
                    elif current_seq_len > target_prompt_len:
                        # During generation: patch the prompt's last token position
                        # This position is now at index (target_prompt_len - 1)
                        prompt_end_idx = target_prompt_len - 1
                        if source.shape[1] > prompt_end_idx:
                            hidden[:, prompt_end_idx, :] = source[:, prompt_end_idx, :].clone()
                elif patch_mode == "all":
                    # Patch all overlapping positions (risky!)
                    min_len = min(current_seq_len, source.shape[1])
                    hidden[:, :min_len, :] = source[:, :min_len, :].clone()
                elif patch_mode == "last":
                    # Patch last token of current sequence
                    if source.shape[1] >= 1:
                        hidden[:, -1, :] = source[:, -1, :].clone()
                elif patch_mode == "first":
                    hidden[:, 0, :] = source[:, 0, :].clone()
                
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden
            return hook
        
        for layer in layers:
            if layer in source_activations:
                h = self.model.model.layers[layer].register_forward_hook(make_patch_hook(layer))
                hooks.append(h)
        
        inputs = target_inputs.to(self.model.device)
        input_len = inputs.input_ids.shape[1]
        
        try:
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,  # CRITICAL: Disable KV cache to allow patching
                )
        finally:
            for h in hooks:
                h.remove()
        
        return self._decode_response(output, input_len)
    
    def generate_baseline(self, prompt: str) -> str:
        """Generate without patching (uses chat_mode if enabled)."""
        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self._decode_response(output, input_len)
    
    def run_patching_experiment(
        self,
        source_prompt: str,
        target_prompt: str,
        layers: List[int],
        check_flip_fn: Callable[[str, str], bool]
    ) -> PatchingResult:
        """
        Run a complete patching experiment.
        
        Args:
            source_prompt: Context to patch FROM
            target_prompt: Context to patch INTO
            layers: Layers to patch
            check_flip_fn: Function(base, patched) -> bool indicating if behavior flipped
            
        Returns:
            PatchingResult
        """
        # Get baseline
        base_response = self.generate_baseline(target_prompt)
        
        # Cache source activations
        source_acts = self.cache_activations(source_prompt, layers)
        
        # Generate with patching
        patched_response = self.patch_and_generate(target_prompt, source_acts, layers)
        
        # Check if flipped
        flipped = check_flip_fn(base_response, patched_response)
        
        return PatchingResult(
            source_context=source_prompt,
            target_context=target_prompt,
            layers_patched=layers,
            base_response=base_response,
            patched_response=patched_response,
            flipped=flipped,
        )
    
    def sweep_layers(
        self,
        source_prompt: str,
        target_prompt: str,
        check_flip_fn: Callable,
        layer_groups: Optional[Dict[str, List[int]]] = None
    ) -> Dict[str, PatchingResult]:
        """
        Sweep patching across layer groups to find causal layers.
        
        Args:
            source_prompt: Source context
            target_prompt: Target context
            check_flip_fn: Function to check if behavior flipped
            layer_groups: Named groups of layers to test
            
        Returns:
            Dict mapping group name to result
        """
        if layer_groups is None:
            layer_groups = {
                "early": list(range(0, 6)),
                "mid": list(range(12, 18)),
                "late": list(range(30, 36)),
                "all": list(range(self.n_layers)),
            }
        
        results = {}
        for name, layers in layer_groups.items():
            results[name] = self.run_patching_experiment(
                source_prompt, target_prompt, layers, check_flip_fn
            )
        
        return results


# Standard check functions
def check_agreement_flip(base: str, patched: str) -> bool:
    """Check if response flipped between agree/disagree."""
    base_lower = base.lower()
    patched_lower = patched.lower()
    
    base_agrees = any(w in base_lower for w in ["yes", "agree", "correct"])
    patched_agrees = any(w in patched_lower for w in ["yes", "agree", "correct"])
    
    return base_agrees != patched_agrees

