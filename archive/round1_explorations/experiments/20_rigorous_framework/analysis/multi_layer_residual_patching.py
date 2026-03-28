"""
Multi-Layer Residual Stream Patching

The key insight: We need to patch residual stream activations at MULTIPLE layers
simultaneously, EARLY in generation (steps 0-50), to affect the distributed ToM circuit.

This addresses:
1. Logit manipulation is insufficient (responses identical)
2. Decision happens at step 0 (need early intervention)
3. Circuit is distributed (need multiple layers)
4. Need activation-level intervention (not just logits)
"""

import torch
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MultiLayerPatchingResult:
    """Result from multi-layer residual stream patching."""
    source_context: str
    target_context: str
    layers_patched: List[int]
    patch_positions: List[int]  # Generation steps where we patched
    base_response: str
    patched_response: str
    flipped: bool
    base_answer: Optional[str]
    patched_answer: Optional[str]


class MultiLayerResidualPatcher:
    """
    Patch residual stream at multiple layers simultaneously, early in generation.
    
    Key features:
    - Patches at multiple layers (distributed circuit)
    - Intervenes early (steps 0-50) when decision forms
    - Patches residual stream (activation-level, not logits)
    - Simultaneous patching at all layers
    
    Example:
        patcher = MultiLayerResidualPatcher(model, tokenizer, chat_mode=True)
        
        result = patcher.patch_early(
            source_prompt="Sally watched Anne move ball to box...",  # TB
            target_prompt="Sally left, Anne moved ball to box...",   # FB
            layers=[20, 24, 28, 32],
            early_steps=[0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        )
    """
    
    CHAT_TEMPLATE = """<|im_start|>system
Think step by step in <think> tags. Then give ONE WORD answer.<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""
    
    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens: int = 500,
        chat_mode: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.chat_mode = chat_mode
        self.n_layers = model.config.num_hidden_layers
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt for chat mode if enabled."""
        if not self.chat_mode:
            return prompt
        return self.CHAT_TEMPLATE.format(user_prompt=prompt)
    
    def cache_residual_stream_multi_layer(
        self,
        prompt: str,
        layers: List[int],
        position: Optional[int] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Cache residual stream activations at multiple layers.
        
        IMPORTANT: We cache from the END of the source prompt, which represents
        the "belief state" after processing the source scenario. This should be
        patched into early generation steps of the target.
        
        Args:
            prompt: Source prompt (e.g., TB scenario)
            layers: Layers to cache
            position: Token position to cache (None = last position)
            
        Returns:
            Dict mapping layer -> residual stream activation (shape: [hidden_size])
        """
        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        prompt_len = inputs.input_ids.shape[1]
        
        cached = {}
        hooks = []
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                # output is (hidden_states,) or hidden_states
                hidden = output[0] if isinstance(output, tuple) else output
                
                if position is None:
                    # Cache last position (end of prompt) - this is the "belief state"
                    cached[layer_idx] = hidden[0, -1, :].detach().clone()
                else:
                    # Cache specific position
                    if position < hidden.shape[1]:
                        cached[layer_idx] = hidden[0, position, :].detach().clone()
            return hook
        
        for layer in layers:
            if layer < self.n_layers:
                # Hook on the layer output (residual stream after layer)
                h = self.model.model.layers[layer].register_forward_hook(make_hook(layer))
                hooks.append(h)
        
        with torch.no_grad():
            _ = self.model(**inputs)
        
        for h in hooks:
            h.remove()
        
        # Verify cached activations have correct shape
        for layer_idx, act in cached.items():
            if len(act.shape) != 1:
                print(f"      Warning: Cached activation at L{layer_idx} has shape {act.shape}, expected 1D", flush=True)
        
        return cached
    
    def patch_early(
        self,
        target_prompt: str,
        source_activations: Dict[int, torch.Tensor],
        layers: List[int],
        early_steps: Optional[List[int]] = None,
        max_early_steps: int = 50
    ) -> MultiLayerPatchingResult:
        """
        Patch residual stream at multiple layers, early in generation.
        
        Args:
            target_prompt: Target prompt (e.g., FB scenario)
            source_activations: Cached activations from source (e.g., TB)
            layers: Layers to patch simultaneously
            early_steps: Specific steps to patch (None = patch steps 0-max_early_steps)
            max_early_steps: Max step to patch if early_steps not provided
            
        Returns:
            MultiLayerPatchingResult
        """
        formatted = self._format_prompt(target_prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        prompt_len = inputs.input_ids.shape[1]
        
        # Determine which steps to patch
        if early_steps is None:
            early_steps = list(range(max_early_steps + 1))
        
        # Get baseline
        print("    Getting baseline...", flush=True)
        sys.stdout.flush()
        with torch.no_grad():
            baseline_outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        base_response = self.tokenizer.decode(
            baseline_outputs[0][prompt_len:],
            skip_special_tokens=True
        )
        print("    Baseline complete!", flush=True)
        sys.stdout.flush()
        
        # Now patch during generation
        print(f"    Patching at {len(early_steps)} early positions across {len(layers)} layers...", flush=True)
        sys.stdout.flush()
        
        patched_ids = inputs.input_ids.clone()
        patch_positions = []
        hooks = []
        step_count = 0
        
        def make_patch_hook(layer_idx):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                current_step = step_count
                
                # Patch if we're at an early step and have source activation
                if current_step in early_steps and layer_idx in source_activations:
                    # CRITICAL: Only patch if shapes match
                    source_act = source_activations[layer_idx]
                    target_act = hidden[0, -1, :]
                    
                    if source_act.shape == target_act.shape:
                        # Try BLENDING instead of replacing (less disruptive)
                        # Blend: 70% source (TB belief) + 30% target (FB current)
                        blend_ratio = 0.7
                        blended = blend_ratio * source_act + (1 - blend_ratio) * target_act
                        hidden[0, -1, :] = blended
                        
                        if current_step not in patch_positions:
                            patch_positions.append(current_step)
                    else:
                        # Shape mismatch - skip this patch
                        print(f"      Warning: Shape mismatch at step {current_step}, layer {layer_idx} (source: {source_act.shape}, hidden: {target_act.shape})", flush=True)
                
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden
            return hook
        
        # Register hooks on all layers
        for layer in layers:
            if layer < self.n_layers and layer in source_activations:
                h = self.model.model.layers[layer].register_forward_hook(make_patch_hook(layer))
                hooks.append(h)
        
        try:
            with torch.no_grad():
                for step in range(self.max_new_tokens):
                    if step % 50 == 0 and step > 0:
                        print(f"      Step {step}...", flush=True)
                        sys.stdout.flush()
                    
                    outputs = self.model(input_ids=patched_ids)
                    logits = outputs.logits
                    
                    # Get next token
                    next_token = logits[0, -1].argmax(dim=-1)
                    next_token = next_token.unsqueeze(0).unsqueeze(0)
                    patched_ids = torch.cat([patched_ids, next_token], dim=-1)
                    step_count += 1
                    
                    if next_token[0, 0].item() == self.tokenizer.eos_token_id:
                        # If EOS too early (before step 50), might be corruption - continue anyway
                        if step_count < 50:
                            print(f"      Warning: Early EOS at step {step_count}, continuing...", flush=True)
                            sys.stdout.flush()
                            # Don't break - force continue generation
                        else:
                            print(f"      EOS at step {step_count}", flush=True)
                            sys.stdout.flush()
                            break
                    
                    # Stop if we've generated enough
                    if step_count > 300:
                        break
        finally:
            for h in hooks:
                h.remove()
        
        print(f"    Complete! Patched at {len(patch_positions)} positions.", flush=True)
        sys.stdout.flush()
        
        patched_response = self.tokenizer.decode(
            patched_ids[0][prompt_len:],
            skip_special_tokens=True
        )
        
        # Extract answers
        base_answer = self._extract_answer(base_response)
        patched_answer = self._extract_answer(patched_response)
        
        # Check if flipped
        flipped = (base_answer != patched_answer and 
                  base_answer is not None and 
                  patched_answer is not None)
        
        return MultiLayerPatchingResult(
            source_context="",  # Will be set by caller
            target_context=target_prompt,
            layers_patched=layers,
            patch_positions=sorted(patch_positions),
            base_response=base_response,
            patched_response=patched_response,
            flipped=flipped,
            base_answer=base_answer,
            patched_answer=patched_answer,
        )
    
    def _extract_answer(self, response: str) -> Optional[str]:
        """Extract answer from response."""
        text = response.lower()
        if "</think>" in text:
            parts = text.split("</think>")
            if len(parts) > 1:
                text = parts[-1]
        
        if "basket" in text:
            return "basket"
        elif "box" in text:
            return "box"
        return None
    
    def run_patching_experiment(
        self,
        source_prompt: str,
        target_prompt: str,
        layers: List[int],
        early_steps: Optional[List[int]] = None,
        max_early_steps: int = 50
    ) -> MultiLayerPatchingResult:
        """
        Run complete patching experiment.
        
        Args:
            source_prompt: Source context (e.g., TB scenario)
            target_prompt: Target context (e.g., FB scenario)
            layers: Layers to patch simultaneously
            early_steps: Steps to patch (None = 0 to max_early_steps)
            max_early_steps: Max step if early_steps not provided
            
        Returns:
            MultiLayerPatchingResult
        """
        # Cache source activations at prompt end
        print("    Caching source activations...", flush=True)
        sys.stdout.flush()
        source_acts = self.cache_residual_stream_multi_layer(
            source_prompt, 
            layers,
            position=None  # Cache last position (prompt end)
        )
        print(f"    Cached activations at {len(source_acts)} layers", flush=True)
        sys.stdout.flush()
        
        # Patch during target generation
        result = self.patch_early(
            target_prompt,
            source_activations=source_acts,
            layers=layers,
            early_steps=early_steps,
            max_early_steps=max_early_steps
        )
        
        result.source_context = source_prompt
        return result

