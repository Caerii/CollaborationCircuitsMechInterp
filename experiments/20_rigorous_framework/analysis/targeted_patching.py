"""
Targeted Activation Patching for Chat Mode

This addresses the fundamental issue: we need to patch at the RIGHT TIME
(when answering, not during reasoning) and potentially at MULTIPLE layers
(since ToM is distributed).

Strategy:
1. Use logit lens to find WHERE in generation the decision crystallizes
2. Patch residual stream at that specific token position
3. Optionally patch multiple layers simultaneously
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class TargetedPatchingResult:
    """Result from targeted patching experiment."""
    source_context: str
    target_context: str
    layers_patched: List[int]
    answer_token_position: int
    base_response: str
    patched_response: str
    flipped: bool
    logit_diff_before: float
    logit_diff_after: float


class TargetedPatcher:
    """
    Targeted activation patching that patches at the answer position only.
    
    Key insight: In chat mode, the model generates reasoning first, then answers.
    We should patch ONLY when generating the answer token, not during reasoning.
    
    Example:
        patcher = TargetedPatcher(model, tokenizer, chat_mode=True)
        
        result = patcher.patch_at_answer_position(
            source_prompt="Sally saw the ball move to the box...",
            target_prompt="Sally didn't see the ball move...",
            layers=[20, 24, 28],  # Multiple layers for distributed circuit
            answer_tokens=["basket", "box"]
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
        max_new_tokens: int = 200,
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
    
    def find_answer_position(
        self,
        prompt: str,
        answer_tokens: List[str],
        max_tokens: int = 200
    ) -> Optional[int]:
        """
        Find the token position where the answer is generated.
        
        Uses logit lens during generation to track when answer probability
        diverges from reasoning tokens.
        
        Returns:
            Token position (relative to prompt start) where answer is generated,
            or None if not found
        """
        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        prompt_len = inputs.input_ids.shape[1]
        
        # Get answer token IDs
        answer_ids = []
        for tok in answer_tokens:
            ids = self.tokenizer.encode(tok, add_special_tokens=False)
            if ids:
                answer_ids.append(ids[0])
        
        if not answer_ids:
            return None
        
        # Track logits during generation
        answer_probs = []  # Track probability of answer tokens at each step
        generated_tokens = []
        
        def logit_hook(module, input, output):
            """Capture logits at each generation step."""
            hidden = output[0] if isinstance(output, tuple) else output
            # Apply layer norm and lm_head
            normed = self.model.model.norm(hidden)
            logits = self.model.lm_head(normed)
            
            # Get probabilities for answer tokens
            probs = torch.softmax(logits[0, -1], dim=-1)
            answer_prob = sum(probs[tok_id].item() for tok_id in answer_ids)
            answer_probs.append(answer_prob)
        
        # Register hook on final layer
        hook = self.model.model.layers[-1].register_forward_hook(logit_hook)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            
            # Decode to find where </think> ends
            full_ids = outputs.sequences[0]
            tokens = [self.tokenizer.decode([t]) for t in full_ids]
            
            # Find answer position (after </think>)
            think_end_pos = None
            for i, tok in enumerate(tokens):
                if "</think>" in tok or "</think>" in tok.lower():
                    think_end_pos = i
                    break
            
            # Find where answer probability spikes
            if think_end_pos and len(answer_probs) > think_end_pos - prompt_len:
                # Look for spike in answer probability after reasoning
                reasoning_end_idx = think_end_pos - prompt_len
                if reasoning_end_idx < len(answer_probs):
                    # Find first position after reasoning where answer prob > threshold
                    threshold = 0.1
                    for i in range(reasoning_end_idx, min(reasoning_end_idx + 20, len(answer_probs))):
                        if answer_probs[i] > threshold:
                            return prompt_len + i
            
            # Fallback: return position after reasoning
            if think_end_pos:
                return think_end_pos + 1
            
        finally:
            hook.remove()
        
        return None
    
    def cache_residual_stream(
        self,
        prompt: str,
        layers: List[int],
        position: Optional[int] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Cache residual stream activations at specific layers.
        
        Args:
            prompt: Source prompt
            layers: Layers to cache
            position: Specific token position to cache (None = last position)
            
        Returns:
            Dict mapping layer -> activation tensor at that position
        """
        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        
        cached = {}
        hooks = []
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if position is None:
                    # Cache last position
                    cached[layer_idx] = hidden[0, -1, :].detach().clone()
                else:
                    # Cache specific position
                    if position < hidden.shape[1]:
                        cached[layer_idx] = hidden[0, position, :].detach().clone()
            return hook
        
        for layer in layers:
            h = self.model.model.layers[layer].register_forward_hook(make_hook(layer))
            hooks.append(h)
        
        with torch.no_grad():
            _ = self.model(**inputs)
        
        for h in hooks:
            h.remove()
        
        return cached
    
    def patch_at_answer_position(
        self,
        target_prompt: str,
        source_activations: Dict[int, torch.Tensor],
        layers: List[int],
        answer_position: int,
        answer_tokens: List[str]
    ) -> Tuple[str, Dict]:
        """
        Patch residual stream at the answer position only.
        
        This is the key innovation: we patch ONLY when generating the answer,
        not during the reasoning phase.
        
        Args:
            target_prompt: Target context
            source_activations: Cached activations from source (at answer position)
            layers: Layers to patch
            answer_position: Token position where answer is generated
            answer_tokens: Answer tokens to track
            
        Returns:
            (generated_text, metadata_dict)
        """
        formatted = self._format_prompt(target_prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        prompt_len = inputs.input_ids.shape[1]
        
        # Convert answer_position to generation step
        # answer_position is absolute, we need relative to prompt
        if answer_position < prompt_len:
            # Answer position is in prompt (shouldn't happen, but handle it)
            gen_step = 0
        else:
            gen_step = answer_position - prompt_len
        
        hooks = []
        patched = [False]  # Use list for closure
        step_count = [0]  # Track generation steps
        
        # Track logits before/after patching
        logit_tracker = {"before": None, "after": None}
        
        answer_ids = []
        for t in answer_tokens:
            ids = self.tokenizer.encode(t, add_special_tokens=False)
            if ids:
                answer_ids.append(ids[0])
        
        def make_patch_hook(layer_idx):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                current_step = step_count[0]
                
                # Only patch at the answer generation step
                if current_step == gen_step and layer_idx in source_activations:
                    # Get logits before patching (first layer only, to avoid double-counting)
                    if layer_idx == layers[0] and logit_tracker["before"] is None:
                        normed = self.model.model.norm(hidden)
                        logits = self.model.lm_head(normed)
                        if answer_ids:
                            logit_tracker["before"] = logits[0, -1, answer_ids[0]].item()
                    
                    # Patch: replace hidden state at last position
                    hidden[0, -1, :] = source_activations[layer_idx].clone()
                    
                    # Get logits after patching (last layer only)
                    if layer_idx == layers[-1]:
                        normed = self.model.model.norm(hidden)
                        logits = self.model.lm_head(normed)
                        if answer_ids:
                            logit_tracker["after"] = logits[0, -1, answer_ids[0]].item()
                    
                    patched[0] = True
                
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden
            return hook
        
        for layer in layers:
            if layer in source_activations:
                h = self.model.model.layers[layer].register_forward_hook(make_patch_hook(layer))
                hooks.append(h)
        
        try:
            # Custom generation loop to track steps
            generated_ids = inputs.input_ids.clone()
            
            with torch.no_grad():
                for _ in range(self.max_new_tokens):
                    outputs = self.model(input_ids=generated_ids)
                    logits = outputs.logits
                    
                    # Get next token
                    next_token = logits[0, -1].argmax(dim=-1).unsqueeze(0)
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    
                    step_count[0] += 1
                    
                    # Stop if we hit EOS
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                    
                    # Stop if we've patched and generated enough tokens
                    if patched and step_count[0] > gen_step + 10:
                        break
        finally:
            for h in hooks:
                h.remove()
        
        response = self.tokenizer.decode(
            generated_ids[0][prompt_len:],
            skip_special_tokens=True
        )
        
        metadata = {
            "patched": patched[0],
            "gen_step": gen_step,
            "logit_before": logit_tracker["before"],
            "logit_after": logit_tracker["after"],
        }
        
        return response, metadata
    
    def run_targeted_experiment(
        self,
        source_prompt: str,
        target_prompt: str,
        layers: List[int],
        answer_tokens: List[str],
        check_flip_fn: Callable[[str, str], bool]
    ) -> TargetedPatchingResult:
        """
        Run complete targeted patching experiment.
        
        Args:
            source_prompt: Context to patch FROM
            target_prompt: Context to patch INTO
            layers: Layers to patch (can be multiple for distributed circuit
            answer_tokens: Tokens that represent the answer (e.g., ["basket", "box"])
            check_flip_fn: Function(base, patched) -> bool
            
        Returns:
            TargetedPatchingResult
        """
        # Get baseline
        formatted_target = self._format_prompt(target_prompt)
        inputs = self.tokenizer(formatted_target, return_tensors="pt").to(self.model.device)
        prompt_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        base_response = self.tokenizer.decode(
            outputs[0][prompt_len:],
            skip_special_tokens=True
        )
        
        # Find answer position in target
        answer_pos = self.find_answer_position(target_prompt, answer_tokens)
        if answer_pos is None:
            # Fallback: use last position
            formatted_source = self._format_prompt(source_prompt)
            source_inputs = self.tokenizer(formatted_source, return_tensors="pt")
            answer_pos = source_inputs.input_ids.shape[1] - 1
        
        # Cache source activations at answer position
        source_acts = self.cache_residual_stream(source_prompt, layers, position=answer_pos)
        
        # Patch at answer position
        patched_response, metadata = self.patch_at_answer_position(
            target_prompt, source_acts, layers, answer_pos, answer_tokens
        )
        
        # Check if flipped
        flipped = check_flip_fn(base_response, patched_response)
        
        return TargetedPatchingResult(
            source_context=source_prompt,
            target_context=target_prompt,
            layers_patched=layers,
            answer_token_position=answer_pos,
            base_response=base_response,
            patched_response=patched_response,
            flipped=flipped,
            logit_diff_before=metadata.get("logit_before", 0.0) or 0.0,
            logit_diff_after=metadata.get("logit_after", 0.0) or 0.0,
        )

