"""
Causal Steering for Representation Validation

Ported from experiment 8's causal steering implementation.
Tests whether a learned representation is FUNCTIONAL (causally affects behavior)
vs merely CORRELATIONAL (present but not used).
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SteeringResult:
    """Result from a steering test."""
    prompt: str
    base_completion: str
    steered_completion: str
    steering_strength: float
    changed: bool
    expected_change: str
    
    def to_dict(self) -> Dict:
        return {
            "prompt": self.prompt[:100] + "..." if len(self.prompt) > 100 else self.prompt,
            "base_completion": self.base_completion[:100],
            "steered_completion": self.steered_completion[:100],
            "steering_strength": self.steering_strength,
            "changed": self.changed,
            "expected_change": self.expected_change,
        }


class CausalSteering:
    """
    Test causal effects of steering model activations.
    
    Key insight from experiment 8: If we can train a probe to classify
    some feature, we can use the probe's weights as a steering vector.
    If steering changes model behavior, the representation is FUNCTIONAL.
    
    Example:
        steering = CausalSteering(model, tokenizer)
        
        # Get steering direction from probe
        direction = probing_pipeline.extract_steering_direction(X, y)
        steering.set_direction(direction, layer=16)
        
        # Test if steering affects behavior
        results = steering.test_effect(test_prompts, strengths=[1.0, 2.0, 3.0])
        
        if results["change_rate"] > 0.5:
            print("Representation is FUNCTIONAL!")
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens: int = 50
    ):
        """
        Initialize steering.
        
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            max_new_tokens: Tokens to generate for testing
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        
        self.steering_direction = None
        self.steering_layer = None
        self.hook_handle = None
    
    def set_direction(
        self,
        direction: np.ndarray,
        layer: int
    ):
        """
        Set the steering direction and layer.
        
        Args:
            direction: Steering direction vector (normalized)
            layer: Layer to apply steering
        """
        self.steering_direction = torch.tensor(direction, dtype=self.model.dtype)
        self.steering_layer = layer
    
    def _create_steering_hook(self, strength: float) -> Callable:
        """Create a forward hook for steering."""
        direction = self.steering_direction.to(self.model.device)
        
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Add steering direction to all positions
            hidden = hidden + strength * direction
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        
        return hook
    
    def generate_with_steering(
        self,
        prompt: str,
        strength: float = 0.0
    ) -> str:
        """
        Generate text with optional steering.
        
        Args:
            prompt: Input prompt
            strength: Steering strength (0 = no steering)
            
        Returns:
            Generated completion
        """
        if self.steering_direction is None:
            raise ValueError("Set steering direction first with set_direction()")
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        # Install hook if steering
        hook_handle = None
        if strength != 0.0:
            hook_handle = self.model.model.layers[self.steering_layer].register_forward_hook(
                self._create_steering_hook(strength)
            )
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            if hook_handle:
                hook_handle.remove()
        
        # Decode completion only
        completion = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return completion.strip()
    
    def test_single_prompt(
        self,
        prompt: str,
        expected_change: str,
        strength: float = 3.0
    ) -> SteeringResult:
        """
        Test steering effect on a single prompt.
        
        Args:
            prompt: Test prompt
            expected_change: What change we expect from steering
            strength: Steering strength
            
        Returns:
            SteeringResult
        """
        # Generate without steering
        base_completion = self.generate_with_steering(prompt, strength=0.0)
        
        # Generate with steering
        steered_completion = self.generate_with_steering(prompt, strength=strength)
        
        # Check if changed
        changed = base_completion.strip() != steered_completion.strip()
        
        return SteeringResult(
            prompt=prompt,
            base_completion=base_completion,
            steered_completion=steered_completion,
            steering_strength=strength,
            changed=changed,
            expected_change=expected_change,
        )
    
    def test_effect(
        self,
        test_prompts: List[Dict],
        strengths: List[float] = [1.0, 2.0, 3.0],
        verbose: bool = False
    ) -> Dict:
        """
        Test steering effect across multiple prompts and strengths.
        
        Args:
            test_prompts: List of {"prompt": ..., "expected_change": ...}
            strengths: Steering strengths to test
            verbose: Print progress
            
        Returns:
            Results summary
        """
        all_results = []
        change_counts = {s: 0 for s in strengths}
        
        for i, prompt_data in enumerate(test_prompts):
            prompt = prompt_data["prompt"]
            expected = prompt_data.get("expected_change", "")
            
            for strength in strengths:
                result = self.test_single_prompt(prompt, expected, strength)
                all_results.append(result)
                
                if result.changed:
                    change_counts[strength] += 1
                
                if verbose:
                    status = "CHANGED" if result.changed else "unchanged"
                    print(f"  Prompt {i+1}, strength={strength}: {status}")
        
        n_prompts = len(test_prompts)
        
        return {
            "n_prompts": n_prompts,
            "n_strengths": len(strengths),
            "change_counts": change_counts,
            "change_rates": {s: c / n_prompts for s, c in change_counts.items()},
            "overall_change_rate": sum(change_counts.values()) / (n_prompts * len(strengths)),
            "results": [r.to_dict() for r in all_results],
        }
    
    def validate_functional_representation(
        self,
        test_prompts: List[Dict],
        min_change_rate: float = 0.3
    ) -> Dict:
        """
        Validate that a representation is functional (causal).
        
        Args:
            test_prompts: Test prompts
            min_change_rate: Minimum change rate to consider functional
            
        Returns:
            Validation result
        """
        results = self.test_effect(test_prompts, strengths=[3.0])
        change_rate = results["change_rates"][3.0]
        
        is_functional = change_rate >= min_change_rate
        
        return {
            "is_functional": is_functional,
            "change_rate": change_rate,
            "threshold": min_change_rate,
            "interpretation": (
                f"Representation IS functional (causal effect on {change_rate:.1%} of prompts)"
                if is_functional else
                f"Representation is NOT functional (only {change_rate:.1%} change rate, need {min_change_rate:.0%})"
            ),
            "details": results,
        }


def create_steering_test_prompts() -> List[Dict]:
    """Create standard test prompts for steering validation."""
    return [
        {
            "prompt": "Alice knows the secret code is 7492. Bob does not know the code. If you ask Alice what the code is, she will say:",
            "expected_change": "Response might attribute knowledge differently",
        },
        {
            "prompt": "Alice knows the password. Bob doesn't know it. Who should you ask for the password?",
            "expected_change": "Might switch from Alice to Bob",
        },
        {
            "prompt": "Only Alice knows where the key is hidden. If Bob looks for the key, he will:",
            "expected_change": "Might change Bob's expected success",
        },
        {
            "prompt": "Alice discovered that the meeting is canceled. Bob hasn't heard yet. Does Bob know the meeting is canceled?",
            "expected_change": "Might flip Yes/No",
        },
        {
            "prompt": "The treasure location is known only to Alice. Bob is searching for it. Bob's chance of finding it is:",
            "expected_change": "Might change probability assessment",
        },
    ]

