"""
Logit Lens Analysis

Apply the final unembedding to intermediate hidden states to track
where the model's prediction emerges and crystallizes.

This is crucial for understanding WHEN decisions are made.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LogitLensResult:
    """Result of logit lens analysis."""
    layers: List[str]
    target_logits: List[float]
    contrast_logits: List[float]
    diffs: List[float]
    predictions: List[str]  # "target" or "contrast"
    decision_layer: str
    decision_layer_idx: int


class LogitLens:
    """
    Logit Lens: Track predictions through layers.
    
    Apply the final layer norm + lm_head to intermediate hidden states
    to see when the model's prediction emerges.
    
    Example:
        lens = LogitLens(model, tokenizer)
        
        result = lens.analyze(
            prompt="Alice put the ball in the drawer. Alice leaves. Bob moves it to basket. Alice returns. Alice looks in the",
            target_token=" drawer",
            contrast_token=" basket"
        )
        
        print(f"Decision crystallizes at {result.decision_layer}")
        print(f"Target vs contrast logit diff: {result.diffs}")
    """
    
    def __init__(self, model, tokenizer):
        """
        Initialize logit lens.
        
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = model.config.num_hidden_layers
    
    def analyze(
        self,
        prompt: str,
        target_token: str,
        contrast_token: str,
    ) -> LogitLensResult:
        """
        Track target vs contrast token logits through layers.
        
        Args:
            prompt: Input prompt
            target_token: Token we expect (e.g., " drawer")
            contrast_token: Contrast token (e.g., " basket")
            
        Returns:
            LogitLensResult with layer-by-layer predictions
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Get token IDs
        target_id = self.tokenizer.encode(target_token, add_special_tokens=False)[0]
        contrast_id = self.tokenizer.encode(contrast_token, add_special_tokens=False)[0]
        
        # Capture hidden states
        hidden_states = []
        
        def capture_hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            hidden_states.append(h.detach())
        
        # Register hooks
        hooks = []
        hooks.append(self.model.model.embed_tokens.register_forward_hook(capture_hook))
        for layer in self.model.model.layers:
            hooks.append(layer.register_forward_hook(capture_hook))
        
        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            for h in hooks:
                h.remove()
        
        # Apply lm_head to each layer's hidden state
        layers = []
        target_logits = []
        contrast_logits = []
        diffs = []
        predictions = []
        
        for i, hidden in enumerate(hidden_states):
            # Apply final layer norm and lm_head
            normed = self.model.model.norm(hidden)
            logits = self.model.lm_head(normed)[0, -1]  # Last token
            
            t_logit = float(logits[target_id])
            c_logit = float(logits[contrast_id])
            diff = t_logit - c_logit
            
            layer_name = "embed" if i == 0 else f"L{i-1}"
            
            layers.append(layer_name)
            target_logits.append(t_logit)
            contrast_logits.append(c_logit)
            diffs.append(diff)
            predictions.append("target" if diff > 0 else "contrast")
        
        # Find decision layer (first stable prediction)
        final_pred = predictions[-1]
        decision_idx = 0
        for i in range(len(predictions) - 1, -1, -1):
            if predictions[i] != final_pred:
                decision_idx = i + 1
                break
        
        return LogitLensResult(
            layers=layers,
            target_logits=target_logits,
            contrast_logits=contrast_logits,
            diffs=diffs,
            predictions=predictions,
            decision_layer=layers[decision_idx],
            decision_layer_idx=decision_idx,
        )
    
    def compare_prompts(
        self,
        prompts: Dict[str, str],
        target_token: str,
        contrast_token: str,
    ) -> Dict[str, LogitLensResult]:
        """
        Compare logit lens results across multiple prompts.
        
        Args:
            prompts: Dict mapping name to prompt
            target_token: Expected token
            contrast_token: Contrast token
            
        Returns:
            Dict mapping name to LogitLensResult
        """
        results = {}
        for name, prompt in prompts.items():
            results[name] = self.analyze(prompt, target_token, contrast_token)
        return results
    
    def find_divergence_point(
        self,
        result_a: LogitLensResult,
        result_b: LogitLensResult,
        threshold: float = 1.0,
    ) -> Optional[int]:
        """
        Find where two prompts start to diverge in prediction.
        
        Args:
            result_a: First logit lens result
            result_b: Second logit lens result
            threshold: Minimum diff in diffs to count as divergence
            
        Returns:
            Layer index where divergence starts, or None
        """
        for i in range(min(len(result_a.diffs), len(result_b.diffs))):
            diff_of_diffs = abs(result_a.diffs[i] - result_b.diffs[i])
            if diff_of_diffs > threshold:
                return i
        return None


def plot_logit_lens(results: Dict[str, LogitLensResult], save_path: Optional[str] = None):
    """
    Plot logit lens results.
    
    Args:
        results: Dict mapping name to LogitLensResult
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return
    
    n_results = len(results)
    fig, axes = plt.subplots(n_results, 1, figsize=(12, 4 * n_results))
    if n_results == 1:
        axes = [axes]
    
    for ax, (name, result) in zip(axes, results.items()):
        x = range(len(result.layers))
        
        ax.plot(x, result.diffs, 'b-', linewidth=2)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=result.decision_layer_idx, color='green', linestyle='--', 
                   label=f'Decision: {result.decision_layer}')
        
        # Color background by prediction
        for i in range(len(result.predictions) - 1):
            color = '#90EE90' if result.predictions[i] == 'target' else '#FFB6C1'
            ax.axvspan(i, i+1, alpha=0.2, color=color)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Target - Contrast Logit')
        ax.set_title(f'{name}: Decision at {result.decision_layer}')
        ax.legend()
        ax.set_xticks(x[::5])
        ax.set_xticklabels([result.layers[i] for i in x[::5]], rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig

