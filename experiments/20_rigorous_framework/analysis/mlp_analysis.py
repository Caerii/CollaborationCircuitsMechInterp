"""
MLP Neuron Analysis

Investigate which MLP neurons activate differently between conditions.
MLPs are crucial - they often store factual knowledge and can override
attention-based reasoning.

Key components in Qwen/Llama style MLPs:
- gate_proj: Gating mechanism
- up_proj: Expands to intermediate dim
- down_proj: Projects back to hidden dim
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class NeuronDiff:
    """Neuron difference between conditions."""
    neuron_idx: int
    diff: float
    value_a: float
    value_b: float


@dataclass
class MLPLayerAnalysis:
    """Analysis of MLP at one layer."""
    layer: int
    gate_top_neurons: List[NeuronDiff]
    down_top_neurons: List[NeuronDiff]
    gate_total_diff: float
    down_total_diff: float


class MLPAnalyzer:
    """
    Analyze MLP neuron activations.
    
    Find which neurons activate differently between conditions,
    which is critical for understanding how knowledge is stored.
    
    Example:
        analyzer = MLPAnalyzer(model, tokenizer)
        
        results = analyzer.compare_conditions(
            condition_a_prompts=["Alice told Bob...", "Carol told Dan..."],
            condition_b_prompts=["Alice announced to Bob...", "Carol announced to Dan..."],
            layers=[32, 33, 34, 35]
        )
        
        for layer_result in results:
            print(f"Layer {layer_result.layer}:")
            print(f"  Top differing gate neuron: {layer_result.gate_top_neurons[0]}")
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = model.config.num_hidden_layers
    
    def get_mlp_activations(
        self,
        prompt: str,
        layers: List[int],
    ) -> Dict[str, torch.Tensor]:
        """
        Get MLP intermediate activations.
        
        Args:
            prompt: Input prompt
            layers: Layers to analyze
            
        Returns:
            Dict mapping "L{layer}_gate", "L{layer}_down" to activations
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        activations = {}
        hooks = []
        
        def make_hook(layer_idx, name):
            def hook(module, input, output):
                activations[f"L{layer_idx}_{name}"] = output[0, -1, :].clone()
            return hook
        
        for layer_idx in layers:
            layer = self.model.model.layers[layer_idx]
            hooks.append(layer.mlp.gate_proj.register_forward_hook(make_hook(layer_idx, "gate")))
            hooks.append(layer.mlp.down_proj.register_forward_hook(make_hook(layer_idx, "down")))
        
        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            for h in hooks:
                h.remove()
        
        return activations
    
    def compare_conditions(
        self,
        condition_a_prompts: List[str],
        condition_b_prompts: List[str],
        layers: List[int],
        top_k: int = 10,
    ) -> List[MLPLayerAnalysis]:
        """
        Compare MLP activations between two conditions.
        
        Args:
            condition_a_prompts: Prompts for condition A
            condition_b_prompts: Prompts for condition B
            layers: Layers to analyze
            top_k: Number of top differing neurons to return
            
        Returns:
            List of MLPLayerAnalysis for each layer
        """
        # Collect activations
        a_activations = [self.get_mlp_activations(p, layers) for p in condition_a_prompts]
        b_activations = [self.get_mlp_activations(p, layers) for p in condition_b_prompts]
        
        results = []
        
        for layer in layers:
            # Average activations across prompts
            a_gate = torch.stack([a[f"L{layer}_gate"] for a in a_activations]).mean(0)
            b_gate = torch.stack([b[f"L{layer}_gate"] for b in b_activations]).mean(0)
            
            a_down = torch.stack([a[f"L{layer}_down"] for a in a_activations]).mean(0)
            b_down = torch.stack([b[f"L{layer}_down"] for b in b_activations]).mean(0)
            
            # Compute differences
            gate_diff = (a_gate - b_gate).abs()
            down_diff = (a_down - b_down).abs()
            
            # Get top neurons
            gate_top = torch.topk(gate_diff, k=min(top_k, len(gate_diff)))
            down_top = torch.topk(down_diff, k=min(top_k, len(down_diff)))
            
            gate_neurons = [
                NeuronDiff(
                    neuron_idx=int(idx),
                    diff=float(val),
                    value_a=float(a_gate[idx]),
                    value_b=float(b_gate[idx]),
                )
                for idx, val in zip(gate_top.indices.tolist(), gate_top.values.tolist())
            ]
            
            down_neurons = [
                NeuronDiff(
                    neuron_idx=int(idx),
                    diff=float(val),
                    value_a=float(a_down[idx]),
                    value_b=float(b_down[idx]),
                )
                for idx, val in zip(down_top.indices.tolist(), down_top.values.tolist())
            ]
            
            results.append(MLPLayerAnalysis(
                layer=layer,
                gate_top_neurons=gate_neurons,
                down_top_neurons=down_neurons,
                gate_total_diff=float(gate_diff.sum()),
                down_total_diff=float(down_diff.sum()),
            ))
        
        return results
    
    def ablate_neurons(
        self,
        prompt: str,
        layer: int,
        neuron_indices: List[int],
        ablate_gate: bool = True,
    ) -> Tuple[str, str]:
        """
        Test ablating specific MLP neurons.
        
        Args:
            prompt: Input prompt
            layer: Layer to ablate
            neuron_indices: Neurons to zero out
            ablate_gate: If True, ablate gate_proj; else down_proj
            
        Returns:
            Tuple of (original_output, ablated_output)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Get original output
        with torch.no_grad():
            orig_out = self.model.generate(
                **inputs, max_new_tokens=5, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        orig_text = self.tokenizer.decode(orig_out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Setup ablation hook
        def ablate_hook(module, input, output):
            out = output.clone()
            for idx in neuron_indices:
                out[0, :, idx] = 0  # Zero out neuron across all positions
            return out
        
        target = self.model.model.layers[layer].mlp.gate_proj if ablate_gate else self.model.model.layers[layer].mlp.down_proj
        handle = target.register_forward_hook(ablate_hook)
        
        try:
            with torch.no_grad():
                abl_out = self.model.generate(
                    **inputs, max_new_tokens=5, do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            abl_text = self.tokenizer.decode(abl_out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        finally:
            handle.remove()
        
        return orig_text.strip(), abl_text.strip()


class AttentionOutputAnalyzer:
    """
    Analyze what attention heads OUTPUT to the residual stream.
    
    The key insight is that attention heads have both WHERE they attend
    and WHAT they output. Two heads can attend identically but output
    very different things.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_heads = model.config.num_attention_heads
        self.hidden_size = model.config.hidden_size
        self.head_dim = self.hidden_size // self.n_heads
    
    def get_head_outputs(
        self,
        prompt: str,
        layers: List[int],
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Get per-head outputs (before combination).
        
        Args:
            prompt: Input prompt
            layers: Layers to analyze
            
        Returns:
            Dict mapping (layer, head) to output vector
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        captured = {}
        hooks = []
        
        def make_hook(layer_idx):
            def hook(module, args, output):
                # args[0] is input to o_proj - (batch, seq, hidden_dim)
                captured[layer_idx] = args[0].detach().cpu()
            return hook
        
        for layer_idx in layers:
            o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
            hook = o_proj.register_forward_hook(make_hook(layer_idx))
            hooks.append(hook)
        
        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            for h in hooks:
                h.remove()
        
        # Extract per-head outputs
        results = {}
        for layer_idx, full_output in captured.items():
            # Reshape: (batch, seq, hidden) -> (batch, seq, n_heads, head_dim)
            reshaped = full_output.view(1, -1, self.n_heads, self.head_dim)
            
            for head_idx in range(self.n_heads):
                head_out = reshaped[0, -1, head_idx, :].numpy()  # Last token
                results[(layer_idx, head_idx)] = head_out
        
        return results
    
    def compare_head_outputs(
        self,
        prompt_a: str,
        prompt_b: str,
        layers: List[int],
        top_k: int = 10,
    ) -> List[Tuple[int, int, float]]:
        """
        Find heads with most different outputs between prompts.
        
        Args:
            prompt_a: First prompt
            prompt_b: Second prompt
            layers: Layers to analyze
            top_k: Number of top heads to return
            
        Returns:
            List of (layer, head, L2_diff) sorted by diff
        """
        outputs_a = self.get_head_outputs(prompt_a, layers)
        outputs_b = self.get_head_outputs(prompt_b, layers)
        
        diffs = []
        for (layer, head), out_a in outputs_a.items():
            out_b = outputs_b[(layer, head)]
            l2_diff = float(np.linalg.norm(out_a - out_b))
            diffs.append((layer, head, l2_diff))
        
        diffs.sort(key=lambda x: x[2], reverse=True)
        return diffs[:top_k]

