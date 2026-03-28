"""
SAE (Sparse Autoencoder) Analysis Module

This module provides tools for training and using SAEs to decompose
model activations into interpretable sparse features.

SAEs help us go from:
  "Layer 12 encodes belief state" (2560 dims)
to:
  "Feature #4723 fires on 'agent has outdated belief'" (interpretable!)

Techniques:
1. Standard SAE: encode residual stream activations
2. Gated SAE: separate feature selection from magnitude estimation  
3. Transcoder: directly map MLP inputs to outputs via sparse features

Key Libraries:
- SAELens: pip install sae-lens (best for pre-trained SAEs)
- dictionary_learning: for custom training
- We provide a simple implementation here for flexibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SAEConfig:
    """Configuration for SAE training."""
    d_model: int  # Model hidden dimension (e.g., 2560 for Qwen3-4B)
    d_sae: int  # SAE dictionary size (typically 4-16x d_model)
    l1_coeff: float = 5e-3  # Sparsity penalty
    lr: float = 1e-4
    dtype: torch.dtype = torch.float32


class SimpleSAE(nn.Module):
    """
    Simple Sparse Autoencoder implementation.
    
    Architecture:
        Encoder: x -> ReLU(W_enc @ (x - b_dec) + b_enc)
        Decoder: f -> W_dec @ f + b_dec
    
    Loss: MSE(x, x_hat) + l1_coeff * L1(f)
    """
    
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        
        # Encoder
        self.W_enc = nn.Parameter(torch.randn(config.d_sae, config.d_model) / np.sqrt(config.d_model))
        self.b_enc = nn.Parameter(torch.zeros(config.d_sae))
        
        # Decoder
        self.W_dec = nn.Parameter(torch.randn(config.d_model, config.d_sae) / np.sqrt(config.d_sae))
        self.b_dec = nn.Parameter(torch.zeros(config.d_model))
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to sparse features."""
        x_centered = x - self.b_dec
        pre_acts = x_centered @ self.W_enc.T + self.b_enc
        return F.relu(pre_acts)
    
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to activations."""
        return f @ self.W_dec.T + self.b_dec
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            x_hat: Reconstructed activations
            f: Sparse feature activations
            loss: Total loss (MSE + L1)
        """
        f = self.encode(x)
        x_hat = self.decode(f)
        
        # MSE reconstruction loss
        mse_loss = F.mse_loss(x_hat, x)
        
        # L1 sparsity loss
        l1_loss = self.config.l1_coeff * f.abs().mean()
        
        total_loss = mse_loss + l1_loss
        
        return x_hat, f, total_loss
    
    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get sparse feature activations for input."""
        return self.encode(x)
    
    def get_top_features(self, x: torch.Tensor, k: int = 10) -> List[Tuple[int, float]]:
        """Get top-k activated features for input."""
        f = self.encode(x)
        if f.dim() > 1:
            f = f.mean(dim=0)  # Average over batch/sequence
        
        values, indices = f.topk(k)
        return [(int(idx), float(val)) for idx, val in zip(indices, values)]


class GatedSAE(nn.Module):
    """
    Gated Sparse Autoencoder.
    
    Addresses shrinkage issue by separating:
    1. Which features to use (gating)
    2. How much to use them (magnitude)
    
    Reference: arxiv.org/abs/2404.16014
    """
    
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        
        # Gating pathway
        self.W_gate = nn.Parameter(torch.randn(config.d_sae, config.d_model) / np.sqrt(config.d_model))
        self.b_gate = nn.Parameter(torch.zeros(config.d_sae))
        
        # Magnitude pathway
        self.W_mag = nn.Parameter(torch.randn(config.d_sae, config.d_model) / np.sqrt(config.d_model))
        self.b_mag = nn.Parameter(torch.zeros(config.d_sae))
        self.r_mag = nn.Parameter(torch.ones(config.d_sae))  # Rescale
        
        # Decoder
        self.W_dec = nn.Parameter(torch.randn(config.d_model, config.d_sae) / np.sqrt(config.d_sae))
        self.b_dec = nn.Parameter(torch.zeros(config.d_model))
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with gating."""
        x_centered = x - self.b_dec
        
        # Gating: which features?
        gate_pre = x_centered @ self.W_gate.T + self.b_gate
        gate = (gate_pre > 0).float()  # Binary gate
        
        # Magnitude: how much?
        mag_pre = x_centered @ self.W_mag.T + self.b_mag
        mag = F.relu(mag_pre) * self.r_mag
        
        return gate * mag
    
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Decode sparse features."""
        return f @ self.W_dec.T + self.b_dec
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with gated encoding."""
        f = self.encode(x)
        x_hat = self.decode(f)
        
        mse_loss = F.mse_loss(x_hat, x)
        l1_loss = self.config.l1_coeff * f.abs().mean()
        
        # Additional loss: encourage gate sparsity
        x_centered = x - self.b_dec
        gate_pre = x_centered @ self.W_gate.T + self.b_gate
        gate_loss = 0.1 * F.relu(gate_pre).mean()
        
        total_loss = mse_loss + l1_loss + gate_loss
        
        return x_hat, f, total_loss


class Transcoder(nn.Module):
    """
    Transcoder: directly map MLP inputs to outputs via sparse features.
    
    Unlike SAE (which encodes/decodes the same activation),
    Transcoder maps input_activation -> output_activation.
    
    Use case: "What computation did MLP layer X perform?"
    
    Architecture:
        MLP_input -> Encoder -> Sparse Features -> Decoder -> MLP_output
    """
    
    def __init__(self, d_in: int, d_out: int, d_sparse: int, l1_coeff: float = 5e-3):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_sparse = d_sparse
        self.l1_coeff = l1_coeff
        
        # Encoder: input -> sparse
        # Use smaller initialization and positive bias to help ReLU
        self.W_enc = nn.Parameter(torch.randn(d_sparse, d_in) * 0.01)
        self.b_enc = nn.Parameter(torch.ones(d_sparse) * 0.1)  # Positive bias!
        
        # Decoder: sparse -> output
        self.W_dec = nn.Parameter(torch.randn(d_out, d_sparse) * 0.01)
        self.b_dec = nn.Parameter(torch.zeros(d_out))
    
    def forward(self, x_in: torch.Tensor, x_out_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x_in: MLP input activations
            x_out_target: Target MLP output activations
            
        Returns:
            x_out_pred: Predicted output
            f: Sparse features
            loss: Total loss
        """
        # Encode with ReLU for sparsity
        pre_act = x_in @ self.W_enc.T + self.b_enc
        f = F.relu(pre_act)
        
        # Decode to output space
        x_out_pred = f @ self.W_dec.T + self.b_dec
        
        # Losses
        mse_loss = F.mse_loss(x_out_pred, x_out_target)
        l1_loss = self.l1_coeff * f.abs().mean()
        
        return x_out_pred, f, mse_loss + l1_loss
    
    def get_features(self, x_in: torch.Tensor) -> torch.Tensor:
        """Get sparse features for input."""
        pre_act = x_in @ self.W_enc.T + self.b_enc
        return F.relu(pre_act)


class SAETrainer:
    """
    Trainer for SAEs/Transcoders.
    
    Usage:
        trainer = SAETrainer(sae, lr=1e-4)
        for batch in data:
            loss = trainer.step(batch)
    """
    
    def __init__(self, model: nn.Module, lr: float = 1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.losses = []
    
    def step(self, x: torch.Tensor, x_out: Optional[torch.Tensor] = None) -> float:
        """Single training step."""
        self.optimizer.zero_grad()
        
        if isinstance(self.model, Transcoder):
            _, _, loss = self.model(x, x_out)
        else:
            _, _, loss = self.model(x)
        
        loss.backward()
        self.optimizer.step()
        
        # Normalize decoder weights (important for SAE stability)
        if hasattr(self.model, 'W_dec'):
            with torch.no_grad():
                self.model.W_dec.data = F.normalize(self.model.W_dec.data, dim=0)
        
        self.losses.append(float(loss))
        return float(loss)


def collect_activations(
    model,
    tokenizer,
    prompts: List[str],
    layer: int,
    component: str = "mlp_out"
) -> torch.Tensor:
    """
    Collect activations from model for SAE training.
    
    Args:
        model: The model
        tokenizer: Tokenizer
        prompts: List of prompts
        layer: Layer to collect from
        component: "mlp_out", "mlp_in", "residual", "attn_out"
        
    Returns:
        Tensor of activations (n_prompts, seq_len, d_model)
    """
    activations = []
    
    def hook(module, input, output):
        if component == "mlp_in":
            activations.append(input[0].detach())
        else:
            activations.append(output.detach())
    
    # Get the right module
    if component in ["mlp_out", "mlp_in"]:
        target_module = model.model.layers[layer].mlp
    elif component == "attn_out":
        target_module = model.model.layers[layer].self_attn
    else:  # residual
        target_module = model.model.layers[layer]
    
    handle = target_module.register_forward_hook(hook)
    
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            model(**inputs)
    
    handle.remove()
    
    # Concatenate all activations
    return torch.cat(activations, dim=0)


def analyze_sae_features(
    sae: SimpleSAE,
    activations: torch.Tensor,
    labels: List[str],
    k: int = 10
) -> Dict:
    """
    Analyze which SAE features are associated with different labels.
    
    Args:
        sae: Trained SAE
        activations: Input activations
        labels: Label for each activation (e.g., "false_belief", "true_belief")
        k: Top-k features to analyze
        
    Returns:
        Dict with feature analysis
    """
    sae.eval()
    
    # Get features for all activations
    with torch.no_grad():
        features = sae.get_feature_activations(activations)
    
    # Analyze by label
    unique_labels = list(set(labels))
    label_features = {}
    
    for label in unique_labels:
        mask = torch.tensor([l == label for l in labels])
        label_feats = features[mask].mean(dim=0)
        
        # Top features for this label
        values, indices = label_feats.topk(k)
        label_features[label] = {
            "top_features": [(int(idx), float(val)) for idx, val in zip(indices, values)],
            "mean_activation": float(label_feats.mean()),
            "sparsity": float((label_feats > 0).float().mean()),
        }
    
    # Find differentially activated features
    if len(unique_labels) == 2:
        l1, l2 = unique_labels
        diff = label_features[l1]["top_features"][0][1] - label_features[l2]["top_features"][0][1]
        # ... more analysis
    
    return {
        "by_label": label_features,
        "n_active_features": int((features > 0).any(dim=0).sum()),
        "mean_sparsity": float((features > 0).float().mean()),
    }


# =====================================================
# PRE-TRAINED SAE LOADING (for popular models)
# =====================================================

def load_pretrained_sae(
    model_name: str,
    layer: int,
    component: str = "residual"
) -> Optional[SimpleSAE]:
    """
    Load pre-trained SAE from SAELens or other sources.
    
    Note: Requires `pip install sae-lens`
    
    For Qwen models, pre-trained SAEs may not be available yet.
    This function provides the interface for when they become available.
    """
    try:
        from sae_lens import SAE
        
        # SAELens uses specific naming conventions
        # e.g., "gpt2-small-res-jb" for GPT-2 residual stream
        sae_id = f"{model_name}-{component}-layer{layer}"
        
        sae = SAE.from_pretrained(sae_id)
        return sae
    except ImportError:
        print("SAELens not installed. Run: pip install sae-lens")
        return None
    except Exception as e:
        print(f"No pre-trained SAE found for {model_name}: {e}")
        return None


# =====================================================
# EXAMPLE USAGE
# =====================================================

def example_usage():
    """
    Example of how to use SAEs for ToM analysis.
    
    1. Collect activations from ToM scenarios
    2. Train SAE on these activations
    3. Analyze which features correlate with belief type
    """
    
    # Configuration for Qwen3-4B
    config = SAEConfig(
        d_model=2560,  # Qwen3-4B hidden size
        d_sae=2560 * 4,  # 4x expansion
        l1_coeff=5e-3,
    )
    
    # Create SAE
    sae = SimpleSAE(config)
    
    # Training loop (pseudocode)
    """
    trainer = SAETrainer(sae)
    
    for batch in activation_batches:
        loss = trainer.step(batch)
        
    # After training, analyze features
    results = analyze_sae_features(
        sae, 
        test_activations,
        labels=["false_belief", "true_belief"]
    )
    
    # Find "belief update" feature
    for label, data in results["by_label"].items():
        print(f"{label}: top features = {data['top_features'][:5]}")
    """
    
    print("See step13_sae_analysis.py for full implementation")


if __name__ == "__main__":
    example_usage()

