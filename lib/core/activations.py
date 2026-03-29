"""Activation extraction via nnsight.

Works with any HuggingFace model — use this for cross-model probing and
representation analysis when circuit-tracer's attribution graphs aren't needed.
"""

import torch
import numpy as np
from nnsight import LanguageModel
from lib.utils.config import MODELS, ExperimentConfig, SYSTEM_PROMPT


def load_nnsight_model(
    model_key: str = "qwen3-4b",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> LanguageModel:
    """Load model via nnsight for activation extraction."""
    spec = MODELS[model_key]
    model = LanguageModel(spec.hf_id, device_map=device, torch_dtype=dtype)
    return model


def extract_residual_stream(
    model: LanguageModel,
    text: str,
    layers: list[int] | None = None,
) -> dict[int, torch.Tensor]:
    """Extract residual stream activations at specified layers.

    Args:
        model: nnsight LanguageModel
        text: Input text
        layers: Layer indices to extract (all if None)

    Returns:
        Dict mapping layer index to activation tensor [seq_len, hidden_dim]
    """
    n_layers = len(model.model.layers)
    if layers is None:
        layers = list(range(n_layers))

    activations = {}

    with model.trace(text) as tracer:
        for layer_idx in layers:
            activations[layer_idx] = model.model.layers[layer_idx].output[0].save()

    return {k: v.value.squeeze(0).detach().cpu() for k, v in activations.items()}


def extract_last_token_activations(
    model: LanguageModel,
    texts: list[str],
    layers: list[int] | None = None,
) -> dict[int, torch.Tensor]:
    """Extract last-token activations for a batch of texts.

    Returns dict mapping layer index to tensor [batch_size, hidden_dim].
    Useful for probing — last token typically contains the model's "answer state".
    """
    n_layers = len(model.model.layers)
    if layers is None:
        layers = list(range(n_layers))

    all_activations = {layer: [] for layer in layers}

    for text in texts:
        acts = extract_residual_stream(model, text, layers)
        for layer in layers:
            # Last token
            all_activations[layer].append(acts[layer][-1])

    return {
        layer: torch.stack(tensors)
        for layer, tensors in all_activations.items()
    }
