"""Direct model loading for mechanistic interpretability work.

Use this when you need activation access, ablation, or circuit-tracer.
For behavioral testing (just running prompts), use chat.py with LM Studio instead.

IMPORTANT: Unload models from LM Studio before using this module — can't share
10GB VRAM between LM Studio and direct PyTorch loading.
"""

import torch
from transformer_lens import HookedTransformer
from circuit_tracer import ReplacementModel

from lib.utils.config import MODELS, ModelSpec, ExperimentConfig


def load_hooked_model(
    model_key: str = "qwen3-4b",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> HookedTransformer:
    """Load a model via TransformerLens for activation hooks and ablation.

    Args:
        model_key: Key from MODELS dict
        device: Device to load on
        dtype: Precision (float16 for 10GB VRAM)

    Returns:
        HookedTransformer with full hook access
    """
    spec = MODELS[model_key]
    model = HookedTransformer.from_pretrained(
        spec.hf_id,
        device=device,
        dtype=dtype,
    )
    return model


def load_circuit_tracer_model(
    model_key: str = "qwen3-4b",
    transcoder_set: str = "",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> ReplacementModel:
    """Load a model via circuit-tracer for attribution graph computation.

    Args:
        model_key: Key from MODELS dict
        transcoder_set: HuggingFace repo for transcoders (auto-detected if empty)
        device: Device to load on
        dtype: Precision

    Returns:
        ReplacementModel ready for attribution
    """
    spec = MODELS[model_key]

    model = ReplacementModel.from_pretrained_and_transcoders(
        spec.hf_id,
        transcoder_set=transcoder_set,
        device=device,
        dtype=dtype,
    )
    return model


def get_model_spec(model_key: str = "qwen3-4b") -> ModelSpec:
    """Get model specifications without loading."""
    return MODELS[model_key]


def check_vram() -> dict:
    """Check current VRAM usage."""
    if not torch.cuda.is_available():
        return {"available": False}

    total = torch.cuda.get_device_properties(0).total_mem / 1e9
    allocated = torch.cuda.memory_allocated(0) / 1e9
    cached = torch.cuda.memory_reserved(0) / 1e9

    return {
        "available": True,
        "gpu": torch.cuda.get_device_name(0),
        "total_gb": round(total, 2),
        "allocated_gb": round(allocated, 2),
        "cached_gb": round(cached, 2),
        "free_gb": round(total - allocated, 2),
    }
