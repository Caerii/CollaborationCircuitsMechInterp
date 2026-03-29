"""Direct model loading for mechanistic interpretability work.

Use this when you need activation access, ablation, or circuit-tracer.
For behavioral testing (just running prompts), use chat.py with LM Studio instead.

IMPORTANT: Unload models from LM Studio before using this module — can't share
10GB VRAM between LM Studio and direct PyTorch loading.
"""

from __future__ import annotations

import torch

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
    from transformer_lens import HookedTransformer

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
    from circuit_tracer import ReplacementModel

    spec = MODELS[model_key]

    # Default transcoder repos per model
    TRANSCODER_REPOS = {
        "qwen3-4b": "mwhanna/qwen3-4b-transcoders",
        "qwen3-8b": "mwhanna/qwen3-8b-transcoders",
        "gemma2-2b": "mwhanna/gemma-scope-transcoders",
        "llama32-1b": "mntss/transcoder-Llama-3.2-1B",
    }
    if not transcoder_set:
        transcoder_set = TRANSCODER_REPOS.get(model_key, "")
    if not transcoder_set:
        raise ValueError(f"No transcoder repo known for {model_key}. Pass transcoder_set explicitly.")

    model = ReplacementModel.from_pretrained(
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

    props = torch.cuda.get_device_properties(0)
    total = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
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
