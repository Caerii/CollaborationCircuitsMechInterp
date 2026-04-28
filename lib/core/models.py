"""Direct model loading for mechanistic interpretability work.

Use this when you need activation access, ablation, or circuit-tracer.
For behavioral testing (just running prompts), use chat.py with LM Studio instead.

IMPORTANT: Unload models from LM Studio before using this module — can't share
10GB VRAM between LM Studio and direct PyTorch loading.
"""

from __future__ import annotations

import torch

from lib.utils.config import MODELS, ModelSpec, ExperimentConfig


TRANSCODER_REPOS = {
    "qwen3-4b": "mwhanna/qwen3-4b-transcoders",
    "qwen3-8b": "mwhanna/qwen3-8b-transcoders",
    "gemma2-2b": "mwhanna/gemma-scope-transcoders",
    "llama32-1b": "mntss/transcoder-Llama-3.2-1B",
}


def get_default_transcoder_set(model_key: str) -> str:
    """Return the default circuit-tracer transcoder set for a model key."""
    return TRANSCODER_REPOS.get(model_key, "")


def transcoder_cache_status(transcoder_set: str) -> dict:
    """Check whether circuit-tracer has a completed local cache for transcoders."""
    from circuit_tracer.utils.caching import get_cached_path, is_cached

    cached_path = get_cached_path(transcoder_set)
    return {
        "transcoder_set": transcoder_set,
        "cached": is_cached(transcoder_set),
        "path": str(cached_path),
    }


def get_mechanistic_hf_id(model_key: str) -> str:
    """Return the TransformerLens/circuit-tracer model id for direct loading."""
    spec = MODELS[model_key]
    return spec.mechanistic_hf_id or spec.hf_id


def validate_transformerlens_model(model_name: str) -> None:
    """Fail early if TransformerLens cannot resolve a model name."""
    from transformer_lens import loading_from_pretrained

    try:
        loading_from_pretrained.get_official_model_name(model_name)
    except ValueError as exc:
        raise RuntimeError(
            f"TransformerLens does not support {model_name}. "
            "Set ModelSpec.mechanistic_hf_id to a supported base model id for circuit tracing."
        ) from exc


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

    model_name = get_mechanistic_hf_id(model_key)
    validate_transformerlens_model(model_name)
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
    )
    return model


def load_circuit_tracer_model(
    model_key: str = "qwen3-4b",
    transcoder_set: str = "",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    require_cached_transcoders: bool = False,
) -> ReplacementModel:
    """Load a model via circuit-tracer for attribution graph computation.

    Args:
        model_key: Key from MODELS dict
        transcoder_set: HuggingFace repo for transcoders (auto-detected if empty)
        device: Device to load on
        dtype: Precision
        require_cached_transcoders: Fail before any network attempt unless the
            circuit-tracer transcoder cache already contains the requested set

    Returns:
        ReplacementModel ready for attribution
    """
    from circuit_tracer import ReplacementModel

    model_name = get_mechanistic_hf_id(model_key)
    validate_transformerlens_model(model_name)

    if not transcoder_set:
        transcoder_set = get_default_transcoder_set(model_key)
    if not transcoder_set:
        raise ValueError(f"No transcoder repo known for {model_key}. Pass transcoder_set explicitly.")

    cache = transcoder_cache_status(transcoder_set)
    if require_cached_transcoders and not cache["cached"]:
        raise RuntimeError(
            "Circuit-tracer transcoders are not cached locally. "
            f"Requested {transcoder_set}; expected cache at {cache['path']}. "
            "Run once with network access or pass --transcoder-set pointing to a cached set."
        )

    try:
        model = ReplacementModel.from_pretrained(
            model_name,
            transcoder_set=transcoder_set,
            device=device,
            dtype=dtype,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Unable to load circuit-tracer transcoders. "
            f"Requested {transcoder_set}; local cache status: "
            f"{'hit' if cache['cached'] else 'miss'} at {cache['path']}. "
            "This usually means Hugging Face is unreachable or the transcoder set is not cached."
        ) from exc
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
