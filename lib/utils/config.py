"""Shared configuration for Round 2 experiments.

Two modes of model access:
- LM Studio (behavioral testing): Fast inference via OpenAI-compatible API.
  Models hosted locally, no Python GPU memory needed.
- Direct loading (mechanistic analysis): Load via transformers/nnsight/circuit-tracer
  for activation extraction, ablation, attribution graphs. Requires GPU memory.
"""

from dataclasses import dataclass, field


# -- LM Studio Configuration --------------------------------------------------

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_API_KEY = "lm-studio"  # LM Studio ignores this but OpenAI client requires it


@dataclass
class ModelSpec:
    name: str
    hf_id: str
    n_layers: int
    n_heads: int
    hidden_size: int
    is_instruct: bool
    circuit_tracer_supported: bool
    lmstudio_id: str = ""  # Model identifier in LM Studio (if different from hf_id)


# All models used in Round 2
MODELS = {
    "qwen3-4b": ModelSpec(
        name="qwen3-4b",
        hf_id="Qwen/Qwen3-4B-Instruct-2507",
        n_layers=36,
        n_heads=32,
        hidden_size=2560,
        is_instruct=True,
        circuit_tracer_supported=True,
    ),
    "gemma2-2b": ModelSpec(
        name="gemma2-2b",
        hf_id="google/gemma-2-2b",
        n_layers=26,
        n_heads=8,
        hidden_size=2304,
        is_instruct=False,
        circuit_tracer_supported=True,
    ),
    "llama32-1b": ModelSpec(
        name="llama32-1b",
        hf_id="meta-llama/Llama-3.2-1B",
        n_layers=16,
        n_heads=32,
        hidden_size=2048,
        is_instruct=False,
        circuit_tracer_supported=True,
    ),
    "qwen3-8b": ModelSpec(
        name="qwen3-8b",
        hf_id="Qwen/Qwen3-8B-Instruct",
        n_layers=36,
        n_heads=32,
        hidden_size=4096,
        is_instruct=True,
        circuit_tracer_supported=True,
    ),
}

PRIMARY_MODEL = "qwen3-4b"
VALIDATION_MODELS = ["gemma2-2b", "llama32-1b"]
SCALE_COMPARISON = ["qwen3-4b", "qwen3-8b"]


@dataclass
class ExperimentConfig:
    """Locked experimental parameters. Do not change without justification."""

    # Generation
    max_new_tokens: int = 1000
    temperature: float = 0.0  # Deterministic for reproducibility

    # Sample sizes
    min_n_per_condition: int = 50
    preferred_n_per_condition: int = 100

    # Statistics
    n_permutations: int = 10_000
    n_bootstrap: int = 10_000
    alpha: float = 0.05
    n_stability_subsets: int = 5
    stability_subset_fraction: float = 0.8

    # Circuit discovery
    attribution_threshold: float = 0.01  # Minimum attribution to include in graph
    circuit_sparsity_target: float = 0.05  # Expect < 5% of features

    # Reproducibility
    random_seed: int = 42

    # Hardware
    dtype: str = "float16"
    device: str = "cuda"


SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Think step by step in <think> tags before giving your final answer."
)
