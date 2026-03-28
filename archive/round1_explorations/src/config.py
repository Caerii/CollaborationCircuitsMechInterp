"""
Configuration for Project A: Self/Other/User Representation Separation
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = PROJECT_ROOT / "cache"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


@dataclass
class ModelConfig:
    """
    Model configuration for Qwen3-4B-Instruct-2507.
    
    This is the SAME model as your LM Studio GGUF, but in HuggingFace format
    which allows us to extract activations for mechanistic interpretability.
    
    Released: August 2025
    - 4B parameters, 2560 hidden size, 36 layers
    - 256K context window
    - Fits in 10GB VRAM (RTX 3080) in FP16
    """
    # HuggingFace model - matches your LM Studio GGUF!
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    
    # Local GGUF path (for reference, not used for mech interp)
    local_gguf_path: str = r"C:\Users\locke\.lmstudio\models\bartowski\qwen_qwen3-4b-instruct-2507"
    
    device: str = "cuda"
    dtype: str = "float16"  # ~8GB VRAM usage on RTX 3080
    max_length: int = 4096  # Can go higher but 4K is good for experiments
    
    # Layers to probe (36 layers total)
    # We sample across the model to find where entity info is encoded
    probe_layers: tuple = (0, 4, 8, 12, 16, 20, 24, 28, 32, 35)
    
    # Model architecture
    hidden_size: int = 2560
    num_layers: int = 36


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    # Data generation
    n_dialogues: int = 300
    min_turns: int = 4
    max_turns: int = 8
    
    # Probing
    probe_hidden_dim: Optional[int] = None  # None = linear probe
    train_split: float = 0.8
    learning_rate: float = 1e-3
    epochs: int = 50
    batch_size: int = 32
    
    # Activation extraction
    extraction_batch_size: int = 2  # Small for VRAM
    
    # Random seed
    seed: int = 42


# Default configs
MODEL_CFG = ModelConfig()
EXP_CFG = ExperimentConfig()


# Entity types for classification
ENTITY_TYPES = {
    "user": 0,
    "self": 1,   # The model itself (Agent A)
    "other": 2,  # Another agent (Agent B)
}

# Dialogue roles
ROLES = {
    "user": "User",
    "agent_a": "Assistant",  # Self - the model we're analyzing
    "agent_b": "Helper",     # Other - a different agent
}

