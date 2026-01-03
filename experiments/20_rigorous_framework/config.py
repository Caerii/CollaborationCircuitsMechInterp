"""
Configuration for rigorous mechanistic interpretability experiments.

This config enforces all methodological requirements learned from experiments 1-19:
- Proper token budgets for reasoning models
- Minimum sample sizes for statistical validity
- Counterbalancing requirements
- Heuristic baseline comparisons
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class ExperimentConfig:
    """
    Configuration for rigorous MI experiments.
    
    All defaults enforce proper methodology:
    - max_tokens=1000: Sufficient for reasoning model <think> tags
    - min_samples_per_condition=50: Statistical power requirement
    - require_counterbalancing=True: 8-scenario design mandatory
    - require_beats_heuristics=True: Model must outperform baselines
    """
    
    # ==================== Model Settings ====================
    model_name: str = "Qwen/Qwen3-4B"
    device_map: str = "auto"
    dtype: str = "float16"  # "float16", "bfloat16", "float32"
    attn_implementation: Optional[str] = "eager"  # Required for attention access
    
    # ==================== Token Budget (CRITICAL) ====================
    # Qwen3-4B is a reasoning model that uses <think> tags.
    # Previous experiments failed because token budget was too small (100-150).
    # 1000 tokens allows full reasoning before answering.
    max_tokens: int = 1000
    
    # ==================== Sample Size Requirements ====================
    # Per PROPER_METHODOLOGY.md and step 56 findings:
    # - n < 30: Confidence intervals too wide for any conclusions
    # - n = 50: Minimum for detecting medium effects (h=0.5)
    # - n = 100+: Recommended for robust findings
    min_samples_per_condition: int = 50
    min_samples_for_circuit_claims: int = 30  # Ablation studies
    recommended_samples: int = 100
    
    # ==================== Statistical Thresholds ====================
    significance_level: float = 0.05  # alpha for hypothesis tests
    min_effect_size: float = 0.2  # Cohen's h minimum for practical significance
    confidence_level: float = 0.95  # For confidence intervals
    
    # ==================== Methodology Enforcement ====================
    # These flags can be set to False for exploratory work,
    # but must be True for any published claims.
    require_counterbalancing: bool = True  # 8-scenario design
    require_beats_heuristics: bool = True  # Must beat first-mention, recency, reality
    require_true_belief_controls: bool = True  # TB paired with FB
    require_novel_names: bool = True  # No Alice/Bob/drawer/basket
    require_statistical_tests: bool = True  # p-values and effect sizes
    
    # ==================== Probing Settings ====================
    # Layers to probe for linear classifiers
    probe_layers: List[int] = field(default_factory=lambda: [4, 8, 12, 16, 20, 24, 28, 32, 35])
    cv_folds: int = 5  # Cross-validation folds
    probe_max_iter: int = 1000  # LogisticRegression max iterations
    
    # ==================== Activation Extraction ====================
    batch_size: int = 8  # Batch size for activation extraction (optimized for 10GB VRAM)
    cache_activations: bool = True  # Save activations to disk
    
    # ==================== Circuit Discovery ====================
    # Number of heads to consider for circuit analysis
    n_heads_per_layer: int = 32  # Qwen3-4B has 32 heads
    n_layers: int = 36  # Qwen3-4B has 36 layers
    ablation_samples: int = 30  # Samples per head for ablation
    
    # ==================== Paths ====================
    results_dir: Path = field(default_factory=lambda: Path("results"))
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.results_dir, str):
            self.results_dir = Path(self.results_dir)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
    
    def validate_for_publication(self) -> List[str]:
        """
        Check if config meets publication standards.
        
        Returns:
            List of violations (empty if valid)
        """
        violations = []
        
        if self.max_tokens < 500:
            violations.append(f"max_tokens={self.max_tokens} too low for reasoning model (need >=500)")
        
        if self.min_samples_per_condition < 50:
            violations.append(f"min_samples={self.min_samples_per_condition} too low (need >=50)")
        
        if not self.require_counterbalancing:
            violations.append("counterbalancing disabled - cannot make ToM claims")
        
        if not self.require_beats_heuristics:
            violations.append("heuristic comparison disabled - cannot claim ToM over shortcuts")
        
        if not self.require_true_belief_controls:
            violations.append("true-belief controls disabled - cannot isolate false belief")
        
        return violations
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "min_samples_per_condition": self.min_samples_per_condition,
            "significance_level": self.significance_level,
            "min_effect_size": self.min_effect_size,
            "require_counterbalancing": self.require_counterbalancing,
            "require_beats_heuristics": self.require_beats_heuristics,
            "require_true_belief_controls": self.require_true_belief_controls,
            "require_novel_names": self.require_novel_names,
            "probe_layers": self.probe_layers,
            "cv_folds": self.cv_folds,
        }


# Default configuration for rigorous experiments
DEFAULT_CONFIG = ExperimentConfig()

# Exploratory configuration (relaxed requirements for initial investigation)
EXPLORATORY_CONFIG = ExperimentConfig(
    min_samples_per_condition=10,
    require_counterbalancing=False,
    require_beats_heuristics=False,
    require_true_belief_controls=False,
    require_novel_names=False,
    require_statistical_tests=False,
)

# Quick test configuration (for development/debugging)
DEBUG_CONFIG = ExperimentConfig(
    max_tokens=200,
    min_samples_per_condition=5,
    min_samples_for_circuit_claims=5,
    require_counterbalancing=False,
    require_beats_heuristics=False,
    require_true_belief_controls=False,
    require_novel_names=False,
    require_statistical_tests=False,
    probe_layers=[8, 16, 24],  # Fewer layers for speed
)

