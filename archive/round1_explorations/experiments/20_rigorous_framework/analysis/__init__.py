"""
Analysis Pipeline for Rigorous MI Research

SIMPLE API (recommended):
    from analysis import validate, accuracy_with_ci, HeuristicBaselines

MECHANISTIC INTERPRETABILITY (core MI techniques):
    from analysis import ActivationPatcher     # Causal patching (gold standard)
    from analysis import CausalSteering        # Steering vectors
    from analysis import CircuitAnalysis       # Head ablation
    from analysis import LogitLens             # Where does the model decide?
    from analysis import MLPAnalyzer           # Which neurons matter?
    from analysis import MinimalPairTester     # Isolate causal factors
    from analysis import compute_layer_mi      # Information theory
    from analysis import null_probe_accuracy   # Statistical null distributions

GEOMETRY & TRANSFER:
    from analysis import analyze_layer_geometry, test_transfer

FULL API (for advanced use):
    from analysis import ResultValidator, ProbingPipeline
"""

# Simple API (recommended)
from .statistics import accuracy_with_ci, compare_accuracies, is_significant, bonferroni
from .simple_validator import SimpleValidator, validate
from .heuristics import HeuristicBaselines, compute_heuristic_predictions

# Mechanistic Interpretability (core MI techniques!)
from .patching import ActivationPatcher, PatchingResult, check_agreement_flip
from .causal_steering import CausalSteering, SteeringResult
from .circuit_analysis import CircuitAnalysis, HeadImportance
# Chat mode circuit analysis (proper ablation + chat evaluation)
from .circuits import ChatModeCircuitAnalyzer, HeadAblator, AblationResult
from .logit_lens import LogitLens, LogitLensResult, plot_logit_lens
from .mlp_analysis import MLPAnalyzer, MLPLayerAnalysis, AttentionOutputAnalyzer
from .minimal_pairs import MinimalPairTester, MinimalPairResult, TOM_MINIMAL_PAIRS, run_standard_tom_minimal_pairs
from .signal_injection import SignalExtractor, HeadAmplifier, InjectionResult
from .geometry import (
    analyze_layer_geometry,
    analyze_geometry_across_layers, 
    find_peak_separation_layer,
    test_transfer,
    GeometryMetrics,
)
from .information_theory import (
    compute_layer_mi,
    compute_layer_mi_sweep,
    compute_redundancy,
    kraskov_mi,
)
from .null_distributions import (
    null_probe_accuracy,
    null_cosine_distribution,
    null_ablation_flip_rate,
    check_significance,
    NullDistribution,
)

# Full API (for advanced use)
from .probing import ProbingPipeline, ProbeResult
from .validator import ResultValidator, ValidationReport

# Extended controls (optional)
from .controls import (
    generate_attention_checks,
    generate_ceiling_scenarios,
    generate_floor_scenarios,
    bootstrap_ci,
    power_analysis,
    bonferroni_correct,
    benjamini_hochberg,
)

__all__ = [
    # Simple API
    "accuracy_with_ci",
    "compare_accuracies",
    "is_significant",
    "bonferroni",
    "SimpleValidator",
    "validate",
    "HeuristicBaselines",
    "compute_heuristic_predictions",
    # Mechanistic Interpretability
    "ActivationPatcher",
    "PatchingResult",
    "check_agreement_flip",
    "CausalSteering",
    "SteeringResult",
    "CircuitAnalysis",
    "HeadImportance",
    # Chat mode circuits
    "ChatModeCircuitAnalyzer",
    "HeadAblator",
    "AblationResult",
    # Logit Lens (where does model decide?)
    "LogitLens",
    "LogitLensResult",
    "plot_logit_lens",
    # MLP Analysis (which neurons matter?)
    "MLPAnalyzer",
    "MLPLayerAnalysis",
    "AttentionOutputAnalyzer",
    # Minimal Pairs (isolate causal factors)
    "MinimalPairTester",
    "MinimalPairResult",
    "TOM_MINIMAL_PAIRS",
    "run_standard_tom_minimal_pairs",
    # Signal Injection & Amplification
    "SignalExtractor",
    "HeadAmplifier",
    "InjectionResult",
    # Geometry
    "analyze_layer_geometry",
    "analyze_geometry_across_layers",
    "find_peak_separation_layer",
    "test_transfer",
    "GeometryMetrics",
    # Information Theory
    "compute_layer_mi",
    "compute_layer_mi_sweep",
    "compute_redundancy",
    "kraskov_mi",
    # Null Distributions
    "null_probe_accuracy",
    "null_cosine_distribution",
    "null_ablation_flip_rate",
    "check_significance",
    "NullDistribution",
    # Full API
    "ProbingPipeline",
    "ProbeResult",
    "ResultValidator",
    "ValidationReport",
    # Extended controls
    "generate_attention_checks",
    "generate_ceiling_scenarios",
    "generate_floor_scenarios",
    "bootstrap_ci",
    "power_analysis",
    "bonferroni_correct",
    "benjamini_hochberg",
]

