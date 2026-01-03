"""
Experiment 20: Rigorous MI Framework for Collaboration Circuits

A comprehensive mechanistic interpretability framework that synthesizes
best practices from experiments 1-19, enforcing proper methodology:
- n >= 50 samples per condition
- 1000 token budget for reasoning models
- 8-scenario counterbalancing
- Heuristic baselines comparison
- Statistical significance testing
- Novel names to break priors
- True-belief controls
- Causal validation for circuit claims

Modules:
    core: Model loading, activation extraction, chat-based experiment running
    scenarios: Scenario generation with counterbalancing and novel names
    analysis: Probing, heuristics, causal steering, validation
    exploration: Systematic discovery, head sweeps, cross-model validation
    visualization: Publication-quality figures
    runners: Complete test suites
"""

from .config import ExperimentConfig, DEFAULT_CONFIG

__version__ = "1.0.0"
__all__ = ["ExperimentConfig", "DEFAULT_CONFIG"]

