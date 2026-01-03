"""
MechInterp Library for Theory of Mind Circuit Analysis

This library provides reusable components for mechanistic interpretability
research on transformer-based language models, specifically targeting
Theory of Mind (ToM) circuits.

Key modules:
- model_utils: Model loading and management
- hooks: Hook management for ablation, amplification, patching
- scenarios: ToM scenario generation
- evaluation: ToM evaluation pipelines
- statistics: Statistical analysis utilities
"""

from .model_utils import QwenModel
from .hooks import HookManager, ActivationCache
from .scenarios import ScenarioGenerator
from .evaluation import ToMEvaluator
from .statistics import compute_accuracy_ci, significance_test

__all__ = [
    'QwenModel',
    'HookManager',
    'ActivationCache',
    'ScenarioGenerator',
    'ToMEvaluator',
    'compute_accuracy_ci',
    'significance_test',
]

__version__ = '0.1.0'


