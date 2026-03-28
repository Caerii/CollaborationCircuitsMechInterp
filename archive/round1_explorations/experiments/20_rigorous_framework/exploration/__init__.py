"""
Systematic Exploration Tools

Provides:
    HeadDiscoverySweep: Systematic search for important attention heads
    PhenomenonDiscovery: Discover patterns and generate hypotheses
    CrossModelValidator: Test findings across model sizes
"""

from .head_sweep import HeadDiscoverySweep
from .discovery import PhenomenonDiscovery, FailurePattern, Hypothesis
from .cross_model import CrossModelValidator

__all__ = [
    "HeadDiscoverySweep",
    "PhenomenonDiscovery",
    "FailurePattern",
    "Hypothesis",
    "CrossModelValidator",
]

