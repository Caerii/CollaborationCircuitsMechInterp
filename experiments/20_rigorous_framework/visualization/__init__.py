"""
Visualization Tools

Provides publication-quality figures for MI research.

2D Figures:
    from visualization import FigureGenerator

3D Animated Visualizations (beautiful for submissions):
    from visualization import Visualization3D
"""

from .figures import FigureGenerator
from .visualization_3d import Visualization3D, VisualizationConfig, COLORS as VIZ_COLORS

__all__ = [
    "FigureGenerator",
    "Visualization3D",
    "VisualizationConfig",
    "VIZ_COLORS",
]

