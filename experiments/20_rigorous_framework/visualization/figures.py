"""
Publication-Quality Figure Generation

Generate figures for MI research papers and presentations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Try to import matplotlib, provide fallback if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


class FigureGenerator:
    """
    Generate publication-quality figures for MI research.
    
    Example:
        fig_gen = FigureGenerator(output_dir="figures/")
        
        # Accuracy comparison
        fig_gen.accuracy_by_condition(results)
        
        # Heuristic comparison
        fig_gen.heuristic_comparison(model_acc, heuristics)
    """
    
    # Style settings for publication
    STYLE = {
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }
    
    # Color palette
    COLORS = {
        "primary": "#2ecc71",    # Green
        "secondary": "#3498db",  # Blue
        "tertiary": "#e74c3c",   # Red
        "quaternary": "#f39c12", # Orange
        "neutral": "#95a5a6",    # Gray
    }
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        style: Optional[Dict] = None
    ):
        """
        Initialize figure generator.
        
        Args:
            output_dir: Directory to save figures
            style: Optional style overrides
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if HAS_MATPLOTLIB:
            plt.rcParams.update(self.STYLE)
            if style:
                plt.rcParams.update(style)
    
    def _check_matplotlib(self):
        """Check if matplotlib is available."""
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib not installed. Install with: pip install matplotlib"
            )
    
    def _save_figure(self, fig, name: str):
        """Save figure to output directory."""
        if self.output_dir:
            path = self.output_dir / f"{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")
        return fig
    
    def accuracy_by_condition(
        self,
        results: Dict[str, Dict],
        title: str = "Accuracy by Condition",
        save_name: Optional[str] = None
    ):
        """
        Bar chart of accuracy by condition.
        
        Args:
            results: Dict mapping condition name to {"accuracy": float, "n": int}
            title: Figure title
            save_name: Filename to save
        """
        self._check_matplotlib()
        
        conditions = list(results.keys())
        accuracies = [results[c].get("accuracy", 0) * 100 for c in conditions]
        ns = [results[c].get("n", results[c].get("n_total", 0)) for c in conditions]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(conditions, accuracies, color=self.COLORS["primary"], edgecolor="black")
        
        # Add value labels
        for bar, n in zip(bars, ns):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%\n(n={n})',
                   ha='center', va='bottom', fontsize=10)
        
        # Add chance line
        ax.axhline(y=50, color=self.COLORS["neutral"], linestyle="--", label="Chance (50%)")
        
        ax.set_xlabel("Condition")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title)
        ax.set_ylim(0, 105)
        ax.legend()
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def heuristic_comparison(
        self,
        model_accuracy: float,
        heuristic_accuracies: Dict[str, float],
        title: str = "Model vs Heuristic Baselines",
        save_name: Optional[str] = None
    ):
        """
        Bar chart comparing model to heuristic baselines.
        
        Args:
            model_accuracy: Model's accuracy
            heuristic_accuracies: Dict mapping heuristic name to accuracy
            title: Figure title
            save_name: Filename to save
        """
        self._check_matplotlib()
        
        names = ["Model"] + list(heuristic_accuracies.keys())
        accuracies = [model_accuracy * 100] + [v * 100 for v in heuristic_accuracies.values()]
        
        colors = [self.COLORS["primary"]] + [self.COLORS["neutral"]] * len(heuristic_accuracies)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(names, accuracies, color=colors, edgecolor="black")
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10)
        
        # Highlight if model beats heuristics
        best_heuristic = max(heuristic_accuracies.values())
        if model_accuracy > best_heuristic:
            ax.set_title(title + " (Model BEATS heuristics)")
        else:
            ax.set_title(title + " (Model does NOT beat heuristics)")
        
        ax.set_xlabel("Method")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def layer_probe_accuracy(
        self,
        probe_results: Dict[int, Dict],
        title: str = "Probe Accuracy by Layer",
        save_name: Optional[str] = None
    ):
        """
        Line plot of probe accuracy across layers.
        
        Args:
            probe_results: Dict mapping layer to {"accuracy": float, "std": float}
            title: Figure title
            save_name: Filename to save
        """
        self._check_matplotlib()
        
        layers = sorted(probe_results.keys())
        accuracies = [probe_results[l].get("accuracy", 0) * 100 for l in layers]
        stds = [probe_results[l].get("std", 0) * 100 for l in layers]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(layers, accuracies, 'o-', color=self.COLORS["primary"], 
                linewidth=2, markersize=8, label="Accuracy")
        ax.fill_between(layers, 
                       [a - s for a, s in zip(accuracies, stds)],
                       [a + s for a, s in zip(accuracies, stds)],
                       alpha=0.2, color=self.COLORS["primary"])
        
        # Chance line
        chance = probe_results[layers[0]].get("chance_level", 0.5) * 100
        ax.axhline(y=chance, color=self.COLORS["neutral"], linestyle="--", label=f"Chance ({chance:.0f}%)")
        
        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title)
        ax.legend()
        ax.set_xticks(layers)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def head_importance_heatmap(
        self,
        head_rankings: List[Tuple[int, int, float]],
        n_layers: int,
        n_heads: int,
        title: str = "Head Importance",
        save_name: Optional[str] = None
    ):
        """
        Heatmap of attention head importance.
        
        Args:
            head_rankings: List of (layer, head, importance)
            n_layers: Total number of layers
            n_heads: Heads per layer
            title: Figure title
            save_name: Filename to save
        """
        self._check_matplotlib()
        
        # Create importance matrix
        matrix = np.zeros((n_layers, n_heads))
        for layer, head, importance in head_rankings:
            if layer < n_layers and head < n_heads:
                matrix[layer, head] = importance
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(matrix, aspect='auto', cmap='Reds')
        
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(title)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Importance")
        
        # Annotate top heads
        top_5 = sorted(head_rankings, key=lambda x: abs(x[2]), reverse=True)[:5]
        for layer, head, imp in top_5:
            if layer < n_layers and head < n_heads:
                ax.plot(head, layer, 'ko', markersize=10, fillstyle='none', linewidth=2)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def steering_effect_plot(
        self,
        steering_results: Dict,
        title: str = "Steering Effect by Strength",
        save_name: Optional[str] = None
    ):
        """
        Line plot of steering effect by strength.
        
        Args:
            steering_results: Results from CausalSteering.test_effect
            title: Figure title
            save_name: Filename to save
        """
        self._check_matplotlib()
        
        change_rates = steering_results.get("change_rates", {})
        
        strengths = sorted(change_rates.keys())
        rates = [change_rates[s] * 100 for s in strengths]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(strengths, rates, 'o-', color=self.COLORS["primary"],
                linewidth=2, markersize=10)
        
        # Threshold line
        ax.axhline(y=30, color=self.COLORS["neutral"], linestyle="--", 
                   label="Functional threshold (30%)")
        
        ax.set_xlabel("Steering Strength")
        ax.set_ylabel("Change Rate (%)")
        ax.set_title(title)
        ax.legend()
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def validation_summary(
        self,
        validation_report: Dict,
        title: str = "Methodology Validation",
        save_name: Optional[str] = None
    ):
        """
        Visual summary of validation checks.
        
        Args:
            validation_report: Report from ResultValidator
            title: Figure title
            save_name: Filename to save
        """
        self._check_matplotlib()
        
        checks = validation_report.get("checks", [])
        
        names = [c["name"] for c in checks]
        passed = [c["passed"] for c in checks]
        
        colors = [self.COLORS["primary"] if p else self.COLORS["tertiary"] for p in passed]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_pos = np.arange(len(names))
        ax.barh(y_pos, [1] * len(names), color=colors, edgecolor="black")
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlim(0, 1.5)
        ax.set_xlabel("")
        ax.set_title(title)
        
        # Add pass/fail labels
        for i, (name, p) in enumerate(zip(names, passed)):
            label = "PASS" if p else "FAIL"
            color = "white" if p else "white"
            ax.text(0.5, i, label, ha='center', va='center', fontsize=12, 
                   fontweight='bold', color=color)
        
        # Overall result
        all_passed = all(passed)
        result_text = "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"
        result_color = self.COLORS["primary"] if all_passed else self.COLORS["tertiary"]
        
        ax.text(1.2, len(names)/2, result_text, ha='center', va='center',
               fontsize=14, fontweight='bold', color=result_color,
               rotation=270)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig

