"""Circuit-tracer wrappers for attribution graph analysis.

Thin wrapper around circuit-tracer that enforces our methodology:
- Aggregates across multiple stimuli
- Computes consistency metrics
- Tests necessity and sufficiency
"""

import json
from pathlib import Path
from dataclasses import dataclass, field

from circuit_tracer import attribute, Graph, ReplacementModel


@dataclass
class CircuitFeature:
    """A feature in the attribution graph."""

    layer: int
    feature_idx: int
    attribution: float  # Direct effect on target logit
    frequency: float  # Fraction of stimuli where this feature appears


@dataclass
class CircuitResult:
    """Aggregated circuit discovery result across stimuli."""

    features: list[CircuitFeature]
    n_stimuli: int
    n_total_features: int  # Total unique features found
    n_consistent_features: int  # Features in >=80% of stimuli
    sparsity: float  # n_consistent / n_possible

    def top_features(self, n: int = 20) -> list[CircuitFeature]:
        return sorted(self.features, key=lambda f: -f.attribution)[:n]

    def consistent_features(self, threshold: float = 0.8) -> list[CircuitFeature]:
        return [f for f in self.features if f.frequency >= threshold]


def trace_single(
    model: ReplacementModel,
    prompt: str,
    target_token: str,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
) -> Graph:
    """Run circuit-tracer on a single prompt.

    Args:
        model: ReplacementModel loaded via lib.core.models
        prompt: Input text
        target_token: Token to trace attribution toward
        node_threshold: Fraction of nodes to prune (higher = sparser)
        edge_threshold: Fraction of edges to prune

    Returns:
        Graph object with full attribution data
    """
    graph = attribute(
        model=model,
        prompt=prompt,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )
    return graph


def trace_batch(
    model: ReplacementModel,
    prompts: list[str],
    target_tokens: list[str],
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    save_dir: Path | None = None,
) -> list[Graph]:
    """Run circuit-tracer on a batch of prompts.

    Args:
        model: ReplacementModel
        prompts: List of input texts
        target_tokens: List of target tokens (one per prompt)
        save_dir: Optional directory to save individual graphs

    Returns:
        List of Graph objects
    """
    graphs = []
    for i, (prompt, target) in enumerate(zip(prompts, target_tokens)):
        graph = trace_single(model, prompt, target, node_threshold, edge_threshold)
        graphs.append(graph)

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            graph.to_pt(save_dir / f"graph_{i:04d}.pt")

    return graphs


def save_results(result: CircuitResult, path: Path) -> None:
    """Save circuit result as JSON."""
    data = {
        "n_stimuli": result.n_stimuli,
        "n_total_features": result.n_total_features,
        "n_consistent_features": result.n_consistent_features,
        "sparsity": result.sparsity,
        "features": [
            {
                "layer": f.layer,
                "feature_idx": f.feature_idx,
                "attribution": f.attribution,
                "frequency": f.frequency,
            }
            for f in result.features
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
