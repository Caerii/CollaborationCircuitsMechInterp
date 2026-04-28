"""Circuit-tracer wrappers for attribution graph analysis.

Thin wrapper around circuit-tracer that enforces our methodology:
- Aggregates across multiple stimuli
- Computes consistency metrics
- Tests necessity and sufficiency
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from circuit_tracer import Graph, ReplacementModel


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


@dataclass
class GraphFeatureContribution:
    """A selected feature's contribution to an attribution graph target."""

    layer: int
    position: int
    feature_idx: int
    activation: float
    direct_effect: float
    total_influence: float

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "position": self.position,
            "feature_idx": self.feature_idx,
            "activation": self.activation,
            "direct_effect": self.direct_effect,
            "total_influence": self.total_influence,
        }


def trace_single(
    model: ReplacementModel,
    prompt: str,
    target_token: str | None = None,
    attribution_targets: list[str] | None = None,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    max_feature_nodes: int | None = None,
) -> Graph:
    """Run circuit-tracer on a single prompt.

    Args:
        model: ReplacementModel loaded via lib.core.models
        prompt: Input text
        target_token: Optional single token to trace attribution toward
        attribution_targets: Optional list of token strings. Overrides target_token.
        max_n_logits: Maximum salient logits if no explicit target is provided
        desired_logit_prob: Salient-logit cumulative probability target
        max_feature_nodes: Optional cap on selected feature nodes

    Returns:
        Graph object with full, unpruned attribution data. Pruning happens later
        with circuit_tracer.graph.prune_graph or frontend graph creation utilities.
    """
    from circuit_tracer import attribute

    targets = attribution_targets
    if targets is None and target_token is not None:
        targets = [target_token]

    graph = attribute(
        model=model,
        prompt=prompt,
        attribution_targets=targets,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
        max_feature_nodes=max_feature_nodes,
    )
    return graph


def trace_batch(
    model: ReplacementModel,
    prompts: list[str],
    target_tokens: list[str | None] | None = None,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    max_feature_nodes: int | None = None,
    save_dir: Path | None = None,
) -> list[Graph]:
    """Run circuit-tracer on a batch of prompts.

    Args:
        model: ReplacementModel
        prompts: List of input texts
        target_tokens: Optional list of target tokens (one per prompt)
        save_dir: Optional directory to save individual graphs

    Returns:
        List of Graph objects
    """
    if target_tokens is None:
        target_tokens = [None] * len(prompts)
    if len(prompts) != len(target_tokens):
        raise ValueError("prompts and target_tokens must have the same length")

    graphs = []
    for i, (prompt, target) in enumerate(zip(prompts, target_tokens)):
        graph = trace_single(
            model,
            prompt,
            target_token=target,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            max_feature_nodes=max_feature_nodes,
        )
        graphs.append(graph)

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            graph.to_pt(save_dir / f"graph_{i:04d}.pt")

    return graphs


def _selected_activation_values(graph: Graph):
    """Return activation values aligned to graph.selected_features."""
    values = graph.activation_values
    if len(values) == len(graph.selected_features):
        return values
    return values[graph.selected_features]


def summarize_graph(graph: Graph, top_n: int = 20) -> dict:
    """Summarize a circuit-tracer graph for downstream comparison.

    The graph stores nodes in the order:
    selected features, reconstruction errors, input token embeddings, logits.
    This summary focuses on selected feature nodes and computes both:
    - direct_effect: weighted direct edge into the graph's logit target nodes
    - total_influence: total influence after graph propagation

    If multiple logit targets are present, effects are weighted by
    graph.logit_probabilities, matching circuit-tracer's graph score logic.
    """
    import torch
    from circuit_tracer.graph import compute_graph_scores, compute_node_influence

    n_logits = len(graph.logit_targets)
    n_tokens = len(graph.input_tokens)
    n_features = len(graph.selected_features)
    n_errors = n_tokens * graph.cfg.n_layers

    if n_logits == 0:
        raise ValueError("Graph has no logit targets")

    logit_weights = torch.zeros(
        graph.adjacency_matrix.shape[0], device=graph.adjacency_matrix.device
    )
    logit_weights[-n_logits:] = graph.logit_probabilities.to(graph.adjacency_matrix.device)

    node_influence = compute_node_influence(graph.adjacency_matrix, logit_weights)
    direct_effect = (
        graph.logit_probabilities.to(graph.adjacency_matrix.device)
        @ graph.adjacency_matrix[-n_logits:, :n_features]
    )

    feature_meta = graph.active_features[graph.selected_features].detach().cpu()
    activation_values = _selected_activation_values(graph).detach().cpu()
    direct_effect_cpu = direct_effect.detach().cpu()
    influence_cpu = node_influence[:n_features].detach().cpu()

    order = torch.argsort(torch.abs(direct_effect_cpu), descending=True)[:top_n].tolist()
    top_features = []
    for idx in order:
        layer, position, feature_idx = feature_meta[idx].tolist()
        top_features.append(GraphFeatureContribution(
            layer=int(layer),
            position=int(position),
            feature_idx=int(feature_idx),
            activation=float(activation_values[idx].item()),
            direct_effect=float(direct_effect_cpu[idx].item()),
            total_influence=float(influence_cpu[idx].item()),
        ).to_dict())

    replacement_score, completeness_score = compute_graph_scores(graph)

    return {
        "input_string": graph.input_string,
        "n_tokens": n_tokens,
        "n_layers": graph.cfg.n_layers,
        "n_selected_features": n_features,
        "n_active_features": int(len(graph.active_features)),
        "n_error_nodes": n_errors,
        "n_logits": n_logits,
        "adjacency_shape": list(graph.adjacency_matrix.shape),
        "replacement_score": float(replacement_score),
        "completeness_score": float(completeness_score),
        "logit_targets": [
            {
                "token": target.token_str,
                "vocab_idx": int(target.vocab_idx),
                "weight": float(graph.logit_probabilities[i].detach().cpu().item()),
            }
            for i, target in enumerate(graph.logit_targets)
        ],
        "top_features": top_features,
    }


def aggregate_feature_summaries(summaries: list[dict], frequency_threshold: float = 0.0) -> list[dict]:
    """Aggregate top-feature summaries across graph summaries.

    Features are keyed by (layer, feature_idx), ignoring token position so that
    the same transcoder feature can be compared across different prompts.
    """
    aggregate = defaultdict(lambda: {
        "count": 0,
        "direct_effect_sum": 0.0,
        "abs_direct_effect_sum": 0.0,
        "total_influence_sum": 0.0,
        "positions": defaultdict(int),
    })

    n_graphs = len(summaries)
    for summary in summaries:
        for feature in summary.get("top_features", []):
            key = (feature["layer"], feature["feature_idx"])
            row = aggregate[key]
            row["count"] += 1
            row["direct_effect_sum"] += feature["direct_effect"]
            row["abs_direct_effect_sum"] += abs(feature["direct_effect"])
            row["total_influence_sum"] += feature["total_influence"]
            row["positions"][feature["position"]] += 1

    rows = []
    for (layer, feature_idx), row in aggregate.items():
        frequency = row["count"] / n_graphs if n_graphs else 0.0
        if frequency < frequency_threshold:
            continue
        rows.append({
            "layer": int(layer),
            "feature_idx": int(feature_idx),
            "count": int(row["count"]),
            "frequency": float(frequency),
            "mean_direct_effect": row["direct_effect_sum"] / row["count"],
            "mean_abs_direct_effect": row["abs_direct_effect_sum"] / row["count"],
            "mean_total_influence": row["total_influence_sum"] / row["count"],
            "positions": {str(k): int(v) for k, v in sorted(row["positions"].items())},
        })

    return sorted(rows, key=lambda r: (-r["frequency"], -r["mean_abs_direct_effect"]))


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
