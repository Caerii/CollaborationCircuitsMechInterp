# Study 1: Circuit Atlas — Mapping the Full ToM Computational Graph

Use Anthropic's circuit-tracer to get complete attribution graphs for Theory of Mind
computation. Replaces the manual head-by-head ablation approach from Round 1.

## Hypotheses

- H1.1: ToM is computed by a sparse subgraph (< 5% of features)
- H1.2: Circuit contains separable subcircuits for agent-binding, belief-content, and divergence-detection
- H1.3: False-belief and true-belief circuits share agent-binding but diverge at belief-content

## Status

- [ ] Stimulus set created (50 FB/TB pairs, counterbalanced, novel names)
- [ ] circuit-tracer running on Qwen3-4B
- [ ] circuit-tracer running on Gemma-2-2B
- [ ] circuit-tracer running on Llama-3.2-1B
- [ ] Attribution graphs aggregated
- [ ] Necessity test (ablate circuit, ToM breaks)
- [ ] Sufficiency test (ablate everything else, ToM survives)
- [ ] Cross-model comparison
- [ ] Results written up
