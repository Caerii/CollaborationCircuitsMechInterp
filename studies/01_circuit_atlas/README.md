# Study 1: Circuit Atlas — Mapping the Full ToM Computational Graph

Use Anthropic's circuit-tracer to get complete attribution graphs for Theory of Mind
computation. Replaces the manual head-by-head ablation approach from Round 1.

## Hypotheses

- H1.1: ToM is computed by a sparse subgraph (< 5% of features)
- H1.2: Circuit contains separable subcircuits for agent-binding, belief-content, and divergence-detection
- H1.3: False-belief and true-belief circuits share agent-binding but diverge at belief-content

## Status

- [x] Pilot stimulus set created (5 base sets x 8 variants = 40 stimuli)
- [x] Pilot behavioral run completed on Qwen3-4B-style instruct setting
- [x] Thin-slice trace script created for FB-correct vs FB-wrong cases
- [x] Graph comparison scaffold created (`thin_slice.py --compare`)
- [ ] Confirmatory stimulus set created (50+ FB/TB pairs, counterbalanced, novel names)
- [ ] circuit-tracer running on Qwen3-4B thin slices
- [ ] circuit-tracer running on Gemma-2-2B
- [ ] circuit-tracer running on Llama-3.2-1B
- [ ] Attribution graphs aggregated
- [ ] Necessity test (ablate circuit, ToM breaks)
- [ ] Sufficiency test (ablate everything else, ToM survives)
- [ ] Cross-model comparison
- [ ] Results written up

## Current Pilot Result

The saved instruct behavioral pilot (`pilot_results/instruct_behavioral.json`) gives:

| Condition | Correct | Accuracy |
|-----------|---------|----------|
| False belief | 4/10 | 40% |
| True belief | 10/10 | 100% |
| Reality check | 10/10 | 100% |
| Explicit belief question | 10/10 | 100% |

This is the key contrast for the next mechanistic step: the model reliably tracks
reality and explicit belief statements, but false-belief inference splits into
belief-based successes and reality-biased failures.

## Immediate Commands

```bash
# No GPU/model load; prints current behavioral slices
python studies/01_circuit_atlas/thin_slice.py --info

# No model load; checks TransformerLens model support, transcoder cache, and VRAM
python studies/01_circuit_atlas/thin_slice.py --preflight

# Explicit network step: fills circuit-tracer's local transcoder cache
python studies/01_circuit_atlas/thin_slice.py --cache-transcoders

# Offline-safe smoke trace: one graph only, requires cached transcoders
python studies/01_circuit_atlas/thin_slice.py --trace --target-mode correct-reality --max-pairs-per-slice 1 --max-graphs 1 --require-cached-transcoders

# Full run: may download circuit-tracer transcoders on first uncached run
python studies/01_circuit_atlas/thin_slice.py --trace --target-mode correct-reality

# After traces exist under thin_slice_results/
python studies/01_circuit_atlas/thin_slice.py --compare
```

The default Qwen3-4B trace uses `mwhanna/qwen3-4b-transcoders`. If the machine
cannot reach Hugging Face, either run after the transcoders have been cached by
`circuit-tracer` or pass `--transcoder-set` explicitly.

Behavioral pilot results were collected from the instruct setting. The direct
circuit-tracer path uses the matching base model id exposed through
`ModelSpec.mechanistic_hf_id` because TransformerLens does not support every
instruct checkpoint name.

`--compare` writes:

- `thin_slice_results/graph_summaries.json`
- `thin_slice_results/graph_comparison.json`
