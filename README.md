# Social Cognition Circuits in LLMs

Mechanistic interpretability research on how language models represent other agents,
track beliefs, and coordinate in multi-agent settings.

## Project Structure

```
studies/                        # Round 2: Systematic studies (current)
├── 01_circuit_atlas/           # Full attribution graphs for ToM circuits
├── 02_social_cognition_circuits/  # Taxonomy of 6 social cognition tasks
├── 03_multi_agent_interference/   # Agent representation interference
├── 04_cross_model_validation/     # Universal vs model-specific findings
└── 05_intervention_and_steering/  # Targeted mechanistic interventions

lib/                            # Shared library
├── core/                       # Model loading, chat evaluation
├── scenarios/                  # Stimulus generation (counterbalanced, novel names)
├── analysis/                   # Statistics, circuit analysis, probing
└── utils/                      # Config, logging

shared/                         # Shared data across studies
├── stimuli/                    # Generated stimulus sets
├── results/                    # Cross-study results
└── figures/                    # Publication figures

archive/round1_explorations/    # December 2025 exploratory work (20 experiments)
```

## Key Documents

- **[EXPERIMENTAL_PLAN.md](EXPERIMENTAL_PLAN.md)** — Pre-registered hypotheses and study designs
- **[METHODOLOGY.md](METHODOLOGY.md)** — Locked methodology protocol (testing format, statistics, controls)
- **[archive/round1_explorations/README.md](archive/round1_explorations/README.md)** — What Round 1 found and its lessons

## Core Question

**What are the computational mechanisms by which LLMs model other agents' mental
states, and how do these mechanisms succeed and fail?**

## Approach

- **Circuit discovery** via Anthropic's `circuit-tracer` (attribution graphs with transcoders)
- **Causal analysis** via Relevance Patching (RelP)
- **Feature decomposition** via SAELens + Temporal SAEs
- **Cross-model validation** on Qwen3-4B, Gemma-2-2B, Llama-3.2-1B
- **Locked methodology**: Pre-registered hypotheses, n>=50, counterbalanced stimuli,
  permutation tests, heuristic baselines, stability analysis

## Key Findings from Round 1 (Informing Round 2)

1. ToM requires chat mode + reasoning budget (50% -> 95% accuracy)
2. Explicit belief parsing works perfectly; implicit belief inference is weak
3. Single-agent and multi-agent ToM use separate circuits
4. Activation patching fails in chat mode (decision is distributed)
5. Probes can overfit to surface features — always run transfer tests

## Setup

```bash
uv sync
# or
pip install -e ".[dev]"
```

## Models

| Model | Role | Hardware |
|-------|------|----------|
| Qwen3-4B-Instruct | Primary | RTX 3080 (10GB) |
| Gemma-2-2B | Validation | RTX 3080 |
| Llama-3.2-1B | Validation | RTX 3080 |
| Qwen3-8B-Instruct | Scale comparison | Needs quantization or cloud |
