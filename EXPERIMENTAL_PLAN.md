# Round 2: Systematic Study of Social Cognition Circuits in LLMs

## Overview

This is a pre-registered experimental plan for mechanistic interpretability research
on how language models implement social cognition — specifically, how they represent
other agents, track beliefs, and coordinate in multi-agent settings.

**Core question**: What are the computational mechanisms by which LLMs model other
agents' mental states, and how do these mechanisms succeed and fail?

**What's different from Round 1**: Pre-registered hypotheses, locked methodology,
proper sample sizes, cross-model validation, and modern tools (circuit-tracer,
Relevance Patching, Temporal SAEs).

---

## Methodology Protocol (Applies to ALL Studies)

### Testing Format
- **Always chat mode** with system/user/assistant roles
- **Always include `<think>` tag instruction** in system prompt
- **Minimum 1000 token generation budget** for reasoning models
- **Never use completion mode** for instruction-tuned models
- Base models (if tested) use completion mode with appropriate prompting

### Sample Sizes
- **Minimum n=50 per condition** for confirmatory experiments
- **n=100 preferred** for primary claims
- **Power analysis**: At expected effect size d=0.5, n=50 gives 80% power (two-tailed)
- Exploratory analyses can use n=20 but must be labeled as exploratory

### Stimulus Design
- **Counterbalanced**: 8-scenario design (2 belief states x 2 location orders x 2 controls)
- **Novel names**: No Alice/Bob/drawer/basket — use generated names to avoid training priors
- **Heuristic baselines built in**: Every stimulus set includes first-mention, recency,
  and reality baselines computable from the stimuli alone

### Statistical Standards
- **Primary test**: Permutation test (10,000 permutations) — no parametric assumptions
- **Effect sizes**: Cohen's d or h with 95% bootstrap CIs (10,000 resamples)
- **Multiple comparisons**: Bonferroni correction within each study; FDR across studies
- **Stability analysis**: Re-run on 5 random 80% subsets of stimuli; report agreement
- **Null baselines**: Compare probe accuracy against null distribution (random labels, matched n and d)
- **Heuristic baselines**: Model must outperform all simple heuristics to claim a capability

### Cross-Model Requirement
- **Primary model**: Qwen3-4B-Instruct (continuity with Round 1)
- **Validation models**: Gemma-2-2B, Llama-3.2-1B (both supported by circuit-tracer)
- **A finding only counts if it replicates across ≥2 model families**
- Report model-specific vs. universal findings separately

### Tools
- **Circuit discovery**: Anthropic's `circuit-tracer` (attribution graphs with transcoders)
- **Causal analysis**: Relevance Patching (RelP) — replaces standard attribution patching
- **Feature decomposition**: SAELens v6 + Temporal SAEs for belief tracking over narratives
- **Activation extraction**: nnsight (works with any HuggingFace model, enables cross-model)
- **Statistical analysis**: scipy, statsmodels, custom permutation tests
- **Visualization**: matplotlib, plotly for interactive circuit graphs

---

## Study 1: Circuit Atlas — Mapping the Full ToM Computational Graph

**Goal**: Use circuit-tracer to get a *complete* attribution graph for ToM computation,
replacing the manual head-by-head ablation approach from Round 1.

### Pre-Registered Hypotheses

**H1.1**: The model's false-belief prediction is computed by a sparse subgraph
(< 5% of total features) that is identifiable via attribution.

**H1.2**: The circuit contains separable subcircuits for:
  - (a) Agent identity binding (WHO has the belief)
  - (b) Belief content representation (WHAT they believe)
  - (c) Belief-reality divergence detection (belief ≠ reality)

**H1.3**: The attribution graph for false-belief vs true-belief scenarios will
share agent-binding features but diverge at belief-content features.

### Method

1. **Prepare stimuli**: 50 matched false-belief / true-belief pairs using the
   counterbalanced 8-scenario design with novel names.

2. **Run circuit-tracer** on each stimulus:
   - Extract full attribution graph (transcoder features → features → output logit)
   - Identify features with highest direct effect on the belief-location token
   - Trace backward to find the full causal chain

3. **Aggregate across stimuli**:
   - Identify features that consistently appear across ≥80% of stimuli
   - Cluster features by function (agent-binding, content, divergence)
   - Build the canonical ToM circuit as the intersection of per-stimulus graphs

4. **Validate**:
   - Ablate the identified circuit: does ToM accuracy drop to chance?
   - Ablate everything EXCEPT the circuit: does ToM accuracy survive?
   - Both directions needed for completeness (per Chen et al. 2025, OR gates)

5. **Cross-model**: Repeat on Gemma-2-2B and Llama-3.2-1B. Compare circuit topology.

### Success Criteria
- Identified circuit achieves ≥90% necessity (ablation breaks ToM) and ≥70% sufficiency
  (circuit alone preserves ToM)
- Circuit is sparse (< 5% of features)
- At least the agent-binding subcircuit replicates across 2+ models

### Outputs
- `studies/01_circuit_atlas/circuit_graph.json` — Full attribution graph
- `studies/01_circuit_atlas/figures/` — Circuit visualizations
- `studies/01_circuit_atlas/results/` — Per-stimulus and aggregate results

---

## Study 2: Social Cognition Circuits — Beyond Sally-Anne

**Goal**: Map circuits for a taxonomy of social cognition tasks, not just false belief.
Test whether the model has modular or shared circuits for different social reasoning.

### Task Taxonomy

| Task | Description | Example |
|------|-------------|---------|
| **False Belief** | Agent has outdated belief | Sally-Anne |
| **Knowledge Attribution** | Who knows what? | "Does X know about Y?" |
| **Intention Reading** | Why did agent do X? | "Why did X move the ball?" |
| **Perspective Taking** | What does X see/experience? | "X is in room A, what can X see?" |
| **Communication Tracking** | What info was transmitted? | "X told Y about Z" |
| **Belief Update** | How does new info change beliefs? | "X learns that Z moved" |

### Pre-Registered Hypotheses

**H2.1** (Modularity): Different social cognition tasks recruit partially distinct
circuits, with a shared "agent-binding" core.

**H2.2** (Explicit > Implicit): Tasks requiring explicit information parsing
(knowledge attribution, false belief with stated beliefs) will show stronger,
more localized circuits than tasks requiring implicit inference (belief update,
intention reading).

**H2.3** (Communication gap): The circuit for "X told Y about Z" → "Y now knows Z"
will be identifiable but weak (low-amplitude features), explaining the
explicit/implicit gap found in Round 1.

### Method

1. **Create 50 stimuli per task** (6 tasks x 50 = 300 total), counterbalanced.

2. **Run circuit-tracer** on each, extract attribution graphs.

3. **Compute circuit overlap**: For each pair of tasks, measure Jaccard similarity
   of the top-50 features in their respective circuits.

4. **Identify the shared core**: Features present in ≥4/6 task circuits.

5. **Identify task-specific modules**: Features unique to each task (present in
   that task but <2 others).

6. **For H2.3 specifically**: Compare feature amplitudes in the belief-update
   circuit vs false-belief circuit. Test whether the same features exist but
   are weaker, or whether different features are involved.

### Success Criteria
- Circuit overlap matrix shows block structure (some tasks cluster, others don't)
- Shared core is identifiable (≥10 features in ≥4/6 tasks)
- Communication→belief-update pathway is identifiable (even if weak)

---

## Study 3: Multi-Agent Interference

**Goal**: Test the original thesis — do representations of multiple agents interfere
with each other, and does interference predict failures?

### Pre-Registered Hypotheses

**H3.1**: When tracking beliefs of 2 agents simultaneously, the model's belief
representations are less separable (lower probe accuracy, more feature overlap)
than when tracking 1 agent.

**H3.2**: Interference (measured as representation overlap) predicts failure:
scenarios where agent representations are more entangled produce more errors.

**H3.3**: Information chain degradation (0/3 facts through 3 agents from Round 1)
is caused by agent representations overwriting each other in shared feature space.

**H3.4**: The model uses different circuits for "my belief" vs "their belief"
(self-other distinction has a mechanistic basis).

### Method

1. **Scaling experiment**: Test ToM accuracy as number of agents increases
   (1, 2, 3, 4, 5 agents), n=50 per condition.

2. **Representation analysis**:
   - Extract activations at each layer for each agent's belief state
   - Measure pairwise cosine similarity between agent representations
   - Train linear probes to distinguish "Agent A's belief" from "Agent B's belief"
   - **Critical control**: Compare probe accuracy to null distribution

3. **Interference-failure correlation**:
   - For each stimulus, compute representation overlap (cosine similarity)
   - For each stimulus, record whether the model answered correctly
   - Test correlation: does higher overlap predict errors?
   - Use logistic regression: P(error) ~ representation_overlap + n_agents

4. **Information chain experiment** (rigorous version):
   - Agent A knows fact F. Agent A tells Agent B. Agent B tells Agent C.
   - n=50 facts, varied domains
   - At each step, use circuit-tracer to see how the fact representation changes
   - Identify where information is lost: at encoding, storage, or retrieval?

5. **Self-other circuit comparison**:
   - Use circuit-tracer on "What do I think?" vs "What does X think?"
   - Compare attribution graphs
   - Identify self-specific and other-specific features

### Success Criteria
- Accuracy decreases monotonically with number of agents (H3.1)
- Representation overlap significantly predicts errors (H3.2, logistic regression p<0.01)
- Information loss is localizable to specific circuit components (H3.3)
- Self vs other circuits have ≥30% non-overlapping features (H3.4)

---

## Study 4: Cross-Model Validation

**Goal**: Determine which findings are universal properties of social cognition in
transformers vs. model-specific artifacts.

### Models

| Model | Size | Family | Notes |
|-------|------|--------|-------|
| Qwen3-4B-Instruct | 4B | Qwen | Primary, continuity with Round 1 |
| Gemma-2-2B | 2B | Google | circuit-tracer supported |
| Llama-3.2-1B | 1B | Meta | circuit-tracer supported, smallest |
| Qwen3-8B-Instruct | 8B | Qwen | Scale comparison within family |

### Pre-Registered Hypotheses

**H4.1**: The agent-binding subcircuit (from Study 1) has a homologous structure
across all 3 model families (same functional role, analogous layer position).

**H4.2**: The explicit/implicit belief gap (from Round 1 and Study 2) exists in
all instruction-tuned models, not just Qwen3.

**H4.3**: Larger models within the same family (Qwen3-4B vs 8B) will show the
same circuit topology but with stronger feature amplitudes (especially for
implicit belief update).

**H4.4**: The RoPE positional encoding is mechanistically involved in tracking
"who was present when" (following Wu et al. 2025, 2504.04238). This will be
testable by examining which features in the ToM circuit connect to positional
encoding parameters.

### Method

1. **Run Study 1 protocol** (circuit-tracer on Sally-Anne) on each model.

2. **Align features across models** using:
   - Functional alignment: Match features by behavioral effect (e.g., "ablating this
     feature breaks false-belief prediction")
   - Atlas-Alignment (Arora et al. 2025): Create concept atlas from Qwen3, align
     other models to it

3. **Compare circuit topology**: Same-layer position? Same connectivity pattern?
   Same functional roles?

4. **RoPE investigation**: For each model, identify features in the ToM circuit
   that have high gradient w.r.t. positional encoding parameters. Perturb
   positional information and measure ToM degradation.

5. **Scale comparison**: Run identical protocol on Qwen3-4B and Qwen3-8B.
   Compare circuit sparsity, feature amplitudes, and belief-update strength.

### Success Criteria
- Agent-binding features found in ≥3/4 models (H4.1)
- Explicit/implicit gap exists in ≥3/4 models (H4.2)
- 8B shows same topology but stronger features than 4B (H4.3)
- Position-connected features are in the top-20 of the ToM circuit in ≥2 models (H4.4)

---

## Study 5: Intervention and Steering

**Goal**: Use mechanistic understanding from Studies 1-4 to develop targeted
interventions that improve social cognition or prevent failures.

### Pre-Registered Hypotheses

**H5.1**: Amplifying features in the belief-update subcircuit (identified in Study 2)
will improve implicit belief tracking accuracy by ≥20 percentage points.

**H5.2**: Ablating interference-causing features (identified in Study 3) will
improve multi-agent accuracy when tracking 3+ agents.

**H5.3**: Steering vectors computed from circuit features will be more effective
than steering vectors computed from raw activations (as in Round 1's experiment 03).

**H5.4**: It is possible to selectively disable deceptive reasoning (agent
models another agent's false belief in order to exploit it) while preserving
cooperative ToM (agent models another agent's false belief in order to help).

### Method

1. **Belief-update amplification**:
   - Identify weak features in the communication→belief-update pathway (Study 2)
   - Scale their activations by 1.5x, 2x, 3x
   - Measure change in implicit belief tracking accuracy
   - Control: Scale random features by same amount

2. **Interference reduction**:
   - Identify high-overlap features from Study 3
   - Ablate or orthogonalize them
   - Measure multi-agent accuracy change
   - Control: Ablate random features

3. **Circuit-informed steering**:
   - Compute steering vectors from circuit features only (not full residual stream)
   - Compare effectiveness to full-activation steering
   - Measure side effects (does ToM steering break other capabilities?)

4. **Selective deception control** (if Studies 1-2 reveal deception-relevant features):
   - Create scenarios with cooperative ToM vs exploitative ToM
   - Identify features that distinguish cooperative from exploitative use
   - Ablate exploitative-specific features
   - Measure: does cooperative ToM survive while exploitative ToM degrades?

### Success Criteria
- Belief-update amplification shows ≥20pp improvement with ≤5pp degradation on controls (H5.1)
- Interference ablation improves 3+ agent accuracy by ≥15pp (H5.2)
- Circuit-informed steering is ≥1.5x more effective than raw steering (H5.3)
- Selective deception control shows ≥2:1 ratio of exploitative vs cooperative degradation (H5.4)

---

## Execution Order and Dependencies

```
Study 1: Circuit Atlas
   │
   ├──→ Study 2: Social Cognition Tasks (needs circuit methodology from Study 1)
   │       │
   │       ├──→ Study 5: Interventions (needs identified circuits from Studies 2-3)
   │       │
   ├──→ Study 3: Multi-Agent Interference (needs agent-binding circuit from Study 1)
   │       │
   │       └──→ Study 5: Interventions
   │
   └──→ Study 4: Cross-Model Validation (can start after Study 1, continues alongside 2-3)
```

**Phase 1** (Studies 1 + 4 in parallel): Map the basic circuit and validate cross-model
**Phase 2** (Studies 2 + 3 in parallel): Extend to full taxonomy and test interference
**Phase 3** (Study 5): Interventions informed by everything above

---

## Infrastructure Setup

### Dependencies to Add

```toml
[project]
dependencies = [
    # Core ML
    "torch>=2.0.0",
    "transformers>=4.40.0",
    "accelerate>=0.27.0",

    # Mechanistic Interpretability
    "nnsight>=0.3.0",
    "sae-lens>=6.0.0",
    "circuit-tracer",          # Anthropic's attribution graphs

    # Analysis
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "scipy>=1.11.0",
    "statsmodels>=0.14.0",

    # Visualization
    "matplotlib>=3.7.0",
    "seaborn>=0.13.0",
    "plotly>=5.18.0",

    # Utilities
    "tqdm",
    "einops",
    "huggingface-hub",
]
```

### Shared Library Structure

```
lib/
├── core/
│   ├── models.py          # Model loading for all supported models
│   ├── chat.py            # Chat-mode evaluation (locked protocol)
│   └── activations.py     # Unified activation extraction via nnsight
├── scenarios/
│   ├── generator.py       # Counterbalanced stimulus generation
│   ├── names.py           # Novel name pools
│   └── taxonomy.py        # Social cognition task definitions
├── analysis/
│   ├── circuits.py        # circuit-tracer wrappers
│   ├── patching.py        # Relevance Patching (RelP) implementation
│   ├── probing.py         # Probes with mandatory null baselines
│   ├── statistics.py      # Permutation tests, CIs, effect sizes
│   └── stability.py       # Subset replication analysis
└── utils/
    ├── config.py          # Shared configuration
    └── logging.py         # Experiment logging
```

---

## What This Plan Does NOT Include (By Design)

- **Completion mode experiments** — Round 1 showed these are misleading for instruct models
- **Single-head stories** — "L12H0 is the ToM head" is not a circuit-level explanation
- **Claims without cross-model validation** — No model-specific findings presented as general
- **Behavioral-only findings** — Every behavioral claim must have mechanistic evidence
- **Post-hoc hypothesis generation presented as confirmation** — Exploratory and confirmatory
  analyses are clearly separated in each study
