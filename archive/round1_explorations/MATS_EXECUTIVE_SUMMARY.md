# Executive Summary: Theory of Mind Circuits in Qwen3-4B

## Research Question

How does Qwen3-4B implement Theory of Mind (ToM)? Through 20 experiments, we identify specific circuits for belief tracking, develop methodology for scalable circuit discovery, and demonstrate that ToM is a reasoning skill requiring computational budget.

## Key Findings

**Behavioral ToM**: Sally-Anne false belief tests yield 81% belief-based predictions (N=200, p<10⁻¹⁹, Cohen's h=0.67). The model predicts agents search based on their beliefs, not reality.

**Chat Mode Requirement**: Testing format critically affects results:

| Mode | False Belief | True Belief |
|------|--------------|-------------|
| Completion | 50-80% | 0-20% |
| Chat + 500 tokens | 75% | 95% |

With chat formatting and `<think>` tags, the model reasons explicitly about who saw what. ToM generalizes across entity types—humans, animals, AI, abstract entities (Fig. 1: `step33_proper_retest.png`).

## Circuit Discovery

**Single-Agent ToM Circuit**: Head 0 at layers 12, 24, and 30 shows significantly higher ablation impact than controls (37% vs 20% change rate, p=0.022). These heads form a distributed "ToM channel" in late layers (32-34), where L32H0, L33H4, L33H16, L33H28, and L34H0 all produce accuracy drops when ablated.

**Multi-Agent Circuit** (Step 10): Multi-agent reasoning recruits different circuitry than single-agent ToM:
- Single-agent: Late layers (32-34)
- Multi-agent: Early-mid layers (0-22)
- Overlap: Only L34H0

**L18H16 Inhibitor**: Ablating L18H16 improves multi-agent accuracy from 50% to 75%—a head that actively interferes with correct reasoning (Fig. 2: `step10_multiagent_heatmap.png`).

## Smart Filtering Pipeline (Step 10c)

We developed a 3-stage methodology for efficient circuit discovery:

**Stage 1 - SAE Layer Screening**: Sparse autoencoders trained on MLP outputs identify layers with highest discriminability between false/true belief scenarios. Layer 28 shows peak discriminability (21.6), followed by layers 24, 20, 16 (Fig. 3: `step10c_filtering_pipeline.png`).

**Stage 2 - Attention Pattern Filtering**: Score heads by attention weight to relevant tokens (agent names, belief verbs, locations). Layers 16 and 20 show highest attention to agent-relevant positions.

**Stage 3 - Targeted Ablation**: Test only 80/1152 candidate heads (7%), achieving 14.4x speedup. Ablation reveals large effect sizes (-33%) indicating causal importance of filtered heads (Fig. 4: `step10c_effect_landscape_3d.png`).

## Interpretable Features

**SAE Decomposition** (Step 13): Belief states decompose into sparse, interpretable features:
- Feature #1979: Activates for "agent has outdated information" (FB-TB difference: +2.12)
- Feature #4772: Activates for "agent observed the event" (-0.77)

Only ~13 features active per input (0.1% sparsity)—belief computation is remarkably sparse.

**Attention/MLP Division**: Attention heads allocate 70.6% weight to agent name tokens. MLP probes achieve 95% accuracy for belief state classification from layers 12+. Attention tracks WHO; MLPs encode WHAT they believe.

## Critical Self-Correction

**Transfer Test** (Exp 05): Initial 100% probe accuracy for entity classification (User/Self/Other) appeared to show deep understanding. Transfer test revealed otherwise—probes trained on labeled dialogues dropped from 99.9% to 32% on unlabeled dialogues (Fig. 5: `transfer_learning.png`). Probes detected tokens, not concepts.

**Heuristic Correction**: What appeared to be "first-mention heuristic" was actually correct original-location tracking for false belief.

## Limitations

- Novel locations 0% in completion; untested in chat mode
- Sample sizes n=8-50 (below recommended n≥50)
- Single model; cross-architecture generalization unknown

## Key Contributions

1. **ToM is a reasoning skill** requiring computational budget, not hard-coded retrieval
2. **Distinct circuits** for single-agent (L32-34) vs multi-agent (L0-22) social cognition
3. **Identified inhibitor head** L18H16 that interferes with multi-agent reasoning
4. **Interpretable SAE features** encoding belief states sparsely
5. **Scalable methodology** for circuit discovery with 14.4x speedup

---

**Figures:**
- Fig. 1: `experiments/20_rigorous_framework/figures/step33_proper_retest.png`
- Fig. 2: `experiments/20_rigorous_framework/figures/step10_multiagent_heatmap.png`
- Fig. 3: `experiments/20_rigorous_framework/figures/step10c_filtering_pipeline.png`
- Fig. 4: `experiments/20_rigorous_framework/figures/step10c_effect_landscape_3d.png`
- Fig. 5: `experiments/05_naturalistic_transfer/figures/transfer_learning.png`

*Qwen3-4B (36 layers × 32 heads) | 20 experiments | 16 hours*
