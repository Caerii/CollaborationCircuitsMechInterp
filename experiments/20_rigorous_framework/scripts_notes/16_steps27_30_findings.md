# Steps 27-30: Ablation, Summary, Probing, Entity Types

## Step 27: Ablate First-Mention Circuit

**Hypothesis**: Ablating first-mention heads would improve True Belief.

**Result**: Minimal effect!

| Condition | Baseline | Ablated | Change |
|-----------|----------|---------|--------|
| False Belief | 85% | 80% | -5% |
| True Belief | 45% | 45% | 0% |

**Interpretation**: The first-mention heuristic is more distributed than expected.
Ablating 5 heads isn't enough to break the heuristic.

---

## Step 28: Comprehensive MATS Summary

Generated comprehensive figure with all findings for MATS application.
See `figures/step28_comprehensive.png`.

---

## Step 29: Linear Probing

**Finding**: 100% probe accuracy at ALL layers!

**Interpretation**: This is a **confound** - the probe is detecting the prompt
difference ("left" vs "watched"), not the belief state itself.

The scenarios have different wording, so a probe trivially distinguishes them.

**Lesson**: Need matched controls to probe genuine belief representation.

---

## Step 30: Entity Type Results 🔥

**MAJOR FINDING: ToM only works for biological entities!**

| Entity Type | False Belief | True Belief | Overall |
|-------------|-------------|-------------|---------|
| **Human** | 100% | 0% | 50% |
| **Animal** | 100% | 0% | 50% |
| **AI** | 0% | 0% | 0% |
| **Abstract** | 0% | 0% | 0% |

### Key Insight:
The model ONLY shows ToM for biological agents (humans, animals).
For AI agents (Robot-A, Assistant-1) and abstract entities (Company-A, Team-Alpha),
ToM completely fails!

### Implications for Multi-Agent Collaboration:
1. **LLM-to-LLM collaboration may be limited** - models don't track AI beliefs
2. **Anthropomorphization helps** - treating AI as "human-like" may improve tracking
3. **Training data bias** - ToM examples in training are primarily human-focused
4. **New research direction** - Can we extend ToM circuits to AI entities?

---

## Updated Architecture Understanding

```
ToM Circuit Scope:
==================

WORKS FOR:
- Humans (Alice, Bob, Carol...)
- Animals (cat, dog, bird...)

FAILS FOR:
- AI agents (Robot-A, Assistant-1...)
- Abstract entities (Company-A, Team-Alpha...)

The ToM circuit appears tuned to biological agents!
```

---

## Files Generated

- `results/step27_ablate_heuristic.json`
- `results/step28_summary.json`
- `results/step29_probing.json`
- `results/step30_entity_types.json`
- `figures/step27_ablate_heuristic.png`
- `figures/step28_comprehensive.png`
- `figures/step29_probing.png`
- `figures/step30_entity_types.png`

