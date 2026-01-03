# MAJOR DISCOVERY: Distributed Inhibitory Network

## Executive Summary

**L24H29 is NOT alone!** There is an entire **inhibitory network** spanning layers 15-25 that suppresses belief update inference. The strongest inhibitor is actually **L17H4** (+48% boost when ablated), not L24H29.

## Key Results

### Top 15 Inhibitory Heads

| Rank | Head | Ablation Accuracy | Boost | Layer Region |
|------|------|-------------------|-------|--------------|
| 1 | **L17H4** | 68% | **+48%** | Early-mid |
| 2 | **L15H12** | 56% | +36% | Early |
| 3 | **L24H29** | 52% | +32% | Mid-late |
| 4 | **L22H11** | 48% | +28% | Mid |
| 5 | **L23H0** | 44% | +24% | Mid |
| 6 | **L20H28** | 44% | +24% | Mid |
| 7 | L23H25 | 40% | +20% | Mid |
| 8 | L23H26 | 40% | +20% | Mid |
| 9 | L17H24 | 40% | +20% | Early-mid |
| 10 | L18H8 | 40% | +20% | Early-mid |
| 11 | L20H0 | 40% | +20% | Mid |
| 12 | L22H0 | 36% | +16% | Mid |
| 13 | L22H10 | 36% | +16% | Mid |
| 14 | L23H30 | 36% | +16% | Mid |
| 15 | L15H24 | 36% | +16% | Early |

**Baseline (no ablation): 20%**

### Inhibitory Network Architecture

```
Layer Distribution of Inhibitory Heads (>10% boost)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

L15: ██████████████████████████  H4, H12*, H24  (3 heads)
L16: ████████████████████████████  H4, H12, H16, H28  (4 heads)
L17: ████████████████████████████████████████████████  H4**, H24  (2 heads - STRONGEST!)
L18: ████████████████████████  H8, H28  (2 heads)
L19: ████████████████  H20  (1 head)
L20: ████████████████████████████  H0, H28*  (2 heads)
L21: ████████████████  H0  (1 head)
L22: ██████████████████████████████  H0, H10, H11*, H16  (4 heads)
L23: ██████████████████████████████  H0*, H25, H26, H30  (4 heads)
L24: ██████████████████████████  H29*  (1 head)
L25: ████████████  H31  (1 head)

* = >20% boost
** = >40% boost (STRONGEST)
```

### Critical Enabling Heads (Ablation HURTS)

| Head | Ablation Accuracy | Impact |
|------|-------------------|--------|
| L22H2 | 8% | -12% |
| L22H9 | 8% | -12% |
| L23H1 | 8% | -12% |
| L17H16 | 8% | -12% |
| L18H0 | 8% | -12% |

These are likely the **processing** heads that actually compute ToM.

## Interpretation

### 1. Multi-Stage Inhibition

The network spans THREE distinct regions:
- **Early (L15-18)**: Contains the STRONGEST inhibitor (L17H4)
- **Mid (L19-23)**: Dense cluster of inhibitors
- **Late (L24-25)**: Final inhibition stage

### 2. Circuit Model Update

```
                          INHIBITORY NETWORK
    ┌──────────────────────────────────────────────────────┐
    │                                                      │
    │   L15-18          L19-23           L24-25           │
    │   ┌─────┐         ┌─────┐          ┌─────┐          │
    │   │L17H4│─────────│L22H11│─────────│L24H29│          │
    │   │+48% │         │+28%  │         │+32%  │          │
    │   └──┬──┘         └──┬───┘         └──┬───┘          │
    │      │               │                │              │
    │      ▼               ▼                ▼              │
    │   BLOCKS          BLOCKS           BLOCKS            │
    │      │               │                │              │
    └──────┼───────────────┼────────────────┼──────────────┘
           │               │                │
           ▼               ▼                ▼
    ┌──────────────────────────────────────────────────────┐
    │              BELIEF UPDATE CIRCUIT                   │
    │   (L22H2, L22H9, L23H1, L17H16, L18H0 - enablers)   │
    └──────────────────────────────────────────────────────┘
                           │
                           ▼
                  BELIEF REPRESENTATION
```

### 3. Why Multiple Inhibitors?

Hypotheses:
1. **Redundancy**: Robust suppression of automatic belief updates
2. **Staged Processing**: Different inhibitors handle different aspects
3. **Feature-Specific**: Each head inhibits based on different cues

### 4. L17H4: The Strongest

L17H4 is in **layer 17** (early-mid network), suggesting:
- Inhibition starts EARLY in processing
- By the time L24H29 acts, much inhibition already happened
- Multi-stage "gating" prevents belief updates

## Implications for Multi-Agent ToM

1. **To enable ToM** without prompting: Need to ablate MULTIPLE heads
2. **Combinatorial intervention**: Which subset gives best results?
3. **Training implications**: This network was learned during training - why?

## Combined Ablation Results

### BREAKTHROUGH: 92.5% ToM Without Prompting!

| Configuration | Accuracy | Boost |
|---------------|----------|-------|
| Baseline (no ablation) | 22.5% | - |
| L17H4 alone | 67.5% | +45% |
| L17H4 + L15H12 | 80.0% | +57.5% |
| L17H4 + L15H12 + L24H29 | **90.0%** | +67.5% |
| All top 6 | **92.5%** | +70% |

### The Magic Trio

These 3 heads, when ablated together, unlock 90% ToM accuracy:

```
   Layer 15          Layer 17          Layer 24
      │                 │                 │
   ┌──▼──┐          ┌──▼──┐          ┌──▼──┐
   │ H12 │          │ H4  │          │ H29 │
   │+30% │          │+45% │          │+30% │
   └─────┘          └─────┘          └─────┘
      │                 │                 │
      └────────────┬────┴────────────────┘
                   │
                   ▼
            COMBINED: 90%
```

### L17H4: The Master Inhibitor

Every top-performing combination includes L17H4:
- Without L17H4: Best combo achieves 72.5%
- With L17H4: Best combo achieves 90%+

This head is **necessary** for strong inhibition.

## Implications

### 1. ToM is Latent, Not Missing

The model CAN do Theory of Mind - it achieves 92.5% when we remove the inhibitors.
This is not a capability gap, it's **active suppression**.

### 2. Surgical Intervention is Possible

We can enable ToM by ablating just 3 specific heads:
- L17H4, L15H12, L24H29

This is a **precise, interpretable intervention**.

### 3. Training Created This Suppression

These heads learned to suppress belief update inference during training.
Why? Possible reasons:
- Avoid over-generalization (not all communication updates beliefs)
- Default to "safe" literal interpretation
- Statistical regularities in training data

### 4. Multi-Agent AI Safety Implications

If we want AI systems to correctly track beliefs in multi-agent settings:
- **Option A**: Use prompting (CoT, few-shot)
- **Option B**: Ablate inhibitory network
- **Option C**: Fine-tune to reduce inhibition

## Next Steps

1. **Attention analysis** on L17H4 (the strongest)
2. **Path patching**: Trace how inhibition flows between heads
3. **Cross-model validation**: Is this network universal?
4. **Test on real multi-agent scenarios**

## Raw Numbers

- Total heads tested: 336
- Baseline accuracy: 22.5%
- Best single ablation: L17H4 -> 67.5% (+45%)
- Best 3-head combo: 90.0%
- Best 6-head combo: 92.5%
- Top 25 inhibitors found
- Search time: 15 minutes

