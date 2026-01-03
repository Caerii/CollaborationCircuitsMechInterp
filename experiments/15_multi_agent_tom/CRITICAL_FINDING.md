# CRITICAL FINDING: Model Requires EXPLICIT Belief Cues

## The Discovery

Running expanded tests on Qwen3-4B revealed a nuanced pattern in ToM performance.

### Test Results Summary

| Test | Belief Cues | Accuracy | 
|------|-------------|----------|
| Standard Sally-Anne | Explicit "doesn't know" | 91% |
| Swapped Order (reality first) | Explicit "doesn't know" | 86% |
| Dialogue (unchanged agent) | Implicit from narrative | **98%** |
| Dialogue (updated agent) | Implicit from narrative | **2%** |
| Divergent beliefs | Explicit per-agent beliefs | 62% |

## Key Insight: EXPLICIT vs IMPLICIT Belief Cues

### When Beliefs Are EXPLICIT: High Performance

Prompts that explicitly state beliefs work well:
- "Sally last saw the ball in the basket and does not know it was moved" → 86-91%
- "Alice believes the ball is in X. Bob believes it is in Y." → 62%

### When Beliefs Must Be INFERRED: Complete Failure

Prompts requiring belief inference from narrative fail:
```
Alice tells Bob: ball in drawer.
Bob leaves.
Carol tells Alice: ball moved to basket.
Bob returns.
Where will Alice look?
```
- Model should infer: Alice LEARNED it moved → Alice believes basket
- Model actually predicts: drawer (original location)
- Result: 2% correct for updated agents!

## The Pattern

| Explicit Statement | Infer from Narrative |
|-------------------|----------------------|
| "Agent believes X" → Uses X | "Agent was told Y while agent was away" → Uses ORIGINAL |
| 86-91% accuracy | 2% accuracy |

The model can FOLLOW explicit belief statements but cannot TRACK belief updates through narrative.

## Implications

### 1. Not a Simple Position Heuristic
The 86% on swapped-order rules out "first-mentioned-location" as the sole explanation.

### 2. Shallow Belief Representation
Model represents explicitly stated beliefs but doesn't maintain them through narrative transformations.

### 3. The Circuit Finding (L12H0, L23H0) Needs Reinterpretation
These heads may implement "explicit belief lookup" rather than "belief tracking from context."

### 4. Dialogue Failure is the Critical Test
The 98% vs 2% asymmetry in dialogue tracking is the clearest evidence of the model's limitation.

## What This Means for MATS

**Strong claim (supported):** Model encodes explicit beliefs.
**Weak claim (unsupported):** Model tracks belief updates through narrative.

The circuit we found (L12H0, L23H0) may be:
- A "belief statement parser" (explicit beliefs → representation)
- NOT a "belief tracker" (events → updated beliefs)

## Next Steps

1. Test if ablating L12H0/L23H0 breaks explicit belief following
2. Investigate what computation would be needed for narrative belief tracking
3. Check if any heads correlate with narrative position tracking

