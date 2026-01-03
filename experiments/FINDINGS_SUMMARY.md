# Scientific Findings Summary

## Executive Summary

This research investigated how language models internally represent different participants (User, Self, Other) in multi-agent dialogue. We made both **positive findings** and a **critical control finding**:

### Initial Findings (Experiments 01-04)
1. ✅ **Entity representations are perfectly linearly separable** (100% probe accuracy)
2. ✅ **Self and Other start nearly identical** (4° apart at layer 0)
3. ✅ **Peak separation occurs at layer 20** (21° apart)
4. ✅ **Self↔Other can be causally manipulated** (~60% flip rate)

### Critical Control Finding (Experiment 05) ⚠️
5. ❌ **Transfer to naturalistic data FAILS** (32% accuracy = chance level)

**The encoding is LEXICAL, not SEMANTIC!**

---

## The Critical Control

### Experiment 05: Naturalistic Transfer

```
┌────────────────────────────────────────────────────────────────┐
│                        KEY RESULT                               │
│                                                                │
│   Within-domain (labeled):     99.9% accuracy                  │
│   Transfer (naturalistic):     32.0% accuracy  ← CHANCE!       │
│   Transfer gap:                +68%                            │
│                                                                │
│   When explicit labels are removed, probes COMPLETELY FAIL.    │
└────────────────────────────────────────────────────────────────┘
```

### What This Means

The 100% probe accuracy we found was detecting **label tokens**, not **entity understanding**:

| What We Thought | What's Actually Happening |
|-----------------|--------------------------|
| Model understands "who" is speaking | Model encodes literal "User:", "Self:", "Other:" tokens |
| U-shaped curve = entity processing | U-shaped curve = token processing |
| Geometric clusters = entity representations | Geometric clusters = token representations |

---

## Revised Interpretation of All Findings

### Experiment 01: Baseline Probing
- **Original**: "100% accuracy proves entity encoding"
- **Revised**: 100% accuracy proves **token encoding** - the model perfectly encodes whether "User:", "You:", or "Helper:" appeared

### Experiment 02: Representation Geometry
- **Original**: "U-shaped curve shows learned Self/Other distinction"
- **Revised**: U-shaped curve shows how the model processes **different label tokens** through layers

### Experiment 03: Causal Steering
- **Original**: "Can manipulate entity perception"
- **Revised**: Can manipulate **token-associated** representations, not abstract entity concepts

### Experiment 04: 3D Visualizations
- **Original**: Shows entity clusters in activation space
- **Revised**: Shows **token** clusters - still beautiful and instructive!

---

## The U-Shaped Curve is Still Real

The phenomenon is real - it just describes something different than we thought:

```
"User:" token processing vs "You:" token processing:

Layer 0:   Nearly identical token embeddings
Layer 20:  Maximum token-specific processing
Layer 35:  Convergence for next-token prediction
```

This is still scientifically interesting! It tells us:
- How the model differentiates between different role labels
- The processing timeline for conversational markers
- Which layers are critical for label processing

---

## Scientific Implications

### For AI Safety 🔒

| Aspect | Implication |
|--------|-------------|
| **Good** | Models don't have deep "theory of mind" |
| **Good** | Multi-agent behavior is more predictable |
| **Caution** | Easy to fool with label manipulation |
| **Caution** | No robust entity tracking without labels |

### For Multi-Agent Systems 🤖

- **Explicit labeling is CRITICAL**
- Can't rely on "natural conversation flow"
- Systems need clear speaker identification
- Context-based inference doesn't work (in Qwen3-4B)

### For Mechanistic Interpretability 🔬

- **Always run transfer controls!**
- Probe accuracy ≠ semantic understanding
- Token encoding ≠ concept encoding
- Lexical confounds are pervasive

---

## Quantitative Summary

| Experiment | Key Metric | Value |
|------------|------------|-------|
| 01: Probing | Classification accuracy | 100% |
| 02: Geometry | Self-Other angle at L20 | 21° |
| 03: Steering | Self→Other flip rate | ~60% |
| 04: Visualization | Animated GIFs created | 7 |
| **05: Transfer** | **Transfer accuracy** | **32% (chance!)** |

---

## Conclusion

This research demonstrates the importance of **transfer controls** in mechanistic interpretability. Without Experiment 05, we would have concluded that models have sophisticated entity tracking. The transfer test revealed that the "entity representations" are actually **label token encodings**.

> **Bottom Line**: Qwen3-4B does NOT semantically understand who is speaking in a conversation. It pattern-matches explicit labels. This has significant implications for multi-agent AI systems.

---

## Future Directions

1. **Other models**: Do larger models (70B+) show semantic transfer?
2. **Fine-tuning**: Can we train models for semantic entity tracking?
3. **Attention analysis**: How do attention patterns differ for labeled vs unlabeled data?
4. **Longer context**: Does semantic understanding emerge with more conversation history?
