# Experiment 03: Causal Steering

## Research Question

**Can we causally manipulate entity representations to change classification?**

Geometric analysis (Experiment 02) shows entities are separated in activation space. But correlation ≠ causation. This experiment tests whether adding/subtracting entity direction vectors actually changes how the model represents entities.

---

## Methodology

### Steering Vector Extraction

For each layer, we compute:
1. **Entity Centroids**: Mean activation vector for User, Self, Other
2. **Steering Vectors**: Normalized difference between centroids
   - `User→Self` = normalize(Self_centroid - User_centroid)
   - `User→Other` = normalize(Other_centroid - User_centroid)
   - `Self→Other` = normalize(Other_centroid - Self_centroid)

### Steering Test

1. Train a classifier on original activations
2. Take activations from entity A
3. Add steering vector × strength
4. Measure "flip rate" - how often classifier now predicts entity B

### Steering Strengths Tested
[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

---

## Results

### Steering Effectiveness at Layer 20 (Peak Separation)

| Direction | Strength=1.0 | Strength=5.0 | Strength=10.0 |
|-----------|--------------|--------------|---------------|
| User → Self | ~0% | ~2% | ~7% |
| User → Other | ~0% | ~3% | ~5% |
| Self → User | ~0% | ~2% | ~5% |
| **Self → Other** | ~0% | ~8% | **~60%** |
| Other → User | ~0% | ~3% | ~7% |
| **Other → Self** | ~0% | ~10% | **~55%** |

### Key Observation

**Self↔Other steering is MOST effective!**

At strength 10.0:
- Self→Other achieves ~60% flip rate
- Other→Self achieves ~55% flip rate
- User→Self/Other only achieves ~5-7% flip rate

This aligns perfectly with our geometric finding that Self and Other are closest to each other.

---

## Interpretation

### Why Self↔Other Steering Works Best

1. **Geometry**: Self and Other centroids are only ~21° apart at layer 20, while User is ~32-35° from both
2. **Magnitude**: The steering vector between Self and Other is smaller, so the same multiplier produces more targeted movement
3. **Semantic Similarity**: Both Self and Other are "AI agents", so their representation subspace overlaps more

### Why User Steering is Harder

The User representation lives in a distinct region of activation space (the "human" subspace). Moving from User to Self/Other requires crossing a larger gap.

---

## Causal Evidence Summary

| Claim | Evidence Type | Strength |
|-------|---------------|----------|
| Entity info is encoded | Probing | Strong (correlational) |
| Representations are geometrically separated | Similarity analysis | Strong (correlational) |
| **Self↔Other can be causally manipulated** | **Steering** | **Moderate (causal)** |
| User↔AI boundary is robust | Steering failure | Moderate (causal) |

---

## Visualization

See: `figures/steering_analysis.png`

---

## Files

- `figures/steering_analysis.png` - Steering effectiveness visualization
- `../../results/advanced_results.json` - Full steering data
