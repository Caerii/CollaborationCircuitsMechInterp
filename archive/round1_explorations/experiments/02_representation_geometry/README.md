# Experiment 02: Representation Geometry

## Research Question

**How do User, Self, and Other representations relate geometrically in activation space?**

Beyond classification accuracy, we want to understand the structure of how entity representations are organized. Are they clustered? How far apart? Do they form meaningful geometric relationships?

---

## Methodology

### Metrics Computed

1. **Cosine Similarity** - Pairwise similarity between entity class means
2. **Angular Separation** - Angle (in degrees) between entity centroid vectors
3. **Discriminability Ratio** - Between-class distance / Within-class variance
4. **PCA Variance** - How many components needed to explain 95% variance

### Approach
- Compute mean activation vector for each entity type at each layer
- Calculate pairwise cosine similarities
- Convert to angular separation (arccos)
- Analyze how separation evolves across layers

---

## Results

### Cosine Similarity Across Layers

| Layer | User↔Self | User↔Other | Self↔Other |
|-------|-----------|------------|------------|
| 0     | 0.922     | 0.926      | **0.998**  |
| 8     | 0.854     | 0.862      | 0.957      |
| 16    | 0.846     | 0.826      | 0.948      |
| **20**| **0.844** | **0.818**  | **0.934**  |
| 24    | 0.868     | 0.851      | 0.944      |
| 35    | 0.925     | 0.928      | 0.978      |

### Angular Separation (Degrees)

| Layer | User-Self | User-Other | Self-Other |
|-------|-----------|------------|------------|
| 0     | 22.6°     | 22.2°      | **4.0°**   |
| 8     | 31.2°     | 30.4°      | 16.8°      |
| 16    | 32.3°     | 34.3°      | 18.5°      |
| **20**| **32.4°** | **35.1°**  | **20.8°**  |
| 24    | 29.7°     | 31.7°      | 19.3°      |
| 35    | 22.3°     | 21.9°      | 12.0°      |

### Discriminability Ratio

| Layer | Ratio | Interpretation |
|-------|-------|----------------|
| 0     | 0.75  | Good separation |
| 8     | 0.72  | Similar |
| 16    | 0.83  | Better |
| **20**| **0.86** | **Best separation** |
| 24    | 0.84  | Good |
| 35    | 0.85  | Good |

---

## Key Findings

### 1. The U-Shaped Separation Curve

```
Layer 0   →   Layer 20   →   Layer 35
   ↓            ↓              ↓
 Similar    DIVERGE       Converge
  (4°)       (21°)          (12°)
     \_______/\______/
        PROCESSING
```

Entity representations follow a U-shaped pattern:
- **Early layers**: Self and Other nearly identical (4° apart)
- **Middle layers**: Maximum separation (21° at layer 20)
- **Late layers**: Partial convergence (12° at output)

### 2. Human vs AI Distinction is Primary

At ALL layers, User is ~22-35° away from both Self and Other.
The model maintains a strong Human/AI boundary throughout processing.

### 3. Self↔Other Distinction is LEARNED

The model doesn't inherently distinguish Self from Other - it LEARNS to do so during middle-layer processing. This is the key finding:

> **At input (layer 0), Self and Other are nearly identical. The model actively refines these representations to distinguish between AI agents.**

---

## Visualization

See: `../../results/experiment_results.png` (bottom-right panel: "Entity Pair Similarity Across Layers")

The teal line (Self↔Other) clearly shows the U-shape: starting high (0.998), dropping to minimum at layer 20 (0.934), then rising again.

---

## Implications

1. **For AI Safety**: If Self/Other distinction is learned mid-network, interventions at these layers might affect multi-agent behavior

2. **For Multi-Agent Systems**: The model has a natural "Human/AI" boundary but a learned "Self/Other" distinction

3. **For Mechanistic Interpretability**: Layer 20 appears to be a critical layer for entity processing

---

## Files

- `../../results/experiment_summary.json` - Full separation analysis
- `../../results/advanced_results.json` - Detailed geometry metrics
- `../../results/steering_analysis.png` - Angle visualization

---

## Next Steps

This geometric analysis is correlational. Experiment 03 (Causal Steering) tests whether we can actually manipulate these representations to change model behavior.

