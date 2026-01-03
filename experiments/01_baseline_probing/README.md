# Experiment 01: Baseline Probing

## Research Question

**Can linear probes classify entity type (User/Self/Other) from model activations?**

If a simple linear classifier can accurately predict who is speaking from the model's internal representations, it provides strong evidence that entity information is encoded in a linearly separable way.

---

## Methodology

### Data
- 200 synthetic multi-party dialogues
- 1,186 individual turns (508 User, 332 Self, 346 Other)
- Dialogues contain explicit role labels ("User:", "You:", "Helper:")

### Approach
1. Extract hidden state activations at the last token of each dialogue turn
2. Sample 10 layers across the model: [0, 4, 8, 12, 16, 20, 24, 28, 32, 35]
3. Train logistic regression classifiers (linear probes) on each layer
4. Evaluate with 80/20 train/test split

### Metrics
- Test accuracy per layer
- Per-class accuracy (User, Self, Other)
- 5-fold cross-validation accuracy

---

## Results

### Probe Accuracy by Layer

| Layer | Train Acc | Test Acc | CV Mean ± Std |
|-------|-----------|----------|---------------|
| 0     | 99.8%     | 100.0%   | 99.6% ± 0.5%  |
| 4     | 100.0%    | 100.0%   | 100.0% ± 0.0% |
| 8     | 100.0%    | 100.0%   | 100.0% ± 0.0% |
| 12    | 100.0%    | 100.0%   | 100.0% ± 0.0% |
| 16    | 100.0%    | 100.0%   | 100.0% ± 0.0% |
| 20    | 100.0%    | 100.0%   | 100.0% ± 0.0% |
| 24    | 100.0%    | 100.0%   | 100.0% ± 0.0% |
| 28    | 100.0%    | 100.0%   | 100.0% ± 0.0% |
| 32    | 100.0%    | 100.0%   | 100.0% ± 0.0% |
| 35    | 100.0%    | 99.2%    | 99.9% ± 0.2%  |

### Per-Class Accuracy (All Layers)
- **User**: 100.0%
- **Self**: 100.0% (99.2% at layer 35)
- **Other**: 100.0% (98.6% at layer 35)

---

## Key Findings

1. **Entity information is perfectly decodable** - Linear probes achieve ~100% accuracy across all layers

2. **Information present from layer 0** - Unlike some features that emerge in later layers, entity identity is encoded immediately

3. **Slight degradation at output** - Layer 35 shows minor accuracy drop (99.2%), suggesting the model optimizes for generation rather than entity tracking at output

4. **All classes equally separable** - No class is harder to detect than others

---

## Interpretation

The near-perfect probe accuracy demonstrates that the model encodes entity identity in a linearly separable way throughout the network. However, this doesn't tell us:
- Whether this is semantic understanding or lexical pattern matching
- How the representations differ geometrically
- Whether we can causally manipulate entity perception

These questions are addressed in subsequent experiments.

---

## Files

- `../../results/probe_results.json` - Detailed probe metrics
- `../../results/probes.pt` - Trained probe weights
- `../../results/experiment_results.png` - Visualization

---

## Citation

```
Experiment conducted as part of: 
"Mechanistic Interpretability of Multi-Agent LLM Collaboration"
MATS 10.0 Application Project
```

