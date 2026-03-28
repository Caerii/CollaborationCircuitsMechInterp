# Experiment 20: Rigorous Collaboration Circuits Investigation

## Research Questions

1. **Does the model have genuine Theory of Mind (ToM)?**
   - First-order: What does X believe?
   - Second-order: What does X think Y believes?
   - Multi-domain: Does it generalize beyond object locations?

2. **What circuits implement ToM?**
   - Which attention heads are critical?
   - Where does the model "decide" (logit lens)?
   - What signals enable/inhibit correct ToM?

3. **Can we causally manipulate ToM?**
   - Does ablating heads change behavior?
   - Can we inject "belief update" signals?
   - Can we steer between belief states?

4. **Does ToM transfer to multi-agent scenarios?**
   - Negotiation: tracking multiple agents' goals
   - Deception: detecting lies vs truth
   - Cooperation vs defection: resource allocation

## Methodology Requirements

- **Sample size**: n ≥ 50 per condition
- **Controls**: True-belief, heuristic baselines
- **Statistics**: Wilson CIs, effect sizes (Cohen's h)
- **Validation**: Cross-model, cross-prompt-style

---

## Steps

### Phase 1: Establish Baselines

| Step | Script | Question | Status |
|------|--------|----------|--------|
| 1 | `step1_baseline_tom.py` | What's base ToM accuracy across formats? | |
| 2 | `step2_heuristic_baselines.py` | What do heuristics predict? | |
| 3 | `step3_higher_order_tom.py` | Does 2nd/3rd order ToM work? | |

### Phase 2: Circuit Discovery

| Step | Script | Question | Status |
|------|--------|----------|--------|
| 4 | `step4_logit_lens.py` | WHERE does the model decide? | |
| 5 | `step5_head_ablation_sweep.py` | WHICH heads matter? | |
| 6 | `step6_mlp_analysis.py` | WHICH neurons differ between conditions? | |

### Phase 3: Causal Testing

| Step | Script | Question | Status |
|------|--------|----------|--------|
| 7 | `step7_signal_injection.py` | Can we inject "belief update" signals? | |
| 8 | `step8_head_amplification.py` | Does amplifying inhibitors make ToM worse? | |
| 9 | `step9_causal_steering.py` | Can we steer between belief states? | |

### Phase 4: Multi-Agent Extension

| Step | Script | Question | Status |
|------|--------|----------|--------|
| 10 | `step10_multi_agent_tom.py` | Does ToM work with multiple agents? | |
| 11 | `step11_negotiation_circuits.py` | What circuits track competing goals? | |
| 12 | `step12_deception_detection.py` | Can we find lie detection circuits? | |

### Phase 5: Cross-Validation

| Step | Script | Question | Status |
|------|--------|----------|--------|
| 13 | `step13_cross_model.py` | Do findings generalize across models? | |
| 14 | `step14_robustness.py` | Do findings hold across prompt styles? | |
| 15 | `step15_final_figures.py` | Publication-ready visualizations | |

---

## Pre-registered Hypotheses

### H1: ToM Accuracy
- **Prediction**: Chat format with reasoning will achieve >80% on first-order FB
- **Baseline**: Raw completion will achieve <50% (chance or heuristic-level)

### H2: Circuit Location
- **Prediction**: Critical heads will be in layers 15-25 (middle layers)
- **Baseline**: Random heads should not affect accuracy

### H3: Causal Manipulation
- **Prediction**: Ablating critical heads will drop accuracy by >20%
- **Control**: Ablating random heads will change accuracy <5%

### H4: Signal Injection
- **Prediction**: Injecting "belief update" signal will increase accuracy on corrupted prompts
- **Effect size**: Cohen's h > 0.3 (small-medium effect)

---

## Output Structure

```
results/
├── step1_baseline_tom.json
├── step2_heuristic_baselines.json
├── ...
└── summary_statistics.json

figures/
├── step1_accuracy_by_format.png
├── step4_logit_lens_evolution.png
├── step5_head_importance_heatmap.png
├── ...
└── publication_figure_main.png
```

