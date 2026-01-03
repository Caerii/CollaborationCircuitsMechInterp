# Step 1: Baseline ToM Performance

**Script**: `scripts/step1_baseline_tom.py`  
**Results**: `results/step1_baseline_tom.json`  
**Figures**: `figures/step1_accuracy_by_format.png`

---

## Key Finding

**Qwen3-4B is an instruction-tuned reasoning model** - it needs chat format with reasoning space to perform ToM tasks.

---

## Results

### Completion Mode (raw next-token prediction)
| Condition | Accuracy | Interpretation |
|-----------|----------|----------------|
| False Belief | 16.7% | Worse than chance |
| True Belief | 83.3% | Tracks reality, not belief |

### Chat Mode (with `<think>` reasoning)
| Condition | Accuracy | Notes |
|-----------|----------|-------|
| False Belief | 66.7% | Much improved |
| True Belief | 66.7% | Consistent |

---

## Heuristic Comparison

| Method | Accuracy |
|--------|----------|
| **First-mention heuristic** | **100%** |
| Chat mode | 66.7% |
| Completion mode | 50% |

**Warning**: First-mention heuristic beats the model. This means our scenarios may be confounded - need better counterbalancing.

---

## Timing Profile

| Mode | Time per Scenario | Tokens/sec |
|------|-------------------|------------|
| Completion | 0.1s | N/A (single forward) |
| Chat | 16.8s | 11.7 tok/s |

The chat mode is slow because the model generates long `<think>` chains before answering.

---

## Implications

1. **Don't test in completion mode** - the model isn't designed for it
2. **Allow sufficient tokens** - 200+ for reasoning
3. **Heuristics are strong** - need discriminating scenarios

