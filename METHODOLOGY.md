# Methodology Protocol

Locked methodology for all Round 2 experiments. No deviations without explicit
justification documented in the study's README.

## 1. Testing Format

**Rule**: Always use chat mode for instruction-tuned models.

```python
# CORRECT
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": scenario},
]
output = model.chat(messages, max_new_tokens=1000)

# WRONG — never do this
output = model.generate("Alice puts the ball in the ___")
```

System prompt template:
```
You are a helpful assistant. Think step by step in <think> tags before answering.
```

Generation budget: 1000 tokens minimum. If the model's reasoning is truncated
(visible in `<think>` tags), increase to 2000.

## 2. Stimulus Design

### Counterbalancing (8-scenario design)

For each base scenario, generate 8 variants:

| # | Belief State | Location Order | Type |
|---|-------------|----------------|------|
| 1 | False Belief | A → B | Test |
| 2 | False Belief | B → A | Test |
| 3 | True Belief | A → B | Test |
| 4 | True Belief | B → A | Test |
| 5 | Reality Check | A → B | Control |
| 6 | Reality Check | B → A | Control |
| 7 | Belief Question | A → B | Control |
| 8 | Belief Question | B → A | Control |

### Novel Names

Never use: Alice, Bob, Charlie, drawer, basket, box, kitchen, garden.

Use generated names from pools:
- **Agents**: Zara, Kael, Priya, Orin, Lumi, Dex, Naia, Voss, ...
- **Objects**: marble, figurine, compass, lantern, crystal, ...
- **Locations**: alcove, cupboard, cabinet, shelf, vault, crate, ...

Rotate through pools so no name appears in >20% of stimuli.

### Heuristic Baselines

Every stimulus must have computable baselines:
- **First-mention**: First location mentioned in the narrative
- **Recency**: Last location mentioned before the question
- **Reality**: Actual current location of the object

Record these as metadata on each stimulus.

## 3. Statistical Analysis

### Primary Tests

```python
from scipy.stats import permutation_test

def compare_conditions(condition_a, condition_b, n_permutations=10000):
    """Two-sided permutation test on accuracy difference."""
    stat = np.mean(condition_a) - np.mean(condition_b)
    # ... permutation logic ...
    return p_value, effect_size, ci_lower, ci_upper
```

### Required Reporting

For every claim, report:
1. Sample size (n per condition)
2. Accuracy with 95% bootstrap CI
3. Effect size (Cohen's d or h) with 95% CI
4. p-value from permutation test
5. Whether it survives Bonferroni correction
6. Comparison to heuristic baselines
7. Stability across 5 random 80% subsets

### Null Baselines for Probes

Before reporting probe accuracy, compute:
```python
null_accuracy = probe_accuracy_on_random_labels(n_samples, n_features, n_folds=5)
```

If your probe accuracy is within 2 SD of null, it's meaningless.

## 4. Circuit Discovery Protocol

### Using circuit-tracer

```python
from circuit_tracer import CircuitTracer

tracer = CircuitTracer(model)
graph = tracer.trace(
    prompt=stimulus,
    target_token=belief_location_token,
    threshold=0.01,  # minimum attribution to include
)
```

### Validation (Both Directions Required)

**Necessity**: Ablate the circuit. If behavior doesn't change, circuit is not necessary.
**Sufficiency**: Ablate everything EXCEPT the circuit. If behavior survives, circuit is sufficient.

Both must pass for a valid circuit claim. Report:
- Necessity score: accuracy_drop / baseline_accuracy
- Sufficiency score: accuracy_preserved / baseline_accuracy

### Cross-Model Comparison

Align features functionally (same behavioral effect when ablated), not by layer number.
A "homologous" circuit means:
- Same functional roles (agent-binding, content, divergence)
- Same connectivity pattern (which roles feed into which)
- Possibly different layer positions (expected across architectures)

## 5. Exploratory vs. Confirmatory

**Confirmatory**: Pre-registered hypotheses tested with the pre-specified protocol.
Results reported regardless of outcome.

**Exploratory**: Any analysis not pre-registered. Clearly labeled as "Exploratory"
in all outputs. Cannot be used to support primary claims. Can generate hypotheses
for future studies.

If an exploratory finding is interesting, it becomes a hypothesis for the NEXT study,
not a result of the current one.

## 6. Reproducibility

- All random seeds fixed and recorded
- Full stimulus sets saved as JSON
- All results saved with timestamps and git commit hashes
- Environment frozen in `uv.lock`
- Every figure regenerable from saved results
