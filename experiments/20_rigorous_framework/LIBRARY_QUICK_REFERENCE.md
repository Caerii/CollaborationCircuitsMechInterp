# Library Quick Reference

## Import Patterns

### Simple (Recommended)
```python
from config import ExperimentConfig
from core import ChatExperimentRunner, load_model_for_chat
from scenarios.templates import generate_n_scenarios
from analysis import (
    accuracy_with_ci, 
    HeuristicBaselines, 
    ResultValidator,
    ChatModeCircuitAnalyzer
)
```

### Full API
```python
from analysis import (
    ProbingPipeline,
    LogitLens,
    CausalSteering,
    MinimalPairTester,
    CircuitAnalysis,
    MLPAnalyzer,
)
```

---

## Common Tasks

### 1. Run ToM Evaluation
```python
from config import ExperimentConfig
from core import ChatExperimentRunner, load_model_for_chat
from scenarios.templates import generate_n_scenarios

config = ExperimentConfig()
model, tokenizer = load_model_for_chat(config.model_name)
runner = ChatExperimentRunner(model, tokenizer, config)

scenarios = generate_n_scenarios(n=50, use_novel_names=True)
results = runner.run_batch(scenarios)
print(f"Accuracy: {results.accuracy:.1%}")
```

### 2. Find Important Heads (Circuit Discovery)
```python
from analysis.circuits import ChatModeCircuitAnalyzer

analyzer = ChatModeCircuitAnalyzer(model, tokenizer, config)
results = analyzer.ablation_sweep(
    scenarios=scenarios,
    layers_to_test=[20, 24, 28, 32],
    heads_per_layer=4
)
significant_heads, correction = analyzer.get_significant_heads(results)
```

### 3. Probe for Belief State
```python
from core import ActivationExtractor
from analysis import ProbingPipeline

extractor = ActivationExtractor(model, tokenizer, config)
activations = extractor.extract(prompts, layers=[8, 16, 24, 32])

pipeline = ProbingPipeline()
probe_results = pipeline.probe_multiple_layers(activations, labels)
```

### 4. Track Decision Formation (Logit Lens)
```python
from analysis import LogitLens

lens = LogitLens(model, tokenizer)
result = lens.analyze(
    prompt="Alice put ball in drawer. Alice left. Bob moved it to basket. Alice looks in the",
    target_token=" drawer",
    contrast_token=" basket"
)
print(f"Decision at layer {result.decision_layer}")
```

### 5. Compare to Heuristics
```python
from analysis import HeuristicBaselines

baselines = HeuristicBaselines()
predictions = baselines.predict_all(scenario)
# Returns: {"first_mention": "...", "recency": "...", "reality": "..."}
```

### 6. Validate Results
```python
from analysis import ResultValidator

validator = ResultValidator(config)
report = validator.validate_tom_results(results)
if report.all_passed:
    print("Results are methodologically sound!")
```

### 7. Statistical Tests
```python
from analysis import accuracy_with_ci, compare_accuracies, bonferroni_correct

# Accuracy with CI
stats = accuracy_with_ci([True, False, True, ...])
# Returns: {"accuracy": 0.75, "ci_low": 0.65, "ci_high": 0.85, "n": 100}

# Compare two accuracies
comparison = compare_accuracies(0.8, 50, 0.6, 50)
# Returns: {"p_value": 0.02, "cohens_h": 0.45, "significant": True}

# Multiple comparisons correction
corrected = bonferroni_correct([0.01, 0.03, 0.04, 0.05], alpha=0.05)
# Returns: {"corrected_alpha": 0.0125, "significant_after_correction": [...]}
```

---

## Key Classes Reference

### Core Infrastructure

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `ChatExperimentRunner` | Run chat-based experiments | `run_single()`, `run_batch()` |
| `ResponseParser` | Parse `<think>` responses | `parse()`, `extract_answer_token()` |
| `ActivationExtractor` | Extract activations | `extract()`, `extract_attention()` |
| `HeadAblator` | Ablate attention heads | `ablate_head()`, `ablate_heads()`, `clear()` |

### Analysis Tools

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `ChatModeCircuitAnalyzer` | Circuit discovery (chat mode) | `ablation_sweep()`, `get_significant_heads()` |
| `CircuitAnalysis` | General circuit analysis | `ablation_sweep()`, `get_attention_patterns()` |
| `LogitLens` | Track decisions through layers | `analyze()` |
| `ProbingPipeline` | Linear probing | `probe_multiple_layers()` |
| `HeuristicBaselines` | Heuristic predictions | `predict_all()`, `evaluate()` |
| `ResultValidator` | Validate methodology | `validate_tom_results()` |
| `CausalSteering` | Test functional representations | `set_direction()`, `test_effect()` |
| `MinimalPairTester` | Isolate causal factors | `test_pairs()` |

### Scenario Generation

| Function/Class | Purpose | Returns |
|----------------|---------|---------|
| `generate_n_scenarios(n)` | Generate n scenarios | List[Dict] |
| `generate_counterbalanced_8(...)` | Generate 8-scenario set | List[Dict] |
| `ToMScenarioGenerator` | Generate ToM scenarios | List[ToMScenario] |
| `get_novel_names()` | Get novel names | Dict[str, List[str]] |

---

## Configuration Options

### ExperimentConfig

```python
config = ExperimentConfig(
    model_name="Qwen/Qwen3-4B",
    max_tokens=1000,                    # Token budget for reasoning
    min_samples_per_condition=50,       # Statistical power requirement
    require_counterbalancing=True,      # 8-scenario design
    require_beats_heuristics=True,      # Must outperform baselines
    require_novel_names=True,           # Break training priors
    require_statistical_tests=True,     # p-values and effect sizes
    significance_level=0.05,            # Alpha for tests
    min_effect_size=0.2,                # Cohen's h minimum
)
```

---

## Data Structures

### Scenario Format
```python
scenario = {
    "story": "Alice put the ball in the drawer. Alice left. Bob moved it to the basket.",
    "question": "Where will Alice look for the ball?",
    "options": ["drawer", "basket"],
    "correct": "drawer",
    "type": "false_belief",  # or "true_belief", "communication", etc.
    "metadata": {...}
}
```

### Result Format
```python
result = {
    "accuracy": 0.75,
    "n": 50,
    "n_correct": 38,
    "ci_low": 0.65,
    "ci_high": 0.85,
    "by_type": {
        "false_belief": {"accuracy": 0.80, "n": 25},
        "true_belief": {"accuracy": 0.70, "n": 25},
    }
}
```

---

## Best Practices

### 1. Always Use Config
```python
config = ExperimentConfig()  # Enforces methodology
```

### 2. Use Chat Mode for Reasoning Models
```python
runner = ChatExperimentRunner(model, tokenizer, config)  # Not completion mode!
```

### 3. Generate Enough Scenarios
```python
scenarios = generate_n_scenarios(n=50)  # Minimum for statistical power
```

### 4. Compare to Heuristics
```python
baselines = HeuristicBaselines()
heuristic_eval = baselines.evaluate(scenarios, model_predictions)
# Model must beat ALL heuristics
```

### 5. Validate Before Making Claims
```python
validator = ResultValidator(config)
report = validator.validate_tom_results(results)
if not report.all_passed:
    print("Cannot make claims - fix methodology issues first!")
```

### 6. Use Multiple Comparisons Correction
```python
from analysis.controls import bonferroni_correct

p_values = [0.01, 0.03, 0.04, 0.05]  # From multiple tests
corrected = bonferroni_correct(p_values, alpha=0.05)
# Only report heads that pass corrected threshold
```

### 7. Use Library Components
```python
# DON'T write custom ablation code
# DO use:
from analysis.circuits import HeadAblator

ablator = HeadAblator(model)
ablator.ablate_head(32, 0)
# ... test ...
ablator.clear()
```

---

## Common Mistakes to Avoid

### ❌ Wrong: Custom Ablation (Broken)
```python
# DON'T do this - uses wrong hook type!
hook = o_proj.register_forward_hook(...)  # WRONG: post-hook on output
```

### ✅ Right: Use Library
```python
# DO this - uses correct pre-hook on input
from analysis.circuits import HeadAblator
ablator = HeadAblator(model)
ablator.ablate_head(32, 0)
```

### ❌ Wrong: Small Sample Size
```python
scenarios = [scenario1, scenario2, scenario3, scenario4]  # n=4 - too small!
```

### ✅ Right: Use Generator
```python
scenarios = generate_n_scenarios(n=50)  # n≥50 required
```

### ❌ Wrong: No Statistical Tests
```python
accuracy1 = 0.75
accuracy2 = 0.65
print(f"Difference: {accuracy1 - accuracy2}")  # No p-value!
```

### ✅ Right: Use Statistical Functions
```python
from analysis import compare_accuracies
comparison = compare_accuracies(0.75, 50, 0.65, 50)
print(f"p={comparison['p_value']:.4f}, h={comparison['cohens_h']:.2f}")
```

---

## Module Dependencies

```
config.py
  ↓
core/
  ├── chat_runner.py → uses response_parser.py
  ├── activation_extractor.py → uses config.py
  └── multi_agent.py → uses chat_runner.py
  ↓
scenarios/
  ├── templates.py → uses novel_names.py, counterbalancing.py
  └── tom_extended.py → uses templates.py
  ↓
analysis/
  ├── circuits/ → uses core/chat_runner.py
  ├── validator.py → uses statistics.py, heuristics.py
  └── controls.py → (standalone)
```

---

## Quick Troubleshooting

### Problem: "Heads already combined" error
**Solution**: Use `HeadAblator` from library (uses pre-hook correctly)

### Problem: Low accuracy (<50%)
**Solution**: Check if using chat mode + sufficient tokens (1000)

### Problem: "No significant heads after correction"
**Solution**: Increase sample size (n≥100) or use less conservative correction (FDR)

### Problem: "Cannot make claims - methodology issues"
**Solution**: Check `ResultValidator` report and fix violations

### Problem: Import errors
**Solution**: Ensure framework root is in `sys.path`:
```python
from pathlib import Path
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))
```

