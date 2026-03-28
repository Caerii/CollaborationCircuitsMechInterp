# Experiment 20: Rigorous MI Framework

A comprehensive mechanistic interpretability framework synthesizing best practices from experiments 1-19.

## Overview

This framework enforces proper methodology for mechanistic interpretability research on Theory of Mind and collaborative reasoning:

- **n >= 50** samples per condition
- **1000 token budget** for reasoning models
- **8-scenario counterbalancing** (FB/TB x Order)
- **Heuristic baseline comparison** (first-mention, recency, reality)
- **Statistical validation** (p-values, effect sizes, confidence intervals)
- **Novel names** to break training priors
- **True-belief controls** paired with false-belief tests

## Installation

```bash
# From repository root
pip install torch transformers scikit-learn scipy numpy matplotlib
```

## Quick Start

```python
from experiments.e20_rigorous_framework import ExperimentConfig
from experiments.e20_rigorous_framework.core import ChatExperimentRunner, load_model_for_chat
from experiments.e20_rigorous_framework.scenarios import ToMScenarioGenerator
from experiments.e20_rigorous_framework.analysis import ResultValidator

# Load model
model, tokenizer = load_model_for_chat("Qwen/Qwen3-4B")

# Generate scenarios with proper methodology
gen = ToMScenarioGenerator(use_novel_names=True)
scenarios = gen.generate_balanced_set(n_per_type=50)

# Run evaluation
runner = ChatExperimentRunner(model, tokenizer, ExperimentConfig())
results = runner.run_batch(scenarios)

# Validate methodology
validator = ResultValidator()
report = validator.validate_tom_results(results.to_dict())
report.print_report()
```

## Directory Structure

```
20_rigorous_framework/
    config.py                 # Central configuration with methodology requirements
    
    core/                     # Core infrastructure
        activation_extractor.py   # Unified activation extraction
        chat_runner.py           # Chat-based experiment running
        response_parser.py       # Parse <think> tag responses
        
    scenarios/                # Scenario generation
        novel_names.py           # Procedural name generation
        counterbalancing.py      # 8-scenario design enforcement
        tom_extended.py          # ToM scenarios (FB, TB, Communication)
        multi_agent.py           # 3+ agent scenarios
        deception.py             # Lie detection, trust calibration
        cooperation.py           # PD, commons, negotiation
        
    analysis/                 # Analysis tools
        heuristics.py            # Baseline heuristics
        probing.py               # Linear probing pipeline
        causal_steering.py       # Steering vector validation
        circuit_analysis.py      # Head ablation analysis
        validator.py             # Methodology enforcement
        
    exploration/              # Systematic discovery
        head_sweep.py            # Find important heads
        discovery.py             # Pattern discovery
        cross_model.py           # Multi-model validation
        
    visualization/            # Figures
        figures.py               # Publication-quality plots
        
    runners/                  # Complete test suites
        run_tom_validation.py    # Standard ToM test
        run_full_suite.py        # All capability tests
```

## Key Features

### 1. Proper Token Budget

```python
config = ExperimentConfig(max_tokens=1000)  # Critical for reasoning models
```

Qwen3-4B uses `<think>` tags for reasoning. Previous experiments failed because token budget was too small (100-150). 1000 tokens allows full reasoning.

### 2. Novel Names

```python
from scenarios import NovelNameGenerator

gen = NovelNameGenerator()
names = gen.generate_set()
# names.agents = ["Zyx", "Qar"]
# names.locations = ["Container-Alpha", "Zone-Beta"]
```

Familiar names (Alice, Bob, drawer, basket) allow models to use training priors instead of actual reasoning.

### 3. Counterbalancing

```python
from scenarios import generate_counterbalanced_set, SALLY_ANNE_TEMPLATE

# 25 tasks * 8 scenarios = 200 scenarios
scenarios = generate_counterbalanced_set(SALLY_ANNE_TEMPLATE, n_tasks=25)
```

Each task generates 8 variants:
1. False Belief, Order A-B
2. False Belief, Order B-A
3. True Belief, Order A-B
4. True Belief, Order B-A
5-8. Control questions

### 4. Heuristic Baselines

```python
from analysis import HeuristicBaselines

baselines = HeuristicBaselines()
evaluation = baselines.evaluate(scenarios, model_predictions)

# Model must beat ALL baselines to claim ToM
assert evaluation["model_beats_heuristics"]
```

### 5. Validation

```python
from analysis import ResultValidator

validator = ResultValidator(config)
report = validator.validate_tom_results(results)

if not report.all_passed:
    print("Cannot make claims - methodology issues:")
    for check in report.checks:
        if not check.passed:
            print(f"  - {check.name}: {check.message}")
```

## Methodology Requirements Summary

| Requirement | Why | Enforcement |
|-------------|-----|-------------|
| n >= 50 | Statistical power | Config, Validator |
| 1000 tokens | Reasoning space | Config, Runner |
| Counterbalancing | Order effects | CounterbalancedScenarioSet |
| Beat heuristics | Prove actual ToM | HeuristicBaselines |
| Novel names | Break priors | NovelNameGenerator |
| True-belief controls | Isolate FB effect | Counterbalancing |
| Statistical tests | Significance | Validator |
| Effect size h >= 0.2 | Practical significance | Validator |

## Running Tests

### Standard ToM Validation

```bash
python runners/run_tom_validation.py
```

### Full Suite

```bash
python runners/run_full_suite.py --n 50
```

### Quick Exploration

```python
from config import DEBUG_CONFIG
from core import ChatExperimentRunner

# Relaxed requirements for exploration
runner = ChatExperimentRunner(model, tokenizer, DEBUG_CONFIG)
```

## Key Findings from Previous Experiments

1. **ToM is possible with proper prompting** (Step 62): 80-90% accuracy with chat format + 1000 tokens
2. **Raw completion fails** (Step 56): 35-50% accuracy, worse than heuristics
3. **Entity encoding is lexical** (Exp 5): 100% probe accuracy reflects token separability, not semantic understanding
4. **Order effects are real** (Step 56): First-mention and recency heuristics often outperform models

## Citation

```
Rigorous MI Framework for Collaboration Circuits
Based on MATS Project A research
```

## License

MIT

