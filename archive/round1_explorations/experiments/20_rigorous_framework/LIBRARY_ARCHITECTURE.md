# Library Architecture Overview

## High-Level Structure

```
experiments/20_rigorous_framework/
├── config.py                    # Central configuration (ExperimentConfig)
│
├── core/                        # Core Infrastructure
│   ├── activation_extractor.py  # Extract activations with caching
│   ├── chat_runner.py          # Run chat-based experiments
│   ├── response_parser.py      # Parse <think> responses
│   ├── prompts.py              # Prompt formatting utilities
│   ├── multi_agent.py          # Multi-agent interaction framework
│   └── cross_model.py          # Cross-model validation
│
├── scenarios/                   # Scenario Generation
│   ├── templates.py            # Simple API (generate_n_scenarios)
│   ├── tom_extended.py         # ToM scenario generator
│   ├── counterbalancing.py     # 8-scenario counterbalancing
│   ├── novel_names.py          # Novel name generation
│   ├── multi_agent.py          # Multi-agent scenarios
│   ├── deception.py            # Deception scenarios
│   ├── cooperation.py           # Cooperation games
│   ├── higher_order_tom.py     # 2nd/3rd order ToM
│   ├── benchmarks.py           # ToMi, FANToM benchmarks
│   └── robustness.py           # Robustness testing
│
├── analysis/                    # Analysis Tools
│   ├── statistics.py           # Basic stats (CI, effect sizes)
│   ├── heuristics.py           # Heuristic baselines
│   ├── validator.py            # Methodology validation
│   ├── simple_validator.py     # Quick validation
│   │
│   ├── circuits/               # Circuit Analysis (NEW)
│   │   ├── ablation.py         # HeadAblator (correct implementation)
│   │   └── chat_circuit_analyzer.py  # ChatModeCircuitAnalyzer
│   │
│   ├── circuit_analysis.py      # General circuit analysis
│   ├── logit_lens.py           # Where does model decide?
│   ├── probing.py              # Linear probing pipeline
│   ├── mlp_analysis.py          # MLP neuron analysis
│   ├── causal_steering.py       # Steering vector validation
│   ├── minimal_pairs.py         # Isolate causal factors
│   ├── signal_injection.py     # Signal extraction & injection
│   ├── patching.py             # Activation patching (completion mode)
│   ├── geometry.py              # Representation geometry
│   ├── information_theory.py   # Mutual information
│   ├── null_distributions.py    # Statistical nulls
│   └── controls.py              # Extended controls (power, bootstrap, corrections)
│
├── exploration/                 # Systematic Discovery
│   ├── head_sweep.py           # Find important heads
│   ├── discovery.py             # Pattern discovery
│   └── cross_model.py           # Cross-model validation
│
├── visualization/               # Figures
│   ├── figures.py              # 2D plots
│   └── visualization_3d.py     # 3D animated visualizations
│
└── runners/                     # Complete Test Suites
    ├── run_tom_validation.py    # Standard ToM test
    └── run_full_suite.py       # All capability tests
```

---

## Core Modules

### 1. `config.py` - Central Configuration

**Purpose**: Enforces methodological requirements

**Key Class**: `ExperimentConfig`
- `min_samples_per_condition=50` - Statistical power requirement
- `max_tokens=1000` - Token budget for reasoning models
- `require_counterbalancing=True` - 8-scenario design
- `require_beats_heuristics=True` - Must outperform baselines
- `require_novel_names=True` - Break training priors
- `validate_for_publication()` - Check if config meets standards

**Usage**:
```python
from config import ExperimentConfig, DEFAULT_CONFIG

config = ExperimentConfig()
# Or use defaults
config = DEFAULT_CONFIG
```

---

### 2. `core/` - Core Infrastructure

#### `chat_runner.py` - Chat-Based Experiment Running

**Purpose**: Run experiments with proper chat formatting for reasoning models

**Key Class**: `ChatExperimentRunner`
- Formats prompts with chat template (`<|im_start|>` tags)
- Instructs model to use `<think>` tags
- Allows sufficient token budget (1000 tokens)
- Parses responses to extract final answers

**Usage**:
```python
from core import ChatExperimentRunner

runner = ChatExperimentRunner(model, tokenizer, config)
result = runner.run_single(scenario)
batch_results = runner.run_batch(scenarios)
```

**Key Insight**: Qwen3-4B needs chat format + 1000 tokens to show ToM (80-90% accuracy). Raw completion only gets 35-50%.

#### `response_parser.py` - Parse Reasoning Responses

**Purpose**: Extract structured info from `<think>` responses

**Key Class**: `ResponseParser`
- Extracts reasoning (inside tags)
- Extracts answer (after tags)
- Validates response format
- Estimates confidence

**Usage**:
```python
from core import ResponseParser

parser = ResponseParser()
parsed = parser.parse(response)
# parsed.reasoning, parsed.answer, parsed.confidence
```

#### `activation_extractor.py` - Unified Activation Extraction

**Purpose**: Extract activations with caching and batching

**Key Class**: `ActivationExtractor`
- Hook-based extraction (efficient)
- Batched processing
- Automatic caching to disk
- Supports layer outputs and attention patterns

**Usage**:
```python
from core import ActivationExtractor

extractor = ActivationExtractor(model, tokenizer, config)
activations = extractor.extract(prompts, layers=[8, 16, 24])
```

#### `multi_agent.py` - Multi-Agent Framework

**Purpose**: Run multi-agent interactions for collaboration circuits

**Key Classes**: `Agent`, `MultiAgentInteraction`
- Agents with personas and history
- Multi-turn conversations
- Activation capture during interactions

---

### 3. `scenarios/` - Scenario Generation

#### `templates.py` - Simple API (Recommended)

**Purpose**: Easy scenario generation

**Key Functions**:
- `generate_n_scenarios(n, use_novel_names=True)` - Generate n scenarios
- `generate_counterbalanced_8(...)` - Generate 8-scenario set
- `get_novel_names()` - Get novel names

**Usage**:
```python
from scenarios.templates import generate_n_scenarios

scenarios = generate_n_scenarios(n=50, use_novel_names=True)
# Returns list of dicts with: story, question, options, correct, type
```

#### `tom_extended.py` - ToM Scenario Generator

**Purpose**: Generate diverse ToM scenarios

**Key Class**: `ToMScenarioGenerator`
- `generate_false_belief(n=50)`
- `generate_true_belief(n=50)`
- `generate_balanced_set(n_per_type=50)`
- `generate_with_counterbalancing(n_tasks=25)`

#### `counterbalancing.py` - 8-Scenario Design

**Purpose**: Enforce proper counterbalancing

**Key Function**: `generate_counterbalanced_set(template, n_tasks=25)`
- Generates 8 scenarios per task:
  - 2 False Belief (A-B order, B-A order)
  - 2 True Belief (A-B order, B-A order)
  - 4 Reality controls

---

### 4. `analysis/` - Analysis Tools

#### `circuits/` - Circuit Analysis (NEW - Fixed!)

**Purpose**: Proper head ablation for chat-mode models

**Key Classes**:
- `HeadAblator`: Correct ablation using pre-hooks on o_proj input
- `ChatModeCircuitAnalyzer`: Combines ablation + chat evaluation + statistics

**Usage**:
```python
from analysis.circuits import ChatModeCircuitAnalyzer

analyzer = ChatModeCircuitAnalyzer(model, tokenizer, config)
results = analyzer.ablation_sweep(scenarios, layers_to_test, ...)
significant_heads, correction = analyzer.get_significant_heads(results)
```

**Key Fix**: Uses `register_forward_pre_hook` on o_proj **input** (heads still separate), not output (heads already combined).

#### `statistics.py` - Basic Statistics

**Purpose**: Essential statistical functions

**Key Functions**:
- `accuracy_with_ci(results)` - Accuracy with Wilson CI
- `compare_accuracies(acc1, n1, acc2, n2)` - Fisher's exact test + Cohen's h
- `bonferroni(p_values)` - Multiple comparisons correction

#### `heuristics.py` - Heuristic Baselines

**Purpose**: Compare model to simple heuristics

**Key Class**: `HeuristicBaselines`
- First-mention: Predict first location mentioned
- Recency: Predict most recent location
- Reality: Predict actual current location

**Usage**:
```python
from analysis import HeuristicBaselines

baselines = HeuristicBaselines()
predictions = baselines.predict_all(scenario)
# Model must beat ALL baselines to claim ToM
```

#### `validator.py` - Methodology Validation

**Purpose**: Enforce methodological requirements

**Key Class**: `ResultValidator`
- Checks sample size (n≥50)
- Checks statistical significance
- Checks effect size (Cohen's h≥0.2)
- Checks beats heuristics
- Checks counterbalancing

**Usage**:
```python
from analysis import ResultValidator

validator = ResultValidator(config)
report = validator.validate_tom_results(results)
if not report.all_passed:
    print("Cannot make claims!")
```

#### `logit_lens.py` - Where Does Model Decide?

**Purpose**: Track predictions through layers

**Key Class**: `LogitLens`
- Applies final unembedding to intermediate states
- Tracks when prediction crystallizes
- Identifies decision layer

**Usage**:
```python
from analysis import LogitLens

lens = LogitLens(model, tokenizer)
result = lens.analyze(prompt, target_token=" drawer", contrast_token=" basket")
# result.decision_layer, result.diffs
```

#### `probing.py` - Linear Probing Pipeline

**Purpose**: Standardized linear probing with cross-validation

**Key Class**: `ProbingPipeline`
- Cross-validated logistic regression
- Multiple layers
- Proper statistics

**Usage**:
```python
from analysis import ProbingPipeline

pipeline = ProbingPipeline()
results = pipeline.probe_multiple_layers(activations, labels)
```

#### `causal_steering.py` - Steering Vector Validation

**Purpose**: Test if representations are functional (causal) vs correlational

**Key Class**: `CausalSteering`
- Extract steering direction from probe
- Test if steering changes behavior
- If yes → representation is functional

#### `minimal_pairs.py` - Isolate Causal Factors

**Purpose**: Vary ONE thing at a time to make causal claims

**Key Class**: `MinimalPairTester`
- Test pairs that differ in one factor
- Isolate what causes behavioral differences

#### `controls.py` - Extended Controls

**Purpose**: Additional methodological controls

**Key Functions**:
- `bonferroni_correct(p_values)` - Multiple comparisons correction
- `benjamini_hochberg(p_values)` - FDR correction (less conservative)
- `power_analysis(effect_size, alpha, power)` - Calculate required n
- `bootstrap_ci(results)` - Bootstrap confidence intervals

---

### 5. `exploration/` - Systematic Discovery

**Purpose**: Tools for discovering patterns

**Key Classes**:
- `HeadDiscoverySweep`: Systematic search for important heads
- `PhenomenonDiscovery`: Discover patterns and generate hypotheses
- `CrossModelValidator`: Test findings across models

---

### 6. `visualization/` - Figures

**Purpose**: Publication-quality visualizations

**Key Classes**:
- `FigureGenerator`: 2D plots
- `Visualization3D`: 3D animated visualizations

---

### 7. `runners/` - Complete Test Suites

**Purpose**: Pre-built test suites with proper methodology

**Key Functions**:
- `run_tom_validation()`: Standard ToM test (n≥50, counterbalancing, etc.)
- `run_full_suite()`: All capability tests

---

## Design Principles

### 1. **Separation of Concerns**
- `core/`: Infrastructure (model loading, activation extraction, running)
- `scenarios/`: Scenario generation
- `analysis/`: Analysis techniques
- `exploration/`: Discovery tools
- `visualization/`: Figures

### 2. **Simple API + Full API**
- Simple API: `from analysis import validate, accuracy_with_ci`
- Full API: `from analysis import ResultValidator, ProbingPipeline`

### 3. **Methodology Enforcement**
- Config enforces requirements (n≥50, tokens≥1000, etc.)
- Validator checks results before allowing claims
- Statistical tests built-in

### 4. **Chat Mode First**
- `ChatExperimentRunner` is the default (not completion mode)
- `ChatModeCircuitAnalyzer` for circuit discovery
- Proper handling of `<think>` tags

### 5. **Reusability**
- Components can be used independently
- Clear interfaces between modules
- No circular dependencies

---

## Common Workflows

### Workflow 1: Basic ToM Test
```python
from config import ExperimentConfig
from core import ChatExperimentRunner, load_model_for_chat
from scenarios.templates import generate_n_scenarios
from analysis import HeuristicBaselines, ResultValidator

# 1. Setup
config = ExperimentConfig()
model, tokenizer = load_model_for_chat(config.model_name)
runner = ChatExperimentRunner(model, tokenizer, config)

# 2. Generate scenarios
scenarios = generate_n_scenarios(n=50, use_novel_names=True)

# 3. Run evaluation
results = runner.run_batch(scenarios)

# 4. Compare to heuristics
baselines = HeuristicBaselines()
heuristic_eval = baselines.evaluate(scenarios, model_predictions)

# 5. Validate
validator = ResultValidator(config)
report = validator.validate_tom_results(results.to_dict())
```

### Workflow 2: Circuit Discovery
```python
from analysis.circuits import ChatModeCircuitAnalyzer
from scenarios.templates import generate_n_scenarios

# 1. Setup
analyzer = ChatModeCircuitAnalyzer(model, tokenizer, config)
scenarios = generate_n_scenarios(n=50)

# 2. Run ablation sweep
results = analyzer.ablation_sweep(
    scenarios=scenarios,
    layers_to_test=[20, 24, 28, 32],
    heads_per_layer=4
)

# 3. Get significant heads (with correction)
significant_heads, correction = analyzer.get_significant_heads(
    results, alpha=0.05, correction="bonferroni"
)
```

### Workflow 3: Probing Analysis
```python
from core import ActivationExtractor
from analysis import ProbingPipeline

# 1. Extract activations
extractor = ActivationExtractor(model, tokenizer, config)
activations = extractor.extract(prompts, layers=[8, 16, 24, 32])

# 2. Probe for belief state
pipeline = ProbingPipeline()
results = pipeline.probe_multiple_layers(activations, labels)
```

---

## Key Insights from Library Design

### 1. **Chat Mode is Critical**
- Completion mode: 35-50% accuracy
- Chat mode with 1000 tokens: 80-90% accuracy
- Library defaults to chat mode

### 2. **Ablation Must Use Pre-Hooks**
- **Wrong**: `register_forward_hook` on o_proj output (heads already combined)
- **Right**: `register_forward_pre_hook` on o_proj input (heads still separate)
- `HeadAblator` implements correct method

### 3. **Methodology is Enforced**
- Config validates requirements
- Validator checks results
- Statistical tests built-in
- Multiple comparisons correction available

### 4. **Scenario Generation is Rich**
- Novel names to break priors
- Counterbalancing for order effects
- Multiple scenario types (FB, TB, Communication, Multi-agent, etc.)
- Benchmarks (ToMi, FANToM)

### 5. **Analysis Tools are Comprehensive**
- Circuit analysis (ablation, attention patterns)
- Probing (linear classifiers)
- Logit lens (where decisions form)
- Causal steering (functional vs correlational)
- Minimal pairs (isolate causal factors)
- Information theory (mutual information)
- Geometry (representation structure)

---

## Potential Improvements

### 1. **Better Organization**
- Consider `analysis/statistics/` submodule for statistical tests
- Consider `analysis/evaluation/` submodule for evaluation utilities
- Consider `core/hooks/` submodule for hook management

### 2. **Documentation**
- Add docstrings to all public functions
- Create usage examples for each module
- Document when to use which tool

### 3. **Testing**
- Unit tests for each module
- Integration tests for workflows
- Validation that ablation actually works

### 4. **Consistency**
- Some scripts still use old patterns
- Should all use library components
- Remove duplicated code

---

## Summary

The library is **well-designed** with:
- ✅ Clear separation of concerns
- ✅ Simple + Full API options
- ✅ Methodology enforcement
- ✅ Chat mode support
- ✅ Comprehensive analysis tools
- ✅ Rich scenario generation

**Main Issues Fixed**:
- ✅ Head ablation bug (now uses correct pre-hook method)
- ✅ Library organization (new `circuits/` submodule)
- ✅ step35 refactored to use library

**Remaining Opportunities**:
- More scripts should use library components
- Better documentation
- More submodules for organization
- Unit tests

