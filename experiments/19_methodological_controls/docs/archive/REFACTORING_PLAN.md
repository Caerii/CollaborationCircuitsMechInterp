# Mechanistic Interpretability Library Refactoring Plan

## Current Problems

### 1. Code Duplication Across 15+ Scripts
Every script has copy-pasted:
- Model loading code
- Hook installation (ablation, amplification)
- Probability calculation functions
- Scenario generation functions
- Statistics computation
- JSON serialization/saving

### 2. No Use of Established MechInterp Libraries

**TransformerLens** - The gold standard:
- Clean APIs for hooking, caching, patching
- Used by Anthropic, DeepMind researchers
- BUT: **Does NOT support Qwen models** (GPT-2, GPT-Neo, GPT-J, Pythia only)

**nnsight** - Alternative:
- From NDIF (National Deep Inference Fabric)
- More model-agnostic
- Good for interventions
- BUT: Steeper learning curve

**nnterp** (2024) - Newest:
- Standardized interface across architectures
- Claims broad model support
- BUT: Very new, less documented

**Conclusion**: Since Qwen is not in TransformerLens, our custom hooks approach is actually appropriate. BUT we should structure them properly.

### 3. Inconsistent Patterns
- Different variable names for same concepts
- Different hook implementations
- Inconsistent result saving formats
- No configuration management

---

## Proposed Library Structure

```
experiments/19_methodological_controls/
├── lib/                              # Reusable library
│   ├── __init__.py
│   ├── model_utils.py               # Model loading, device management
│   ├── hooks.py                     # Hook management (ablation, amplification, patching)
│   ├── scenarios.py                 # Scenario generation (Sally-Anne, multi-agent, etc.)
│   ├── evaluation.py                # Probability evaluation, accuracy computation
│   ├── statistics.py                # Statistical analysis (CI, significance tests)
│   └── visualization.py             # Plotting utilities
│
├── configs/                         # Experiment configurations
│   ├── model_config.yaml            # Model-specific settings
│   ├── circuit_config.yaml          # Known circuit components
│   └── experiment_configs/
│       ├── robustness.yaml
│       ├── path_patching.yaml
│       └── multiagent.yaml
│
├── experiments/                     # Clean experiment scripts
│   ├── exp01_baseline_tom.py
│   ├── exp02_ablation_sweep.py
│   ├── exp03_path_patching.py
│   ├── exp04_multiagent.py
│   └── exp05_robustness.py
│
├── results/                         # Experiment outputs
├── docs/                            # Documentation
└── README.md
```

---

## Library Module Specifications

### 1. `lib/model_utils.py`

```python
class QwenModel:
    """Wrapper for Qwen model with mechinterp utilities."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Instruct-2507"):
        self.model = None
        self.tokenizer = None
        self.n_heads = None
        self.n_layers = None
        self.head_dim = None
        
    def load(self, dtype=torch.float16, device_map="auto"):
        """Load model and tokenizer."""
        
    @property
    def device(self):
        """Get model device."""
        
    def get_logits(self, text: str) -> torch.Tensor:
        """Get logits for text."""
        
    def get_probs(self, text: str, target_tokens: list) -> dict:
        """Get probabilities for specific tokens."""
```

### 2. `lib/hooks.py`

```python
class HookManager:
    """Manage PyTorch hooks for mechinterp interventions."""
    
    def __init__(self, model: QwenModel):
        self.model = model
        self.hooks = []
        
    def clear(self):
        """Remove all hooks."""
        
    def ablate_heads(self, heads: list[tuple[int, int]]):
        """Zero out specified (layer, head) pairs."""
        
    def amplify_heads(self, heads_scales: list[tuple[int, int, float]]):
        """Scale specified heads by factor."""
        
    def patch_heads(self, heads: list, source_cache: dict):
        """Replace head outputs with cached values (path patching)."""
        
    def cache_activations(self, layers: list = None) -> dict:
        """Cache activations for all/specified layers."""


class ActivationCache:
    """Store and retrieve cached activations."""
    
    def __init__(self):
        self.cache = {}
        
    def store(self, layer: int, head: int, activation: torch.Tensor):
        """Store activation."""
        
    def retrieve(self, layer: int, head: int) -> torch.Tensor:
        """Retrieve activation."""
        
    def clear(self):
        """Clear cache."""
```

### 3. `lib/scenarios.py`

```python
class ScenarioGenerator:
    """Generate ToM test scenarios."""
    
    @staticmethod
    def sally_anne(n: int, seed: int = 42) -> list[dict]:
        """Generate Sally-Anne false belief scenarios."""
        
    @staticmethod
    def belief_update_implicit(n: int, seed: int = 42) -> list[dict]:
        """Generate scenarios requiring implicit belief update inference."""
        
    @staticmethod  
    def belief_update_explicit(n: int, seed: int = 42) -> list[dict]:
        """Generate scenarios with explicit belief cues."""
        
    @staticmethod
    def multiagent_implicit(n: int, seed: int = 42) -> list[dict]:
        """Generate multi-agent implicit scenarios."""
        
    @staticmethod
    def template_variations(n: int, templates: list = None) -> list[dict]:
        """Generate same scenario with different templates."""
        
    @staticmethod
    def negative_controls(n: int, seed: int = 42) -> list[dict]:
        """Generate negative controls (when ToM SHOULD fail)."""
```

### 4. `lib/evaluation.py`

```python
class ToMEvaluator:
    """Evaluate ToM performance."""
    
    def __init__(self, model: QwenModel, hook_manager: HookManager = None):
        self.model = model
        self.hooks = hook_manager
        
    def evaluate_scenario(self, scenario: dict) -> dict:
        """Evaluate single scenario, return detailed results."""
        
    def evaluate_batch(self, scenarios: list, use_intervention: bool = False) -> dict:
        """Evaluate batch of scenarios."""
        
    def compare_interventions(self, scenarios: list, interventions: list) -> dict:
        """Compare multiple interventions on same scenarios."""
```

### 5. `lib/statistics.py`

```python
def compute_accuracy_ci(results: list[bool], confidence: float = 0.95) -> dict:
    """Compute accuracy with Wilson score confidence interval."""
    
def significance_test(baseline: list[bool], intervention: list[bool]) -> dict:
    """McNemar's test for paired binary outcomes."""
    
def effect_size(baseline: float, intervention: float, n: int) -> dict:
    """Cohen's h effect size for proportions."""
```

---

## Configuration Schema

### `circuit_config.yaml`
```yaml
# Known circuit components for Qwen3-4B
model: Qwen/Qwen3-4B-Instruct-2507

inhibitors:
  - layer: 17
    head: 4
    strength: 0.40  # ablation boost
  - layer: 18
    head: 11
    strength: 0.47
  - layer: 18
    head: 14
    strength: 0.33
  - layer: 19
    head: 30
    strength: 0.40
  - layer: 21
    head: 17
    strength: 0.33

enablers:
  - layer: 15
    head: 9
    critical: true  # ablation kills ToM
  - layer: 19
    head: 2
    critical: true
  - layer: 19
    head: 15
    critical: true

circuit_layers:
  core: [15, 16, 17, 18, 19, 20, 21]
  extended: [14, 15, 16, 17, 18, 19, 20, 21, 22]
```

---

## Best Practices We Should Follow

### 1. Reproducibility
- **Fixed seeds** for all random generation
- **Fixed scenario sets** saved as JSON for reuse
- **Versioned configs** for each experiment

### 2. Statistical Rigor
- **Confidence intervals** on all accuracy measures
- **Significance tests** when comparing conditions
- **Effect sizes** reported alongside p-values
- **Sample sizes** of N ≥ 50 for reliable estimates

### 3. Proper Controls
- **Negative controls**: When should ToM fail?
- **Positive controls**: When should ToM succeed?
- **Random baselines**: What's chance performance?

### 4. Clean Experiments
- **One independent variable** per experiment
- **Clear hypotheses** stated before running
- **Pre-registered** analysis plans (in code)

---

## Why NOT TransformerLens?

TransformerLens is excellent but has limitations for our use case:

1. **Model Support**: Only GPT-2, GPT-Neo, GPT-J, Pythia, etc.
   - Qwen is NOT supported
   - Adding Qwen would require significant work

2. **Our hooks work well**: We've validated our custom hooks achieve:
   - 100% ToM with ablation
   - 92% on implicit multi-agent
   - Correct path patching results

3. **Flexibility**: Our custom approach lets us:
   - Handle GQA architecture
   - Custom amplification patterns
   - Combined interventions

**Conclusion**: Keep custom hooks but organize them properly in a library.

---

## Implementation Priority

### Phase 1: Core Library (HIGH PRIORITY)
1. Create `lib/model_utils.py` - Standardized model loading
2. Create `lib/hooks.py` - Unified hook management
3. Create `lib/evaluation.py` - Standard evaluation pipeline

### Phase 2: Scenario & Stats (MEDIUM)
4. Create `lib/scenarios.py` - All scenario generators
5. Create `lib/statistics.py` - Statistical utilities

### Phase 3: Configs (MEDIUM)
6. Create `configs/` with YAML configurations
7. Migrate hardcoded values to configs

### Phase 4: Refactor Experiments (LOWER)
8. Refactor existing scripts to use library
9. Create clean experiment scripts

---

## Immediate Next Steps

Before implementing the library, we should:

1. **Run the robustness test** (step15) to validate our current findings
2. **Document the current circuit** fully in one place
3. **Then** implement the library with confidence in what we're abstracting

This ensures we don't prematurely abstract something that might change.


