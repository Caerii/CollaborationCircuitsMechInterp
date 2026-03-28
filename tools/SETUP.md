# Tools & Setup Reference

## Installed Stack

| Package | Version | Purpose |
|---------|---------|---------|
| circuit-tracer | 0.4.1 | Attribution graphs (Anthropic methodology) |
| sae-lens | 6.39.0 | Sparse autoencoder training & analysis |
| transformer-lens | 2.18.0 | Hooked transformer for MI |
| nnsight | 0.6.1 | Universal activation extraction |
| openai | 2.30.0 | LM Studio API client |
| torch | 2.9.1+cu130 | GPU compute |
| transformers | 4.57.3 | Model loading (pinned for circuit-tracer) |
| statsmodels | 0.14.6 | Statistical tests |

## Two Modes of Operation

### 1. Behavioral Testing → LM Studio API

For running prompts and measuring accuracy. LM Studio manages GPU memory.

```python
from lib.core.chat import run_scenario

resp = run_scenario(
    scenario_text="Zara puts the marble in the alcove...",
    question="Where will Zara look for the marble?",
)
print(resp.thinking)  # Chain of thought
print(resp.answer)    # Final answer
```

**When to use**: Measuring ToM accuracy, testing stimuli, behavioral experiments.

**Setup**: Load a model in LM Studio, it serves on http://localhost:1234/v1

### 2. Mechanistic Analysis → Direct Model Loading

For activation extraction, ablation, attribution graphs. Loads model into GPU directly.

```python
# circuit-tracer (attribution graphs)
from circuit_tracer import attribute

# nnsight (activation extraction)
import nnsight

# transformer-lens (hooked transformers)
from transformer_lens import HookedTransformer
```

**When to use**: Circuit discovery, probing, ablation, SAE analysis.

**Setup**: Unload model from LM Studio first (can't share 10GB VRAM).

## Supported Models for circuit-tracer

| Model | Sizes | Transcoder |
|-------|-------|------------|
| Qwen-3 | 0.6B, 1.7B, **4B**, 8B, 14B | PLT |
| Gemma-2 | 2B | PLT, CLT |
| Llama-3.2 | 1B | PLT, CLT |

## Workflow

1. **Design stimuli** → `lib/scenarios/generator.py` (no GPU needed)
2. **Behavioral baseline** → LM Studio + `lib/core/chat.py`
3. **Close LM Studio** → Free GPU memory
4. **Mechanistic analysis** → circuit-tracer / nnsight / transformer-lens
5. **Statistical analysis** → `lib/analysis/statistics.py` (no GPU needed)
