# Round 1: Initial Explorations (December 2025)

Exploratory mechanistic interpretability research on Theory of Mind and multi-agent
collaboration circuits in Qwen3-4B. 20 experiments over ~16 hours.

## What Was Done

- **Experiments 01-05**: Entity representation probing (User/Self/Other classification)
- **Experiments 06-08**: Multi-agent collaboration and belief tracking
- **Experiments 09-10**: Sally-Anne false belief paradigm
- **Experiments 11-13**: Circuit discovery via head ablation and information theory
- **Experiments 14-16**: Statistical reboot with larger sample sizes
- **Experiments 17-19**: Methodological controls and critical re-evaluation
- **Experiment 20**: Rigorous framework attempt with 36+ step scripts

## Key Lessons Learned

### Methodological Mistakes (Don't Repeat)

1. **Tested instruction-tuned model in completion mode** — Qwen3-4B needs chat mode
   with `<think>` tags and 500+ token budget. Completion mode gives misleading results.

2. **Probe overfitting** — 100% probe accuracy collapsed to 32% on transfer test.
   Probes detected label tokens ("User:", "Helper:"), not semantic concepts.

3. **Tiny sample sizes** — Most experiments used n=8-20. The "inhibitory network"
   finding (90% accuracy from ablating 3 heads) was contradicted at n=50 in experiment 19.

4. **No pre-registration** — Without pre-committed hypotheses, every experiment
   generated "findings" that couldn't be distinguished from noise.

5. **No cross-model validation** — All findings are Qwen3-4B specific.

6. **Activation patching doesn't work in chat mode** — Decision is distributed across
   early reasoning tokens, not localizable to single positions.

### Genuine Findings Worth Investigating Further

1. **Chat mode + reasoning budget transforms performance** (50% -> 95% on ToM)
2. **Explicit vs implicit belief gap** — Model parses "X believes Y" (100%) but can't
   infer belief updates from communication (2-17% without scaffolding)
3. **Separate circuits for single-agent (L32-34) vs multi-agent (L0-22) ToM**
4. **SAE features are sparse** — ~13 features active per input for belief computation
5. **Framing effects** — 50% allocation swing from cooperative vs competitive framing
6. **Information chains degrade** — 0/3 facts preserved through 3-agent chains

## Structure

```
experiments/          # 20 numbered experiment directories
docs/                 # Research documentation and analysis
scripts/              # Top-level experiment scripts
src/                  # Core library (model.py, probing.py, config.py, data_generation.py)
results/              # Aggregated results and visualizations
data/                 # Generated datasets
MATS_DETAILED_WRITEUP.md    # Full research narrative
MATS_EXECUTIVE_SUMMARY.md   # 600-word summary
```

## Technical Setup

- **Model**: Qwen3-4B-Instruct-2507 (36 layers x 32 heads)
- **Hardware**: RTX 3080 (10GB VRAM), FP16
- **Tools**: PyTorch, HuggingFace Transformers, nnsight, custom hooks
