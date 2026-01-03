# Theory of Mind Circuits in Qwen3-4B

Mechanistic interpretability research investigating how language models implement Theory of Mind (ToM)—the ability to reason about mental states that differ from reality.

**MATS 10.0 Application Project** | 20 Experiments | ~16 Hours

## Research Question

> How does Qwen3-4B implement Theory of Mind? Is it genuine reasoning or surface-level heuristics?

## Key Findings

### 1. ToM is a Reasoning Skill
- **Chat mode + reasoning tokens required**: 75% FB / 95% TB accuracy with `<think>` tags
- **Completion mode fails**: 50-80% FB / 0-20% TB accuracy
- ToM requires computational budget, not hard-coded retrieval

### 2. Distinct Circuits for Single vs Multi-Agent
- **Single-agent ToM**: Late layers (32-34)
- **Multi-agent ToM**: Early-mid layers (0-22)
- **Overlap**: Only L34H0

### 3. Inhibitor Head Discovery
- **L18H16**: Ablating improves multi-agent accuracy from 50% to 75%
- A head that actively interferes with correct reasoning

### 4. Statistical Validation
- Sally-Anne: 81% belief-based predictions (N=200, p<10⁻¹⁹, Cohen's h=0.67)
- L12/24/30 H0 ablation: p=0.022 vs controls

### 5. Critical Self-Correction
- Transfer test revealed probes detected tokens, not concepts (99.9% → 32%)
- "First-mention heuristic" was actually correct original-location tracking

## Experiment Structure

```
experiments/
├── 01-05: Entity Representation (probing, geometry, steering, transfer)
├── 06-08: Collaboration & Belief Tracking
├── 09-10: False Belief / Sally-Anne Paradigm
├── 11-13: Circuit Discovery (attention, information theory, causal)
├── 14-16: Rigorous Reboot (statistical corrections)
├── 17-18: Presence Tracking & Validation
├── 19: Methodological Controls (breakthrough: chat mode matters)
└── 20: Rigorous Framework (comprehensive library + final analysis)
```

## Technical Setup

| Component | Specification |
|-----------|---------------|
| Model | Qwen3-4B (36 layers × 32 heads) |
| Hardware | RTX 3080 10GB |
| Framework | PyTorch + TransformerLens |
| Methods | Ablation, SAE decomposition, linear probing |

## Quick Start

### Setup (Windows)

```powershell
# Install uv
irm https://astral.sh/uv/install.ps1 | iex

# Create environment
uv venv .venv --python 3.11
.\.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA 12.1
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project
uv pip install -e ".[dev]"
```

### Run Experiments

```powershell
# Experiment 20 (rigorous framework) - recommended
cd experiments/20_rigorous_framework
python scripts/step1_behavioral.py    # Behavioral ToM tests
python scripts/step5_ablation.py      # Single-agent circuit discovery
python scripts/step10_multiagent.py   # Multi-agent circuits
python scripts/step10c_smart_circuit_hunt.py  # Smart filtering pipeline
```

## Key Results Files

| File | Description |
|------|-------------|
| `experiments/20_rigorous_framework/FINAL_RESEARCH_SUMMARY.md` | Comprehensive findings |
| `experiments/20_rigorous_framework/results/step10_multiagent_ablation.json` | Multi-agent circuit data |
| `experiments/20_rigorous_framework/figures/` | All visualizations |
| `MATS_EXECUTIVE_SUMMARY.md` | 600-word application summary |
| `MATS_DETAILED_WRITEUP.md` | Full research narrative |

## Methodology Highlights

### 3-Stage Smart Filtering Pipeline (Step 10c)
1. **SAE Layer Screening**: Identify layers with highest FB/TB discriminability
2. **Attention Pattern Filtering**: Score heads by attention to relevant tokens
3. **Targeted Ablation**: Test only 7% of heads (80/1152), achieving 14.4x speedup

### SAE Feature Decomposition
- Feature #1979: "Agent has outdated information" (FB-TB diff: +2.12)
- Feature #4772: "Agent observed the event" (-0.77)
- Only ~13 features active per input (0.1% sparsity)

## Limitations

- Sample sizes below recommended (n=8-50 vs target n≥50)
- Single model; cross-architecture generalization unknown
- Novel locations untested in chat mode
- Activation patching corrupts chat-mode generation

## Repository Structure

```
├── experiments/           # All 20 experiments with scripts, data, results
├── src/                   # Core library code
├── docs/                  # Research documentation
├── MATS_EXECUTIVE_SUMMARY.md   # Application summary (600 words)
├── MATS_DETAILED_WRITEUP.md    # Full writeup (3000+ words)
└── pyproject.toml         # Dependencies
```

## Documentation

- [MATS Executive Summary](MATS_EXECUTIVE_SUMMARY.md) - 600-word overview
- [MATS Detailed Writeup](MATS_DETAILED_WRITEUP.md) - Full research narrative
- [Experiment 20 README](experiments/20_rigorous_framework/README.md) - Final framework docs
- [Final Research Summary](experiments/20_rigorous_framework/FINAL_RESEARCH_SUMMARY.md) - Comprehensive findings

## Citation

```
Jakir, A. (2025). Theory of Mind Circuits in Qwen3-4B: A Mechanistic Interpretability Analysis.
MATS 10.0 Application Project.
```
