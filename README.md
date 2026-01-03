# Collaboration Circuits: Mechanistic Interpretability

**Project A: Self/Other/User Representation Separation**

Testing whether LLMs form distinct internal representations for different entities (User, Self, Other agents) in multi-party conversations.

## Quick Start

### 1. Setup (Windows with RTX 3080)

```powershell
# Install uv if needed
irm https://astral.sh/uv/install.ps1 | iex

# Create environment and install with CUDA
uv venv .venv --python 3.11
.\.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA 12.1 (for RTX 3080)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project
uv pip install -e ".[dev]"
```

Or run the setup script:
```powershell
.\scripts\setup.ps1
```

### 2. Verify Setup

```powershell
python scripts/test_setup.py
```

### 3. Run Experiment

```powershell
# Full pipeline
python scripts/run_experiment.py

# Or run phases separately
python scripts/run_experiment.py --phase generate   # Create dialogues
python scripts/run_experiment.py --phase extract    # Extract activations
python scripts/run_experiment.py --phase probe      # Train probes
python scripts/run_experiment.py --phase analyze    # Generate results
```

## Project Structure

```
├── src/
│   ├── config.py           # Configuration
│   ├── model.py            # Model loading + activation extraction
│   ├── data_generation.py  # Synthetic dialogue generator
│   └── probing.py          # Linear probing pipeline
├── scripts/
│   ├── setup.ps1           # Windows setup script
│   ├── test_setup.py       # Verify installation
│   ├── extract_test.py     # Test activation extraction
│   └── run_experiment.py   # Main experiment runner
├── data/                   # Generated dialogues + activations
├── results/                # Probe results + visualizations
├── docs/                   # Research documentation
└── pyproject.toml          # Dependencies (uv/pip)
```

## Hardware

- **GPU**: RTX 3080 10GB
- **Model**: Qwen2.5-3B-Instruct (HuggingFace, ~6GB FP16)
- **Note**: Your LM Studio GGUF model is great for inference, but mech interp needs HuggingFace for activation access

## Research Question

> When processing multi-party conversations (User + Agent A + Agent B), does the model form **distinct internal representations** for each party?

### Expected Outputs

1. **Probe accuracy heatmap** - Which layers encode entity info?
2. **Representation similarity** - How similar are User/Self/Other representations?
3. **Causal test** (optional) - Does steering entity representations change behavior?

## Documentation

See [docs/](docs/) for full research plan:
- [application-project.md](docs/application-project.md) - 20-hour MATS project plan
- [thesis.md](docs/thesis.md) - Core thesis
- [neel-alignment.md](docs/neel-alignment.md) - Alignment with research interests
