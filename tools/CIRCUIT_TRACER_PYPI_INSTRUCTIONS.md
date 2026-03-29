# Instructions: Converting circuit-tracer into a PyPI Library

These instructions are for another Claude instance to fork and publish
`circuit-tracer` as a proper PyPI package, modeled after the `neural-assemblies`
library at https://github.com/Caerii/assemblies.

---

## Context

- **Source repo**: https://github.com/decoderesearch/circuit-tracer
- **Current version**: 0.4.1
- **Current build system**: hatchling (already has pyproject.toml)
- **NOT on PyPI**: `pip install circuit-tracer` does not work
- **Goal**: Fork it, clean it up, publish to PyPI as `circuit-tracer`
- **Template to follow**: https://github.com/Caerii/assemblies (`neural-assemblies` on PyPI)

---

## Step 1: Fork and Clone

```bash
# Fork https://github.com/decoderesearch/circuit-tracer on GitHub first
git clone https://github.com/YOUR_USERNAME/circuit-tracer.git
cd circuit-tracer
```

---

## Step 2: Understand the Current Structure

The package is already well-organized:

```
circuit_tracer/
├── __init__.py                    # Lazy-loading entry point
├── __main__.py                    # CLI: `circuit-tracer attribute ...`
├── graph.py                       # Graph data structure
├── attribution/                   # Core attribution computation
│   ├── attribute.py               # Unified interface (routes to backend)
│   ├── attribute_nnsight.py       # nnsight backend
│   ├── attribute_transformerlens.py  # TransformerLens backend
│   ├── context_nnsight.py
│   ├── context_transformerlens.py
│   └── targets.py                 # LogitTarget specs
├── replacement_model/             # Model wrappers
│   ├── replacement_model.py       # Factory class
│   ├── replacement_model_nnsight.py
│   └── replacement_model_transformerlens.py
├── transcoder/                    # Transcoder loading
│   ├── single_layer_transcoder.py
│   ├── cross_layer_transcoder.py
│   └── activation_functions.py
├── frontend/                      # Web visualization server
│   ├── local_server.py
│   ├── graph_models.py
│   ├── feature_models.py
│   └── assets/                    # HTML/CSS/JS for interactive UI
└── utils/
    ├── create_graph_files.py
    ├── hf_utils.py
    ├── caching.py
    ├── demo_utils.py
    ├── tl_nnsight_mapping.py
    ├── disk_offload.py
    └── salient_logits.py
```

**Public API** (3 main exports):
1. `ReplacementModel` — Factory for backend-agnostic model wrappers
2. `Graph` — Attribution result data structure
3. `attribute()` — Main function to compute attribution graphs

**CLI**: `circuit-tracer attribute --prompt "..." --transcoder_set gemma`

---

## Step 3: Examine and Update pyproject.toml

The current pyproject.toml uses hatchling. You have two choices:

### Option A: Keep hatchling (minimal changes)
The current build system works. Just ensure metadata is complete for PyPI.

### Option B: Switch to setuptools (match neural-assemblies pattern)
Switch to setuptools for consistency with our other projects.

**Recommended: Option A** (less risk of breaking things).

### Required pyproject.toml Changes

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "circuit-tracer"
version = "0.4.1"  # Keep existing version initially
description = "Attribution graphs and circuit tracing for language model interpretability"
readme = "README.md"
license = { text = "MIT" }  # Check their actual license!
requires-python = ">=3.10"
authors = [
    # Keep original authors, add yourself as maintainer
    { name = "Decode Research" },
    { name = "YOUR_NAME", email = "YOUR_EMAIL" },
]
keywords = [
    "mechanistic-interpretability", "circuit-tracing", "attribution-graphs",
    "sparse-autoencoders", "transcoders", "transformer-lens",
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "einops>=0.8.0",
    "huggingface-hub<1.0.0",
    "nnsight>=0.6.0",
    "numpy>=1.24.0",
    "pydantic>=2.0.0",
    "safetensors>=0.5.0",
    "tokenizers>=0.21.0",
    "torch>=2.0.0",
    "tqdm>=4.60.0",
    "transformer-lens>=2.16.0",
    "transformers>=4.56.0,<=4.57.3",
]

[project.urls]
Homepage = "https://github.com/YOUR_USERNAME/circuit-tracer"
Documentation = "https://github.com/YOUR_USERNAME/circuit-tracer#readme"
Repository = "https://github.com/YOUR_USERNAME/circuit-tracer"
"Original Repository" = "https://github.com/decoderesearch/circuit-tracer"

[project.scripts]
circuit-tracer = "circuit_tracer.__main__:main"

[project.optional-dependencies]
viz = [
    "ipykernel>=6.29.5,<7.0.0",
    "ipywidgets>=8.1.7",
    "seaborn>=0.13.2",
]
dev = [
    "pytest>=8.0.0",
    "ruff>=0.12.7",
    "pyright>=1.1.403",
    "ipython>=8.37.0",
]
all = [
    "circuit-tracer[viz,dev]",
]
```

### Key Changes from Original:
1. **Split optional deps**: `ipykernel`, `ipywidgets`, `seaborn` moved to `[viz]` extra
   (users who just want the Python API shouldn't need Jupyter)
2. **Added classifiers and keywords** for PyPI discoverability
3. **Added project.urls** including link to original repo (give credit)
4. **Added project.scripts** for CLI entry point
5. **Loosened nnsight pin** from `==0.6.1` to `>=0.6.0` (exact pins cause conflicts)

---

## Step 4: Fix __init__.py for Version

Add `__version__` to `circuit_tracer/__init__.py`. The current init uses lazy
loading and doesn't expose a version. Add:

```python
__version__ = "0.4.1"  # Keep in sync with pyproject.toml
```

The existing `__init__.py` uses `__getattr__` for lazy loading of
`ReplacementModel`, `Graph`, and `attribute`. Keep that pattern but add
the version string.

---

## Step 5: Add MANIFEST.in

Create `MANIFEST.in` to include frontend assets in source distributions:

```
include LICENSE
include README.md
include pyproject.toml

# Frontend assets for visualization server
recursive-include circuit_tracer/frontend/assets *.html *.css *.js *.json *.svg

# Documentation
recursive-include circuit_tracer/utils *.md

global-exclude __pycache__
global-exclude *.py[cod]
global-exclude .pytest_cache
prune .venv
prune dist
```

**Important**: The frontend/ directory contains HTML/CSS/JS assets that
the visualization server needs at runtime. These MUST be included in the
distribution.

Also ensure hatchling includes them. Add to pyproject.toml:

```toml
[tool.hatch.build.targets.wheel]
packages = ["circuit_tracer"]

[tool.hatch.build.targets.sdist]
include = [
    "circuit_tracer/",
    "LICENSE",
    "README.md",
]
```

---

## Step 6: Add py.typed Marker

Create an empty file for PEP 561 type checking support:

```bash
touch circuit_tracer/py.typed
```

---

## Step 7: Add GitHub Actions for PyPI Publishing

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]
  workflow_dispatch:

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Lint
        run: ruff check circuit_tracer/
      - name: Test
        run: pytest tests/ -x -q
        continue-on-error: true  # Tests may need GPU; don't block publish

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install build tools
        run: pip install build
      - name: Build
        run: python -m build
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  publish:
    needs: build
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write  # Required for Trusted Publishers
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1
```

### One-Time PyPI Setup (Trusted Publishers):
1. Go to https://pypi.org/manage/account/publishing/
2. Add a new pending publisher:
   - PyPI project name: `circuit-tracer`
   - Owner: `YOUR_GITHUB_USERNAME`
   - Repository: `circuit-tracer`
   - Workflow name: `publish.yml`
   - Environment: `pypi`

---

## Step 8: Test the Build Locally

```bash
# Install build tools
pip install build twine

# Build
python -m build

# Check the distribution
twine check dist/*

# Verify wheel contents
unzip -l dist/circuit_tracer-0.4.1-py3-none-any.whl | head -30

# Test install from wheel
pip install dist/circuit_tracer-0.4.1-py3-none-any.whl

# Verify
python -c "from circuit_tracer import attribute, Graph, ReplacementModel; print('OK')"
python -c "import circuit_tracer; print(circuit_tracer.__version__)"
```

---

## Step 9: Test Upload to TestPyPI First

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ circuit-tracer
```

---

## Step 10: Publish

Option A (manual):
```bash
twine upload dist/*
```

Option B (automated — preferred):
```bash
git tag v0.4.1
git push origin v0.4.1
# Then create a GitHub Release from the tag
# The publish.yml workflow will handle the rest
```

---

## Checklist Before First Publish

- [ ] Fork created and cloned
- [ ] License checked (MIT? Apache? — respect original)
- [ ] `__version__` added to `__init__.py`
- [ ] pyproject.toml updated with full metadata
- [ ] MANIFEST.in created (include frontend assets!)
- [ ] py.typed marker created
- [ ] `python -m build` succeeds
- [ ] `twine check dist/*` passes
- [ ] `pip install dist/*.whl` works
- [ ] `from circuit_tracer import attribute, Graph, ReplacementModel` works
- [ ] CLI works: `circuit-tracer --help`
- [ ] Frontend assets included in wheel (check with `unzip -l`)
- [ ] .github/workflows/publish.yml created
- [ ] Trusted Publisher configured on PyPI
- [ ] TestPyPI upload successful
- [ ] README credits original authors

---

## Version Strategy Going Forward

- **0.4.1**: First PyPI release (matches current version)
- **0.4.2+**: Bug fixes and compatibility improvements
- **0.5.0**: Any API changes or new features you add
- **1.0.0**: When API is stable

Keep `__version__` in `__init__.py` in sync with `version` in `pyproject.toml`.
Consider using `importlib.metadata` instead of hardcoding:

```python
# In __init__.py
from importlib.metadata import version
__version__ = version("circuit-tracer")
```

---

## Things to Watch Out For

1. **Frontend assets**: The `circuit_tracer/frontend/assets/` directory contains
   the web UI. If these aren't in the wheel, the `--server` flag will fail.
   Verify with `unzip -l dist/*.whl | grep assets`.

2. **transformers pin**: `<=4.57.3` is a hard upper bound. This WILL cause
   dependency conflicts for users with newer transformers. Consider loosening
   to `>=4.56.0` and testing, or adding a warning instead of a hard pin.

3. **nnsight exact pin**: Original uses `==0.6.1`. Loosen to `>=0.6.0` to
   avoid conflicts. Test that 0.6.1+ still works.

4. **torch not pinned to CUDA version**: Good — let users manage their own
   torch installation. Don't add CUDA-specific torch deps.

5. **Original authors**: Credit them prominently in README and maintain their
   license. This is a fork, not a steal.

6. **Name availability**: https://pypi.org/project/circuit-tracer/ currently
   returns "No project description provided" with no releases — meaning the
   name is registered but empty. You may need to contact the original authors
   or choose a different name like `circuit-tracer-mi` if you can't claim it.
   CHECK THIS FIRST before doing all the work.
