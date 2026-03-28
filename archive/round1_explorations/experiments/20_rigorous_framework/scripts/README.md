# Experiment Scripts

This folder contains the actual experiment scripts that USE the framework.

## Structure

```
scripts/
├── run_all.py          # Master runner script
├── step1_baseline_tom.py       # Phase 1: Baselines
├── step4_logit_lens.py         # Phase 2: Circuit Discovery  
├── step5_head_ablation_sweep.py
└── README.md
```

## Running Experiments

### Run All Steps
```bash
python scripts/run_all.py
```

### Run Specific Steps
```bash
python scripts/run_all.py step1          # Just step 1
python scripts/run_all.py step1 step4    # Steps 1 and 4
```

### Run Individual Script Directly
```bash
python scripts/step1_baseline_tom.py
```

## Output

Each step produces:
- **Results**: `results/stepN_*.json` - Raw data and statistics
- **Figures**: `figures/stepN_*.png` - Publication-ready plots

## Adding New Steps

1. Create `stepN_your_analysis.py` following the template
2. Add entry to `STEPS` list in `run_all.py`
3. Document hypothesis and methodology in docstring

### Step Template

```python
"""
Step N: Your Analysis Name

HYPOTHESIS: What you're testing

METHODOLOGY:
- Sample sizes
- Controls
- Statistics

OUTPUT: results/stepN_*.json, figures/stepN_*.png
"""

import sys
from pathlib import Path

FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig
from analysis import ...
from scenarios import ...

def main():
    print("STEP N: YOUR ANALYSIS")
    
    config = ExperimentConfig()
    
    # Your analysis code here
    
    # Save results
    # Generate figures

if __name__ == "__main__":
    main()
```

