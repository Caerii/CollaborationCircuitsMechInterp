#!/usr/bin/env python
"""
Runner script for the ultra-fast sweep.
Execute this with: uv run experiments/19_methodological_controls/runners/run_ultrafast.py
"""

import sys
import os
from pathlib import Path

# Add the parent directory to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiments" / "19_methodological_controls"))

# Import and run
from scripts.step19_ultrafast_sweep import run_ultrafast_sweep

if __name__ == "__main__":
    print("=" * 70)
    print("ULTRA-FAST GARGANTUAN SWEEP - Runner Script")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    run_ultrafast_sweep()



