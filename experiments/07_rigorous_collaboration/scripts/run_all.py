"""
Run All Steps
=============
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

steps = [
    "step1_generate_data.py",
    "step2_extract_activations.py",
    "step3_analyze.py",
    "step4_visualize.py",
]

print("=" * 60)
print("RUNNING ALL STEPS")
print("=" * 60)

for i, step in enumerate(steps, 1):
    print(f"\n{'=' * 60}")
    print(f"STEP {i}: {step}")
    print("=" * 60)
    
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / step)],
        cwd=str(SCRIPTS_DIR.parent)
    )
    
    if result.returncode != 0:
        print(f"\n[ERROR] Step {i} failed!")
        sys.exit(1)

print("\n" + "=" * 60)
print("ALL STEPS COMPLETE!")
print("=" * 60)
























