"""Run all belief tracking experiment steps."""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

steps = [
    "step1_generate_data.py",
    "step2_extract.py", 
    "step3_analyze.py",
    "step4_visualize.py",
]

for step in steps:
    print(f"\n{'='*60}")
    print(f"RUNNING: {step}")
    print("="*60 + "\n")
    
    result = subprocess.run(
        [sys.executable, "-u", str(SCRIPTS_DIR / step)],
        cwd=SCRIPTS_DIR.parent.parent.parent,  # Project root
    )
    
    if result.returncode != 0:
        print(f"\n[ERROR] {step} failed with code {result.returncode}")
        sys.exit(1)

print("\n" + "="*60)
print("ALL STEPS COMPLETE!")
print("="*60)
























