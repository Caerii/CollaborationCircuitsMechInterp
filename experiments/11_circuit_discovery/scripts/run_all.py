"""
Run All Circuit Discovery Steps
================================

Orchestrates the full pipeline with timing.
"""

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent.parent.parent
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

STEPS = [
    ("efficient_circuit_analysis.py", "Layer & Head Analysis"),
    ("causal_head_ablation.py", "Causal Ablation Test"),
]


def run_step(script: str, name: str) -> float:
    """Run a step and return time taken."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {name}")
    print(f"{'='*60}\n")
    
    start = time.perf_counter()
    
    result = subprocess.run(
        [str(PYTHON), "-u", str(SCRIPTS_DIR / script)],
        cwd=str(PROJECT_ROOT),
    )
    
    elapsed = time.perf_counter() - start
    
    if result.returncode != 0:
        print(f"\n[FAILED] {script} returned {result.returncode}")
    else:
        print(f"\n[OK] {name} completed in {elapsed:.1f}s")
    
    return elapsed


def main():
    total_start = time.perf_counter()
    times = {}
    
    print("=" * 60)
    print("CIRCUIT DISCOVERY PIPELINE")
    print("=" * 60)
    
    for script, name in STEPS:
        times[name] = run_step(script, name)
    
    total_time = time.perf_counter() - total_start
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("\nTiming Summary:")
    for name, t in times.items():
        print(f"  {name:<30} {t:>6.1f}s")
    print(f"  {'TOTAL':<30} {total_time:>6.1f}s")


if __name__ == "__main__":
    main()























