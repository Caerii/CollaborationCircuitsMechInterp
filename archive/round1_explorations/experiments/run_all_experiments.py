"""
Master Experiment Runner
=========================

Runs all experiments in order with timing and logging.

Experiments:
1. 10_proper_tom: Rigorous ToM methodology
2. 11_circuit_discovery: Find ToM heads
3. 12_information_theory: MI analysis
4. 13_causal_steering: Causal interventions
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = Path(__file__).parent
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
LOG_FILE = EXPERIMENTS_DIR / "experiment_log.json"


EXPERIMENTS = [
    # (folder, script, name)
    ("11_circuit_discovery", "scripts/efficient_circuit_analysis.py", "Circuit Analysis"),
    ("12_information_theory", "scripts/information_analysis.py", "Information Theory"),
    ("13_causal_steering", "scripts/steering_experiments.py", "Causal Steering"),
    ("13_causal_steering", "scripts/activation_patching.py", "Activation Patching"),
]


def run_experiment(folder: str, script: str, name: str) -> dict:
    """Run a single experiment and return results."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {name}")
    print(f"{'='*60}\n")
    
    script_path = EXPERIMENTS_DIR / folder / script
    
    if not script_path.exists():
        print(f"  [SKIP] Script not found: {script_path}")
        return {"name": name, "status": "skipped", "time": 0}
    
    start = time.perf_counter()
    
    try:
        result = subprocess.run(
            [str(PYTHON), "-u", str(script_path)],
            cwd=str(PROJECT_ROOT),
            timeout=600,  # 10 minute timeout per experiment
        )
        status = "success" if result.returncode == 0 else f"failed (code {result.returncode})"
    except subprocess.TimeoutExpired:
        status = "timeout"
    except Exception as e:
        status = f"error: {str(e)}"
    
    elapsed = time.perf_counter() - start
    
    print(f"\n[{status.upper()}] {name} completed in {elapsed:.1f}s")
    
    return {
        "name": name,
        "folder": folder,
        "script": script,
        "status": status,
        "time_seconds": elapsed,
    }


def main():
    print("=" * 60)
    print("MASTER EXPERIMENT RUNNER")
    print("=" * 60)
    print(f"\nStarted at: {datetime.now().isoformat()}")
    print(f"Running {len(EXPERIMENTS)} experiments\n")
    
    total_start = time.perf_counter()
    results = []
    
    for folder, script, name in EXPERIMENTS:
        result = run_experiment(folder, script, name)
        results.append(result)
    
    total_time = time.perf_counter() - total_start
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Experiment':<30} {'Status':<15} {'Time':<10}")
    print("-" * 55)
    
    for r in results:
        print(f"{r['name']:<30} {r['status']:<15} {r['time_seconds']:.1f}s")
    
    print("-" * 55)
    print(f"{'TOTAL':<30} {'':<15} {total_time:.1f}s")
    
    # Save log
    log = {
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": total_time,
        "experiments": results,
        "success_count": sum(1 for r in results if r["status"] == "success"),
        "total_count": len(results),
    }
    
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)
    
    print(f"\n[OK] Log saved to {LOG_FILE}")
    
    # Status
    successes = sum(1 for r in results if r["status"] == "success")
    print(f"\nCompleted: {successes}/{len(results)} experiments successful")


if __name__ == "__main__":
    main()























