"""
Run All Rigorous Experiments
=============================

Execute the methodologically fixed experiment pipeline.
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SCRIPTS_DIR = Path(__file__).parent
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def run_script(name: str, script: str) -> bool:
    """Run a script and return success status."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {name}")
    print(f"{'='*60}\n")
    
    script_path = SCRIPTS_DIR / script
    
    if not script_path.exists():
        print(f"  [SKIP] Script not found: {script_path}")
        return False
    
    start = time.perf_counter()
    
    try:
        result = subprocess.run(
            [str(PYTHON), "-u", str(script_path)],
            cwd=str(PROJECT_ROOT),
            timeout=1200,  # 20 min timeout
        )
        success = result.returncode == 0
    except subprocess.TimeoutExpired:
        print("  [TIMEOUT]")
        success = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        success = False
    
    elapsed = time.perf_counter() - start
    status = "OK" if success else "FAILED"
    print(f"\n[{status}] {name} completed in {elapsed:.1f}s")
    
    return success


def main():
    print("=" * 60)
    print("RIGOROUS REBOOT - METHODOLOGICALLY FIXED EXPERIMENTS")
    print("=" * 60)
    
    total_start = time.perf_counter()
    results = []
    
    # Step 1: Generate large dataset
    results.append(("Generate Dataset (N=800)", run_script(
        "Step 1: Generate Large Dataset",
        "step1_generate_large_dataset.py"
    )))
    
    # Step 2: Compute null distributions
    results.append(("Null Distributions", run_script(
        "Step 2: Null Distributions",
        "step2_null_distributions.py"
    )))
    
    # Step 3: Behavioral ToM test
    results.append(("Behavioral ToM", run_script(
        "Step 3: Behavioral ToM Test",
        "step3_behavioral_tom_test.py"
    )))
    
    # Step 4: Proper ablation
    results.append(("Proper Ablation", run_script(
        "Step 4: Proper Head Ablation",
        "step4_proper_ablation.py"
    )))
    
    # Summary
    total_time = time.perf_counter() - total_start
    
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    
    for name, success in results:
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {name}")
    
    print(f"\n  Total time: {total_time/60:.1f} minutes")
    print(f"  Success: {sum(1 for _, s in results if s)}/{len(results)}")


if __name__ == "__main__":
    main()






















