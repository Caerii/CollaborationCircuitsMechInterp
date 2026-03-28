"""
Run all multi-agent ToM experiments in sequence.
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

scripts = [
    "step1_generate_recursive_scenarios.py",
    "step2_behavioral_test.py",
    "step3_per_agent_probing.py",
]

def main():
    print("=" * 60)
    print("RUNNING MULTI-AGENT TOM EXPERIMENTS")
    print("=" * 60)
    
    for script in scripts:
        print(f"\n{'='*60}")
        print(f"Running: {script}")
        print("=" * 60)
        
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / script)],
            cwd=str(SCRIPTS_DIR.parent),
        )
        
        if result.returncode != 0:
            print(f"[!] {script} failed with code {result.returncode}")
            return result.returncode
        
        print(f"[OK] {script} completed")
    
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())



