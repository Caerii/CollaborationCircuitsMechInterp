"""
Run All Experiments

Master script to run all experiment steps in sequence.
Can also run individual steps by name.

Usage:
    python run_all.py              # Run all steps
    python run_all.py step1        # Run just step 1
    python run_all.py step1 step4  # Run steps 1 and 4
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

SCRIPTS_DIR = Path(__file__).parent

# Define all steps in order
STEPS = [
    ("step1", "step1_baseline_tom.py", "Baseline ToM Performance"),
    # ("step2", "step2_heuristic_baselines.py", "Heuristic Baselines"),  # TODO
    # ("step3", "step3_higher_order_tom.py", "Higher-Order ToM"),  # TODO
    ("step4", "step4_logit_lens.py", "Logit Lens Analysis"),
    ("step5", "step5_head_ablation_sweep.py", "Head Ablation Sweep"),
    # Add more steps as they're created
]


def run_step(step_name, script_name, description):
    """Run a single step and capture output."""
    print(f"\n{'='*70}")
    print(f"RUNNING: {step_name} - {description}")
    print(f"{'='*70}")
    
    script_path = SCRIPTS_DIR / script_name
    
    if not script_path.exists():
        print(f"WARNING: Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,  # Show output in real-time
            cwd=str(SCRIPTS_DIR.parent),
        )
        
        if result.returncode == 0:
            print(f"\n✓ {step_name} completed successfully")
            return True
        else:
            print(f"\n✗ {step_name} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n✗ {step_name} failed with exception: {e}")
        return False


def main():
    print("=" * 70)
    print("EXPERIMENT 20: RIGOROUS COLLABORATION CIRCUITS")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    
    # Determine which steps to run
    if len(sys.argv) > 1:
        # Run specific steps
        steps_to_run = sys.argv[1:]
        steps = [(name, script, desc) for name, script, desc in STEPS if name in steps_to_run]
        
        if not steps:
            print(f"Error: No valid steps found. Available: {[s[0] for s in STEPS]}")
            return 1
    else:
        # Run all steps
        steps = STEPS
    
    print(f"\nSteps to run: {[s[0] for s in steps]}")
    
    # Run each step
    results = {}
    for step_name, script_name, description in steps:
        success = run_step(step_name, script_name, description)
        results[step_name] = success
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    for step_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {step_name}")
    
    n_success = sum(results.values())
    n_total = len(results)
    print(f"\nCompleted: {n_success}/{n_total} steps")
    
    return 0 if n_success == n_total else 1


if __name__ == "__main__":
    sys.exit(main())

