"""Study 1 Pilot: End-to-end pipeline validation.

Run this FIRST to validate the full pipeline works before committing to n=50+ runs.

Steps:
  1. Generate 5 stimulus pairs (10 scenarios total)
  2. Run behavioral baseline via LM Studio
  3. Run circuit-tracer attribution on 2 examples
  4. Print summary

Usage:
  # Step 1: Start LM Studio with Qwen3-4B loaded
  # Step 2: Run behavioral baseline
  python studies/01_circuit_atlas/pilot.py --behavioral

  # Step 3: Close LM Studio to free VRAM
  # Step 4: Run circuit-tracer
  python studies/01_circuit_atlas/pilot.py --mechanistic

  # Or run just stimulus generation (no GPU needed)
  python studies/01_circuit_atlas/pilot.py --stimuli-only
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.scenarios.generator import generate_false_belief_set, Stimulus
from lib.utils.config import ExperimentConfig, SYSTEM_PROMPT

PILOT_DIR = Path(__file__).parent / "pilot_results"


def generate_stimuli() -> list[Stimulus]:
    """Generate a small pilot stimulus set."""
    stimuli = generate_false_belief_set(base_id="pilot", n_sets=5, seed=42)
    print(f"Generated {len(stimuli)} stimuli ({len(stimuli)//8} base scenarios x 8 variants)")

    # Show a couple examples
    for s in stimuli[:2]:
        print(f"\n--- {s.scenario_id} ({s.condition}) ---")
        print(f"Text: {s.text}")
        print(f"Question: {s.question}")
        print(f"Correct: {s.correct_answer}")
        print(f"Heuristics: first={s.first_mention_answer}, recency={s.recency_answer}, reality={s.reality_answer}")

    # Save
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    stimuli_data = [
        {
            "id": s.scenario_id,
            "text": s.text,
            "question": s.question,
            "correct": s.correct_answer,
            "condition": s.condition,
            "first_mention": s.first_mention_answer,
            "recency": s.recency_answer,
            "reality": s.reality_answer,
        }
        for s in stimuli
    ]
    with open(PILOT_DIR / "stimuli.json", "w") as f:
        json.dump(stimuli_data, f, indent=2)
    print(f"\nSaved to {PILOT_DIR / 'stimuli.json'}")

    return stimuli


def run_behavioral(stimuli: list[Stimulus]) -> None:
    """Run stimuli through LM Studio and evaluate accuracy."""
    from lib.core.chat import run_scenario

    config = ExperimentConfig()
    results = []

    print(f"\nRunning {len(stimuli)} scenarios through LM Studio...")
    print("(Make sure Qwen3-4B is loaded in LM Studio)\n")

    for i, s in enumerate(stimuli):
        try:
            resp = run_scenario(
                scenario_text=s.text,
                question=s.question,
                config=config,
            )

            # Check if correct answer appears in response
            answer_lower = resp.answer.lower()
            correct = s.correct_answer.lower() in answer_lower
            wrong_loc = (
                s.reality_answer.lower() in answer_lower
                if s.condition == "false_belief"
                else s.first_mention_answer.lower() in answer_lower
            )

            results.append({
                "id": s.scenario_id,
                "condition": s.condition,
                "correct_answer": s.correct_answer,
                "model_answer": resp.answer[:200],
                "thinking": resp.thinking[:500] if resp.thinking else "",
                "got_correct": correct,
                "got_wrong": wrong_loc,
            })

            status = "CORRECT" if correct else ("WRONG" if wrong_loc else "UNCLEAR")
            print(f"  [{i+1}/{len(stimuli)}] {s.scenario_id}: {status}")

        except Exception as e:
            print(f"  [{i+1}/{len(stimuli)}] {s.scenario_id}: ERROR - {e}")
            results.append({
                "id": s.scenario_id,
                "condition": s.condition,
                "error": str(e),
            })

    # Summary
    by_condition = {}
    for r in results:
        cond = r["condition"]
        if cond not in by_condition:
            by_condition[cond] = {"correct": 0, "total": 0}
        by_condition[cond]["total"] += 1
        if r.get("got_correct"):
            by_condition[cond]["correct"] += 1

    print("\n=== BEHAVIORAL PILOT RESULTS ===")
    for cond, counts in by_condition.items():
        acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
        print(f"  {cond}: {counts['correct']}/{counts['total']} ({acc:.0%})")

    # Heuristic comparison
    with open(PILOT_DIR / "stimuli.json") as f:
        stimuli_data = json.load(f)

    for heuristic in ["first_mention", "recency", "reality"]:
        correct = sum(
            1 for s in stimuli_data
            if s["condition"] == "false_belief" and s[heuristic] == s["correct"]
        )
        total = sum(1 for s in stimuli_data if s["condition"] == "false_belief")
        print(f"  {heuristic} heuristic on FB: {correct}/{total} ({correct/total:.0%})")

    # Save
    with open(PILOT_DIR / "behavioral_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {PILOT_DIR / 'behavioral_results.json'}")


def run_mechanistic() -> None:
    """Run circuit-tracer on a couple examples to validate the pipeline."""
    import torch
    from lib.core.models import load_circuit_tracer_model, check_vram

    print("\n=== VRAM CHECK ===")
    vram = check_vram()
    print(f"  GPU: {vram.get('gpu', 'N/A')}")
    print(f"  Free: {vram.get('free_gb', 'N/A')} GB")

    if vram.get("free_gb", 0) < 6:
        print("\n  WARNING: Less than 6GB free. Close LM Studio first!")
        print("  Aborting mechanistic pilot.")
        return

    # Load
    print("\nLoading Qwen3-4B with circuit-tracer...")
    model = load_circuit_tracer_model("qwen3-4b")
    print("  Model loaded.")

    vram = check_vram()
    print(f"  VRAM after load: {vram.get('allocated_gb', '?')} GB used")

    # Load stimuli
    stimuli_path = PILOT_DIR / "stimuli.json"
    if not stimuli_path.exists():
        print("  No stimuli found. Run --stimuli-only first.")
        return

    with open(stimuli_path) as f:
        stimuli_data = json.load(f)

    # Pick one false belief and one true belief
    fb = next(s for s in stimuli_data if s["condition"] == "false_belief")
    tb = next(s for s in stimuli_data if s["condition"] == "true_belief")

    for label, stim in [("FALSE_BELIEF", fb), ("TRUE_BELIEF", tb)]:
        prompt = f"{stim['text']}\n\n{stim['question']}"
        print(f"\n--- Circuit-tracer: {label} ---")
        print(f"  Prompt: {prompt[:100]}...")
        print(f"  Correct answer: {stim['correct']}")

        try:
            from circuit_tracer import attribute
            graph = attribute(model=model, prompt=prompt)
            print(f"  Graph computed successfully!")
            print(f"  Saving to {PILOT_DIR / f'{label.lower()}_graph.pt'}...")
            graph.to_pt(PILOT_DIR / f"{label.lower()}_graph.pt")
            print(f"  Done.")
        except Exception as e:
            print(f"  ERROR: {e}")
            print(f"  This is expected if transcoders aren't cached yet.")
            print(f"  The error message should indicate what to download.")

    print("\n=== MECHANISTIC PILOT COMPLETE ===")
    print("If graphs were computed, the pipeline works end-to-end.")
    print("Proceed to full Study 1 with n=50 stimulus sets.")


def main():
    parser = argparse.ArgumentParser(description="Study 1 Pilot")
    parser.add_argument("--stimuli-only", action="store_true", help="Just generate stimuli")
    parser.add_argument("--behavioral", action="store_true", help="Run behavioral via LM Studio")
    parser.add_argument("--mechanistic", action="store_true", help="Run circuit-tracer")
    args = parser.parse_args()

    if not any([args.stimuli_only, args.behavioral, args.mechanistic]):
        parser.print_help()
        print("\nRun with --stimuli-only first, then --behavioral, then --mechanistic")
        return

    stimuli = generate_stimuli()

    if args.behavioral:
        run_behavioral(stimuli)

    if args.mechanistic:
        run_mechanistic()


if __name__ == "__main__":
    main()
