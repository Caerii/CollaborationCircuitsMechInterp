"""Thin-slice mechanistic analysis: FB-correct vs FB-wrong via circuit-tracer.

The key question: what's different in the attribution graph when the model
correctly tracks a false belief vs when it defaults to reality?

We have matched pairs from the behavioral pilot:
- Slice A: Model gets BOTH FB and TB correct (genuine ToM)
- Slice B: Model gets TB correct but FB wrong (reality bias)

Same model, same architecture — the difference must be in which circuits activate.

Usage:
  # 1. Close LM Studio (need the VRAM)
  # 2. Run this script
  python studies/01_circuit_atlas/thin_slice.py
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = Path(__file__).parent / "thin_slice_results"


def load_slices() -> dict:
    """Load the curated thin slices from behavioral pilot."""
    stimuli_path = Path(__file__).parent / "pilot_results" / "stimuli.json"
    results_path = Path(__file__).parent / "pilot_results" / "instruct_behavioral.json"

    with open(stimuli_path) as f:
        stimuli = json.load(f)
    with open(results_path) as f:
        results = json.load(f)

    stim_by_id = {s["id"]: s for s in stimuli}
    result_by_id = {r["id"]: r for r in results}

    # Curate slices based on behavioral results
    slice_a = []  # FB correct (genuine ToM)
    slice_b = []  # FB wrong (reality bias)

    for base_i in range(5):
        for order in ["AB", "BA"]:
            fb_id = f"pilot_{base_i:03d}_FB_{order}"
            tb_id = f"pilot_{base_i:03d}_TB_{order}"
            fb_stim = stim_by_id[fb_id]
            fb_result = result_by_id[fb_id]

            pair = {
                "fb_id": fb_id,
                "tb_id": tb_id,
                "fb_stimulus": fb_stim,
                "tb_stimulus": stim_by_id[tb_id],
                "fb_correct": fb_result["got_correct"],
            }

            if fb_result["got_correct"]:
                slice_a.append(pair)
            else:
                slice_b.append(pair)

    print(f"Slice A (FB correct, genuine ToM): {len(slice_a)} pairs")
    print(f"Slice B (FB wrong, reality bias):  {len(slice_b)} pairs")
    return {"genuine_tom": slice_a, "reality_bias": slice_b}


def build_prompt(stimulus: dict) -> str:
    """Build the prompt exactly as the model saw it during behavioral testing."""
    return f"{stimulus['text']}\n\n{stimulus['question']} Answer with just the location name."


def run_circuit_tracer(slices: dict) -> None:
    """Run circuit-tracer on thin slices and compare."""
    from lib.core.models import check_vram

    print("\n=== VRAM CHECK ===")
    vram = check_vram()
    print(f"  GPU: {vram.get('gpu', 'N/A')}, Free: {vram.get('free_gb', 'N/A')} GB")
    if vram.get("free_gb", 0) < 6:
        print("  Not enough VRAM. Close LM Studio first!")
        return

    print("\n=== Loading model with circuit-tracer ===")
    from circuit_tracer import attribute
    from lib.core.models import load_circuit_tracer_model

    # Load Qwen3-4B with transcoders
    print("  Loading Qwen3-4B + transcoders (this may download ~2GB first time)...")
    start = time.time()
    model = load_circuit_tracer_model("qwen3-4b")
    print(f"  Loaded in {time.time() - start:.0f}s")

    vram = check_vram()
    print(f"  VRAM after load: {vram.get('allocated_gb', '?')} GB")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run on a few examples from each slice
    for slice_name, pairs in slices.items():
        print(f"\n=== Processing slice: {slice_name} ({len(pairs)} pairs) ===")

        for i, pair in enumerate(pairs[:3]):  # Start with 3 per slice
            fb_stim = pair["fb_stimulus"]
            tb_stim = pair["tb_stimulus"]

            for label, stim in [("FB", fb_stim), ("TB", tb_stim)]:
                prompt = build_prompt(stim)
                stim_id = stim["id"]
                print(f"\n  [{slice_name}] {stim_id}:")
                print(f"    Prompt: {prompt[:80]}...")
                print(f"    Correct: {stim['correct']}")

                try:
                    start = time.time()
                    graph = attribute(
                        model=model,
                        prompt=prompt,
                    )
                    elapsed = time.time() - start
                    print(f"    Graph computed in {elapsed:.1f}s")

                    # Save the graph
                    out_path = RESULTS_DIR / f"{slice_name}_{stim_id}_graph.pt"
                    graph.to_pt(out_path)
                    print(f"    Saved to {out_path.name}")

                except Exception as e:
                    print(f"    ERROR: {e}")
                    # Save the error for debugging
                    error_path = RESULTS_DIR / f"{slice_name}_{stim_id}_error.txt"
                    error_path.write_text(str(e))

    print("\n=== DONE ===")
    print(f"Results saved to {RESULTS_DIR}")
    print("\nNext step: compare attribution graphs between genuine_tom and reality_bias slices")


def compare_graphs() -> None:
    """Compare saved attribution graphs between slices.

    Run this after run_circuit_tracer has completed.
    """
    import torch

    graph_files = sorted(RESULTS_DIR.glob("*_graph.pt"))
    if not graph_files:
        print("No graphs found. Run with --trace first.")
        return

    print(f"Found {len(graph_files)} graph files:")
    for f in graph_files:
        print(f"  {f.name}")

    # TODO: Load graphs, extract top features, compute overlap between slices
    # This is where the actual science happens — will implement after we see
    # what the graph structure looks like.
    print("\nGraph comparison not yet implemented — need to see graph structure first.")
    print("Run --trace first, then we'll build the comparison.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Thin-slice mechanistic analysis")
    parser.add_argument("--trace", action="store_true", help="Run circuit-tracer on thin slices")
    parser.add_argument("--compare", action="store_true", help="Compare saved attribution graphs")
    parser.add_argument("--info", action="store_true", help="Show slice info without loading model")
    args = parser.parse_args()

    if not any([args.trace, args.compare, args.info]):
        parser.print_help()
        print("\nStart with --info to see the slices, then --trace to run circuit-tracer")
        return

    slices = load_slices()

    if args.info:
        print("\n=== SLICE DETAILS ===")
        for name, pairs in slices.items():
            print(f"\n{name}:")
            for p in pairs:
                fb = p["fb_stimulus"]
                print(f"  {fb['id']}: {fb['text'][:60]}... -> {fb['correct']}")
        return

    if args.trace:
        run_circuit_tracer(slices)

    if args.compare:
        compare_graphs()


if __name__ == "__main__":
    main()
