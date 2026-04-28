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
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.analysis.circuits import aggregate_feature_summaries, summarize_graph

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


def behavioral_summary() -> dict:
    """Summarize the saved behavioral pilot results."""
    results_path = Path(__file__).parent / "pilot_results" / "instruct_behavioral.json"
    with open(results_path) as f:
        results = json.load(f)

    by_condition = defaultdict(lambda: {"correct": 0, "total": 0})
    false_belief_rows = []
    for row in results:
        condition = row["condition"]
        by_condition[condition]["total"] += 1
        if row["got_correct"]:
            by_condition[condition]["correct"] += 1
        if condition == "false_belief":
            false_belief_rows.append({
                "id": row["id"],
                "correct_answer": row["correct_answer"],
                "got_correct": row["got_correct"],
                "model_output": row["model_output"].strip(),
            })

    condition_rows = {}
    for condition, counts in sorted(by_condition.items()):
        total = counts["total"]
        condition_rows[condition] = {
            "correct": counts["correct"],
            "total": total,
            "accuracy": counts["correct"] / total if total else 0.0,
        }

    return {
        "conditions": condition_rows,
        "false_belief_rows": false_belief_rows,
    }


def attribution_targets_for(stimulus: dict, target_mode: str) -> list[str] | None:
    """Return circuit-tracer attribution targets for a stimulus."""
    if target_mode == "salient":
        return None
    if target_mode == "correct":
        return [stimulus["correct"]]
    if target_mode == "correct-reality":
        targets = [stimulus["correct"]]
        if stimulus["reality"] != stimulus["correct"]:
            targets.append(stimulus["reality"])
        return targets
    raise ValueError(f"Unknown target_mode: {target_mode}")


def resolve_transcoder_set(model_key: str, transcoder_set: str) -> str:
    """Resolve an explicit or default transcoder set for a model."""
    from lib.core.models import get_default_transcoder_set

    return transcoder_set or get_default_transcoder_set(model_key)


def cache_transcoders(model_key: str, transcoder_set: str = "") -> None:
    """Download and store circuit-tracer transcoders in the local cache."""
    from circuit_tracer.utils.caching import get_cached_path, is_cached, save_transcoders_to_cache

    resolved_transcoder_set = resolve_transcoder_set(model_key, transcoder_set)
    if not resolved_transcoder_set:
        print(f"No default transcoder set for {model_key}; pass --transcoder-set.")
        return

    cached_path = get_cached_path(resolved_transcoder_set)
    print(f"Transcoder set: {resolved_transcoder_set}")
    print(f"Cache path: {cached_path}")
    if is_cached(resolved_transcoder_set):
        print("Already cached.")
        return

    print("Downloading transcoders into the circuit-tracer cache...")
    path = save_transcoders_to_cache(resolved_transcoder_set)
    print(f"Cached transcoders at {path}")


def preflight(model_key: str, transcoder_set: str = "") -> None:
    """Check circuit-tracer prerequisites without loading the full model."""
    from lib.core.models import (
        check_vram,
        get_mechanistic_hf_id,
        transcoder_cache_status,
        validate_transformerlens_model,
    )

    resolved_transcoder_set = resolve_transcoder_set(model_key, transcoder_set)
    model_name = get_mechanistic_hf_id(model_key)
    print(f"Model key: {model_key}")
    print(f"Mechanistic model id: {model_name}")
    try:
        validate_transformerlens_model(model_name)
        print("TransformerLens support: ok")
    except RuntimeError as exc:
        print(f"TransformerLens support: ERROR: {exc}")

    if resolved_transcoder_set:
        cache = transcoder_cache_status(resolved_transcoder_set)
        print(f"Transcoder set: {resolved_transcoder_set}")
        print(f"Transcoder cache: {'hit' if cache['cached'] else 'miss'} at {cache['path']}")
    else:
        print("Transcoder set: none resolved")

    vram = check_vram()
    if vram.get("available"):
        print(f"GPU: {vram.get('gpu')}, free {vram.get('free_gb')} GB")
    else:
        print("GPU: unavailable")


def run_circuit_tracer(
    slices: dict,
    max_pairs_per_slice: int = 3,
    model_key: str = "qwen3-4b",
    transcoder_set: str = "",
    target_mode: str = "salient",
    skip_existing: bool = True,
    max_feature_nodes: int | None = None,
    max_graphs: int | None = None,
    require_cached_transcoders: bool = False,
) -> None:
    """Run circuit-tracer on thin slices and compare."""
    from lib.core.models import (
        check_vram,
        load_circuit_tracer_model,
        transcoder_cache_status,
    )

    print("\n=== VRAM CHECK ===")
    vram = check_vram()
    print(f"  GPU: {vram.get('gpu', 'N/A')}, Free: {vram.get('free_gb', 'N/A')} GB")
    if vram.get("free_gb", 0) < 6:
        print("  Not enough VRAM. Close LM Studio first!")
        return

    print("\n=== Loading model with circuit-tracer ===")
    resolved_transcoder_set = resolve_transcoder_set(model_key, transcoder_set)
    if not resolved_transcoder_set:
        print(f"  No default transcoder set for {model_key}; pass --transcoder-set.")
        return

    cache = transcoder_cache_status(resolved_transcoder_set)
    print(f"  Model key: {model_key}")
    print(f"  Transcoder set: {resolved_transcoder_set}")
    print(f"  Transcoder cache: {'hit' if cache['cached'] else 'miss'} at {cache['path']}")
    if require_cached_transcoders and not cache["cached"]:
        print("  Required cached transcoders, but this set is not cached. Aborting before network load.")
        return

    print("  Loading model + transcoders (may download on first uncached run)...")
    start = time.time()
    try:
        model = load_circuit_tracer_model(
            model_key,
            transcoder_set=resolved_transcoder_set,
            require_cached_transcoders=require_cached_transcoders,
        )
    except RuntimeError as exc:
        print(f"  ERROR: {exc}")
        return
    print(f"  Loaded in {time.time() - start:.0f}s")

    vram = check_vram()
    print(f"  VRAM after load: {vram.get('allocated_gb', '?')} GB")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    from circuit_tracer import attribute

    # Run on a few examples from each slice
    graphs_attempted = 0
    for slice_name, pairs in slices.items():
        print(f"\n=== Processing slice: {slice_name} ({len(pairs)} pairs) ===")

        for i, pair in enumerate(pairs[:max_pairs_per_slice]):
            fb_stim = pair["fb_stimulus"]
            tb_stim = pair["tb_stimulus"]

            for label, stim in [("FB", fb_stim), ("TB", tb_stim)]:
                if max_graphs is not None and graphs_attempted >= max_graphs:
                    print(f"\nReached --max-graphs={max_graphs}; stopping trace run.")
                    return

                prompt = build_prompt(stim)
                stim_id = stim["id"]
                out_path = RESULTS_DIR / f"{slice_name}_{stim_id}_graph.pt"
                print(f"\n  [{slice_name}] {stim_id}:")
                print(f"    Prompt: {prompt[:80]}...")
                print(f"    Correct: {stim['correct']}")

                if skip_existing and out_path.exists():
                    print(f"    Skipping existing graph: {out_path.name}")
                    continue

                graphs_attempted += 1
                targets = attribution_targets_for(stim, target_mode)
                if targets:
                    print(f"    Attribution targets: {targets}")

                try:
                    start = time.time()
                    graph = attribute(
                        model=model,
                        prompt=prompt,
                        attribution_targets=targets,
                        max_feature_nodes=max_feature_nodes,
                    )
                    elapsed = time.time() - start
                    print(f"    Graph computed in {elapsed:.1f}s")

                    # Save the graph
                    graph.to_pt(out_path)
                    print(f"    Saved to {out_path.name}")

                except ValueError as e:
                    if not targets:
                        print(f"    ERROR: {e}")
                        error_path = RESULTS_DIR / f"{slice_name}_{stim_id}_error.txt"
                        error_path.write_text(str(e))
                        continue
                    print(f"    Targeted attribution failed: {e}")
                    print("    Retrying with salient logits.")
                    try:
                        start = time.time()
                        graph = attribute(
                            model=model,
                            prompt=prompt,
                            max_feature_nodes=max_feature_nodes,
                        )
                        elapsed = time.time() - start
                        print(f"    Graph computed in {elapsed:.1f}s")
                        graph.to_pt(out_path)
                        print(f"    Saved to {out_path.name}")
                    except Exception as retry_error:
                        print(f"    ERROR: {retry_error}")
                        error_path = RESULTS_DIR / f"{slice_name}_{stim_id}_error.txt"
                        error_path.write_text(str(retry_error))
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
    graph_files = sorted(RESULTS_DIR.glob("*_graph.pt"))
    if not graph_files:
        print("No graphs found. Run with --trace first.")
        print(f"Expected graph files under {RESULTS_DIR}")
        return

    from circuit_tracer import Graph

    print(f"Found {len(graph_files)} graph files:")
    for f in graph_files:
        print(f"  {f.name}")

    stimuli_path = Path(__file__).parent / "pilot_results" / "stimuli.json"
    results_path = Path(__file__).parent / "pilot_results" / "instruct_behavioral.json"
    with open(stimuli_path) as f:
        stimuli = json.load(f)
    with open(results_path) as f:
        results = json.load(f)

    stim_by_id = {s["id"]: s for s in stimuli}
    result_by_id = {r["id"]: r for r in results}

    graph_summaries = []
    errors = []

    for path in graph_files:
        slice_name, stim_id = parse_graph_filename(path.name)
        try:
            graph = Graph.from_pt(str(path), map_location="cpu")
            summary = summarize_graph(graph, top_n=25)
            stim = stim_by_id.get(stim_id, {})
            result = result_by_id.get(stim_id, {})
            summary.update({
                "file": path.name,
                "slice": slice_name,
                "stimulus_id": stim_id,
                "condition": stim.get("condition"),
                "location_order": stim.get("location_order"),
                "correct_answer": stim.get("correct"),
                "reality_answer": stim.get("reality"),
                "model_got_correct": result.get("got_correct"),
            })
            graph_summaries.append(summary)
        except Exception as exc:
            errors.append({"file": path.name, "error": str(exc)})

    by_slice = defaultdict(list)
    for summary in graph_summaries:
        by_slice[summary["slice"]].append(summary)

    comparison = {
        "n_graphs": len(graph_summaries),
        "n_errors": len(errors),
        "slices": {},
        "feature_sets": {},
        "errors": errors,
    }

    for slice_name, rows in sorted(by_slice.items()):
        replacement_scores = [r["replacement_score"] for r in rows]
        completeness_scores = [r["completeness_score"] for r in rows]
        aggregate = aggregate_feature_summaries(rows, frequency_threshold=0.0)
        comparison["slices"][slice_name] = {
            "n_graphs": len(rows),
            "mean_replacement_score": mean(replacement_scores),
            "mean_completeness_score": mean(completeness_scores),
            "top_features_by_frequency": aggregate[:50],
        }
        comparison["feature_sets"][slice_name] = [
            [row["layer"], row["feature_idx"]]
            for row in aggregate
            if row["frequency"] >= 0.25
        ]

    if "genuine_tom" in comparison["feature_sets"] and "reality_bias" in comparison["feature_sets"]:
        genuine = {tuple(x) for x in comparison["feature_sets"]["genuine_tom"]}
        reality = {tuple(x) for x in comparison["feature_sets"]["reality_bias"]}
        comparison["feature_overlap"] = {
            "shared": sorted([list(x) for x in genuine & reality]),
            "genuine_only": sorted([list(x) for x in genuine - reality]),
            "reality_bias_only": sorted([list(x) for x in reality - genuine]),
            "jaccard": len(genuine & reality) / len(genuine | reality) if genuine | reality else 0.0,
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "graph_summaries.json"
    comparison_path = RESULTS_DIR / "graph_comparison.json"
    with open(summary_path, "w") as f:
        json.dump(graph_summaries, f, indent=2)
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n=== GRAPH SUMMARY ===")
    for slice_name, row in comparison["slices"].items():
        print(
            f"  {slice_name}: n={row['n_graphs']}, "
            f"replacement={row['mean_replacement_score']:.3f}, "
            f"completeness={row['mean_completeness_score']:.3f}"
        )
    if errors:
        print(f"  Errors: {len(errors)} graph(s); see {comparison_path.name}")
    print(f"\nSaved graph summaries to {summary_path}")
    print(f"Saved graph comparison to {comparison_path}")
    return



def parse_graph_filename(filename: str) -> tuple[str, str]:
    """Parse filenames like genuine_tom_pilot_000_FB_AB_graph.pt."""
    stem = filename.removesuffix("_graph.pt")
    for prefix in ["genuine_tom_", "reality_bias_"]:
        if stem.startswith(prefix):
            return prefix.removesuffix("_"), stem[len(prefix):]
    return "unknown", stem


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Thin-slice mechanistic analysis")
    parser.add_argument("--trace", action="store_true", help="Run circuit-tracer on thin slices")
    parser.add_argument("--compare", action="store_true", help="Compare saved attribution graphs")
    parser.add_argument("--info", action="store_true", help="Show slice info without loading model")
    parser.add_argument("--preflight", action="store_true", help="Check model support, cache, and VRAM without loading the model")
    parser.add_argument("--cache-transcoders", action="store_true", help="Download transcoders into circuit-tracer's local cache")
    parser.add_argument("--max-pairs-per-slice", type=int, default=3, help="Pairs to trace from each slice")
    parser.add_argument("--model-key", default="qwen3-4b", help="Model key from lib.utils.config.MODELS")
    parser.add_argument("--transcoder-set", default="", help="Override circuit-tracer transcoder set")
    parser.add_argument(
        "--require-cached-transcoders",
        action="store_true",
        help="Fail fast unless the requested transcoder set is already in circuit-tracer's cache",
    )
    parser.add_argument(
        "--target-mode",
        choices=["salient", "correct", "correct-reality"],
        default="salient",
        help="Which logits to use as attribution targets",
    )
    parser.add_argument("--max-feature-nodes", type=int, default=None, help="Optional feature-node cap")
    parser.add_argument("--max-graphs", type=int, default=None, help="Stop after attempting this many graphs")
    parser.add_argument("--rerun-existing", action="store_true", help="Overwrite existing graph files")
    args = parser.parse_args()

    if not any([args.trace, args.compare, args.info, args.preflight, args.cache_transcoders]):
        parser.print_help()
        print("\nStart with --info to see the slices, then --trace to run circuit-tracer")
        return

    if args.preflight:
        preflight(args.model_key, args.transcoder_set)
        return

    if args.cache_transcoders:
        cache_transcoders(args.model_key, args.transcoder_set)
        return

    slices = load_slices()

    if args.info:
        summary = behavioral_summary()
        print("\n=== BEHAVIORAL PILOT SUMMARY ===")
        for condition, counts in summary["conditions"].items():
            print(
                f"  {condition}: {counts['correct']}/{counts['total']} "
                f"({counts['accuracy']:.0%})"
            )

        print("\n=== SLICE DETAILS ===")
        for name, pairs in slices.items():
            print(f"\n{name}:")
            for p in pairs:
                fb = p["fb_stimulus"]
                print(f"  {fb['id']}: {fb['text'][:60]}... -> {fb['correct']}")
        return

    if args.trace:
        run_circuit_tracer(
            slices,
            max_pairs_per_slice=args.max_pairs_per_slice,
            model_key=args.model_key,
            transcoder_set=args.transcoder_set,
            target_mode=args.target_mode,
            skip_existing=not args.rerun_existing,
            max_feature_nodes=args.max_feature_nodes,
            max_graphs=args.max_graphs,
            require_cached_transcoders=args.require_cached_transcoders,
        )

    if args.compare:
        compare_graphs()


if __name__ == "__main__":
    main()
