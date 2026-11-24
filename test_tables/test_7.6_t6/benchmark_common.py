"""Table 6 controller: fixed selected plans with reduced buffer capacities."""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import math
import multiprocessing as mp
from pathlib import Path

from process_runner import check_inputs, run_case, require_same_identity

QUERIES = ("Q1.1", "Q2.1", "Q3.1")
REDUCTIONS = (0.0, 0.1, 0.3, 0.5)


def validate_query(rows, query, reductions):
    """Match the author's cross-cell selection check; never guess a fallback."""
    if len(rows) != len(reductions) or not rows or reductions[0] != 0:
        raise ValueError("A complete baseline and all requested reduction cells are required.")
    require_same_identity(rows)
    baseline = rows[0]
    for row, amount in zip(rows, reductions):
        if row["query"] != query or row["variant"] != amount:
            raise ValueError("Query/reduction identity differs from this request.")
        for key in ("normal_selection_sha256", "original_candidate_sha256",
                    "reference_identity_sha256", "input_content_sha256"):
            if row[key] != baseline[key]:
                raise ValueError(f"Normal planning or original input changed between capacity cells: {key}.")
        for key in ("baseline_capacity", "baseline_field_words", "buffer", "alignment", "exact_path"):
            if row["capacity_stress"][key] != baseline["capacity_stress"][key]:
                raise ValueError(f"Baseline buffer contract changed between capacity cells: {key}.")
        calls = row["measured_capacity_exact_branch_calls"]
        if (len(calls) != 5 or any(type(count) is not int or count not in (0, 1) for count in calls)
                or type(row["fallback_triggered"]) is not bool
                or row["fallback_triggered"] != any(calls)):
            raise ValueError("Fallback marking differs from the five returned measured-call observations.")
        if not math.isfinite(row["median_ms"]) or row["median_ms"] <= 0:
            raise ValueError("Invalid measured latency.")
    return rows


def print_table(context, args):
    labels = ["Baseline" if amount == 0 else f"-{100 * amount:g}%" for amount in args.reductions]
    width = 17
    print("Query " + " ".join(f"{label + ' (ms)':>{width}}" for label in labels), flush=True)
    identity = None
    for query in args.queries:
        rows = validate_query(
            [run_case(context, args, query, amount) for amount in args.reductions],
            query, args.reductions,
        )
        current = rows[0]["source_id"], rows[0]["dataset_metadata_id"], rows[0]["input_content_sha256"]
        if identity is not None and identity != current:
            raise ValueError("Artifact or SF100 data identity changed between queries.")
        identity = current
        values = [f"{row['median_ms']:.2f}" + ("\u2020" if row["fallback_triggered"] else "") for row in rows]
        print(f"{query:<5} " + " ".join(f"{value:>{width}}" for value in values), flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Sectile Table 6: SF100 complete-query latency with reduced intermediate capacities."
    )
    parser.add_argument("--artifact-root", type=Path,
                        default=Path.home() / "test/sectile-artifact-EF6C")
    parser.add_argument("--work", type=Path, default=Path.home() / "test/sectile-work")
    parser.add_argument("--queries", nargs="+", choices=QUERIES, default=QUERIES)
    parser.add_argument("--reductions", nargs="+", type=float, choices=REDUCTIONS, default=REDUCTIONS,
                        help="Removed fractions; baseline is always included (default: 0 0.1 0.3 0.5).")
    parser.add_argument("--timeout", type=float, default=3600,
                        help="Maximum seconds per query/reduction (default: 3600).")
    parser.add_argument("--check-only", action="store_true",
                        help="Check existing files, artifact and TPU without running full queries.")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive.")
    args.artifact_root = args.artifact_root.expanduser().resolve()
    args.work = args.work.expanduser().resolve()
    args.queries = tuple(dict.fromkeys(args.queries))
    args.reductions = tuple(sorted({0.0, *args.reductions}))
    try:
        check_inputs(args.artifact_root, args.work, args.queries)
        context = mp.get_context("spawn")
        if args.check_only:
            info = run_case(context, args)
            print(f"Preflight passed: {info['device']}; JAX {info['jax']}; existing SF100 inputs.")
            return
        print_table(context, args)
    except (Exception, KeyboardInterrupt) as error:
        raise SystemExit(str(error) or "Benchmark interrupted.") from error
