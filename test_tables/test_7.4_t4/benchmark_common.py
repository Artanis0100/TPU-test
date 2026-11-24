"""Terminal tables for the paper's forced-map and prediction experiments."""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import math
import multiprocessing as mp
from pathlib import Path

from process_runner import check_inputs, run_case, require_same_identity

T4_QUERIES = ("Q2.3", "Q3.3")
T4_VARIANTS = ("compiler_only", "hash", "bitmap", "bitmap_rank")
T5_VARIANTS = ("bitmap", "bitmap_rank", "hash")
LABELS = {"compiler_only": "Compiler-managed", "hash": "All-Hash",
          "bitmap": "All-Bitmap", "bitmap_rank": "All-Bitmap-Rank"}
CANDIDATES = {"bitmap": "Bitmap", "bitmap_rank": "Bitmap-Rank", "hash": "Hash"}


def print_table4(context, args):
    widths = [max(16, len(LABELS[variant]) + 5) for variant in args.variants]
    print("Query " + " ".join(f"{LABELS[v] + ' (ms)':>{width}}"
                             for v, width in zip(args.variants, widths)), flush=True)
    identity = None
    for query in args.queries:
        rows = [run_case(context, args, query, variant) for variant in args.variants]
        require_same_identity(rows)
        current = rows[0]["source_id"], rows[0]["dataset_metadata_id"]
        if identity is not None and identity != current:
            raise ValueError("Artifact or dataset identity changed between queries.")
        identity = current
        values = " ".join(f"{row['median_ms']:>{width}.2f}" for row, width in zip(rows, widths))
        print(f"{query:<5} {values}", flush=True)


def validate_table5(rows):
    if len(rows) != 3 or tuple(row["variant"] for row in rows) != T5_VARIANTS:
        raise ValueError("Table 5 requires all three preselected plans in the declared order.")
    require_same_identity(rows)
    for key in ("comparison_identity", "aggregate_sha256", "ordered_rows_sha256"):
        if any(row[key] != rows[0][key] for row in rows[1:]):
            raise ValueError(f"Table 5 plans differ in required comparison evidence: {key}.")
    if len({row["physical_key"] for row in rows}) != 3:
        raise ValueError("Table 5 did not return three distinct physical plans.")
    if any(type(row["planner_selected"]) is not bool for row in rows):
        raise ValueError("Planner choice evidence must contain explicit booleans.")
    if sum(row["planner_selected"] for row in rows) != 1:
        raise ValueError("The full planner did not select exactly one of these three plans; no star is invented.")
    if any(not math.isfinite(row["predicted_ms"]) or row["predicted_ms"] <= 0 for row in rows):
        raise ValueError("Invalid model prediction.")
    return rows


def print_table5(context, args):
    rows = validate_table5([run_case(context, args, "Q2.3", variant) for variant in T5_VARIANTS])
    print("Candidate       Predicted (ms)   Measured (ms)   Rank (pred./meas.)", flush=True)
    for row in rows:
        # Equal values receive equal competition ranks; use unrounded values.
        predicted_rank = 1 + sum(other["predicted_ms"] < row["predicted_ms"] for other in rows)
        measured_rank = 1 + sum(other["median_ms"] < row["median_ms"] for other in rows)
        label = CANDIDATES[row["variant"]] + ("*" if row["planner_selected"] else "")
        print(f"{label:<15} {row['predicted_ms']:>14.2f} {row['median_ms']:>15.2f} "
              f"{str(predicted_rank) + '/' + str(measured_rank):>20}", flush=True)


def main(table):
    if table not in ("t4", "t5"):
        raise ValueError("Unknown paper table.")
    parser = argparse.ArgumentParser(
        description=("Sectile Table 4: complete-query SF100 map representations."
                     if table == "t4" else "Sectile Table 5: Q2.3 prediction vs numeric TPU latency.")
    )
    parser.add_argument("--artifact-root", type=Path,
                        default=Path.home() / "test/sectile-artifact-EF6C")
    parser.add_argument("--work", type=Path, default=Path.home() / "test/sectile-work")
    parser.add_argument("--timeout", type=float, default=3600,
                        help="Maximum seconds per query/configuration (default: 3600).")
    parser.add_argument("--check-only", action="store_true",
                        help="Check existing files, artifact and TPU without running full queries.")
    if table == "t4":
        parser.add_argument("--queries", nargs="+", choices=T4_QUERIES, default=T4_QUERIES)
        parser.add_argument("--variants", nargs="+", choices=T4_VARIANTS, default=T4_VARIANTS)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive.")
    args.artifact_root = args.artifact_root.expanduser().resolve()
    args.work = args.work.expanduser().resolve()
    queries = tuple(dict.fromkeys(args.queries)) if table == "t4" else ("Q2.3",)
    if table == "t4":
        args.queries = queries
        args.variants = tuple(dict.fromkeys(args.variants))
    try:
        check_inputs(args.artifact_root, args.work, queries)
        context = mp.get_context("spawn")
        if args.check_only:
            info = run_case(context, args)
            print(f"Preflight passed: {info['device']}; JAX {info['jax']}; existing SF100 inputs.")
            return
        if table == "t4":
            print_table4(context, args)
        else:
            print_table5(context, args)
    except (Exception, KeyboardInterrupt) as error:
        raise SystemExit(str(error) or "Benchmark interrupted.") from error
