"""Console-only adapter for the author's Figure 4a experiment (SF100).

The original planner, query kernels, guarded execution, host SQL result ABI,
measurement functions and correctness checks are reused without source edits.
Data and exact CPU references must already exist; see README.md.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALL_QUERIES = (
    "Q1.1", "Q1.2", "Q1.3", "Q2.1", "Q2.2", "Q2.3",
    "Q3.1", "Q3.2", "Q3.3", "Q3.4", "Q4.1", "Q4.2", "Q4.3",
)
SF = 100


def check_inputs(work: Path, queries):
    """Fail before initializing TPU if input preparation has not been done."""
    source = work / "data/date_pk/sf100"
    cache = work / "cache/date_pk/sf100-v1"
    paths = [source / "manifest.txt", cache / "manifest.json"]
    paths.extend(source / f"{table}.tbl" for table in (
        "lineorder", "date", "customer", "supplier", "part"
    ))
    for query in queries:
        reference = work / "references/date_pk/sf100" / query.replace(".", "")
        paths.extend(reference / name for name in (
            "result.json", "aggregates.npz", "sql-rows.json"
        ))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(
            "SF100 inputs/references are missing. Prepare them using README.md.\n"
            + "\n".join(missing[:8])
        )
    fields = dict(
        line.strip().split("=", 1)
        for line in (source / "manifest.txt").read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    )
    expected = {"format_version": "3", "scale_factor": "100", "physical_order": "date_pk",
                "source_preordered_layout": "1"}
    if any(fields.get(key) != value for key, value in expected.items()):
        raise ValueError("Figure 4a requires the author's date_pk sorted source layout.")
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    missing_arrays = [str(cache / entry["file"]) for entry in manifest["arrays"]
                      if not (cache / entry["file"]).is_file()]
    if missing_arrays:
        raise ValueError("Integer cache is incomplete: " + missing_arrays[0])


def import_artifact(root: Path):
    """Use the supplied artifact, including its immutable content verification."""
    if not (root / "content-manifest.json").is_file():
        raise ValueError("--artifact-root must point to sectile-artifact-EF6C.")
    sys.path[:0] = [str(root / "src"), str(root)]
    from reproduce import common
    if common.ROOT.resolve() != root:
        raise ValueError("A different reproduce package was imported; check PYTHONPATH.")
    return common


def run_cell(args, query):
    """The same call/settings as reproduce.main_matrix.run_cell, without files."""
    from console_artifacts import NullText, discard_author_artifacts

    common = import_artifact(args.artifact_root)
    common.outside_repository(args.work)
    identity = common.configure(SF)
    os.environ.update(PROFILE_HARDWARE_TRACE="0", PROFILE_STAGE_TRACE="0",
                      PYTHONDONTWRITEBYTECODE="1", JAX_ENABLE_COMPILATION_CACHE="false")
    # configure() must precede imports that initialize JAX/libtpu.
    jax = common.require_tpu()
    jax.config.update("jax_enable_compilation_cache", False)
    device = jax.devices()[0]
    if "v6e" not in device.device_kind.lower() and "v6 lite" not in device.device_kind.lower():
        raise ValueError(f"Expected one TPU v6e chip, found {device.device_kind}.")
    from jax.experimental.pallas import tpu as pltpu
    if getattr(pltpu.get_tpu_info(), "sparse_core", None) is None:
        raise ValueError("The TPU runtime does not expose programmable SparseCore.")

    # Only source/input reads happen here; never call data preparation implicitly.
    with redirect_stdout(NullText()):
        inputs, source, _ = common.load_inputs(args.work, SF)
    from reproduce.query_plans import paper_plan
    from reproduce.reference_cache import saved_oracles
    from reproduce.execution_guard.integration import install
    from tpu_db.benchmark.production import run_resident_query

    reference = args.work / "references/date_pk/sf100" / query.replace(".", "")
    output_root = Path(__file__).resolve().parent / ".console-artifacts-not-created"
    # Suppress only author audit output. Compute, timers, guard and validators stay intact.
    with discard_author_artifacts(output_root), redirect_stdout(NullText()):
        with saved_oracles(args.work, SF), install():
            record, _ = run_resident_query(
                paper_plan(query) if query == "Q4.3" else query,
                inputs, source_directory=source, output_dir=output_root / query,
                planner_mode="exact_int64", reference_dir=reference,
                runs=5, maximum_calibrations=0, cost_model=common.make_model(),
                experimental_calibration=False, host_sql_rows=True,
            )
    common.check_result(record)
    common.verify_contents(source_id=identity)
    latency = float(record["median_ms"])
    if not math.isfinite(latency) or latency <= 0:
        raise ValueError("The experiment did not return a valid measured latency.")
    return {"query": query, "latency_ms": latency}


def main(queries=ALL_QUERIES):
    parser = argparse.ArgumentParser(
        description="Sectile Figure 4a: SF100, one warmup, five SQL-ready calls, median ms."
    )
    parser.add_argument("--work", type=Path, default=PROJECT_ROOT / "sectile-work",
                        help="Existing author data/cache/references work directory.")
    parser.add_argument("--artifact-root", type=Path,
                        default=PROJECT_ROOT / "sectile-artifact-EF6C")
    parser.add_argument("--cell", choices=ALL_QUERIES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.work = args.work.expanduser().resolve()
    args.artifact_root = args.artifact_root.expanduser().resolve()
    selected = (args.cell,) if args.cell else tuple(queries)
    try:
        if not (args.artifact_root / "content-manifest.json").is_file():
            raise ValueError("Original artifact is missing; set --artifact-root. See README.md.")
        check_inputs(args.work, selected)
        if args.cell:
            # Machine-readable stdout is used only between parent and child processes.
            print(json.dumps(run_cell(args, args.cell)), flush=True)
            return
        print("Query    Latency (ms)", flush=True)
        script = Path(__file__).with_name("run_benchmark.py")
        for query in selected:
            child = subprocess.run([
                sys.executable, "-B", str(script), "--cell", query,
                "--work", str(args.work), "--artifact-root", str(args.artifact_root),
            ], capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
            if child.returncode:
                raise RuntimeError(
                    f"{query} failed; no latency is reported for that query.\n"
                    + child.stderr.strip() + "\n" + child.stdout.strip()
                )
            row = json.loads(child.stdout.strip())
            latency = float(row["latency_ms"])
            if row["query"] != query or not math.isfinite(latency) or latency <= 0:
                raise ValueError("Unexpected measurement returned by the child process.")
            print(f"{query:<8} {latency:>12.3f}", flush=True)
    except (ValueError, RuntimeError, ImportError, OSError) as error:
        raise SystemExit(str(error)) from error
