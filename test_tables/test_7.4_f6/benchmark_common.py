"""Console controller for Figure 6; all TPU work runs in isolated children."""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path
import time

ALL_QUERIES = (
    "Q1.1", "Q1.2", "Q1.3", "Q2.1", "Q2.2", "Q2.3",
    "Q3.1", "Q3.2", "Q3.3", "Q3.4", "Q4.1", "Q4.2", "Q4.3",
)
VARIANTS = ("compiler_only", "full_library")


def check_inputs(artifact, work, queries):
    """Check existing files only; never generate or repair data implicitly."""
    if not (artifact / "content-manifest.json").is_file():
        raise ValueError("Original artifact is missing; set --artifact-root.")
    source = work / "data/date_pk/sf100"
    cache = work / "cache/date_pk/sf100-v1"
    paths = [source / "manifest.txt", cache / "manifest.json"]
    paths.extend(source / f"{name}.tbl" for name in (
        "lineorder", "date", "customer", "supplier", "part"
    ))
    for query in queries:
        reference = work / "references/date_pk/sf100" / query.replace(".", "")
        paths.extend(reference / name for name in (
            "result.json", "aggregates.npz", "sql-rows.json"
        ))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError("Existing SF100 inputs/references are missing:\n" + "\n".join(missing[:8]))
    fields = dict(line.strip().split("=", 1)
                  for line in (source / "manifest.txt").read_text(encoding="utf-8").splitlines()
                  if "=" in line and not line.lstrip().startswith("#"))
    expected = {"format_version": "3", "scale_factor": "100", "physical_order": "date_pk",
                "source_preordered_layout": "1"}
    if any(fields.get(key) != value for key, value in expected.items()):
        raise ValueError("Figure 6 requires the author's SF100 date_pk sorted inputs.")
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if not manifest["arrays"] or any(not (cache / item["file"]).is_file()
                                     for item in manifest["arrays"]):
        raise ValueError("The existing SF100 integer cache is incomplete.")


def receive(connection, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if connection.poll(0.2):
            try:
                return connection.recv()
            except EOFError as error:
                raise RuntimeError("Worker closed its pipe before returning a result.") from error
        if not process.is_alive():
            if connection.poll():
                continue
            raise RuntimeError(f"Worker exited with status {process.exitcode} before returning a result.")
    raise TimeoutError(f"Worker exceeded {timeout:g} seconds; adjust --timeout if needed.")


def run_case(context, args, query=None, variant=None):
    """One process per query/configuration; logs and records never use files."""
    from benchmark_worker import run_worker
    from console_output import capture_console

    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=run_worker, args=(
        child, args.artifact_root, args.work, query, variant,
    ))
    try:
        # Linux spawn inherits quiet native stdout/stderr. This also covers
        # startup and finalizer logs outside the worker's diagnostic context.
        with capture_console():
            process.start()
        child.close()
        message = receive(parent, process, args.timeout)
        if message.get("event") == "error":
            label = f"{query} / {variant}" if query else "Preflight"
            raise RuntimeError(f"{label} failed; no value is substituted.\n"
                               + message["message"] + message.get("logs", ""))
        if message.get("event") != "ok":
            raise RuntimeError(f"Unexpected worker message: {message!r}")
        process.join(30)
        if process.is_alive() or process.exitcode:
            raise RuntimeError("Worker did not exit cleanly after returning its result.")
        row = message["result"]
        if query is not None:
            if (row["query"] != query or row["variant"] != variant
                    or not math.isfinite(row["median_ms"]) or row["median_ms"] <= 0):
                raise ValueError("Worker returned an invalid latency or query/configuration identity.")
        return row
    finally:
        child.close()
        if process.is_alive():
            process.terminate()
            process.join(10)
        if process.is_alive():
            process.kill()
            process.join(5)
        parent.close()
        if not process.is_alive():
            process.close()


def main(default_queries=ALL_QUERIES):
    parser = argparse.ArgumentParser(
        description="Sectile Figure 6: SF100, compiler-managed only vs full library, median ms."
    )
    parser.add_argument("--artifact-root", type=Path,
                        default=Path.home() / "test/sectile-artifact-EF6C")
    parser.add_argument("--work", type=Path, default=Path.home() / "test/sectile-work")
    parser.add_argument("--queries", nargs="+", choices=ALL_QUERIES, default=default_queries)
    parser.add_argument("--timeout", type=float, default=3600,
                        help="Maximum seconds per isolated query/configuration (default: 3600).")
    parser.add_argument("--check-only", action="store_true",
                        help="Check existing files, artifact and TPU; do not load or run full queries.")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive.")
    args.artifact_root = args.artifact_root.expanduser().resolve()
    args.work = args.work.expanduser().resolve()
    args.queries = tuple(dict.fromkeys(args.queries))
    try:
        check_inputs(args.artifact_root, args.work, args.queries)
        context = mp.get_context("spawn")
        if args.check_only:
            info = run_case(context, args)
            print(f"Preflight passed: {info['device']}; JAX {info['jax']}; existing SF100 inputs.")
            return
        print("Query   Compiler-managed only (ms)   Full library (ms)", flush=True)
        identity = None
        for query in args.queries:
            rows = [run_case(context, args, query, variant) for variant in VARIANTS]
            for row in rows:
                current = row["source_id"], row["dataset_metadata_id"]
                if identity is not None and identity != current:
                    raise RuntimeError("Artifact or dataset identity changed between measurements.")
                identity = current
            print(f"{query:<6} {rows[0]['median_ms']:>26.3f} {rows[1]['median_ms']:>19.3f}", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        raise SystemExit(str(error) or "Benchmark interrupted.") from error
