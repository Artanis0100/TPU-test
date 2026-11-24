#!/usr/bin/env python3
"""Sectile section 7.3 Table 3: real SF100 TPU counters, stdout only."""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

QUERIES = ("Q1.1", "Q2.1", "Q3.1", "Q3.2", "Q4.1")


def check_paths(artifact, work, queries):
    """Read existing inputs only; never regenerate or relocate them."""
    if not (artifact / "content-manifest.json").is_file():
        raise ValueError(f"Original artifact not found: {artifact}; set --artifact-root.")
    source = work / "data/date_pk/sf100"
    cache = work / "cache/date_pk/sf100-v1"
    required = [source / "manifest.txt", cache / "manifest.json"]
    required.extend(source / f"{name}.tbl" for name in (
        "lineorder", "date", "customer", "supplier", "part"
    ))
    for query in queries:
        directory = work / "references/date_pk/sf100" / query.replace(".", "")
        required.extend(directory / name for name in (
            "result.json", "aggregates.npz", "sql-rows.json"
        ))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("Existing SF100 inputs/references are missing:\n" + "\n".join(missing[:8]))
    fields = dict(line.strip().split("=", 1)
                  for line in (source / "manifest.txt").read_text().splitlines()
                  if "=" in line and not line.lstrip().startswith("#"))
    if any(fields.get(key) != value for key, value in {
        "format_version": "3", "scale_factor": "100", "physical_order": "date_pk",
        "source_preordered_layout": "1",
    }.items()):
        raise ValueError("The experiment requires the author's sorted SF100 date_pk inputs.")
    manifest = json.loads((cache / "manifest.json").read_text())
    if any(not (cache / entry["file"]).is_file() for entry in manifest["arrays"]):
        raise ValueError("Existing SF100 integer cache is incomplete.")


def receive(connection, process, expected, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if connection.poll(0.2):
            try:
                message = connection.recv()
            except EOFError as error:
                raise RuntimeError("TPU worker closed its control pipe unexpectedly.") from error
            if message.get("event") == "error":
                raise RuntimeError(message.get("message", "TPU worker failed."))
            if message.get("event") != expected:
                raise RuntimeError(f"Unexpected worker state: {message!r}")
            return message
        if not process.is_alive():
            if connection.poll():
                continue
            raise RuntimeError(f"TPU worker exited with status {process.exitcode}.")
    raise TimeoutError(f"Timed out waiting for worker state {expected!r}.")


def cleanup(connection, process):
    if process.is_alive():
        try:
            connection.send({"command": "abort"})
        except (OSError, EOFError):
            pass
        process.join(15)
        if process.is_alive():
            process.terminate()
            process.join(10)
        if process.is_alive():
            process.kill()
            process.join(5)
    connection.close()
    if not process.is_alive():
        process.close()


def hardware_check(connection, artifact):
    """Run the device check in a fresh process, without reading large arrays."""
    try:
        os.environ["JAX_PLATFORMS"] = "tpu"
        sys.path[:0] = [str(artifact / "src"), str(artifact)]
        from reproduce import common
        common.configure(100, profile=True)
        jax = common.require_tpu()
        device = jax.devices()[0]
        if "v6e" not in device.device_kind.lower() and "v6 lite" not in device.device_kind.lower():
            raise ValueError(f"Expected TPU v6e, found {device.device_kind}.")
        common.make_model()
        connection.send({"event": "checked", "device": device.device_kind,
                         "jax": jax.__version__})
    except BaseException:
        import traceback
        connection.send({"event": "error", "message": traceback.format_exc()})
    finally:
        connection.close()


def run_query(context, args, query, capture, summarize, options):
    from query_worker import run_worker
    parent, child = context.Pipe()
    process = context.Process(target=run_worker, args=(
        child, query, args.artifact_root, args.work, args.port,
    ))
    process.start()
    child.close()
    try:
        ready = receive(parent, process, "ready", args.ready_timeout)
        if ready["query"] != query or ready["port"] != args.port:
            raise RuntimeError("Worker query/port identity differs from this request.")
        parent.send({"command": "start"})
        receive(parent, process, "running", 120)
        # Three independent RPCs; conversion/analysis starts only after stopping
        # the loop, so it cannot steal host CPU during the measured windows.
        raw_windows = [capture(f"127.0.0.1:{args.port}", options) for _ in range(3)]
        parent.send({"command": "stop"})
        validated = receive(parent, process, "validated", 300)
        if validated["query"] != query or validated["completed_invocations"] < 1:
            raise RuntimeError("Steady loop did not validate this query.")
        if validated["identity"] != ready["identity"]:
            raise RuntimeError("The selected execution identity changed during capture.")
        process.join(30)
        if process.is_alive() or process.exitcode:
            raise RuntimeError("TPU worker did not exit successfully after validation.")
        result = summarize(raw_windows, args.artifact_root, verbose=args.verbose)
        result["query"] = query
        result["identity"] = ready["identity"]
        return result
    finally:
        cleanup(parent, process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path,
                        default=Path.home() / "test/sectile-artifact-EF6C")
    parser.add_argument("--work", type=Path, default=Path.home() / "test/sectile-work")
    parser.add_argument("--queries", nargs="+", choices=QUERIES, default=QUERIES)
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--ready-timeout", type=float, default=1800)
    parser.add_argument("--check-only", action="store_true",
                        help="Check paths, capture/conversion APIs and the TPU; do not profile.")
    parser.add_argument("--verbose", action="store_true",
                        help="Show native XProf INFO logs during counter conversion.")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.ready_timeout <= 0:
        parser.error("Use a valid TCP port and positive readiness timeout.")
    args.artifact_root = args.artifact_root.expanduser().resolve()
    args.work = args.work.expanduser().resolve()
    args.queries = tuple(dict.fromkeys(args.queries))
    # The coordinator only captures/decodes bytes. Each spawned worker explicitly
    # selects TPU before importing JAX. There is never more than one TPU worker.
    os.environ.update(JAX_PLATFORMS="cpu", JAX_ENABLE_COMPILATION_CACHE="false",
                      LIBTPU_INIT_ARGS="", XLA_FLAGS="", PYTHONDONTWRITEBYTECODE="1")
    try:
        check_paths(args.artifact_root, args.work, args.queries)
        sys.path[:0] = [str(args.artifact_root / "src"), str(args.artifact_root)]
        from counter_capture import preflight as capture_preflight, capture_window
        from counter_metrics import preflight as metric_preflight, summarize_windows
        capture_preflight()
        metric_preflight(verbose=args.verbose)
        from tpu_db.benchmark.steady_state_profiling import xprof_capture_options
        context = mp.get_context("spawn")
        if args.check_only:
            parent, child = context.Pipe()
            process = context.Process(target=hardware_check, args=(child, args.artifact_root))
            process.start()
            child.close()
            try:
                checked = receive(parent, process, "checked", 180)
                process.join(30)
                if process.is_alive() or process.exitcode:
                    raise RuntimeError("Hardware preflight process did not exit cleanly.")
                print(f"Preflight passed: {checked['device']}; JAX {checked['jax']}; "
                      "existing SF100 inputs; in-memory XProf capture/conversion APIs.")
            finally:
                cleanup(parent, process)
            return
        print("Query   HBM peak (%)  Vector-ALU issue (%)  XLU busy (%)  SC issue N0/N1 (%)", flush=True)
        for query in args.queries:
            row = run_query(context, args, query, capture_window,
                            summarize_windows, xprof_capture_options())
            print(f"{query:<6} {row['hbm_peak_pct']:>12.2f} "
                  f"{row['vector_alu_issue_pct']:>21.2f} {row['xlu_busy_pct']:>13.2f} "
                  f"{row['sc_n0_issue_pct']:>8.2f}/{row['sc_n1_issue_pct']:.2f}", flush=True)
            if row.get("trace_loss_present"):
                print(f"{query}: required counters are complete; trace-loss evidence "
                      "was present in the captured XSpace.", file=sys.stderr, flush=True)
    except Exception as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
