"""Console-only Figure 7 controller; the original microbenchmark prepares once."""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import math
import multiprocessing as mp
from pathlib import Path
import time

WIDTHS = {
    "aggregation": (1, 5, 10, 15, 20, 25, 26, 27, 28, 30, 32, 33, 34, 35, 36, 40, 48),
    "projection": (1, 2, 4, 6, 8, 16, 24, 32),
}
TABLES = {
    "aggregation": ("lineorder", "customer", "supplier", "date"),
    "projection": ("lineorder", "part", "supplier"),
}
BACKENDS = ("jax_xla", "sparsecore", "mxu")


def resolve_data(args, operation):
    """Read existing raw SF1 tables; never derive them from SF100."""
    if not (args.artifact_root / "content-manifest.json").is_file():
        raise ValueError("Original artifact is missing; set --artifact-root.")
    candidates = ((args.data_dir,) if args.data_dir else (
        args.work / "data/date_pk/sf1", args.work / "figure7-data/sf1",
    ))
    for directory in candidates:
        if not all((directory / f"{table}.tbl").is_file()
                   and (directory / f"{table}.tbl").stat().st_size > 0 for table in TABLES[operation]):
            continue
        manifest = directory / "manifest.txt"
        if manifest.is_file():
            fields = dict(line.strip().split("=", 1)
                          for line in manifest.read_text(encoding="utf-8").splitlines()
                          if "=" in line and not line.lstrip().startswith("#"))
            if fields.get("scale_factor") != "1":
                raise ValueError("The input manifest is not SF1; SF100 cannot reproduce Figure 7.")
        return directory.resolve()
    attempted = "\n".join(str(path) for path in candidates)
    raise ValueError(
        "Figure 7 requires genuine SF1 raw tables; no complete input directory was found:\n"
        + attempted + "\nUse --data-dir EXISTING_SF1 or explicitly run python -B prepare_sf1.py "
        "to prepare separate inputs with the VM's existing generator."
    )


def receive(connection, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if connection.poll(0.2):
            try:
                return connection.recv()
            except EOFError as error:
                raise RuntimeError("Worker closed its pipe without a result.") from error
        if not process.is_alive():
            if connection.poll():
                continue
            raise RuntimeError(f"Worker exited with status {process.exitcode} before returning its result.")
    raise TimeoutError(f"The sweep exceeded {timeout:g} seconds; adjust --timeout if needed.")


def run_operation(context, args, check_only=False):
    from runtime import run_worker
    from console_output import capture_console

    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=run_worker, args=(
        child, args.artifact_root, args.data_dir, args.widths, check_only,
    ))
    try:
        # Inherited quiet fd 1/2 cover native startup and finalizer messages too.
        with capture_console():
            process.start()
        child.close()
        message = receive(parent, process, args.timeout)
        if message.get("event") == "error":
            raise RuntimeError("Microbenchmark failed; no value is substituted.\n"
                               + message["message"] + message.get("logs", ""))
        if message.get("event") != "ok":
            raise RuntimeError(f"Unexpected worker message: {message!r}")
        process.join(30)
        if process.is_alive() or process.exitcode:
            raise RuntimeError("The worker did not exit cleanly after returning its result.")
        return message["result"]
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


def validate_result(result, args, operation):
    rows, manifest = result["rows"], result["metadata"]
    key = "measure_counts" if operation == "aggregation" else "projection_widths"
    if (manifest["scale_factor"] != 1 or manifest["runs"] != 30 or manifest["warmups"] != 3
            or tuple(manifest[key]) != args.widths
            or Path(manifest["data_dir"]).resolve() != args.data_dir
            or manifest["git"]["artifact_identity"] != result["source_id"]
            or len(result["raw_rows"]) != 3 * len(args.widths)):
        raise ValueError("The returned data/protocol/sweep identity differs from the request.")
    if tuple(row["width"] for row in rows) != args.widths:
        raise ValueError("A requested width is missing, repeated or out of order.")
    for row in rows:
        if set(row) != {"width", *BACKENDS} or type(row["width"]) is not int:
            raise ValueError("Unexpected width/backend result schema.")
        if any(not math.isfinite(row[name]) or row[name] <= 0 for name in BACKENDS):
            raise ValueError("Invalid measured latency.")
    return rows


def main(operation):
    if operation not in WIDTHS:
        raise ValueError("Unknown Figure 7 operation.")
    parser = argparse.ArgumentParser(
        description=f"Sectile Figure 7 {'(a)' if operation == 'aggregation' else '(b)'}: "
                    f"SF1 {operation}, three backends, median device latency in ms."
    )
    parser.add_argument("--artifact-root", type=Path,
                        default=Path.home() / "test/sectile-artifact-EF6C")
    parser.add_argument("--work", type=Path, default=Path.home() / "test/sectile-work")
    parser.add_argument("--data-dir", type=Path, help="Existing genuine SF1 raw-table directory.")
    parser.add_argument("--widths", type=int, nargs="+", choices=WIDTHS[operation], default=WIDTHS[operation],
                        help="Subset of the paper's widths; omitted for the full sweep.")
    parser.add_argument("--timeout", type=float, default=3600,
                        help="Maximum seconds for the entire sweep (default: 3600).")
    parser.add_argument("--check-only", action="store_true",
                        help="Check input paths, source and TPU; do not load tables or measure kernels.")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive.")
    args.artifact_root = args.artifact_root.expanduser().resolve()
    args.work = args.work.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve() if args.data_dir else None
    args.widths = tuple(sorted(set(args.widths)))
    try:
        args.data_dir = resolve_data(args, operation)
        result = run_operation(mp.get_context("spawn"), args, check_only=args.check_only)
        if args.check_only:
            print(f"Preflight passed: {result['device']}; JAX {result['jax']}; existing SF1 raw tables.")
            return
        rows = validate_result(result, args, operation)
        print("Terms    JAX/XLA (ms)    Pallas SparseCore (ms)    Pallas MXU (ms)", flush=True)
        for row in rows:
            print(f"{row['width']:>5} {row['jax_xla']:>15.3f} {row['sparsecore']:>25.3f} "
                  f"{row['mxu']:>18.3f}", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        raise SystemExit(str(error) or "Benchmark interrupted.") from error
