"""Read-only SF100 checks and serial process orchestration for Table 6."""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import json
import math
from pathlib import Path
import time


def check_inputs(artifact, work, queries):
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
        raise ValueError("This experiment requires the author's SF100 date_pk sorted inputs.")
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
    from runtime import run_worker
    from console_output import capture_console

    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=run_worker, args=(
        child, args.artifact_root, args.work, query, variant,
    ))
    try:
        # Inherited quiet fd 1/2 cover native startup and finalizer messages.
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
                raise ValueError("Worker returned an invalid latency or experiment identity.")
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


def require_same_identity(rows):
    identities = {(row["source_id"], row["dataset_metadata_id"]) for row in rows}
    if len(identities) != 1:
        raise ValueError("Artifact or dataset identity differs between measurements.")
