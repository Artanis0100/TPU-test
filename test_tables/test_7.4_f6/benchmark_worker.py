"""Reuse the author's exact Figure 6 and Figure 4a entries without source edits."""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import importlib
import math
import os
from pathlib import Path
from statistics import median
import traceback
from unittest.mock import patch


def prepare_runtime(artifact_root, work):
    """Configure libtpu before importing any JAX-dependent artifact writers."""
    os.environ.update(
        JAX_PLATFORMS="tpu", PYTHONDONTWRITEBYTECODE="1",
        JAX_ENABLE_COMPILATION_CACHE="false", JAX_DUMP_IR_TO="",
        PROFILE_HARDWARE_TRACE="0", PROFILE_STAGE_TRACE="0",
        PROFILE_CORE_ABLATION="0", RUN_CORE_ABLATION="0", RUN_COMPONENT_ABLATIONS="0",
    )
    sys.path[:0] = [str(artifact_root / "src"), str(artifact_root)]
    from reproduce import common

    if common.ROOT.resolve() != artifact_root:
        raise ValueError("A different Sectile artifact was imported; check PYTHONPATH.")
    common.outside_repository(work)
    identity = common.configure(100)
    jax = common.require_tpu()
    jax.config.update("jax_enable_compilation_cache", False)
    device = jax.devices()[0]
    kind = device.device_kind.lower()
    if "v6e" not in kind and "v6 lite" not in kind:
        raise ValueError(f"Expected one TPU v6e chip, found {device.device_kind}.")
    from jax.experimental.pallas import tpu as pltpu

    if getattr(pltpu.get_tpu_info(), "sparse_core", None) is None:
        raise ValueError("The runtime does not expose programmable SparseCore.")
    common.make_model()
    return common, jax, identity


def measure(common, query, variant, work, identity):
    from console_artifacts import discard_author_artifacts

    entries = {"compiler_only": "reproduce.compiler_only", "full_library": "reproduce.main_matrix"}
    if variant not in entries:
        raise ValueError(f"Unknown Figure 6 configuration: {variant}")
    entry = importlib.import_module(entries[variant])
    captured = []
    output_root = Path(__file__).resolve().parent / ".console-artifacts-not-created"
    # Pass the yielded virtual path: compiler_only calls output.mkdir directly.
    # Only result printing/persistence is adapted; author timing and checks run.
    with discard_author_artifacts(output_root) as output:
        with patch.object(common, "print_result", captured.append):
            if variant == "compiler_only":
                entry.run_cell(work, output, query, 100)
            else:
                entry.run_cell(work, output, 100, query)
    if len(captured) != 1:
        raise RuntimeError("The author entry did not return exactly one validated record.")
    record = captured[0]
    common.check_result(record)
    selection = "forced_compiler_managed_only" if variant == "compiler_only" else "production_planner_auto"
    expected = {"query": query, "sf": 100, "source_id": identity,
                "selection_mode": selection, "arithmetic_mode": "exact_int64",
                "result_finalization": "tpu"}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Returned record differs from the requested Figure 6 protocol.")
    samples = [float(value) for value in record["samples_ms"]]
    latency = float(record["median_ms"])
    if (len(samples) != 5 or any(not math.isfinite(value) or value <= 0 for value in samples)
            or not math.isfinite(latency) or latency <= 0 or latency != median(samples)):
        raise ValueError("Invalid measured samples or median; no latency is fabricated.")
    common.verify_contents(source_id=identity)
    return {"query": query, "variant": variant, "median_ms": latency,
            "samples_ms": samples, "source_id": identity,
            "dataset_metadata_id": record["dataset_metadata_id"]}


def run_worker(connection, artifact_root, work, query=None, variant=None):
    """Spawn target. All successful Python/native logs stay in anonymous RAM."""
    from console_output import capture_console

    try:
        with capture_console() as logs:
            try:
                artifact_root = Path(artifact_root).expanduser().resolve()
                work = Path(work).expanduser().resolve()
                common, jax, identity = prepare_runtime(artifact_root, work)
                if query is None:
                    result = {"device": jax.devices()[0].device_kind, "jax": jax.__version__}
                else:
                    result = measure(common, query, variant, work, identity)
                connection.send({"event": "ok", "result": result})
            except BaseException:
                connection.send({"event": "error", "message": traceback.format_exc(),
                                 "logs": logs.tail_text()})
    except BaseException:
        try:
            connection.send({"event": "error", "message": traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()
