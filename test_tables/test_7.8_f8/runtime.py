"""One fresh TPU process per query/layout; never load both layouts together."""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import os
from pathlib import Path
import traceback


def prepare_runtime(artifact_root, work, layout="date_pk"):
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
    identity = common.configure(100, layout)
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


def run_worker(connection, artifact_root, work, query=None, layout=None, order_descriptor=None, bundle=None):
    from console_output import capture_console

    try:
        with capture_console() as logs:
            try:
                common, jax, identity = prepare_runtime(
                    Path(artifact_root).resolve(), Path(work).resolve(), layout or "date_pk"
                )
                if query is None:
                    result = {"device": jax.devices()[0].device_kind, "jax": jax.__version__,
                              "source_id": identity,
                              "hbm_bytes_limit": (jax.devices()[0].memory_stats() or {}).get("bytes_limit")}
                else:
                    from experiment import measure
                    result = measure(common, work, query, layout, identity, order_descriptor, bundle)
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
