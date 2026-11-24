"""One TPU query worker for the in-memory Table 3 counter controller.

Only the parent process captures counters. This process binds and validates
the author's selected query once, then repeatedly invokes that same binding.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch


QUERIES = ("Q1.1", "Q2.1", "Q3.1", "Q3.2", "Q4.1")


class _BindingCaptured(BaseException):
    """Exit production before its end-of-query JIT-cache cleanup."""


def _load_steady_tool(artifact_root):
    path = artifact_root / "scripts/evaluation/run_steady_state_profile.py"
    name = "_sectile_table3_steady_worker"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load the author's steady-state helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _capture_binding(common, inputs, source, work, query, output_root):
    from console_artifacts import discard_author_artifacts
    from reproduce.execution_guard.integration import install
    from reproduce.reference_cache import saved_oracles
    from tpu_db.benchmark.production import run_resident_query
    from tpu_db.result_boundary import measurement, oracle

    state = {}
    reference = work / "references/date_pk/sf100" / query.replace(".", "")
    with discard_author_artifacts(output_root), saved_oracles(work, 100):
        with install() as executions:
            safety_measure = measurement.measure_host_sql_candidate

            def capture(bound, **kwargs):
                # Preserve all six formal calls, guards, retry policy and exact
                # validators. Their normal audit-file writes are discarded.
                record = safety_measure(bound, **kwargs)
                oracle_rows, _ = oracle.query_rows(bound.logical, inputs)
                state.update(
                    bound=bound, execution=executions.get(id(bound)), record=record,
                    expected=kwargs["expected"], expected_rows=kwargs["expected_rows"],
                    oracle_rows=oracle_rows,
                )
                raise _BindingCaptured()

            with patch.object(measurement, "measure_host_sql_candidate", capture):
                try:
                    run_resident_query(
                        query, inputs, source_directory=source,
                        output_dir=output_root / query, planner_mode="exact_int64",
                        reference_dir=reference, runs=5, maximum_calibrations=0,
                        cost_model=common.make_model(), experimental_calibration=False,
                        host_sql_rows=True,
                    )
                except _BindingCaptured:
                    pass
    if not state:
        raise RuntimeError("The production planner did not provide a measured binding.")
    # Do not retain the capture exception or its traceback, and do not clear JIT
    # caches: historical callables own warmed nested executables.
    gc.collect()
    common.check_result(state["record"])
    return state


def _profile_binding(state):
    bound, execution = state["bound"], state["execution"]
    if execution is None:
        return bound
    from reproduce.execution_guard.profile import PreparedProfileResult, strip_flag

    def device_call(*arguments):
        if len(arguments) != len(bound.arguments) or any(
            actual is not expected
            for actual, expected in zip(arguments, bound.arguments)
        ):
            raise ValueError("The profile must reuse the selected resident operands.")
        result, outside = execution.device_call()
        # The exact author profile ABI: Python PyTree packing only; no extra
        # kernel, host flag observation, D2H or row construction inside the loop.
        return result._replace(source=result.source._replace(
            diagnostics=(*result.source.diagnostics, outside)
        ))

    return replace(
        bound, function=device_call, prepared=PreparedProfileResult(bound.prepared),
        restore=lambda host: bound.restore(strip_flag(host)), warm_result=None,
    )


def _validate_result(profile, state, rows, host):
    from tpu_db.benchmark.overflow_diagnostics import complete_overflow_diagnostics

    _, checks = profile.original.validate(
        profile.restore(host), state["expected"], state["expected_rows"]
    )
    checks = complete_overflow_diagnostics(checks)
    if rows != state["oracle_rows"]:
        raise AssertionError("Profile SQL rows differ from the independent exact oracle.")
    for name in (
        "device_fallback", "continuation_used", "fallback_or_continuation",
        "map_capacity_overflow", "map_probe_overflow", "range_fallback",
        "page_overflows", "lookup_fallback_count", "range_guard_outside",
        "external_exact_fallback",
    ):
        if checks.get(name, False):
            raise RuntimeError(f"Reject this success-path counter capture: {name}.")
    if any(checks.get("map_overflows", ())):
        raise RuntimeError("Reject this counter capture: a map overflow occurred.")


def _run(connection, query, artifact_root, work, port):
    if query not in QUERIES or not 1 <= port <= 65535:
        raise ValueError("Expected a Table 3 query and a valid profiler port.")
    artifact_root = Path(artifact_root).expanduser().resolve()
    work = Path(work).expanduser().resolve()
    sys.path[:0] = [str(artifact_root / "src"), str(artifact_root)]
    from reproduce import common

    if common.ROOT.resolve() != artifact_root:
        raise ValueError("A different Sectile artifact was imported.")
    common.outside_repository(work)
    identity = common.configure(100, profile=True)
    # Keep configure()'s enhanced barrier and transpose-trace runtime options.
    os.environ.update(
        XLA_FLAGS="", JAX_ENABLE_COMPILATION_CACHE="false", JAX_DUMP_IR_TO="",
        PROFILE_HARDWARE_TRACE="0", PROFILE_STAGE_TRACE="0",
        PROFILE_CORE_ABLATION="0", RUN_CORE_ABLATION="0", RUN_COMPONENT_ABLATIONS="0",
    )
    jax = common.require_tpu()
    jax.config.update("jax_enable_compilation_cache", False)
    device = jax.devices()[0]
    kind = device.device_kind.lower()
    if "v6e" not in kind and "v6 lite" not in kind:
        raise ValueError(f"Expected one v6e TPU chip, found {device.device_kind}.")
    from jax.experimental.pallas import tpu as pltpu

    if getattr(pltpu.get_tpu_info(), "sparse_core", None) is None:
        raise RuntimeError("The TPU runtime does not expose programmable SparseCore.")
    inputs, source, _ = common.load_inputs(work, 100)
    output_root = Path(__file__).resolve().parent / ".console-artifacts-not-created"
    state = _capture_binding(common, inputs, source, work, query, output_root)
    common.verify_contents(source_id=identity)
    profile = _profile_binding(state)
    descriptor = json.dumps(profile.descriptor.to_dict(), sort_keys=True)
    selected_function, arguments = profile.function, profile.arguments
    producer = state["bound"].function
    tool = _load_steady_tool(artifact_root)

    def require_unchanged():
        if (profile.function is not selected_function
                or state["bound"].function is not producer
                or len(profile.arguments) != len(arguments)
                or any(a is not b for a, b in zip(profile.arguments, arguments))
                or json.dumps(profile.descriptor.to_dict(), sort_keys=True) != descriptor):
            raise RuntimeError("The selected profile binding changed during capture.")

    def complete_call():
        with jax.enable_x64(profile.original.is_typed):
            rows, host, _ = tool._complete_host_call(
                selected_function, arguments, profile.prepared, jax
            )
        _validate_result(profile, state, rows, host)

    for _ in range(2):
        complete_call()
    require_unchanged()
    details = profile.descriptor.to_dict()
    selected_identity = {
        "candidate": details["candidate_label"],
        "physical_key": details["latency_prediction"]["physical_key"],
        "device": device.device_kind, "device_id": device.id,
        "jax": jax.__version__, "source_id": identity,
        "dataset_metadata_id": inputs.catalog.dataset_id,
        "guarded": state["execution"] is not None,
    }
    server = None
    try:
        server = jax.profiler.start_server(port)
        connection.send({"event": "ready", "query": query, "port": port,
                         "identity": selected_identity})
        command = connection.recv()
        if command != {"command": "start"}:
            raise RuntimeError(f"Expected the start command, received {command!r}.")
        checks, running = 0, False

        def stop_requested():
            nonlocal checks, running
            # The second entry follows one complete synchronized invocation.
            if checks and not running:
                connection.send({"event": "running", "query": query})
                running = True
            checks += 1
            stop = False
            while connection.poll():
                message = connection.recv()
                if message == {"command": "stop"}:
                    stop = True
                elif message == {"command": "abort"}:
                    raise RuntimeError("The parent aborted the counter loop.")
                else:
                    raise RuntimeError(f"Unexpected counter-loop command: {message!r}.")
            return stop

        with jax.enable_x64(profile.original.is_typed):
            last, intervals = tool.run_steady_iterations(
                selected_function, arguments, jax, stop_requested=stop_requested,
                annotation=jax.profiler.TraceAnnotation,
            )
        # Counters have stopped before the controller sends stop. Observe only
        # the final device result, as in the author's immutable-snapshot protocol.
        host = jax.device_get(last)
        rows = profile.prepared.rows(host)
        profile.prepared.validate_host_abi(host)
        _validate_result(profile, state, rows, host)
        for _ in range(5):
            complete_call()
        require_unchanged()
        common.verify_contents(source_id=identity)
        connection.send({"event": "validated", "query": query,
                         "completed_invocations": len(intervals),
                         "extra_warmups": 2, "post_loop_calls": 5,
                         "identity": selected_identity})
    finally:
        if server is not None:
            jax.profiler.stop_server()


def run_worker(connection, query, artifact_root: Path, work: Path, port: int):
    """Spawn target; communicate status in memory and never create result files."""
    sys.dont_write_bytecode = True
    os.environ.update(JAX_PLATFORMS="tpu", PYTHONDONTWRITEBYTECODE="1",
                      JAX_ENABLE_COMPILATION_CACHE="false")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from console_artifacts import NullText
        with redirect_stdout(NullText()):
            _run(connection, query, artifact_root, work, port)
    except BaseException:
        try:
            connection.send({"event": "error", "query": query,
                             "message": traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError, ValueError):
            pass
    finally:
        connection.close()
