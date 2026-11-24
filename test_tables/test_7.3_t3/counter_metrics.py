"""Table 3 metrics from three in-memory XPlane capture windows.

The author validates 228 counters and records trace loss. This adapter also
requires the nine raw inputs of the selected XProf metrics so missing inputs
cannot silently become zero. No files, converter caches or reports are written.

Official XProf sources inspected at the 2026-10-03 revision:
https://github.com/openxla/xprof/blob/fd1b530a3e1a20526dd172dcf831e178439a6611/xprof/convert/tpu_generic_utilization_utils.cc
https://github.com/openxla/xprof/blob/fd1b530a3e1a20526dd172dcf831e178439a6611/xprof/utils/tpu_counter_ids_v6e.h
https://github.com/openxla/xprof/blob/fd1b530a3e1a20526dd172dcf831e178439a6611/xprof/convert/repository.h
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from contextlib import contextmanager
import importlib
import json
import math
import os
from pathlib import Path
import re


_TC_PREFIX = "VF_CHIP_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_"
_EXTRA_COUNTERS = (
    *(_TC_PREFIX + f"VECTOR_ALU_INSTRUCTION_{i}" for i in range(4)),
    *(_TC_PREFIX + f"XLU_BUSY_{i}" for i in range(3)),
    *(f"VF_CHIP_SC_{i}_SCT_0_SC_STATS_COUNTERS_UNPRIVILEGED_COUNT_CYCLES" for i in range(2)),
)
_METRICS = {
    ("Vector ALUs", 0): "vector_alu_issue_pct",
    ("Avg XLU Busy", 0): "xlu_busy_pct",
    ("SC Vector Issue", 0): "sc_n0_issue_pct",
    ("SC Vector Issue", 1): "sc_n1_issue_pct",
}
_CONVERSION_OPTIONS = {"use_saved_result": False, "group_by": "program"}
# Before Abseil logging initializes, its seconds field can contain Unix time.
_NATIVE_INFO = re.compile(rb"^I\d{4} \d{2}:\d{2}:\d+\.\d+\s+\d+\s")


@contextmanager
def _native_logs(*, verbose=False):
    """Filter native INFO on Linux; preserve diagnostics without disk files.

    XProf writes directly to fd 2, bypassing Python's redirect_stderr. An
    anonymous memory descriptor avoids pipe backpressure while C++ holds the
    GIL. Conversion runs after all measured windows and the TPU worker exits.
    """
    if verbose or not hasattr(os, "memfd_create"):
        yield
        return
    sys.stderr.flush()
    with os.fdopen(os.memfd_create("sectile-xprof-logs", os.MFD_CLOEXEC), "w+b") as logs:
        saved_stderr = os.dup(2)
        failed = True
        try:
            os.dup2(logs.fileno(), 2)
            yield
            failed = False
        finally:
            try:
                sys.stderr.flush()
            finally:
                os.dup2(saved_stderr, 2)
                os.close(saved_stderr)
            logs.seek(0)
            # Replay warnings, errors and unrecognized lines. On failure keep
            # every captured line, including INFO, for troubleshooting.
            with os.fdopen(os.dup(2), "wb") as destination:
                destination.writelines(
                    line for line in logs if failed or not _NATIVE_INFO.match(line)
                )


def _byte_converter():
    for module_name in (
        "xprof.convert._pywrap_profiler_plugin",
        "xprof.pywrap._pywrap_profiler_plugin",
    ):
        try:
            module = importlib.import_module(module_name)
        except (ImportError, OSError, AttributeError):
            continue
        function = getattr(module, "xspace_to_tools_data_from_byte_string", None)
        if callable(function):
            return function
    raise RuntimeError(
        "XProf's in-memory XSpace converter is unavailable in the current environment. "
        "Keep the existing environment and report this error; see README.md."
    )


def preflight(*, verbose=False) -> None:
    """Check the installed native byte-conversion API without capturing data."""
    with _native_logs(verbose=verbose):
        _byte_converter()


def _author_modules(artifact_root):
    root = Path(artifact_root).expanduser().resolve()
    sys.path[:0] = [str(root / "src"), str(root)]
    contracts = importlib.import_module("tpu_db.benchmark.steady_state_profiling")
    reader = importlib.import_module("tpu_db.benchmark.xplane_reader")
    accepted = importlib.import_module("reproduce.protocols.extract_accepted_counters")
    xplane = importlib.import_module("tpu_db.benchmark.steady_state_xplane")
    for module in (contracts, reader, accepted, xplane):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError("A counter-analysis module came from a different artifact.")
    return contracts, reader, accepted, xplane


def _extra_counter_metadata(raw, reader, accepted_record):
    """Validate additional formula inputs; this is stricter than the author gate."""
    space = reader.XSpace.FromString(raw)
    identities = {}
    seen_ids = {row["performance_counter_id"] for row in accepted_record["rows"]}
    for plane in space.planes:
        for line in plane.lines:
            if line.name != "counters_0":
                continue
            for event in line.events:
                metadata = plane.event_metadata[event.metadata_id]
                name = metadata.name
                if name not in _EXTRA_COUNTERS:
                    continue
                if name in identities or metadata.id != event.metadata_id:
                    raise ValueError(f"Repeated or inconsistent metric input: {name}")
                ms = reader.resolved_stats(metadata.stats, plane)
                es = reader.resolved_stats(event.stats, plane)
                counter_id = ms.get("performance_counter_id")
                description = ms.get("performance_counter_description")
                if (type(counter_id) is not int or counter_id < 0 or counter_id in seen_ids
                        or not isinstance(description, str) or not description.strip()):
                    raise ValueError(f"Invalid metric input metadata: {name}")
                seen_ids.add(counter_id)
                values = [stats["counter_value"] for stats in (ms, es) if "counter_value" in stats]
                if (not values or any(type(value) is not int or value < 0 for value in values)
                        or any(value != values[0] for value in values[1:])):
                    raise ValueError(f"Invalid or ambiguous metric input value: {name}")
                if name.endswith("_CYCLES") and values[0] <= 0:
                    raise ValueError(f"SC metric cycle denominator must be positive: {name}")
                if line.timestamp_ns or event.offset_ps or event.duration_ps:
                    raise ValueError(f"Metric input is not an untimed end sample: {name}")
                identities[name] = (counter_id, description)
    missing = set(_EXTRA_COUNTERS) - set(identities)
    if missing:
        raise ValueError("Missing raw Table 3 metric inputs: " + ", ".join(sorted(missing)))
    return identities


def _selected_metrics(rows):
    selected = {}
    for row in rows:
        node = row["node"]
        if type(node) not in (int, float) or not math.isfinite(node):
            continue
        key = row["name"], node
        if key not in _METRICS:
            continue
        column = _METRICS[key]
        if column in selected:
            raise ValueError(f"Ambiguous Table 3 metric identity: {key}")
        if not math.isfinite(row["peak"]) or row["peak"] <= 0:
            raise ValueError(f"Table 3 metric has no positive peak: {key}")
        expected_unit = "cycles" if key[0] == "Avg XLU Busy" else "instructions"
        if row["unit"] != expected_unit:
            raise ValueError(f"Unexpected Table 3 metric unit: {key}")
        selected[column] = row
    if set(selected) != set(_METRICS.values()):
        missing = set(_METRICS.values()) - set(selected)
        raise ValueError("Missing Table 3 utilization metrics: " + ", ".join(sorted(missing)))
    return selected


def summarize_windows(raw_windows: list[bytes], artifact_root: Path, *, verbose=False) -> dict:
    """Return the five Table 3 percentages and per-window loss metadata in RAM."""
    if len(raw_windows) != 3 or any(type(raw) is not bytes or not raw for raw in raw_windows):
        raise ValueError("Table 3 requires exactly three nonempty raw XPlane byte strings.")
    with _native_logs(verbose=verbose):
        converter = _byte_converter()
    contracts, reader, accepted, xplane = _author_modules(artifact_root)
    records, ledgers, payloads, extra_identities, selected_windows = [], [], [], [], []
    for window, raw in enumerate(raw_windows, start=1):
        record = accepted.extract_bytes(raw, contracts, reader)
        extra_identities.append(_extra_counter_metadata(raw, reader, record))
        # Preloaded XSpaces disable SessionSnapshot's filesystem/cache access.
        # The empty locator is intentional: no profile file needs to exist.
        with _native_logs(verbose=verbose):
            data, success = converter([raw], [""], "utilization_viewer", dict(_CONVERSION_OPTIONS))
            if not success:
                raise RuntimeError(f"XProf utilization conversion failed for window {window}.")
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise ValueError("XProf utilization output must be a JSON table object.")
        selected = _selected_metrics(contracts.utilization_viewer_rows(payload))
        if any(not math.isfinite(100.0 * row["achieved"] / row["peak"])
               for row in selected.values()):
            raise ValueError(f"Table 3 metric ratio is not finite in window {window}.")
        records.append(record)
        ledgers.append(record["hbm_rows"])
        payloads.append(payload)
        selected_windows.append(selected)
    if any(identity != extra_identities[0] for identity in extra_identities[1:]):
        raise ValueError("Raw Table 3 metric input metadata differs between windows.")
    xplane.require_matching_counter_metadata(ledgers)
    hbm = contracts.summarize_hbm_counter_replicates(ledgers)
    utilization = contracts.summarize_utilization_viewers(payloads)
    result = {"hbm_peak_pct": float(hbm["summary"]["achieved_over_peak_percent"]["median"])}
    selected_summary = _selected_metrics(
        [dict(row, peak=row["peak_median"]) for row in utilization["metrics"]]
    )
    for column, row in selected_summary.items():
        result[column] = float(row["whole_window_achieved_over_peak_percent_median"])
    if any(not math.isfinite(value) or value < 0 for value in result.values()):
        raise ValueError("Table 3 conversion produced an invalid percentage.")
    windows = []
    for index, (record, selected) in enumerate(zip(records, selected_windows), start=1):
        metrics = {column: 100.0 * row["achieved"] / row["peak"]
                   for column, row in selected.items()}
        metrics["hbm_peak_pct"] = hbm["replicate_metrics"][index - 1]["achieved_over_peak_percent"]
        windows.append({
            "window": index,
            "metrics": metrics,
            "warnings": record["warnings"],
            "plane_loss_evidence": record["plane_loss_evidence"],
            "trace_buffers_dropped_markers": record["trace_buffers_dropped_markers"],
            "legacy_gate_would_reject": record["legacy_gate_would_reject"],
            "legacy_gate_reasons": record["legacy_gate_reasons"],
            "required_counter_count": record["required_counter_count"],
            "additional_metric_input_count": len(_EXTRA_COUNTERS),
        })
    result["windows"] = windows
    result["trace_loss_present"] = any(record["legacy_gate_would_reject"] for record in records)
    return result
