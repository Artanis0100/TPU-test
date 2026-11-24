"""Table 6: unchanged fixed-plan capacity stress with evidence retained in RAM."""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import hashlib
import importlib
import json
import math
from pathlib import Path
from statistics import median
from unittest.mock import patch


def _fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate(record, query, amount, identity):
    # common.check_result rejects every fallback; the fallback exercised here
    # is intentional. Retain its exact-result requirements and the author's
    # dedicated capacity/scan-range/map classification instead.
    expected = {
        "status": "PASSED", "source_id": identity, "query": query, "sf": 100,
        "requested_reduction": amount, "warmup_calls": 1, "measured_calls": 5,
        "correctness_verified": True, "aggregate_values_exact": True,
        "ordered_sql_rows_verified": True, "query_h2d_bytes": 0,
        "arithmetic_mode": "exact_int64", "result_finalization": "tpu",
        "selection_mode": "evaluation_fixed_plan_capacity_stress", "all_six_calls_exact": True,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Returned record differs from the paper's Table 6 protocol.")
    if record["independent_sql_oracle"]["version"] != "1.4.4":
        raise ValueError("The ordered SQL oracle must be the original DuckDB 1.4.4 reference.")
    measured = record["per_call_diagnostics"]
    if len(measured) != 5:
        raise ValueError("Five independently validated measured calls are required.")
    for call in (record["warmup_diagnostics"], *measured):
        if call.get("correctness_verified") is not True:
            raise ValueError("A capacity-stress call failed exact validation.")
        if call.get("external_exact_fallback") or call.get("range_guard_outside"):
            raise ValueError("A scan-range retry confounds the capacity experiment.")
        if (any(call.get("map_overflows", ())) or call.get("map_capacity_overflow")
                or call.get("map_probe_overflow")):
            raise ValueError("A join-map overflow confounds the capacity experiment.")
        if type(call.get("capacity_exact_branch_calls")) is not int or call["capacity_exact_branch_calls"] not in (0, 1):
            raise ValueError("The capacity fallback count must be zero or one per call.")
    samples = [float(value) for value in record["samples_ms"]]
    latency = float(record["median_ms"])
    if (len(samples) != 5 or any(not math.isfinite(value) or value <= 0 for value in samples)
            or not math.isfinite(latency) or latency <= 0 or latency != median(samples)):
        raise ValueError("Invalid measured samples or median; no latency is substituted.")
    note = record["capacity_stress"]
    flags = {"structure_fixed": True, "model_reranked": False,
             "estimates_for_cost_model_unchanged": True}
    if (any(note.get(key) != value for key, value in flags.items())
            or note["requested_fraction"] != 1 - amount
            or not 0 < note["capacity"] <= note["baseline_capacity"]
            or not math.isclose(note["actual_fraction"], note["capacity"] / note["baseline_capacity"],
                                rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError("Capacity intervention changed the fixed-plan experiment contract.")
    return samples, latency


def measure(common, work, query, variant, identity):
    """Run one original misestimation cell; accept its intended exact branch."""
    amount = float(variant)
    if query not in ("Q1.1", "Q2.1", "Q3.1") or not math.isfinite(amount) or not 0 <= amount < 1:
        raise ValueError("Expected a Table 6 query and a finite reduction in [0, 1).")
    from memory_artifacts import memory_artifacts

    protocol = importlib.import_module("reproduce.misestimation")
    root = Path(__file__).resolve().parent / ".memory-artifacts-not-created"
    captured = []
    with memory_artifacts(root, extra_path_modules=(protocol,)) as store:
        with patch.object(common, "print_result", captured.append):
            protocol.run_cell(work, store.root, 100, query, amount)
        if len(captured) != 1:
            raise RuntimeError("The author cell did not return exactly one completed record.")
        record = store.read_json(store.root / "result.json")
        if _fingerprint(record) != _fingerprint(captured[0]):
            raise ValueError("The returned and serialized author records differ.")
        selection = store.read_json(store.root / "normal-selection.json")
        original = store.read_json(store.root / "original-candidate.json")
        changed = store.read_json(store.root / "capacity-candidate.json")
        injection = store.read_json(store.root / "capacity-injection.json")
    samples, latency = _validate(record, query, amount, identity)
    if injection != record["capacity_stress"]:
        raise ValueError("Capacity injection evidence differs from the validated record.")
    # read_saved_rows in the original entry authenticated this proof against
    # inputs.catalog.dataset_id. The author result itself omits that field.
    reference = work / "references/date_pk/sf100" / query.replace(".", "") / "result.json"
    proof = json.loads(reference.read_text(encoding="utf-8"))
    cache = json.loads((work / "cache/date_pk/sf100-v1/manifest.json").read_text(encoding="utf-8"))
    dataset = proof["dataset_metadata_id"]
    if cache["dataset_metadata_id"] != dataset:
        raise ValueError("The authenticated reference and input cache dataset identities differ.")
    calls = [call["capacity_exact_branch_calls"] for call in record["per_call_diagnostics"]]
    common.verify_contents(source_id=identity)
    return {
        "query": query, "variant": variant, "median_ms": latency, "samples_ms": samples,
        "source_id": identity, "dataset_metadata_id": dataset,
        "normal_selection_sha256": _fingerprint(selection),
        "original_candidate_sha256": _fingerprint(original),
        "capacity_candidate_sha256": _fingerprint(changed),
        "reference_identity_sha256": _fingerprint(record["reference_identity"]),
        "input_content_sha256": record["reference_identity"]["input_content_sha256"],
        "capacity_stress": injection, "fallback_triggered": any(calls),
        "measured_capacity_exact_branch_calls": calls,
        "warmup_capacity_exact_branch_calls": record["warmup_diagnostics"]["capacity_exact_branch_calls"],
    }
