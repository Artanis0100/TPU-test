"""Table 5: unchanged author Q2.3 preselection and numeric-ready timing."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import math
from pathlib import Path
from statistics import median
import sys
from unittest.mock import patch


def _fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def measure(common, work, query, variant, identity):
    """Run the original formal protocol, retaining its output bytes in RAM."""
    if query != "Q2.3" or variant not in ("bitmap", "bitmap_rank", "hash"):
        raise ValueError("Table 5 requires Q2.3 and one of its three preselected families.")
    import numpy as np
    from memory_artifacts import memory_artifacts
    from reproduce.reference_cache import saved_oracles

    protocol = importlib.import_module("reproduce.protocols.table4")
    root = Path(__file__).resolve().parent / ".console-artifacts-not-created"
    arguments = [
        protocol.__file__, "--source", str(common.ROOT), "--parameters", str(common.PARAMETERS),
        "--root", str(root), "--family", variant,
        "--cache-root", str(work / "cache/date_pk"),
        "--data-root", str(work / "data/date_pk"),
        "--reference-root", str(work / "references/date_pk"),
    ]
    # The author's statistics serialization, hash readback, preflight and final
    # record creation all execute unchanged. Only their persistence is in RAM.
    with memory_artifacts(root, extra_path_modules=(
        protocol, "tpu_db.benchmark.evaluation_preflight", "tpu_db.benchmark.evaluation_inventory",
    )) as store:
        with patch.object(sys, "argv", arguments), saved_oracles(work, 100):
            protocol.main()
        record = store.read_json(root / "result.json")
        preselection = store.read_json(root / "preselection.json")
        ordered = store.read_json(root / "untimed-output-preflight/ordered-rows.json")
        with np.load(io.BytesIO(store.read_bytes(root / "aggregates.npz")), allow_pickle=False) as arrays:
            actual, reference = arrays["actual"], arrays["reference"]
            np.testing.assert_array_equal(actual, reference)
            if actual.dtype != np.int64:
                raise ValueError("Table 5 numeric output must preserve exact INT64 aggregates.")
            aggregate_sha = _fingerprint({"dtype": actual.dtype.str, "shape": list(actual.shape),
                                          "sha256": hashlib.sha256(actual.tobytes(order="C")).hexdigest()})
        if hashlib.sha256(store.read_bytes(root / record["statistics_file"])).hexdigest() != record["statistics_sha256"]:
            raise ValueError("The in-memory model statistics changed during measurement.")
    common.check_result(record)
    expected = {
        "source_id": identity, "sf": 100, "query": query, "family": variant,
        "arithmetic_mode": "exact_int64", "prediction_units": "milliseconds",
        "timing_contract": "cost_model_candidate_dependent_tpu_v1", "timing_interval": "A",
        "tpu_finalization_in_primary_timer": False, "host_sql_rows_in_primary_timer": False,
        "host_numeric_output_equivalence_verified": True,
        "coefficients_promoted": False, "query_measurements_used_for_fit": False,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("The returned record differs from the paper's Table 5 numeric interval.")
    if ordered["actual"] != ordered["oracle"]:
        raise ValueError("The untimed ordered SQL outputs differ from the independent oracle.")
    if (preselection["source_id"] != identity
            or preselection["parameter_sha256"] != record["parameter_sha256"]
            or set(preselection["candidates"]) != {"bitmap", "bitmap_rank", "hash"}):
        raise ValueError("The three-family frozen preselection identity is inconsistent.")
    samples = [float(value) for value in record["samples_ms"]]
    latency = float(record["median_ms"])
    predicted = float(record["planner_predicted_cost"]["milliseconds"])
    if (len(samples) != 5 or any(not math.isfinite(value) or value <= 0 for value in samples)
            or not math.isfinite(latency) or latency != median(samples)
            or not math.isfinite(predicted) or predicted <= 0
            or type(record["planner_selected"]) is not bool):
        raise ValueError("The measured samples, prediction or planner-choice flag are invalid.")
    common.verify_contents(source_id=identity)
    return {
        "query": query, "variant": variant, "median_ms": latency, "samples_ms": samples,
        "predicted_ms": predicted, "planner_selected": record["planner_selected"],
        "source_id": identity, "dataset_metadata_id": record["dataset_metadata_id"],
        "physical_key": record["physical_key"],
        "aggregate_sha256": aggregate_sha, "ordered_rows_sha256": _fingerprint(ordered),
        "comparison_identity": {
            "parameter_sha256": record["parameter_sha256"],
            "statistics_sha256": record["statistics_sha256"],
            "input_cache_manifest_sha256": record["input_cache_manifest_sha256"],
            "preselection_sha256": _fingerprint(preselection),
            "date_contract_sha256": _fingerprint(preselection["date_invariant"]),
            "runtime_id": record["runtime_id"],
            "aggregate_sha256": aggregate_sha,
            "ordered_rows_sha256": _fingerprint(ordered),
        },
    }
