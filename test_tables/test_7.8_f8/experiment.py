"""Author complete-query timing and validation against one RAM resident layout."""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

from contextlib import nullcontext
import hashlib
import importlib
import json
import math
from pathlib import Path
from statistics import median


def measure(common, work, query, layout, identity, order_descriptor, bundle):
    if layout not in ("original", "date_pk") or query not in common.QUERIES:
        raise ValueError("Expected one Figure 8 layout and an SSB query.")
    import jax
    from memory_artifacts import memory_artifacts
    from reference_bridge import inherited_reference
    from resident_inputs import build_inputs
    from source_order import attach_order
    from reproduce.query_plans import paper_plan
    from reproduce.execution_guard.integration import install
    from reproduce.execution_guard.evaluation import layout_coverage
    from tpu_db.benchmark.production import run_resident_query

    evaluation = importlib.import_module("tpu_db.benchmark.layout_evaluation")
    metadata = order_descriptor["metadata"]
    attached = attach_order(order_descriptor) if layout == "original" else nullcontext((None, metadata))
    with attached as (permutation, metadata):
        inputs, source = build_inputs(work, layout, permutation, jax_api=jax,
            permutation_sha256=metadata["permutation_sha256"],
            expected_cache_sha256=metadata["cache_manifest_sha256"])
        del permutation
    root = Path(__file__).resolve().parent / ".memory-artifacts-not-created"
    with memory_artifacts(root, extra_path_modules=(evaluation,)) as store:
        with inherited_reference(store, inputs, query, bundle, metadata, layout) as reference:
            with common.model_context() as model, install(), layout_coverage():
                with evaluation.capture_bound_fact_ranges() as captured:
                    record, _ = run_resident_query(
                        paper_plan(query) if query == "Q4.3" else query, inputs,
                        source_directory=source, output_dir=store.root / "query",
                        planner_mode="exact_int64", reference_dir=reference, runs=5,
                        maximum_calibrations=0, cost_model=model,
                        experimental_calibration=False, host_sql_rows=True)
                descriptor = store.read_json(store.root / "query/descriptor.json")
                rows = store.read_json(store.root / "query" / record["selected_result_directory"] / "sql-rows.json")["rows"]
                scan = evaluation.scan_record(descriptor, inputs, record, captured)
    common.check_result(record)
    expected = {"status": "PASSED", "source_id": identity, "query": query, "sf": 100, "fact_order": layout,
                "planner_mode": "exact_int64", "arithmetic_mode": "exact_int64",
                "result_finalization": "tpu", "timing_contract": "warm_prepared_host_sql_rows_ready_v1"}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Returned record differs from the Figure 8 complete-query boundary.")
    samples = [float(value) for value in record["samples_ms"]]
    latency = float(record["median_ms"])
    if (len(samples) != 5 or any(not math.isfinite(value) or value <= 0 for value in samples)
            or not math.isfinite(latency) or latency <= 0 or latency != median(samples)):
        raise ValueError("Invalid measured samples or median; no latency is substituted.")
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    common.verify_contents(source_id=identity)
    return dict(query=query, variant=layout, median_ms=latency, samples_ms=samples, source_id=identity,
        dataset_metadata_id=record["dataset_metadata_id"], output_digest=digest,
        permutation_sha256=metadata["permutation_sha256"], base_input_sha256=metadata["input_content_sha256"],
        scanned_rows_per_call=scan["scanned_rows_per_call"])
