"""Reuse authenticated SQL oracles across one proven physical row permutation."""

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def authenticate_references(common, work, queries, metadata):
    """Input files were hashed once by source_order; apply author oracle checks."""
    import numpy as np
    from reproduce.query_plans import paper_plan
    from reproduce.reference_cache import read_saved_rows

    if fingerprint(metadata["input_files_sha256"]) != metadata["input_content_sha256"]:
        raise ValueError("Authenticated input content identity differs.")
    code = {str(path.relative_to(common.ROOT)): common.sha256(path)
            for path in sorted((common.ROOT / "src").rglob("*.py"))}
    identity = {name: metadata[name] for name in ("input_files_sha256", "input_content_sha256")}
    identity["reference_code_sha256"] = fingerprint(code)
    base = SimpleNamespace(catalog=SimpleNamespace(dataset_id=metadata["base_dataset_id"]))
    bundles = {}
    for query in queries:
        logical = paper_plan(query)
        directory = Path(work) / "references/date_pk/sf100" / query.replace(".", "")
        rows, oracle = read_saved_rows(directory, logical, base, identity, 100)
        record_bytes = (directory / "result.json").read_bytes()
        record = json.loads(record_bytes)
        arrays = (directory / "aggregates.npz").read_bytes()
        if hashlib.sha256(arrays).hexdigest() != record["files_sha256"]["aggregates.npz"]:
            raise ValueError("The authenticated aggregate reference changed.")
        with np.load(io.BytesIO(arrays), allow_pickle=False) as values:
            if values["reference"].dtype != np.int64:
                raise ValueError("The aggregate oracle is not exact INT64.")
            np.testing.assert_array_equal(values["actual"], values["reference"])
        bundles[query] = dict(query=query, logical_plan_sha256=fingerprint(asdict(logical)),
            rows=rows, oracle=oracle, aggregates_npz=arrays, joined_rows=record["joined_rows"],
            base_dataset_id=metadata["base_dataset_id"],
            base_reference_sha256=hashlib.sha256(record_bytes).hexdigest())
    return bundles


@contextmanager
def inherited_reference(store, inputs, query, bundle, metadata, layout):
    """Keep true base provenance; INT64 SQL results survive a fact-row bijection."""
    import numpy as np
    from resident_inputs import derived_dataset_id
    from tpu_db import storage
    from tpu_db.result_boundary import measurement, oracle

    specs = inputs.catalog.expected_specs
    facts = [spec for spec in specs if spec.table == "lineorder"]
    expected_id = derived_dataset_id(metadata["base_dataset_id"], layout, metadata["permutation_sha256"])
    if (inputs.catalog.dataset_id != expected_id or storage.fact_layout(specs) != layout
            or bundle["query"] != query or bundle["base_dataset_id"] != metadata["base_dataset_id"]
            or metadata.get("bijection_verified") is not True
            or metadata.get("order_key_sequence_verified") is not True
            or len(facts) != 9 or any(spec.logical_row_count != metadata["rows"] for spec in facts)):
        raise ValueError("The inherited oracle does not match this proven resident relation.")
    inheritance = dict(method="all nine fact columns share one proven row bijection; exact INT64 SQL",
        base_dataset_id=metadata["base_dataset_id"], resident_dataset_id=expected_id, layout=layout,
        permutation_sha256=metadata["permutation_sha256"], base_reference_sha256=bundle["base_reference_sha256"])
    directory = store.root / "inherited-reference"
    directory.mkdir()
    (directory / "result.json").write_text(json.dumps(dict(status="PASSED", dataset_metadata_id=expected_id,
        arithmetic_mode="exact_int64", joined_rows=bundle["joined_rows"],
        purpose="Inherited authenticated oracle across proven physical row permutation",
        row_order_equivalence=inheritance)))
    aggregate_path = directory / "aggregates.npz"
    # np.load calls builtins.open. Adapt this exact virtual input, not ordinary files.
    original_load = np.load

    def load(file, *args, **kwargs):
        if isinstance(file, (str, Path)) and str(file) == str(aggregate_path):
            return original_load(io.BytesIO(bundle["aggregates_npz"]), *args, **kwargs)
        return original_load(file, *args, **kwargs)

    def query_rows(logical, actual_inputs):
        if (actual_inputs is not inputs or logical.name != query
                or fingerprint(asdict(logical)) != bundle["logical_plan_sha256"]):
            raise ValueError("SQL oracle requested for another logical plan or input catalog.")
        return list(bundle["rows"]), dict(bundle["oracle"], row_order_equivalence=inheritance)

    with patch.object(np, "load", load), patch.object(oracle, "query_rows", query_rows), \
            patch.object(measurement, "query_rows", query_rows):
        yield directory
