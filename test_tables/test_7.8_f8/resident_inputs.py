"""Build both resident layouts from the existing read-only date_pk cache.

The source-order catalog is a RAM row permutation of all fact columns, not a
new raw-table snapshot. The original source directory is only the namespace
for PreorderedColumnReader, which projects the actual resident host arrays.
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def _require_digest(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"Invalid {label} SHA-256 identity.")


def derived_dataset_id(base_id, layout, permutation_sha256=None):
    """A reordered physical catalog has its own explicit reconstruction ID."""
    _require_digest(base_id, "base dataset")
    if layout == "date_pk":
        return base_id
    if layout != "original":
        raise ValueError("Expected original or date_pk layout.")
    _require_digest(permutation_sha256, "row permutation")
    value = {"contract": "sectile_ram_source_order_v1", "base_dataset_id": base_id,
             "permutation_sha256": permutation_sha256, "fact_order": "original"}
    return _digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def load_host_cache(work, *, expected_cache_sha256=None):
    """Validate and mmap existing columns without allocating any TPU arrays.

    Match the author's load_integer_inputs metadata, schema, dtype and shape
    checks. Full input authentication and the permutation proof are performed
    once by the controller, before any measured query is launched.
    """
    from tpu_db import storage
    from tpu_db.benchmark import typed_cache

    work = Path(work).expanduser().resolve()
    source, cache = work / "data/date_pk/sf100", work / "cache/date_pk/sf100-v1"
    payload = (cache / "manifest.json").read_bytes()
    if expected_cache_sha256 is not None and _digest(payload) != expected_cache_sha256:
        raise ValueError("The authenticated date_pk cache manifest changed.")
    manifest = json.loads(payload)
    required = {"kind": "typed_integer_input_cache", "schema_version": 1,
                "status": "READY", "scale_factor": 100, "fact_order": "date_pk"}
    if (any(manifest.get(key) != value for key, value in required.items())
            or Path(manifest["source_directory"]).resolve() != source.resolve()
            or len(manifest["arrays"]) != 26):
        raise ValueError("The existing date_pk integer cache is incomplete or incompatible.")
    specs = tuple(storage._column_spec_from_manifest(entry["spec"]) for entry in manifest["arrays"])
    storage.canonical_schema_contract(specs)
    if ({(spec.table, spec.physical_column) for spec in specs}
            != set(storage.SSB_PHYSICAL_SCHEMA_BY_KEY)
            or len({spec.semantic for spec in specs}) != 26
            or sum(spec.table == "lineorder" for spec in specs) != 9
            or storage.fact_layout(specs) != "date_pk"
            or any(spec.dataset_id != manifest["dataset_metadata_id"] for spec in specs)):
        raise ValueError("The cached canonical column identities differ from the source schema.")
    if typed_cache._sources(source, specs) != manifest["source_files"]:
        raise ValueError("A source table changed since date_pk cache preparation.")
    layout_metadata = typed_cache._source_layout_metadata(source, fact_order="date_pk", scale_factor=100)
    if ("source_layout_metadata" in manifest
            and manifest["source_layout_metadata"] != layout_metadata):
        raise ValueError("The source data manifest changed since cache preparation.")
    host = {}
    for entry, spec in zip(manifest["arrays"], specs, strict=True):
        filename = f"{spec.table}.{spec.physical_column}.npy"
        path = cache / filename
        if entry["file"] != filename or typed_cache._stamp(path) != entry["stamp"]:
            raise ValueError(f"A cached column changed: {filename}.")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if (values.shape != (spec.logical_row_count,) or values.dtype != np.dtype(spec.storage_dtype)
                or values.dtype.kind not in "iu"):
            raise ValueError(f"Cached column shape/dtype is incompatible: {filename}.")
        values.flags.writeable = False
        host[spec.semantic] = values
    return manifest, specs, MappingProxyType(host), source


def _check_permutation(permutation, rows, chunk_rows):
    permutation = np.asarray(permutation)
    if permutation.shape != (rows,) or permutation.dtype.kind not in "iu":
        raise ValueError("The proven row permutation has the wrong shape or dtype.")
    for start in range(0, rows, chunk_rows):
        block = permutation[start:start + chunk_rows]
        if block.size and (int(block.min()) < 0 or int(block.max()) >= rows):
            raise ValueError("The row permutation contains an out-of-range source index.")
    return permutation


def _take_column(values, permutation, chunk_rows):
    output = np.empty(values.shape, dtype=values.dtype)
    for start in range(0, values.size, chunk_rows):
        stop = min(values.size, start + chunk_rows)
        # Bounded gather temporaries avoid a second complete 600M-row column.
        np.take(values, permutation[start:stop], out=output[start:stop], mode="raise")
    output.flags.writeable = False
    return output


def build_inputs(work, layout, permutation=None, *, jax_api, permutation_sha256=None,
                 expected_cache_sha256=None, chunk_rows=1 << 20):
    """Upload exactly one chosen layout, preserving the author resident ABI."""
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("The gather chunk size must be a positive integer.")
    if layout not in ("original", "date_pk"):
        raise ValueError("Expected original or date_pk layout.")
    from tpu_db import storage
    from tpu_db.benchmark.frozen import _pad_execution_column
    from tpu_db.benchmark.typed import TypedResidentInputs
    from tpu_db.planning.range_scan import ResidentZoneMap, ZoneMapIndex

    manifest, base_specs, base_host, source = load_host_cache(
        work, expected_cache_sha256=expected_cache_sha256
    )
    dataset_id = derived_dataset_id(manifest["dataset_metadata_id"], layout, permutation_sha256)
    specs = tuple(replace(spec, dataset_id=dataset_id,
                          order="original" if layout == "original" and spec.table == "lineorder" else spec.order)
                  for spec in base_specs)
    storage.canonical_schema_contract(specs)
    row_counts = {spec.logical_row_count for spec in specs if spec.table == "lineorder"}
    if len(row_counts) != 1:
        raise ValueError("Fact columns have different logical row extents.")
    if layout == "original":
        if permutation is None:
            raise ValueError("Original order requires the authenticated source-row permutation.")
        permutation = _check_permutation(permutation, row_counts.pop(), chunk_rows)
    elif permutation is not None:
        raise ValueError("Date-sorted execution must use the existing cache without a permutation.")
    limit = (jax_api.devices()[0].memory_stats() or {}).get("bytes_limit")
    capacity = min(24 << 30, int(limit * .8)) if limit else 24 << 30
    if sum(spec.storage_bytes for spec in specs) > capacity:
        raise ValueError("The resident input catalog exceeds the author's TPU admission ceiling.")
    catalog = storage.ResidentScaleFactor(dataset_id, capacity, expected_specs=specs)
    host, zones = {}, {}
    with catalog.preload_scope(jax_api.device_put):
        for spec in specs:
            values = base_host[spec.semantic]
            if layout == "original" and spec.table == "lineorder":
                values = _take_column(values, permutation, chunk_rows)
            host[spec.semantic] = values
            column = catalog.preload(spec, _pad_execution_column(values, spec))
            if spec.semantic.relation == "lineorder" and spec.semantic.name == "orderdate":
                # Rebuild zones from the actual row order, bound to its precise
                # new resident handle; never reuse date_pk pruning metadata.
                zones[spec.semantic] = ResidentZoneMap(column, ZoneMapIndex.from_array(values))
    inputs = TypedResidentInputs(catalog, MappingProxyType(host), None, MappingProxyType(zones))
    return inputs, source
