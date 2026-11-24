"""Table 4: run author fixed-skeleton controls with evidence retained in RAM."""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import importlib
import importlib.util
import math
from pathlib import Path
from statistics import median
from unittest.mock import patch


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load the author's adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _compiler(common, work, query, store):
    from reproduce import compiler_only

    captured = []
    with patch.object(common, "print_result", captured.append):
        compiler_only.run_cell(work, store.root, query, 100)
    if len(captured) != 1:
        raise RuntimeError("The compiler-managed control did not return exactly one record.")
    return captured[0]


def _fixed(common, work, query, family, store, protocol):
    from reproduce.reference_cache import saved_oracles
    from tpu_db.planning import cost_model

    arguments = [
        "--source", str(common.ROOT), "--parameters", str(common.PARAMETERS),
        "--root", str(store.root), "--query", query, "--family", family,
        "--cache", str(work / "cache/date_pk/sf100-v1"),
        "--data", str(work / "data/date_pk/sf100"),
        "--reference-root", str(work / "references/date_pk"),
    ]
    # Keep the original experiment-local model injection scoped as well.
    with saved_oracles(work, 100), patch.object(
        cost_model, "production_cost_model", cost_model.production_cost_model
    ):
        if (query, family) == ("Q2.3", "hash"):
            # Original artifact recovery: Part 1 chunk, Supplier 2 chunks.
            # Its stage-signature, DATE invariance and topology checks read the
            # exact JSON bytes produced by the unchanged base protocol in RAM.
            adapter = _load_module(
                "_sectile_t4_q23_hash_formal",
                common.ROOT / "evaluation/table3/table3_q23_hash_chunked_formal.py",
            )
            store.patch_module(adapter)
            load = adapter.load_module

            def load_with_ram_paths(name, path):
                return store.patch_module(load(name, path))

            with patch.object(adapter, "load_module", load_with_ram_paths):
                adapter.main([
                    *arguments, "--base-adapter",
                    str(common.ROOT / "reproduce/protocols/table3.py"),
                ])
            completed = store.read_json(store.root / "external-q23-chunked-completed.json")
            if completed["status"] != "BASE_FORMAL_MEASUREMENT_COMPLETED":
                raise ValueError("The original chunked-Hash completion checks did not pass.")
        else:
            with patch.object(sys, "argv", [str(protocol.__file__), *arguments]):
                protocol.main()
    cell = store.root / "fixed-formal-sf100" / query.replace(".", "") / family
    record = store.read_json(cell / "result.json")
    protocol.validate_formal_record(record)
    return record


def measure(common, work, query, variant, identity):
    """Return one real complete-query median; no numeric output is hardcoded."""
    if query not in ("Q2.3", "Q3.3") or variant not in (
        "compiler_only", "hash", "bitmap", "bitmap_rank"
    ):
        raise ValueError("Expected a Table 4 query and representation control.")
    from memory_artifacts import memory_artifacts

    protocol = importlib.import_module("reproduce.protocols.table3")
    output_root = Path(__file__).resolve().parent / ".memory-artifacts-not-created"
    with memory_artifacts(output_root, extra_path_modules=(protocol,)) as store:
        record = (_compiler(common, work, query, store) if variant == "compiler_only"
                  else _fixed(common, work, query, variant, store, protocol))
        common.check_result(record)
        expected = {
            "query": query, "sf": 100, "source_id": identity,
            "arithmetic_mode": "exact_int64", "result_finalization": "tpu",
            "selection_mode": ("forced_compiler_managed_only" if variant == "compiler_only"
                               else "formal_fixed_skeleton_nondate_representation_control"),
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("Returned record differs from the requested Table 4 protocol.")
        if variant != "compiler_only" and (record.get("family") != variant
                                             or record.get("formal_paper_data") is not True):
            raise ValueError("The forced fixed-skeleton representation was not measured.")
        samples = [float(value) for value in record["samples_ms"]]
        latency = float(record["median_ms"])
        if (len(samples) != 5 or any(not math.isfinite(value) or value <= 0 for value in samples)
                or not math.isfinite(latency) or latency <= 0 or latency != median(samples)):
            raise ValueError("Invalid measured samples or median; no value is substituted.")
        result = {"query": query, "variant": variant, "median_ms": latency,
                  "samples_ms": samples, "source_id": identity,
                  "dataset_metadata_id": record["dataset_metadata_id"]}
    common.verify_contents(source_id=identity)
    return result
