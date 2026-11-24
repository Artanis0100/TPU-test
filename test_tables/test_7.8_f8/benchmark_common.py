"""Figure 8, Sectile only: recover source order in RAM from existing date_pk SF100."""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from statistics import fmean

from process_runner import run_case

QUERIES = ("Q1.1", "Q1.2", "Q1.3", "Q2.1", "Q2.2", "Q2.3", "Q3.1", "Q3.2",
           "Q3.3", "Q3.4", "Q4.1", "Q4.2", "Q4.3")
LAYOUTS = ("original", "date_pk")
TABLES = ("lineorder", "date", "customer", "supplier", "part")


def check_inputs(artifact, work, queries):
    """Metadata-only preflight; the existing date_pk layout is the only disk input."""
    if not (artifact / "content-manifest.json").is_file():
        raise ValueError("Original artifact is missing; set --artifact-root.")
    source, cache = work / "data/date_pk/sf100", work / "cache/date_pk/sf100-v1"
    paths = [source / "manifest.txt", cache / "manifest.json"]
    paths.extend(source / f"{name}.tbl" for name in TABLES)
    for query in dict.fromkeys(("Q1.1", *queries)):
        directory = work / "references/date_pk/sf100" / query.replace(".", "")
        paths.extend(directory / name for name in ("result.json", "aggregates.npz", "sql-rows.json"))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError("Existing date_pk SF100 inputs are incomplete:\n" + "\n".join(missing[:10]))
    from source_order import COMMIT
    fields = dict(line.strip().split("=", 1) for line in (source / "manifest.txt").read_text().splitlines()
                  if "=" in line and not line.lstrip().startswith("#"))
    expected = dict(format_version="3", scale_factor="100", physical_order="date_pk",
                    source_preordered_layout="1", dbgen_commit=COMMIT)
    if any(fields.get(key) != value for key, value in expected.items()):
        raise ValueError("Expected the pinned date_pk SF100 raw-table manifest.")
    manifest = json.loads((cache / "manifest.json").read_text())
    if (manifest.get("kind") != "typed_integer_input_cache" or manifest.get("schema_version") != 1
            or manifest.get("status") != "READY" or manifest.get("scale_factor") != 100
            or manifest.get("fact_order") != "date_pk" or len(manifest.get("arrays", ())) != 26
            or Path(manifest["source_directory"]).resolve() != source.resolve()):
        raise ValueError("Existing date_pk integer-cache metadata differs.")
    if any(not (cache / entry["file"]).is_file() for entry in manifest["arrays"]):
        raise ValueError("One or more of the 26 existing cached arrays are missing.")
    for query in dict.fromkeys(("Q1.1", *queries)):
        directory = work / "references/date_pk/sf100" / query.replace(".", "")
        record = json.loads((directory / "result.json").read_text())
        if (record.get("status") != "PASSED" or record.get("sf") != 100 or record.get("query") != query
                or record.get("dataset_metadata_id") != manifest["dataset_metadata_id"]):
            raise ValueError(f"The existing exact reference metadata differs: {query}.")
    return manifest


def geometric_mean(values):
    values = tuple(values)
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Geometric mean requires finite positive completed timings.")
    return math.exp(fmean(math.log(value) for value in values))


def run_table(context, args, order, bundles, checked):
    from resident_inputs import derived_dataset_id
    from source_order import assert_unchanged

    medians = {layout: [] for layout in LAYOUTS}
    metadata = order.metadata
    print("Query     Source order (ms)     Date-sorted (ms)", flush=True)
    for query in args.queries:
        rows = []
        for layout in LAYOUTS:
            descriptor = order.descriptor() if layout == "original" else {"metadata": metadata}
            row = run_case(context, args, query, layout, descriptor, bundles[query])
            expected = dict(source_id=checked["source_id"],
                dataset_metadata_id=derived_dataset_id(metadata["base_dataset_id"], layout, metadata["permutation_sha256"]),
                permutation_sha256=metadata["permutation_sha256"], base_input_sha256=metadata["input_content_sha256"])
            if any(row.get(key) != value for key, value in expected.items()):
                raise ValueError("The measured source, catalog or row-permutation identity changed.")
            rows.append(row)
            medians[layout].append(row["median_ms"])
        if rows[0]["output_digest"] != rows[1]["output_digest"]:
            raise ValueError(f"Exact ordered SQL outputs differ between layouts: {query}.")
        print(f"{query:<7} {rows[0]['median_ms']:>18.3f} {rows[1]['median_ms']:>20.3f}", flush=True)
    assert_unchanged(args.work, metadata)
    original, date_sorted = (geometric_mean(medians[layout]) for layout in LAYOUTS)
    label = "Geomean" if args.queries == QUERIES else f"Subset({len(args.queries)})"
    print(f"{label:<12} {original:>13.3f} {date_sorted:>20.3f}", flush=True)
    print(f"Sectile speedup (source/date-sorted): {original / date_sorted:.3f}x", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path.home() / "test/sectile-artifact-EF6C")
    parser.add_argument("--work", type=Path, default=Path.home() / "test/sectile-work")
    parser.add_argument("--queries", nargs="+", choices=QUERIES, default=QUERIES)
    parser.add_argument("--timeout", type=float, default=3600, help="Maximum seconds per query/layout.")
    parser.add_argument("--check-only", action="store_true", help="Check existing paths, RAM and TPU; no raw-table scan or query.")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive.")
    args.artifact_root = args.artifact_root.expanduser().resolve()
    args.work = args.work.expanduser().resolve()
    args.queries = tuple(dict.fromkeys(args.queries))
    try:
        # The controller authenticates CPU data only. Fresh children explicitly
        # select TPU before importing JAX; no parent TPU client holds HBM.
        os.environ.update(JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1")
        manifest = check_inputs(args.artifact_root, args.work, args.queries)
        sys.path[:0] = [str(args.artifact_root / "src"), str(args.artifact_root)]
        from reproduce import common
        if common.ROOT.resolve() != args.artifact_root:
            raise ValueError("A different author artifact was imported; check PYTHONPATH.")
        common.outside_repository(args.work)
        identity = common.verify_contents()
        from resident_inputs import load_host_cache
        if load_host_cache(args.work)[0] != manifest:
            raise ValueError("Existing cache metadata changed after preflight.")
        from source_order import check_memory, source_order
        resources = check_memory(manifest)
        context = mp.get_context("spawn")
        checked = run_case(context, args)
        if checked["source_id"] != identity:
            raise ValueError("Controller and TPU worker imported different artifacts.")
        hbm_limit = checked["hbm_bytes_limit"]
        capacity = min(24 << 30, int(hbm_limit * .8)) if hbm_limit else 24 << 30
        if resources["cached_storage_bytes"] > capacity:
            raise ValueError("Cached resident inputs exceed the author's TPU HBM admission ceiling.")
        if args.check_only:
            print(f"Preflight passed: {checked['device']}; existing date_pk SF100; "
                  f"RAM {resources['available_host_bytes'] / (1 << 30):.1f} GiB available, "
                  f"{resources['required_host_bytes'] / (1 << 30):.1f} GiB required.")
            return
        from reference_bridge import authenticate_references
        with source_order(args.work, manifest) as order:
            bundles = authenticate_references(common, args.work, args.queries, order.metadata)
            run_table(context, args, order, bundles, checked)
    except (Exception, KeyboardInterrupt) as error:
        raise SystemExit(str(error) or "Benchmark interrupted.") from error
