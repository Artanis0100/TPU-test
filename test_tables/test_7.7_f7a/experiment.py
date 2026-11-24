"""Figure 7a: execute the author's wide-aggregation microbenchmark unchanged."""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import csv
import importlib.util
import io
import json
import math
from pathlib import Path
from statistics import median
from unittest.mock import patch

BACKENDS = ("jax_xla", "sparsecore", "mxu")
WIDTHS = (1, 5, 10, 15, 20, 25, 26, 27, 28, 30, 32, 33, 34, 35, 36, 40, 48)


def _load(path):
    name = "_sectile_figure7a_aggregation"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load author microbenchmark: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def measure(common, data_dir, widths, identity):
    """Prepare SF1 once, then retain every original backend/timer/validation."""
    from microbenchmark_artifacts import output_artifacts

    widths = tuple(widths)
    if not widths or any(width not in WIDTHS for width in widths) or tuple(sorted(set(widths))) != widths:
        raise ValueError("Use unique increasing Figure 7a measure counts.")
    path = common.ROOT / "ablation_studies/mxu/aggregation.py"
    module = _load(path)
    root = Path(__file__).resolve().parent / ".memory-artifacts-not-created"
    arguments = [str(path), "--data-dir", str(data_dir), "--sf", "1",
                 "--runs", "30", "--warmups", "3", "--measure-counts",
                 ",".join(str(width) for width in widths), "--output-dir", str(root)]
    metadata = lambda repository: dict(source="artifact_content_manifest",
                                       artifact_identity=common.verify_contents())
    with output_artifacts(module, root) as store:
        with patch.object(sys, "argv", arguments), patch.object(module, "git_metadata", metadata):
            if module.main() != 0:
                raise RuntimeError("The aggregation microbenchmark did not finish successfully.")
        raw_rows = list(csv.DictReader(io.StringIO(store.read_text(root / "results.csv"))))
        manifest = store.read_json(root / "manifest.json")
    if (manifest["benchmark_protocol"] != "outer-jit-v2"
            or manifest["scale_factor"] != 1 or manifest["runs"] != 30 or manifest["warmups"] != 3
            or tuple(manifest["measure_counts"]) != widths
            or manifest["git"]["artifact_identity"] != identity
            or manifest["survivor_capacity"] != 262_144
            or manifest["output_slots"] != 1024):
        raise ValueError("Aggregation manifest differs from the original SF1 protocol.")
    grouped = {width: {"width": width} for width in widths}
    seen = set()
    for row in raw_rows:
        width, backend = int(row["measure_count"]), row["backend"]
        key = width, backend
        if width not in grouped or backend not in BACKENDS or key in seen:
            raise ValueError("Unexpected or duplicate aggregation measurement identity.")
        if row["measurement_scope"] != "aggregation_only" or row["correctness"] != "passed":
            raise ValueError("Aggregation output did not pass the original correctness check.")
        samples = [float(value) for value in json.loads(row["samples_ms"])]
        latency = float(row["median_ms"])
        if (len(samples) != 30 or any(not math.isfinite(value) or value <= 0 for value in samples)
                or not math.isfinite(latency) or latency <= 0
                or not math.isclose(latency, median(samples), rel_tol=0, abs_tol=1.1e-6)):
            raise ValueError("Invalid measured aggregation samples or median.")
        seen.add(key)
        grouped[width][backend] = latency
    if seen != {(width, backend) for width in widths for backend in BACKENDS}:
        raise ValueError("A requested aggregation width/backend result is missing.")
    common.verify_contents(source_id=identity)
    return {"rows": [grouped[width] for width in widths], "metadata": manifest,
            "raw_rows": raw_rows, "source_id": identity}
