"""Figure 7b: reuse the author's full online join-plus-projection timings."""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

from contextlib import contextmanager
import csv
import importlib.util
import io
import json
import math
from pathlib import Path
from statistics import median
from unittest.mock import patch


BACKENDS = ("jax_xla", "sparsecore", "mxu")


@contextmanager
def _author_module(path):
    name = "_sectile_figure7b_projection"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load the author's projection benchmark: {path}")
    module = importlib.util.module_from_spec(spec)
    absent = object()
    previous = sys.modules.get(name, absent)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        if previous is absent:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def _validated_rows(raw_rows, widths):
    grouped = {width: {"width": width} for width in widths}
    if len(raw_rows) != 3 * len(widths):
        raise ValueError("The author sweep did not return all three backends for every width.")
    for row in raw_rows:
        width, backend = int(row["projection_width"]), row["backend"]
        if (width not in grouped or backend not in BACKENDS or backend in grouped[width]
                or row["measurement_scope"] != "online_supplier_join_plus_projection"
                or row["correctness"] != "passed"):
            raise ValueError("Unexpected, repeated or unvalidated join-projection measurement.")
        samples = [float(value) for value in json.loads(row["samples_ms"])]
        elapsed = float(row["median_ms"])
        if (len(samples) != 30 or any(not math.isfinite(value) or value <= 0 for value in samples)
                or not math.isfinite(elapsed) or elapsed <= 0
                or not math.isclose(elapsed, median(samples), rel_tol=1e-9, abs_tol=1.1e-6)):
            raise ValueError("Invalid 30-sample median in the author projection results.")
        grouped[width][backend] = elapsed
    if any(set(row) != {"width", *BACKENDS} for row in grouped.values()):
        raise ValueError("The projection sweep is missing a backend.")
    return [grouped[width] for width in widths]


def measure(common, data_dir, widths, identity):
    """Run the original main once; only its file persistence is redirected."""
    from microbenchmark_artifacts import output_artifacts

    widths = tuple(widths)
    if (not widths or tuple(sorted(set(widths))) != widths
            or any(type(width) is not int or not 1 <= width <= 32 for width in widths)):
        raise ValueError("Projection widths must be unique increasing integers in 1..32.")
    data_dir = Path(data_dir).expanduser().resolve()
    path = common.ROOT / "ablation_studies/mxu/projection.py"
    root = Path(__file__).resolve().parent / ".memory-artifacts-not-created"
    arguments = [str(path), "--data-dir", str(data_dir), "--sf", "1",
                 "--runs", "30", "--warmups", "3", "--output-dir", str(root),
                 "--projection-widths", ",".join(map(str, widths))]
    with _author_module(path) as module:
        # Match reproduce.mxu: artifact content identity replaces Git/deployment
        # metadata only. Kernels, routing, validation and timers stay unchanged.
        with output_artifacts(module, root) as store:
            with patch.object(module, "git_metadata", lambda repository: {
                "source": "artifact_content_manifest", "artifact_identity": common.verify_contents(),
            }), patch.object(sys, "argv", arguments):
                status = module.main()
            if status != 0:
                raise RuntimeError(f"The author's projection benchmark returned status {status}.")
            raw_rows = list(csv.DictReader(io.StringIO(store.read_text(root / "results.csv"))))
            metadata = store.read_json(root / "manifest.json")
    expected = {"benchmark_protocol": "independent-reference-outer-jit-v2", "scale_factor": 1,
                "projection_widths": list(widths), "warmups": 3, "runs": 30, "tile_rows": 1024,
                "backends": list(BACKENDS)}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("The author manifest differs from the Figure 7b protocol.")
    if (Path(metadata["data_dir"]).resolve() != data_dir
            or metadata["git"] != {"source": "artifact_content_manifest", "artifact_identity": identity}):
        raise ValueError("The projection result uses different data or artifact identity.")
    rows = _validated_rows(raw_rows, widths)
    common.verify_contents(source_id=identity)
    return {"rows": rows, "metadata": metadata, "source_id": identity, "raw_rows": raw_rows}
