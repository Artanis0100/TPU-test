"""Prepare real SF100 inputs using the bundled, pinned author generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

sys.dont_write_bytecode = True

from benchmark_common import PROJECT_ROOT, import_artifact

HERE = Path(__file__).resolve().parent
GENERATOR = HERE / "generator"
COMMIT = "ae1e254aa4d603d8ef1f44078e5abed011634b23"


def checked_path(root, relative):
    """Reject path escapes and links before reading or copying source files."""
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"Unsafe generator source path: {relative!r}")
    parts = PurePosixPath(relative)
    if (not parts.parts or parts.is_absolute()
            or str(parts) != relative or ".." in parts.parts
            or "\\" in relative or ":" in relative):
        raise ValueError(f"Unsafe generator source path: {relative!r}")
    path = root
    for part in ("", *parts.parts):
        if part:
            if path.exists() and not path.is_dir():
                raise ValueError(f"Generator parent is not a directory: {path}")
            path = path / part
        if path.is_symlink():
            raise ValueError(f"Generator source links are not allowed: {path}")
    return path


def check_file(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        raise ValueError(f"Generator source size/type differs: {path}")
    payload = path.read_bytes()
    git_blob = b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
    if (hashlib.sha256(payload).hexdigest() != entry["sha256"]
            or hashlib.sha1(git_blob).hexdigest() != entry["git_blob_sha1"]):
        raise ValueError(f"Generator source checksum differs: {path}")


def check_links(root):
    if root.is_symlink():
        raise ValueError(f"Generator source links are not allowed: {root}")
    if root.exists() and not root.is_dir():
        raise ValueError(f"Generator source directory is not a directory: {root}")
    if root.exists():
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Generator source links are not allowed: {path}")


def prepare_generator(work):
    """Verify all 33 upstream files, then copy without overwriting changes."""
    index_path = GENERATOR / "ssb-dbgen-source.json"
    if index_path.is_symlink():
        raise ValueError("The generator source index must not be a link.")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    repository = str(index.get("repository", "")).rstrip("/").removesuffix(".git")
    if (repository not in {"eyalroz/ssb-dbgen", "https://github.com/eyalroz/ssb-dbgen"}
            or index.get("commit") != COMMIT):
        raise ValueError("Bundled generator provenance differs from the author's pin.")
    files = index.get("files")
    if not isinstance(files, dict) or len(files) != 33:
        raise ValueError("The bundled generator must contain all 33 upstream files.")
    source, destination = GENERATOR / "ssb-dbgen", work / "ssb-dbgen"
    if (source.resolve().is_relative_to(destination.resolve())
            or destination.resolve().is_relative_to(source.resolve())):
        raise ValueError("The generator work copy must be outside the bundled source.")
    check_links(source)
    check_links(destination)
    for relative, entry in files.items():
        if (not isinstance(entry, dict) or type(entry.get("size")) is not int
                or entry["size"] < 0
                or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", "")))
                or not re.fullmatch(r"[0-9a-f]{40}", str(entry.get("git_blob_sha1", "")))):
            raise ValueError(f"Invalid generator source index entry: {relative!r}")
        check_file(checked_path(source, relative), entry)
        target = checked_path(destination, relative)
        if target.exists():
            check_file(target, entry)  # Check every existing file before any writes.
    observed = {p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()}
    if observed != set(files):
        raise ValueError("The bundled generator file inventory differs from its index.")
    for relative, entry in files.items():
        target = checked_path(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            try:
                with target.open("xb") as output:
                    output.write((source / relative).read_bytes())
            except FileExistsError:
                pass  # A concurrent copy must still pass the checks below.
        check_file(checked_path(destination, relative), entry)
    return destination


def prepare(common, work, dbgen):
    """Same generation, integer preload and array hashes as the author entry."""
    identity = common.configure(100, "date_pk")
    jax = common.require_tpu()
    environment = dict(os.environ, SCALE_FACTORS="100", PHYSICAL_ORDER="date_pk",
        DATA_ROOT=str(work / "data/date_pk"), DBGEN_DIR=str(dbgen), DBGEN_COMMIT=COMMIT,
        GCS_ROOT="", ALLOW_DBGEN_FALLBACK="1", PUBLISH_GENERATED_DATA="0",
        SOURCE_PREORDERED_LAYOUT="1", DATASET_ID=f"ssb-dbgen-{COMMIT[:12]}-date-pk",
        PREORDER_SOURCE_ROOT="", PYTHONDONTWRITEBYTECODE="1")
    subprocess.run(["bash", str(GENERATOR / "prepare_local_ssb.sh")],
                   env=environment, check=True)
    from tpu_db.benchmark.typed_cache import load_or_preload_integer_inputs
    cache = work / "cache/date_pk/sf100-v1"
    _, manifest, _ = load_or_preload_integer_inputs(cache,
        source_directory=work / "data/date_pk/sf100", scale_factor=100,
        source_id=identity, capacity_bytes=min(24 << 30,
            int(jax.devices()[0].memory_stats()["bytes_limit"] * .8)),
        device_put=jax.device_put, fact_order="date_pk")
    hashes = {entry["file"]: common.sha256(cache / entry["file"])
              for entry in manifest["arrays"]}
    (cache / "array-hashes.json").write_text(json.dumps(hashes, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps(dict(sf=100, layout="date_pk", arrays=len(hashes), cache=str(cache))),
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=PROJECT_ROOT / "sectile-work")
    parser.add_argument("--artifact-root", type=Path,
                        default=PROJECT_ROOT / "sectile-artifact-EF6C")
    parser.add_argument("--with-references", action="store_true",
                        help="Also generate exact references for all 13 queries.")
    parser.add_argument("--prepare-cell", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.work = args.work.expanduser().resolve()
    args.artifact_root = args.artifact_root.expanduser().resolve()
    try:
        common = import_artifact(args.artifact_root)
        common.outside_repository(args.work)
        common.verify_contents()
        for local, original in (
            ("prepare_frozen_ssb_data.sh", "scripts/prepare_frozen_ssb_data.sh"),
            ("prepare_data.original.py", "reproduce/prepare_data.py"),
        ):
            copied = checked_path(GENERATOR, local)
            if common.sha256(copied) != common.sha256(args.artifact_root / original):
                raise ValueError(f"The bundled author source copy differs: {local}")
        dbgen = prepare_generator(args.work)
        if args.prepare_cell:
            prepare(common, args.work, dbgen)
            return
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=os.pathsep.join((str(args.artifact_root / "src"),
                                       str(args.artifact_root))))
        # Release the preparation process's TPU allocation before reference generation.
        subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
            "--prepare-cell", "--work", str(args.work), "--artifact-root",
            str(args.artifact_root)], env=environment, check=True)
        if args.with_references:
            subprocess.run([sys.executable, "-B", "-m", "reproduce.references",
                "--work", str(args.work), "--scales", "100", "--layouts", "date_pk"],
                env=environment, check=True)
    except (ValueError, OSError, ImportError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
