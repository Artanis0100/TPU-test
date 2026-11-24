#!/usr/bin/env python3
"""Explicit offline SF1 input preparation using the existing pinned generator.

The benchmark never invokes this helper automatically. This command creates
new input tables and their data manifest, without recreating SF100 or writing
benchmark reports. An existing destination is always refused.
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

COMMIT = "ae1e254aa4d603d8ef1f44078e5abed011634b23"
SCRIPT = "scripts/prepare_frozen_ssb_data.sh"
TABLES = ("lineorder", "date", "customer", "supplier", "part")
_BASH = r'''set -euo pipefail
source "$1"
ensure_generator() {
  if [[ ! -x "$DBGEN_DIR/build/dbgen" || ! -f "$DBGEN_DIR/dists.dss" ]]; then
    echo "The existing dbgen executable and dists.dss are required; no build or download is performed." >&2
    return 1
  fi
  GENERATOR_READY=1
}
main
'''


def _script_identity(artifact):
    script = artifact / SCRIPT
    if script.is_symlink() or not script.resolve().is_relative_to(artifact):
        raise ValueError("The original preparation script must be inside the artifact.")
    inventory = json.loads((artifact / "content-manifest.json").read_text(encoding="utf-8"))
    expected = inventory["files"][SCRIPT]
    if hashlib.sha256(script.read_bytes()).hexdigest() != expected:
        raise ValueError("The original preparation script differs from its content manifest.")
    return script, expected


def _environment(work, dbgen, destination):
    environment = dict(os.environ)
    for name in ("DSS_CONFIG", "DSS_DIST", "DSS_ADHOC"):
        environment.pop(name, None)
    environment.update(
        SCALE_FACTORS="1", PHYSICAL_ORDER="date_pk", SOURCE_PREORDERED_LAYOUT="1",
        DATA_ROOT=str(destination), DATASET_ID=f"ssb-dbgen-{COMMIT[:12]}-date-pk",
        DBGEN_DIR=str(dbgen), DBGEN_COMMIT=COMMIT, DSS_PATH=".",
        GCS_ROOT="", ALLOW_DBGEN_FALLBACK="1", PUBLISH_GENERATED_DATA="0",
        PREORDER_SOURCE_ROOT="", RAW_LINEORDER_STAGING_ROOT="",
        LINEORDER_SORT_PARTITIONS="1", SORT_BUFFER_SIZE="4G", PYTHONDONTWRITEBYTECODE="1",
    )
    return environment


def prepare(artifact, work, dbgen, destination):
    """Create only a fresh data root, retaining author sorting/verification."""
    raw_destination = Path(destination).expanduser()
    if os.path.lexists(raw_destination):
        raise FileExistsError(f"SF1 destination already exists; use a new --data-root: {raw_destination}")
    artifact, work, dbgen = (Path(value).expanduser().resolve() for value in (artifact, work, dbgen))
    destination = raw_destination.resolve()
    if destination == artifact or artifact in destination.parents:
        raise ValueError("SF1 inputs must be outside the original artifact.")
    if os.path.lexists(destination):
        raise FileExistsError(f"SF1 destination already exists: {destination}")
    script, digest = _script_identity(artifact)
    binary = dbgen / "build/dbgen"
    if not binary.is_file() or not os.access(binary, os.X_OK) or not (dbgen / "dists.dss").is_file():
        raise ValueError(f"Existing offline generator is incomplete: {dbgen}; set --dbgen-dir.")
    # Reserve a fresh root ourselves. The unchanged author script's owned
    # staging/cleanup protocol is then confined to this new absolute prefix.
    destination.mkdir(parents=True, exist_ok=False)
    result = subprocess.run(
        ["bash", "-c", _BASH, "sectile-figure7-sf1", str(script)],
        cwd=destination, env=_environment(work, dbgen, destination),
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    if result.returncode:
        tail = (result.stdout + "\n" + result.stderr)[-32768:]
        raise RuntimeError(f"Offline SF1 preparation failed (status {result.returncode}).\n{tail}")
    if hashlib.sha256(script.read_bytes()).hexdigest() != digest:
        raise ValueError("The original preparation script changed during generation.")
    output = destination / "sf1"
    required = [output / "manifest.txt", *(output / f"{name}.tbl" for name in TABLES)]
    if any(not path.is_file() for path in required):
        raise ValueError("The author preparation command did not complete all five SF1 tables.")
    fields = dict(line.strip().split("=", 1)
                  for line in (output / "manifest.txt").read_text(encoding="utf-8").splitlines()
                  if "=" in line and not line.lstrip().startswith("#"))
    expected = {"format_version": "3", "scale_factor": "1", "physical_order": "date_pk",
                "source_preordered_layout": "1", "dbgen_commit": COMMIT}
    if any(fields.get(key) != value for key, value in expected.items()):
        raise ValueError("Prepared SF1 manifest differs from the pinned date_pk input contract.")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path,
                        default=Path.home() / "test/sectile-artifact-EF6C")
    parser.add_argument("--work", type=Path, default=Path.home() / "test/sectile-work")
    parser.add_argument("--dbgen-dir", type=Path,
                        help="Existing generator root containing build/dbgen and dists.dss.")
    parser.add_argument("--data-root", type=Path,
                        help="New data parent directory (default: WORK/figure7-data).")
    args = parser.parse_args()
    work = args.work.expanduser().resolve()
    try:
        output = prepare(args.artifact_root, work, args.dbgen_dir or work / "ssb-dbgen",
                         args.data_root or work / "figure7-data")
    except (Exception, KeyboardInterrupt) as error:
        raise SystemExit(str(error) or "SF1 preparation interrupted.") from error
    print(f"SF1 data ready: {output}", flush=True)


if __name__ == "__main__":
    main()
