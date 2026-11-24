"""Recover dbgen order once, keeping only a sealed row index in anonymous RAM.

At the pinned dbgen revision, driver.gen_tbl increases the order index,
build.ez_sparse(index, 0) is strictly increasing, and print.pr_line emits
line numbers 1..N. Thus numeric (LO_ORDERKEY, LO_LINENUMBER) is source order.
No raw table, sorted run, integer cache or reference file is written here.
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

from contextlib import contextmanager
import hashlib
import io
import json
import mmap
import os
from pathlib import Path

COMMIT = "ae1e254aa4d603d8ef1f44078e5abed011634b23"
CHUNK_ROWS = 1 << 20


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stamp(path):
    value = Path(path).stat()
    return {"size": value.st_size, "mtime_ns": value.st_mtime_ns}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_linux():
    if not sys.platform.startswith("linux") or not hasattr(os, "memfd_create"):
        raise RuntimeError("RAM reconstruction requires the Linux TPU VM and anonymous memory descriptors.")


def _seal_commands():
    """Use Linux UAPI values if Python was built without the seal symbols.

    https://github.com/torvalds/linux/blob/master/include/uapi/linux/fcntl.h
    https://github.com/torvalds/linux/blob/master/include/uapi/asm-generic/fcntl.h
    This only supplies missing names; kernel support is tested separately.
    """
    import fcntl
    import platform

    defaults = {"F_ADD_SEALS": 1033, "F_GET_SEALS": 1034, "F_SEAL_SEAL": 1,
                "F_SEAL_SHRINK": 2, "F_SEAL_GROW": 4, "F_SEAL_WRITE": 8}
    if (any(not hasattr(fcntl, name) for name in defaults)
            and (not sys.platform.startswith("linux")
                 or platform.machine().lower() not in ("x86_64", "amd64", "aarch64", "arm64"))):
        raise RuntimeError("Missing fcntl seal symbols on an unverified architecture; Linux x86_64/aarch64 is required.")
    constants = {name: getattr(fcntl, name, value) for name, value in defaults.items()}
    mask = (constants["F_SEAL_SEAL"] | constants["F_SEAL_SHRINK"]
            | constants["F_SEAL_GROW"] | constants["F_SEAL_WRITE"])
    return constants["F_ADD_SEALS"], constants["F_GET_SEALS"], mask


def check_sealing():
    """Probe the exact anonymous-memory ABI before reading or sorting SF100."""
    require_linux()
    import fcntl

    add_command, get_command, mask = _seal_commands()
    descriptor = None
    try:
        descriptor = os.memfd_create("sectile-seal-selfcheck", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        os.ftruncate(descriptor, 4)
        with mmap.mmap(descriptor, 4, access=mmap.ACCESS_WRITE) as mapping:
            mapping[:] = b"SSB1"
        # F_SEAL_WRITE requires every shared writable mapping to be closed.
        fcntl.fcntl(descriptor, add_command, mask)
        if (fcntl.fcntl(descriptor, get_command) & mask) != mask:
            raise RuntimeError("The kernel did not apply all required memory seals.")
        with mmap.mmap(descriptor, 4, access=mmap.ACCESS_READ) as mapping:
            if mapping[:] != b"SSB1":
                raise RuntimeError("The sealed-memory readback differs.")
    except (OSError, AttributeError) as error:
        raise RuntimeError(f"Linux memfd sealing self-check failed before SF100 preparation: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def check_memory(manifest):
    """Conservative host headroom, including preparation and query bindings."""
    check_sealing()
    rows = {entry["spec"]["logical_row_count"] for entry in manifest["arrays"]
            if entry["spec"]["table"] == "lineorder"}
    if len(rows) != 1:
        raise ValueError("The cached fact columns have different row counts.")
    rows = rows.pop()
    if not 0 < rows <= (1 << 32):
        raise ValueError("Source-order indices must fit the validated uint32 row domain.")
    import numpy as np
    logical_bytes = sum(entry["spec"]["logical_row_count"] * np.dtype(
        entry["spec"]["storage_dtype"]).itemsize for entry in manifest["arrays"])
    storage_bytes = sum(entry["spec"]["storage_row_count"] * np.dtype(
        entry["spec"]["storage_dtype"]).itemsize for entry in manifest["arrays"])
    # Three catalogs' logical bytes cover host operands and binding work;
    # 28 bytes/row cover key/order/index sorting, plus a 16 GiB reserve.
    required = 3 * logical_bytes + 28 * rows + (16 << 30)
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
    available = int(fields["MemAvailable"].split()[0]) * 1024
    if available < required:
        raise ValueError(f"Insufficient host RAM: conservatively require {required / (1 << 30):.1f} GiB "
                         f"available, found {available / (1 << 30):.1f} GiB. Stop other jobs before retrying.")
    return {"rows": rows, "required_host_bytes": required, "available_host_bytes": available,
            "permutation_bytes": rows * 4, "cached_logical_bytes": logical_bytes,
            "cached_storage_bytes": storage_bytes}


class HashReader(io.RawIOBase):
    """Hash all raw bytes while pandas reads only the two primary-key fields."""
    def __init__(self, source):
        self.source, self.digest, self.byte_count = source, hashlib.sha256(), 0

    def readable(self):
        return True

    def readinto(self, buffer):
        count = self.source.readinto(buffer)
        if count:
            self.digest.update(memoryview(buffer)[:count])
            self.byte_count += count
        return count


def read_keys(path, rows, expected_sha, expected_bytes, *, chunk_rows=CHUNK_ROWS):
    import numpy as np
    import pandas as pd

    keys = np.empty(rows, dtype=np.uint64)
    offset = 0
    with Path(path).open("rb", buffering=0) as raw:
        hashing = HashReader(raw)
        with io.BufferedReader(hashing, buffer_size=1 << 20) as source:
            chunks = pd.read_csv(source, sep="|", header=None, usecols=[0, 1],
                                 names=["orderkey", "linenumber"], dtype=np.int64,
                                 chunksize=chunk_rows, memory_map=False, compression=None)
            with chunks:
                for frame in chunks:
                    order = frame["orderkey"].to_numpy(copy=False)
                    line = frame["linenumber"].to_numpy(copy=False)
                    stop = offset + order.size
                    if (stop > rows or np.any(order <= 0) or np.any((order & 24) != 0)
                            or np.any(order > (np.iinfo(np.int64).max >> 3))
                            or np.any(line < 1) or np.any(line > 7)):
                        raise ValueError("Raw primary keys do not match base-data dbgen source order.")
                    keys[offset:stop] = (order.astype(np.uint64) << np.uint64(3)) | line.astype(np.uint64)
                    offset = stop
            # Exhaust any unconsumed buffered bytes, so the hash always covers
            # the complete file rather than just parsed records.
            while source.read(1 << 20):
                pass
        if (offset != rows or hashing.byte_count != expected_bytes
                or hashing.digest.hexdigest() != expected_sha):
            raise ValueError("The raw LINEORDER row count, bytes or SHA differs from its frozen manifest.")
    return keys


def validate_sorted_keys(keys, order, *, expected_orders=None, chunk_rows=CHUNK_ROWS):
    """Check uniqueness and the exact base-generator order/line sequence."""
    import numpy as np

    previous_order, previous_line, previous_key = 0, 0, 0
    for start in range(0, len(order), chunk_rows):
        selected = keys[order[start:start + chunk_rows]]
        if not selected.size:
            continue
        if int(selected[0]) <= previous_key or np.any(selected[1:] <= selected[:-1]):
            raise ValueError("Duplicate or unordered raw primary keys; source order cannot be proved.")
        raw_order, lines = selected >> np.uint64(3), selected & np.uint64(7)
        indices = ((raw_order >> np.uint64(5)) << np.uint64(3)) | (raw_order & np.uint64(7))
        prior_orders = np.empty_like(indices)
        prior_lines = np.empty_like(lines)
        prior_orders[0], prior_lines[0] = previous_order, previous_line
        prior_orders[1:], prior_lines[1:] = indices[:-1], lines[:-1]
        same = indices == prior_orders
        if (np.any(same & (lines != prior_lines + 1))
                or np.any(~same & ((indices != prior_orders + 1) | (lines != 1)))):
            raise ValueError("Primary keys are not the complete unrefreshed dbgen order/line sequence.")
        previous_order, previous_line, previous_key = int(indices[-1]), int(lines[-1]), int(selected[-1])
    if expected_orders is not None and previous_order != expected_orders:
        raise ValueError("The recovered source order has the wrong number of generated orders.")


class OrderHandle:
    def __init__(self, descriptor, metadata):
        self.fd, self.metadata = descriptor, metadata

    def descriptor(self):
        from multiprocessing.reduction import DupFd
        return {"fd": DupFd(self.fd), "metadata": self.metadata}


def assert_unchanged(work, metadata):
    """Cheap checks between cells: immutable base paths must retain their stamps."""
    work = Path(work).resolve()
    source, cache = work / "data/date_pk/sf100", work / "cache/date_pk/sf100-v1"
    if (metadata["source_stamps"] != {name: stamp(source / f"{name}.tbl") for name in metadata["source_stamps"]}
            or metadata["cache_stamps"] != {name: stamp(cache / name) for name in metadata["cache_stamps"]}
            or sha256_file(source / "manifest.txt") != metadata["raw_manifest_sha256"]
            or sha256_file(cache / "manifest.json") != metadata["cache_manifest_sha256"]):
        raise ValueError("The existing SF100 source or cache changed during the experiment.")


@contextmanager
def source_order(work, manifest):
    """Authenticate the base cache once and yield its sealed source-row index."""
    import numpy as np
    import fcntl

    require_linux()
    add_command, get_command, seal_mask = _seal_commands()
    work = Path(work).resolve()
    resource = check_memory(manifest)
    source, cache = work / "data/date_pk/sf100", work / "cache/date_pk/sf100-v1"
    raw_payload, cache_payload = (source / "manifest.txt").read_bytes(), (cache / "manifest.json").read_bytes()
    if json.loads(cache_payload) != manifest:
        raise ValueError("Cache manifest changed after preflight.")
    fields = dict(line.strip().split("=", 1) for line in raw_payload.decode().splitlines()
                  if "=" in line and not line.lstrip().startswith("#"))
    expected = {"format_version": "3", "scale_factor": "100", "physical_order": "date_pk",
                "source_preordered_layout": "1", "dbgen_commit": COMMIT}
    if any(fields.get(key) != value for key, value in expected.items()):
        raise ValueError("Recovering source order requires the pinned, date_pk SF100 manifest.")
    source_stamps = {name: stamp(source / f"{name}.tbl") for name in
                     ("lineorder", "date", "customer", "supplier", "part")}
    cache_stamps = {entry["file"]: stamp(cache / entry["file"]) for entry in manifest["arrays"]}
    for entry in manifest["arrays"]:
        if cache_stamps[entry["file"]] != entry["stamp"]:
            raise ValueError("A cache array differs from its prepared stamp.")
    if source_stamps != manifest["source_files"]:
        raise ValueError("A source table differs from its prepared cache stamp.")
    hashes = {entry["file"]: sha256_file(cache / entry["file"]) for entry in manifest["arrays"]}
    anchor = json.loads((work / "references/date_pk/sf100/Q11/result.json").read_text())
    if hashes != anchor["input_files_sha256"] or fingerprint(hashes) != anchor["input_content_sha256"]:
        raise ValueError("The cached input values differ from the preserved exact-reference inputs.")
    rows = resource["rows"]
    if rows != int(fields["lineorder_rows"]):
        raise ValueError("Cached and raw fact row counts differ.")
    keys = read_keys(source / "lineorder.tbl", rows, fields["lineorder_sha256"], int(fields["lineorder_bytes"]))
    order = np.argsort(keys, kind="stable")
    validate_sorted_keys(keys, order, expected_orders=150_000_000)
    descriptor = os.memfd_create("sectile-source-row-index", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        os.ftruncate(descriptor, rows * 4)
        digest = hashlib.sha256()
        mapping = mmap.mmap(descriptor, rows * 4, access=mmap.ACCESS_WRITE)
        try:
            permutation = np.ndarray((rows,), dtype="<u4", buffer=mapping)
            for start in range(0, rows, CHUNK_ROWS):
                stop = min(rows, start + CHUNK_ROWS)
                permutation[start:stop] = order[start:stop]
                digest.update(permutation[start:stop].tobytes())
            del permutation
        finally:
            mapping.close()
        del order, keys
        fcntl.fcntl(descriptor, add_command, seal_mask)
        if (fcntl.fcntl(descriptor, get_command) & seal_mask) != seal_mask:
            raise RuntimeError("The source-order index did not acquire all required memory seals.")
        if (source_stamps != {name: stamp(source / f"{name}.tbl") for name in source_stamps}
                or cache_stamps != {name: stamp(cache / name) for name in cache_stamps}
                or raw_payload != (source / "manifest.txt").read_bytes()
                or cache_payload != (cache / "manifest.json").read_bytes()):
            raise ValueError("Base data changed during RAM reconstruction.")
        metadata = {"contract": "dbgen_source_order_from_primary_keys_v1", "rows": rows,
                    "permutation_sha256": digest.hexdigest(), "bijection_verified": True,
                    "order_key_sequence_verified": True, "base_dataset_id": manifest["dataset_metadata_id"],
                    "cache_manifest_sha256": hashlib.sha256(cache_payload).hexdigest(),
                    "raw_manifest_sha256": hashlib.sha256(raw_payload).hexdigest(),
                    "raw_lineorder_sha256": fields["lineorder_sha256"],
                    "input_files_sha256": hashes, "input_content_sha256": fingerprint(hashes),
                    "source_stamps": source_stamps, "cache_stamps": cache_stamps,
                    "dbgen_commit": COMMIT, "permutation_dtype": "<u4"}
        yield OrderHandle(descriptor, metadata)
        assert_unchanged(work, metadata)
    finally:
        os.close(descriptor)


@contextmanager
def attach_order(descriptor):
    """Map an immutable inherited index; no path or disk copy is involved."""
    import numpy as np
    import fcntl

    require_linux()
    _, get_command, seal_mask = _seal_commands()
    metadata = descriptor["metadata"]
    fd = descriptor["fd"].detach()
    mapping = None
    try:
        rows = metadata["rows"]
        if (metadata.get("bijection_verified") is not True or metadata.get("order_key_sequence_verified") is not True
                or metadata.get("contract") != "dbgen_source_order_from_primary_keys_v1"
                or metadata.get("permutation_dtype") != "<u4" or os.fstat(fd).st_size != rows * 4
                or (fcntl.fcntl(fd, get_command) & seal_mask) != seal_mask):
            raise ValueError("The inherited source-order index is not the sealed, verified permutation.")
        mapping = mmap.mmap(fd, rows * 4, access=mmap.ACCESS_READ)
        values = np.ndarray((rows,), dtype="<u4", buffer=mapping)
        values.flags.writeable = False
        try:
            yield values, metadata
        finally:
            del values
    finally:
        if mapping is not None:
            mapping.close()
        os.close(fd)
