"""Keep author evidence and its readbacks in RAM beneath one reserved path.

Only known module-local Path aliases and NumPy/JSON output entrypoints are
adapted. Source, cache and reference reads use the original filesystem paths.
"""

from contextlib import ExitStack, contextmanager
import fnmatch
import importlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
from unittest.mock import patch


_WRITERS = (
    "tpu_db.benchmark.production", "tpu_db.benchmark.production_runtime",
    "tpu_db.result_boundary.binding", "tpu_db.result_boundary.device",
    "tpu_db.result_boundary.measurement", "reproduce.execution_guard.integration",
    "reproduce.execution_guard.measurement", "reproduce.execution_guard.evaluation",
    "tpu_db.benchmark.evaluation_inventory", "tpu_db.benchmark.evaluation_preflight",
    "tpu_db.benchmark.q23_q33_fixed_skeleton",
)


def _snapshot_fd(payload, descriptor=None):
    """Expose RAM bytes to the author's fsync calls without a disk pathname."""
    created = descriptor is None
    if created:
        if not hasattr(os, "memfd_create"):
            raise OSError("RAM report fileno requires Linux anonymous memory descriptors.")
        descriptor = os.memfd_create("sectile-layout-report", os.MFD_CLOEXEC)
    try:
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("Could not snapshot the RAM report descriptor.")
            remaining = remaining[written:]
        return descriptor
    except BaseException:
        if created:
            os.close(descriptor)
        raise


def _matches(parts, pattern):
    if not pattern:
        return not parts
    if pattern[0] == "**":
        return _matches(parts, pattern[1:]) or bool(parts) and _matches(parts[1:], pattern)
    return bool(parts) and fnmatch.fnmatchcase(parts[0], pattern[0]) and _matches(parts[1:], pattern[1:])


@contextmanager
def memory_artifacts(root, extra_path_modules=()):
    """Yield a store with root/read_json/read_bytes/patch_module; never persist it."""
    reserved = Path(os.path.abspath(os.fspath(root)))
    if os.path.lexists(reserved):
        raise FileExistsError(f"RAM output root must not exist: {reserved}")
    files, directories = {}, {reserved}

    def native(value):
        return Path(os.path.abspath(os.fspath(value)))

    def inside(value):
        try:
            path = native(value)
        except (TypeError, ValueError):
            return False
        return path == reserved or reserved in path.parents

    def key(value):
        if not inside(value):
            raise ValueError(f"RAM output escaped its reserved root: {value}")
        return native(value)

    def put(value, payload):
        path = key(value)
        if path.parent not in directories:
            raise FileNotFoundError(f"RAM output parent is absent: {path.parent}")
        if path in directories:
            raise IsADirectoryError(str(path))
        files[path] = bytes(payload)

    class MemoryBytes(io.BytesIO):
        def __init__(self, path, payload, writable):
            super().__init__(payload)
            self.path, self.commit = path, writable
            self.descriptor = None

        def fileno(self):
            if self.closed:
                raise ValueError("fileno of closed RAM report")
            self.descriptor = _snapshot_fd(self.getvalue(), self.descriptor)
            os.lseek(self.descriptor, self.tell(), os.SEEK_SET)
            return self.descriptor

        def close(self):
            if self.closed:
                return
            try:
                if self.commit:
                    put(self.path, self.getvalue())
            finally:
                try:
                    if self.descriptor is not None:
                        os.close(self.descriptor)
                        self.descriptor = None
                finally:
                    super().close()

    class MemoryPath(type(Path())):
        def mkdir(self, mode=0o777, parents=False, exist_ok=False):
            path = key(self)
            if path in files or path in directories and not exist_ok and path != reserved:
                raise FileExistsError(str(path))
            if path.parent not in directories and path != reserved and not parents:
                raise FileNotFoundError(str(path.parent))
            directories.add(path)
            if parents:
                directories.update(parent for parent in path.parents if inside(parent))

        def exists(self, *, follow_symlinks=True):
            path = key(self)
            return path in files or path in directories

        def is_file(self, *, follow_symlinks=True):
            return key(self) in files

        def is_dir(self, *, follow_symlinks=True):
            return key(self) in directories

        def is_symlink(self):
            key(self)
            return False

        def resolve(self, strict=False):
            path = key(self)
            if strict and not self.exists():
                raise FileNotFoundError(str(path))
            return MemoryPath(path)

        def stat(self, *, follow_symlinks=True):
            path = key(self)
            if not self.exists():
                raise FileNotFoundError(str(path))
            mode = stat.S_IFDIR if path in directories else stat.S_IFREG
            return os.stat_result((mode | 0o600, 0, 0, 1, 0, 0, len(files.get(path, b"")), 0, 0, 0))

        def lstat(self):
            return self.stat()

        def read_bytes(self):
            path = key(self)
            if path not in files:
                raise FileNotFoundError(str(path))
            return files[path]

        def read_text(self, encoding=None, errors=None):
            return self.read_bytes().decode(encoding or "utf-8", errors or "strict")

        def write_bytes(self, data):
            payload = bytes(memoryview(data))
            put(self, payload)
            return len(payload)

        def write_text(self, data, encoding=None, errors=None, newline=None):
            if not isinstance(data, str):
                raise TypeError("write_text requires str")
            with self.open("w", encoding=encoding, errors=errors, newline=newline) as output:
                return output.write(data)

        def open(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
            path = key(self)
            if mode not in {"r", "rt", "rb", "w", "wt", "wb", "x", "xt", "xb", "a", "at", "ab"}:
                raise ValueError(f"Unsupported RAM output open mode: {mode}")
            if path in directories:
                raise IsADirectoryError(str(path))
            if "x" in mode and path in files:
                raise FileExistsError(str(path))
            if "r" in mode and path not in files:
                raise FileNotFoundError(str(path))
            writable = "r" not in mode
            if writable and path.parent not in directories:
                raise FileNotFoundError(str(path.parent))
            payload = files.get(path, b"") if "r" in mode or "a" in mode else b""
            output = MemoryBytes(path, payload, writable)
            if "a" in mode:
                output.seek(0, io.SEEK_END)
            if "b" in mode:
                return output
            return io.TextIOWrapper(output, encoding=encoding or "utf-8", errors=errors or "strict", newline=newline)

        def iterdir(self):
            base = key(self)
            if base not in directories:
                raise NotADirectoryError(str(base))
            return iter(MemoryPath(path) for path in sorted(set(files) | directories)
                        if path != base and path.parent == base)

        def glob(self, pattern, **kwargs):
            base = key(self)
            if PurePosixPath(pattern).is_absolute() or ".." in PurePosixPath(pattern).parts:
                raise ValueError("RAM glob must remain beneath its directory")
            return iter(MemoryPath(path) for path in sorted(set(files) | directories)
                        if path != base and base in path.parents and
                        _matches(PurePosixPath(path.relative_to(base).as_posix()).parts,
                                 PurePosixPath(pattern).parts))

        def rglob(self, pattern, **kwargs):
            return self.glob("**/" + str(pattern), **kwargs)

    def output_path(*args, **kwargs):
        path = Path(*args, **kwargs)
        if inside(path):
            return MemoryPath(native(path))
        if any(isinstance(value, MemoryPath) for value in args):
            raise ValueError(f"RAM output escaped its root: {path}")
        return path

    with ExitStack() as stack:
        class Store:
            root = MemoryPath(reserved)

            def read_bytes(self, path):
                return MemoryPath(key(path)).read_bytes()

            def read_json(self, path):
                return json.loads(self.read_bytes(path))

            def patch_module(self, module):
                if isinstance(module, str):
                    module = importlib.import_module(module)
                if hasattr(module, "Path"):
                    stack.enter_context(patch.object(module, "Path", output_path))
                return module

        store = Store()
        for module in (*_WRITERS, *extra_path_modules):
            store.patch_module(module)
        typed = importlib.import_module("tpu_db.benchmark.typed")
        guarded = importlib.import_module("reproduce.execution_guard.measurement")
        numpy = importlib.import_module("numpy")

        def atomic_wrapper(original):
            def write(path, value):
                if not inside(path):
                    return original(path, value)
                payload = json.dumps(value, indent=2, allow_nan=False).encode()
                put(path, payload)
            return write

        def numpy_wrapper(original):
            def write(file, *args, **kwargs):
                if not inside(file):
                    return original(file, *args, **kwargs)
                destination = output_path(file)
                if not str(destination).endswith(".npz"):
                    destination = output_path(str(destination) + ".npz")
                buffer = io.BytesIO()
                original(buffer, *args, **kwargs)
                put(destination, buffer.getvalue())
            return write

        stack.enter_context(patch.object(typed, "atomic_json", atomic_wrapper(typed.atomic_json)))
        stack.enter_context(patch.object(guarded, "atomic_json", atomic_wrapper(guarded.atomic_json)))
        for name in ("savez", "savez_compressed"):
            stack.enter_context(patch.object(numpy, name, numpy_wrapper(getattr(numpy, name))))
        yield store
