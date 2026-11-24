"""Discard only the author's artifact writes beneath one reserved output path."""

from contextlib import ExitStack, contextmanager
import importlib
import io
import os
from pathlib import Path
from unittest.mock import patch


# These are the output writers reached by the unchanged production/guard entry.
PATH_MODULES = (
    "tpu_db.benchmark.production",
    "tpu_db.benchmark.production_runtime",
    "tpu_db.result_boundary.binding",
    "tpu_db.result_boundary.device",
    "tpu_db.result_boundary.measurement",
    "reproduce.execution_guard.integration",
    "reproduce.execution_guard.measurement",
)


class NullText(io.TextIOBase):
    def writable(self):
        return True

    def write(self, value):
        if self.closed:
            raise ValueError("write to closed discarded log")
        if not isinstance(value, str):
            raise TypeError("a text log requires str")
        return len(value)


@contextmanager
def discard_author_artifacts(root):
    """Yield a virtual output root; restore all author aliases on exit.

    The root must not exist. Input/source/reference paths remain native paths.
    This adapter changes artifact persistence, not query execution or timing.
    """
    reserved = Path(os.path.abspath(os.fspath(root)))
    if os.path.lexists(reserved):
        raise FileExistsError(f"discarded output root must not exist: {reserved}")

    def inside(value):
        try:
            path = Path(os.path.abspath(os.fspath(value)))
        except (TypeError, ValueError):
            return False
        return path == reserved or reserved in path.parents

    class _OutputPath(type(Path())):
        def _require_output(self):
            if not inside(self):
                raise ValueError(f"virtual output path escaped its root: {self}")

        def _unexpected(self, *args, **kwargs):
            raise OSError(f"unexpected access to discarded output: {self}")

        def mkdir(self, mode=0o777, parents=False, exist_ok=False):
            self._require_output()

        def open(self, mode="r", buffering=-1, encoding=None, errors=None,
                 newline=None):
            self._require_output()
            if mode not in {"w", "wt", "a", "at", "x", "xt"}:
                return self._unexpected()
            return NullText()

        def write_text(self, data, encoding=None, errors=None, newline=None):
            self._require_output()
            if not isinstance(data, str):
                raise TypeError("write_text requires str")
            return len(data)

        def write_bytes(self, data):
            self._require_output()
            return memoryview(data).nbytes

        # Fail if the author starts depending on output files or new I/O forms.
        read_text = read_bytes = stat = lstat = _unexpected
        exists = is_file = is_dir = is_symlink = _unexpected
        iterdir = glob = rglob = readlink = resolve = _unexpected
        touch = unlink = rmdir = rename = replace = _unexpected
        chmod = lchmod = symlink_to = hardlink_to = _unexpected

    def output_path(*args, **kwargs):
        path = Path(*args, **kwargs)
        if inside(path):
            return _OutputPath(path)
        if any(isinstance(value, _OutputPath) for value in args):
            raise ValueError(f"virtual output path escaped its root: {path}")
        return path

    # Load everything first, including modules that capture atomic_json aliases.
    modules = [importlib.import_module(name) for name in PATH_MODULES]
    typed = importlib.import_module("tpu_db.benchmark.typed")
    guarded_measurement = modules[-1]
    numpy = importlib.import_module("numpy")
    original_atomic = typed.atomic_json
    original_guard_atomic = guarded_measurement.atomic_json
    original_savez = numpy.savez

    def discard_or_delegate(function, destination, *args, **kwargs):
        if inside(destination):
            return None
        if isinstance(destination, _OutputPath):
            raise ValueError(f"virtual output path escaped its root: {destination}")
        return function(destination, *args, **kwargs)

    def atomic_json(path, value):
        return discard_or_delegate(original_atomic, path, value)

    def guard_atomic_json(path, value):
        return discard_or_delegate(original_guard_atomic, path, value)

    def savez(file, *args, **kwargs):
        return discard_or_delegate(original_savez, file, *args, **kwargs)

    with ExitStack() as stack:
        for module in modules:
            stack.enter_context(patch.object(module, "Path", output_path))
        stack.enter_context(patch.object(typed, "atomic_json", atomic_json))
        stack.enter_context(patch.object(guarded_measurement, "atomic_json", guard_atomic_json))
        stack.enter_context(patch.object(numpy, "savez", savez))
        yield _OutputPath(reserved)
