"""Keep a microbenchmark's three report files in RAM; delegate input reads."""

from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
from unittest.mock import patch


@contextmanager
def output_artifacts(module, root):
    """Adapt only this source module's Path alias beneath a nonexistent root."""
    reserved = Path(os.path.abspath(os.fspath(root)))
    if os.path.lexists(reserved):
        raise FileExistsError(f"RAM output root must not exist: {reserved}")
    files = {}

    def key(value):
        path = Path(os.path.abspath(os.fspath(value)))
        if path != reserved and reserved not in path.parents:
            raise ValueError(f"RAM output escaped its root: {value}")
        return path

    def inside(value):
        try:
            key(value)
            return True
        except (TypeError, ValueError):
            return False

    class MemoryBytes(io.BytesIO):
        def __init__(self, path):
            super().__init__()
            self.path = path

        def close(self):
            if not self.closed:
                files[self.path] = self.getvalue()
            super().close()

    class OutputPath(type(Path())):
        def mkdir(self, mode=0o777, parents=False, exist_ok=False):
            key(self)

        def open(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
            path = key(self)
            if mode not in {"w", "wt", "wb"}:
                raise OSError(f"Unexpected RAM report open mode: {mode}")
            buffer = MemoryBytes(path)
            if "b" in mode:
                return buffer
            return io.TextIOWrapper(buffer, encoding=encoding or "utf-8",
                                    errors=errors or "strict", newline=newline)

        def write_text(self, data, encoding=None, errors=None, newline=None):
            with self.open("w", encoding=encoding, errors=errors, newline=newline) as stream:
                return stream.write(data)

        def write_bytes(self, data):
            payload = bytes(memoryview(data))
            files[key(self)] = payload
            return len(payload)

    def output_path(*args, **kwargs):
        path = Path(*args, **kwargs)
        if inside(path):
            return OutputPath(key(path))
        if any(isinstance(value, OutputPath) for value in args):
            raise ValueError(f"RAM output escaped its root: {path}")
        return path

    class Store:
        root = OutputPath(reserved)

        def read_text(self, path):
            return files[key(path)].decode("utf-8")

        def read_json(self, path):
            return json.loads(self.read_text(path))

    with patch.object(module, "Path", output_path):
        yield Store()
