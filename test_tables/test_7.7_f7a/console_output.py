"""Capture Python and native console logs in anonymous RAM, without files."""

from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
import ctypes
import io
import os
import sys


def _flush():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
    if hasattr(os, "memfd_create"):
        ctypes.CDLL(None).fflush(None)


class _Capture:
    def __init__(self, stream, descriptor=None):
        self.stream, self.descriptor = stream, descriptor

    def tail_text(self, limit=32768):
        """Read the last UTF-8 bytes while the capture context remains active."""
        if type(limit) is not int or limit <= 0:
            raise ValueError("Console tail limit must be a positive integer.")
        _flush()
        if self.descriptor is None:
            raw = self.stream.getvalue().encode("utf-8")[-limit:]
        else:
            size = os.fstat(self.descriptor).st_size
            # pread leaves the shared write offset unchanged for later logs.
            raw = os.pread(self.descriptor, min(size, limit), max(0, size - limit))
        return raw.decode("utf-8", errors="replace")


@contextmanager
def capture_console():
    """Yield an in-memory diagnostic tail; restore both streams on every exit.

    Linux fd 1/2 capture includes C++ libraries. The portable Python-only
    fallback supports local CPU mocks. Anonymous multiprocessing control pipes
    have separate descriptors and are never redirected.
    """
    if not hasattr(os, "memfd_create"):
        with io.StringIO() as stream, redirect_stdout(stream), redirect_stderr(stream):
            yield _Capture(stream)
        return
    _flush()
    with ExitStack() as stack:
        descriptor = os.memfd_create("sectile-console", os.MFD_CLOEXEC)
        stack.callback(os.close, descriptor)
        for number in (1, 2):
            saved = os.dup(number)
            stack.callback(os.close, saved)
            stack.callback(os.dup2, saved, number)
            os.dup2(descriptor, number)
        stream = stack.enter_context(io.TextIOWrapper(
            os.fdopen(os.dup(descriptor), "wb", buffering=0),
            encoding="utf-8", errors="backslashreplace", newline="\n", write_through=True,
        ))
        stack.enter_context(redirect_stdout(stream))
        stack.enter_context(redirect_stderr(stream))
        try:
            yield _Capture(stream, descriptor)
        finally:
            _flush()
