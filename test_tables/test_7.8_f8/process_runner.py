"""Anonymous Pipe transport; native logs remain quiet throughout each child."""

import sys
sys.dont_write_bytecode = True

import math
import time


def receive(connection, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if connection.poll(0.2):
            try:
                return connection.recv()
            except EOFError as error:
                raise RuntimeError("TPU worker closed its pipe without a result.") from error
        if not process.is_alive():
            if connection.poll():
                continue
            raise RuntimeError(f"TPU worker exited with status {process.exitcode} before returning a result.")
    raise TimeoutError(f"Worker exceeded {timeout:g} seconds; adjust --timeout if needed.")


def run_case(context, args, query=None, layout=None, order_descriptor=None, bundle=None):
    from runtime import run_worker
    from console_output import capture_console

    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=run_worker, args=(
        child, args.artifact_root, args.work, query, layout, order_descriptor, bundle,
    ))
    try:
        with capture_console():
            process.start()
        child.close()
        message = receive(parent, process, args.timeout)
        if message.get("event") == "error":
            label = f"{query} / {layout}" if query else "Preflight"
            raise RuntimeError(f"{label} failed; no value is substituted.\n"
                               + message["message"] + message.get("logs", ""))
        if message.get("event") != "ok":
            raise RuntimeError(f"Unexpected worker message: {message!r}")
        process.join(30)
        if process.is_alive() or process.exitcode:
            raise RuntimeError("Worker did not exit cleanly after returning its result.")
        result = message["result"]
        if query is not None and (result["query"] != query or result["variant"] != layout
                or not math.isfinite(result["median_ms"]) or result["median_ms"] <= 0):
            raise ValueError("Worker returned an invalid latency or query/layout identity.")
        return result
    finally:
        child.close()
        if process.is_alive():
            process.terminate()
            process.join(10)
        if process.is_alive():
            process.kill()
            process.join(5)
        parent.close()
        if not process.is_alive():
            process.close()
