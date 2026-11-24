#!/usr/bin/env python3
"""Approximate Sectile section 3.1, Table 1, on one GCP v6e TPU chip.

Run: python -B run_benchmark.py
Requires an already installed JAX TPU environment; never installs/downloads.
Samples stay in memory. Only the terminal table is produced; no result files.
Inputs are synthetic, not the paper's original SSB trace. See README.md.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

sys.dont_write_bytecode = True

PROBE_COUNTS = (128, 2_048, 5_466_180)
TABLE_SIZE = 2_001
QUALIFYING_SUPPLIERS = 449
PATHS = ("jax", "sparsecore")


def build_sparsecore_gather(rows: int, table_size: int):
    """Author's kernel: reproduce/protocols/probe.py, unchanged implementation."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.experimental.pallas import tpu_sc as plsc

    if rows <= 0 or not 1 <= table_size <= 4096:
        raise ValueError("SparseCore gather requires positive rows and table <=4096")
    windows = (rows + 127) // 128
    table_words = (table_size + 127) // 128 * 128
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core", subcore_axis_name="subcore"
    )

    @pl.kernel(
        out_type=jax.ShapeDtypeStruct((windows, 128), jnp.int32),
        mesh=mesh,
        scratch_types=[pltpu.VMEM((table_words,), jnp.int32)],
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
        name="q31_supplier_gather_sc_v1",
    )
    def kernel(table_hbm, keys_hbm, output_hbm, table_local):
        pltpu.sync_copy(table_hbm, table_local)

        def body(keys_ref, output_ref):
            @plsc.parallel_loop(0, 16)
            def group(i):
                start = i * 8
                keys = keys_ref[0, pl.ds(start, 8)]
                keys = jnp.clip(keys, jnp.int32(0), jnp.int32(table_size - 1))
                output_ref[0, pl.ds(start, 8)] = plsc.load_gather(
                    table_local, [keys]
                )

        spec = pl.BlockSpec((1, 128), lambda i: (i, 0))
        pltpu.emit_pipeline(
            body,
            grid=(windows,),
            in_specs=[spec],
            out_specs=[spec],
            core_axis_name=("core", "subcore"),
            dimension_semantics=(pltpu.PARALLEL,),
        )(keys_hbm, output_hbm)

    @jax.jit
    def run(table, keys):
        if (
            table.shape != (table_size,)
            or keys.shape != (rows,)
            or table.dtype != jnp.int32
            or keys.dtype != jnp.int32
        ):
            raise ValueError("standalone gather operand contract differs")
        padded_table = jnp.pad(table, (0, table_words - table_size))
        padded_keys = jnp.pad(keys, (0, windows * 128 - rows)).reshape(
            windows, 128
        )
        with jax.enable_x64(False):
            result = kernel(padded_table, padded_keys)
        return result.reshape(-1)[:rows]

    return run


def make_inputs(np, seed: int):
    """Match SF1 map cardinality/selectivity; approximate the probe distribution."""
    rng = np.random.default_rng(seed)
    table = np.zeros(TABLE_SIZE, dtype=np.int32)
    selected = rng.choice(np.arange(1, TABLE_SIZE), QUALIFYING_SUPPLIERS, replace=False)
    # selected is the set of keys that will be probed; the rest of the table is zero.
    table[selected] = rng.integers(1, 6, size=QUALIFYING_SUPPLIERS, dtype=np.int32)
    keys = rng.integers(1, TABLE_SIZE, size=max(PROBE_COUNTS), dtype=np.int32)
    # key is the supplier key to probe; some keys will be zero (no match) and some will be non-zero (match).
    return table, keys


def select_device(jax, pltpu):
    try:
        devices = jax.local_devices(backend="tpu")
    except RuntimeError as error:
        raise RuntimeError("Run in an existing GCP v6e TPU VM JAX environment.") from error
    if not devices:
        raise RuntimeError("No local TPU found; a GCP v6e TPU VM is required.")
    device = devices[0]
    kind = device.device_kind.lower()
    if "v6e" not in kind and "v6 lite" not in kind:
        raise RuntimeError(f"Expected v6e (TPU v6 lite), found {device.device_kind}.")
    with jax.default_device(device):
        info = pltpu.get_tpu_info()
    if getattr(info, "sparse_core", None) is None:
        raise RuntimeError("The selected TPU/runtime does not expose programmable SparseCore.")
    return device


def measure_case(jax, np, table_host, keys_host, table_device, device, args, offset):
    keys_device = jax.device_put(keys_host, device)
    jax.block_until_ready((table_device, keys_device))

    @jax.jit
    def probe_jax(table, keys):
        return table[keys]

    # Recent XLA versions can offload ordinary gather to SparseCore themselves.
    # Keep the TensorCore baseline intended by the paper's JAX/XLA column.
    # Official JAX discussion: https://github.com/jax-ml/jax/issues/40553
    baseline_options = {
        "xla_tpu_offload_gather_to_sparsecore": "false",
        "xla_tpu_offload_all_supported_gathers_to_sparsecore": "false",
    }
    try:
        baseline = probe_jax.lower(table_device, keys_device).compile(
            compiler_options=baseline_options
        )
    except Exception as error:
        raise RuntimeError(
            "JAX/XLA baseline compilation failed. The runtime must support disabling "
            "SparseCore gather offload; see README.md for the artifact's versions. "
            "No fallback or fabricated timing is used."
        ) from error
    sparsecore = build_sparsecore_gather(len(keys_host), len(table_host)).lower(
        table_device, keys_device
    ).compile()
    compiled = {"jax": baseline, "sparsecore": sparsecore}
    samples = {name: [] for name in PATHS}
    outputs = {}

    # All H2D/D2H transfers, compilation and validation are outside these intervals.
    with (
        jax.transfer_guard_host_to_device("disallow_explicit"),
        jax.transfer_guard_device_to_host("disallow_explicit"),
    ):
        for iteration in range(args.warmups):
            order = PATHS if (iteration + offset) % 2 == 0 else PATHS[::-1]
            for name in order:
                outputs[name] = jax.block_until_ready(
                    compiled[name](table_device, keys_device)
                )
        for iteration in range(args.samples):
            order = PATHS if (iteration + offset) % 2 == 0 else PATHS[::-1]
            for name in order:
                start = time.perf_counter_ns()
                result = compiled[name](table_device, keys_device)
                result.block_until_ready()
                elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
                samples[name].append(elapsed_ms)
                outputs[name] = result

    expected = table_host[keys_host]
    for name in PATHS:
        actual = np.asarray(outputs[name])
        if actual.shape != expected.shape or actual.dtype != np.dtype(np.int32):
            raise AssertionError(f"{name}: incorrect output shape or dtype")
        np.testing.assert_array_equal(actual, expected)
    return {
        "probes": len(keys_host),
        "samples_ms": samples,
        "median_ms": {name: statistics.median(samples[name]) for name in PATHS},
    }


def print_table(results):
    border = "+-------------+---------------+----------------+"
    print(border)
    print("|      Probes |  JAX/XLA (ms) | Pallas/SC (ms) |")
    print(border)
    for row in results:
        medians = row["median_ms"]
        print(
            f"| {row['probes']:>11,} | {medians['jax']:>13.3f} | "
            f"{medians['sparsecore']:>14.3f} |"
        )
    print(border)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20, help="Even sample count per path.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.warmups < 1 or args.samples < 2 or args.samples % 2 or args.seed < 0:
        parser.error("Use --warmups >= 1, an even --samples >= 2, and --seed >= 0.")

    try:
        import jax
        import numpy as np
        from jax.experimental.pallas import tpu as pltpu
        from jax.experimental.pallas import tpu_sc as plsc
    except ImportError as error:
        raise SystemExit(
            "Requires an already installed JAX TPU environment with Pallas SparseCore "
            "and NumPy. See README.md. Nothing has been installed or downloaded."
        ) from error
    if not all(hasattr(plsc, name) for name in ("VectorSubcoreMesh", "load_gather", "parallel_loop")):
        raise SystemExit("Installed JAX lacks the required Pallas SparseCore API; see README.md.")

    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_enable_compilation_cache", False)
    device = select_device(jax, pltpu)
    table_host, all_keys = make_inputs(np, args.seed)
    results = []
    with jax.default_device(device):
        table_device = jax.device_put(table_host, device)
        jax.block_until_ready(table_device)
        for offset, rows in enumerate(PROBE_COUNTS):
            results.append(measure_case(
                jax, np, table_host, all_keys[:rows], table_device, device, args, offset
            ))

    print(f"Device: {device.device_kind} (one chip); JAX: {jax.__version__}")
    print(f"Synthetic SF1-like inputs; seed={args.seed}; warmups={args.warmups}; samples={args.samples}.")
    print("Median warm call + synchronization; exact int32 validation passed for both paths.")
    print_table(results)


if __name__ == "__main__":
    main()
