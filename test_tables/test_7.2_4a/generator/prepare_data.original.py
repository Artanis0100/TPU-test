"""Generate SSB and use the original loader for dictionary columns and statistics."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from reproduce import common


def prepare(work, sf, layout):
    identity = common.configure(sf, layout)
    jax = common.require_tpu()
    work = work.resolve()
    environment = dict(os.environ, SCALE_FACTORS=str(sf),
        PHYSICAL_ORDER='date_pk' if layout == 'date_pk' else 'standard',
        DATA_ROOT=str(work / 'data' / layout), DBGEN_DIR=str(work / 'ssb-dbgen'),
        GCS_ROOT='', ALLOW_DBGEN_FALLBACK='1', PUBLISH_GENERATED_DATA='0')
    subprocess.run(['bash', str(common.ROOT / 'scripts/prepare_frozen_ssb_data.sh')],
        env=environment, check=True)
    from tpu_db.benchmark.typed_cache import load_or_preload_integer_inputs
    cache = work / 'cache' / layout / f'sf{sf}-v1'
    _, manifest, _ = load_or_preload_integer_inputs(cache,
        source_directory=work / 'data' / layout / f'sf{sf}', scale_factor=sf,
        source_id=identity, capacity_bytes=min(24 << 30,
            int(jax.devices()[0].memory_stats()['bytes_limit'] * .8)),
        device_put=jax.device_put, fact_order=layout)
    hashes = {entry['file']: common.sha256(cache / entry['file']) for entry in manifest['arrays']}
    # Content hashing is offline, never part of a measured query call.
    (cache / 'array-hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps(dict(sf=sf, layout=layout, arrays=len(hashes), cache=str(cache))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=common.outside_repository, default=Path('../sectile-work'))
    parser.add_argument('--scales', nargs='+', type=int, choices=common.SCALES, default=[1])
    parser.add_argument('--layouts', nargs='+', choices=('date_pk', 'original'), default=['date_pk'])
    parser.add_argument('--cell', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cell:
        if len(args.scales) != 1 or len(args.layouts) != 1:
            parser.error('--cell requires one scale and layout')
        prepare(args.work, args.scales[0], args.layouts[0])
    else:
        common.verify_contents()
        for layout in args.layouts:
            for sf in args.scales:
                subprocess.run([sys.executable, '-B', '-m', 'reproduce.prepare_data', '--cell',
                    '--work', str(args.work.resolve()), '--scales', str(sf), '--layouts', layout], check=True)


if __name__ == '__main__':
    main()
