#!/usr/bin/env bash
set -euo pipefail

# Preserve the author's generation, sorting, verification and recovery code.
# Only replace Git setup with a build of source verified by prepare_data.py.
if [[ "${SCALE_FACTORS:-}" != "100" || "${PHYSICAL_ORDER:-}" != "date_pk" || \
      "${DBGEN_COMMIT:-}" != "ae1e254aa4d603d8ef1f44078e5abed011634b23" || \
      -z "${DBGEN_DIR:-}" ]]; then
  echo "Run ../prepare_data.py to configure and verify the local SF100 generator." >&2
  exit 2
fi

preparation_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$preparation_directory/prepare_frozen_ssb_data.sh"

ensure_generator() {
  if [[ "$GENERATOR_READY" == "1" ]]; then
    return
  fi
  cmake -S "$DBGEN_DIR" -B "$DBGEN_DIR/build" || return 1
  cmake --build "$DBGEN_DIR/build" -j"$(processor_count)" || return 1
  GENERATOR_READY=1
}

main "$@"
