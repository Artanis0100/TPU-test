#!/usr/bin/env bash
set -euo pipefail

# Prepare and verify the immutable SSB inputs shared by TPU, DuckDB, and
# Sirius. The manifest is part of the dataset contract: a scale factor is
# usable only when every table matches its recorded rows, bytes, and SHA-256.
SCALE_FACTORS="${SCALE_FACTORS:-1 5 10 15 20}"
DBGEN_COMMIT="${DBGEN_COMMIT:-ae1e254aa4d603d8ef1f44078e5abed011634b23}"
DBGEN_DIR="${DBGEN_DIR:-$HOME/ssb-dbgen-standard}"
PHYSICAL_ORDER="${PHYSICAL_ORDER:-standard}"
if [[ "$PHYSICAL_ORDER" == "date_pk" ]]; then
  DEFAULT_DATASET_ID="ssb-dbgen-${DBGEN_COMMIT:0:12}-date-pk"
  EXPECTED_SOURCE_PREORDERED_LAYOUT=1
elif [[ "$PHYSICAL_ORDER" == "standard" ]]; then
  DEFAULT_DATASET_ID="ssb-dbgen-${DBGEN_COMMIT:0:12}"
  EXPECTED_SOURCE_PREORDERED_LAYOUT=0
else
  echo "PHYSICAL_ORDER must be standard or date_pk." >&2
  exit 2
fi
SOURCE_PREORDERED_LAYOUT="${SOURCE_PREORDERED_LAYOUT:-$EXPECTED_SOURCE_PREORDERED_LAYOUT}"
DATASET_ID="${DATASET_ID:-$DEFAULT_DATASET_ID}"
DATA_ROOT="${DATA_ROOT:-$HOME/tpu-db-data/${DATASET_ID}}"
GCS_ROOT="${GCS_ROOT:-}"
ALLOW_DBGEN_FALLBACK="${ALLOW_DBGEN_FALLBACK:-0}"
PUBLISH_GENERATED_DATA="${PUBLISH_GENERATED_DATA:-0}"
SORT_BUFFER_SIZE="${SORT_BUFFER_SIZE:-96G}"
# Large generated LINEORDER inputs may be staged on a separate filesystem
# (for example a RAM-backed mount) while the sorted, verified snapshot is
# written to DATA_ROOT. This avoids requiring space for both physical orders
# on the durable dataset disk. The raw staging directory is always unique and
# is removed only after the sorted output has replaced its symlink.
RAW_LINEORDER_STAGING_ROOT="${RAW_LINEORDER_STAGING_ROOT:-}"
if [[ -n "$RAW_LINEORDER_STAGING_ROOT" ]]; then
  DEFAULT_LINEORDER_SORT_PARTITIONS=8
else
  DEFAULT_LINEORDER_SORT_PARTITIONS=1
fi
# Range partitions bound durable-disk amplification during date+PK ordering.
# They are especially useful when the unsorted LINEORDER input is on a
# separate filesystem. All unsorted range buckets exist together. During each
# sort, the growing output plus remaining buckets stays equal to the total
# partition bytes, with one additional sorted bucket and sort temporary space.
# The original LINEORDER also remains until final replacement: regular input
# charges those bytes to DATA_ROOT, while a raw symlink charges them to its
# target filesystem. DATE skew can make the largest bucket nearly as large as
# LINEORDER. These operands are measured and reported below. Boundaries come
# from the observed DATE range, not from a scale factor or query identity.
LINEORDER_SORT_PARTITIONS="${LINEORDER_SORT_PARTITIONS:-$DEFAULT_LINEORDER_SORT_PARTITIONS}"
PREORDER_SOURCE_ROOT="${PREORDER_SOURCE_ROOT:-}"
TABLES=(lineorder date customer supplier part)
GENERATOR_READY=0

# Every unpublished directory created by this invocation carries a marker for
# this session.  Cleanup only removes an exact, validated path whose marker
# matches the owning session; it never guesses ownership from a wildcard.
PREPARATION_OWNER_MARKER=.tpu-db-preparation-owner
PREPARATION_RECOVERY_MARKER=.tpu-db-recovery-generation
PREPARATION_LOCK_BASENAME=.tpu-db-preparation-lock
PREPARATION_LOCK_OWNER_MARKER=.tpu-db-preparation-lock-owner
PREPARATION_SESSION_ID="$(date -u +%Y%m%dT%H%M%SZ).$$.${RANDOM}"
PREPARATION_OWNER_PID="$$"
PREPARATION_LOCK_HELD=0
PREPARATION_ACTIVE_INCOMING=""
PREPARATION_ACTIVE_INCOMING_SF=""
PREPARATION_ACTIVE_RAW=""
PREPARATION_ACTIVE_RAW_SF=""
PREPARATION_ACTIVE_ROLLBACK=""
PREPARATION_ACTIVE_ROLLBACK_SF=""
PREPARATION_ACTIVE_ROLLBACK_DESTINATION=""
PREPARATION_CLEANUP_RUNNING=0

if [[ "$ALLOW_DBGEN_FALLBACK" != "0" && "$ALLOW_DBGEN_FALLBACK" != "1" ]]; then
  echo "ALLOW_DBGEN_FALLBACK must be 0 or 1." >&2
  exit 2
fi
if [[ "$PUBLISH_GENERATED_DATA" != "0" && "$PUBLISH_GENERATED_DATA" != "1" ]]; then
  echo "PUBLISH_GENERATED_DATA must be 0 or 1." >&2
  exit 2
fi
if [[ "$SOURCE_PREORDERED_LAYOUT" != "$EXPECTED_SOURCE_PREORDERED_LAYOUT" ]]; then
  echo "SOURCE_PREORDERED_LAYOUT=$SOURCE_PREORDERED_LAYOUT contradicts" \
    "PHYSICAL_ORDER=$PHYSICAL_ORDER (expected" \
    "$EXPECTED_SOURCE_PREORDERED_LAYOUT)." >&2
  exit 2
fi
if [[ -n "$RAW_LINEORDER_STAGING_ROOT" && "$PHYSICAL_ORDER" != "date_pk" ]]; then
  echo "RAW_LINEORDER_STAGING_ROOT requires PHYSICAL_ORDER=date_pk." >&2
  exit 2
fi
if ! [[ "$LINEORDER_SORT_PARTITIONS" =~ ^[1-9][0-9]*$ ]] || \
    (( LINEORDER_SORT_PARTITIONS > 64 )); then
  echo "LINEORDER_SORT_PARTITIONS must be an integer from 1 through 64." >&2
  exit 2
fi
for sf in $SCALE_FACTORS; do
  if ! [[ "$sf" =~ ^[1-9][0-9]*$ ]]; then
    echo "Every SCALE_FACTORS value must be a positive integer: $sf" >&2
    exit 2
  fi
done

ensure_generator() {
  if [[ "$GENERATOR_READY" == "1" ]]; then
    return
  fi
  if [[ ! -d "$DBGEN_DIR/.git" ]]; then
    git clone https://github.com/eyalroz/ssb-dbgen.git "$DBGEN_DIR" || return 1
  fi
  git -C "$DBGEN_DIR" fetch origin || return 1
  git -C "$DBGEN_DIR" checkout --detach "$DBGEN_COMMIT" || return 1
  cmake -S "$DBGEN_DIR" -B "$DBGEN_DIR/build" || return 1
  cmake --build "$DBGEN_DIR/build" -j"$(processor_count)" || return 1
  GENERATOR_READY=1
}

processor_count() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  else
    getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1
  fi
}

table_row_count() {
  local path="$1"
  local newline_count
  local final_byte
  newline_count="$(LC_ALL=C wc -l < "$path" | tr -d '[:space:]')" || \
    return 1
  final_byte="$(
    LC_ALL=C tail -c 1 "$path" 2>/dev/null | \
      od -An -tu1 | tr -d '[:space:]'
  )" || return 1
  if [[ -n "$final_byte" && "$final_byte" != "10" ]]; then
    newline_count="$((newline_count + 1))"
  fi
  printf '%s\n' "$newline_count"
}

file_byte_count() {
  local byte_count
  # Manifest and capacity bytes describe file content. Explicitly dereference
  # symlinks because GNU stat otherwise reports the link-text length.
  if byte_count="$(stat -L -c %s "$1" 2>/dev/null)"; then
    printf '%s\n' "$byte_count"
  else
    stat -L -f %z "$1"
  fi
}

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{ print $1 }'
  else
    shasum -a 256 "$1" | awk '{ print $1 }'
  fi
}

manifest_value() {
  local manifest="$1"
  local key="$2"
  awk -F= -v key="$key" '
    $1 == key {
      value = substr($0, index($0, "=") + 1)
      sub(/^[[:space:]]+/, "", value)
      sub(/[[:space:]\r]+$/, "", value)
      print value
    }
  ' "$manifest"
}

manifest_structure_is_valid() {
  local manifest="$1"
  local format_version="$2"
  LC_ALL=C awk -v format_version="$format_version" '
    BEGIN {
      allowed["format_version"] = 1
      allowed["generator"] = 1
      allowed["dbgen_commit"] = 1
      allowed["scale_factor"] = 1
      allowed["dataset_id"] = 1
      allowed["physical_order"] = 1
      allowed["source_preordered_layout"] = 1
      allowed["source_manifest_sha256"] = 1
      required["format_version"] = 1
      required["generator"] = 1
      required["dbgen_commit"] = 1
      required["scale_factor"] = 1
      tables[1] = "lineorder"
      tables[2] = "date"
      tables[3] = "customer"
      tables[4] = "supplier"
      tables[5] = "part"
      for (table_index = 1; table_index <= 5; table_index++) {
        table = tables[table_index]
        allowed[table "_rows"] = 1
        allowed[table "_bytes"] = 1
        allowed[table "_sha256"] = 1
        required[table "_rows"] = 1
        required[table "_bytes"] = 1
        required[table "_sha256"] = 1
      }
      if (format_version == "3") {
        required["dataset_id"] = 1
        required["physical_order"] = 1
        required["source_preordered_layout"] = 1
      }
    }
    {
      if ($0 == "" || $0 ~ /^[[:space:]]/ ||
          $0 ~ /[[:space:]]$/ || index($0, "=") == 0) {
        invalid = 1
        next
      }
      separator = index($0, "=")
      key = substr($0, 1, separator - 1)
      value = substr($0, separator + 1)
      if (!(key in allowed) || key in seen || value == "") {
        invalid = 1
        next
      }
      seen[key] = 1
    }
    END {
      for (key in required) {
        if (!(key in seen)) invalid = 1
      }
      exit invalid ? 1 : 0
    }
  ' "$manifest"
}

process_start_token() {
  local pid="$1"
  if [[ -r "/proc/${pid}/stat" ]]; then
    awk '{ print $22 }' "/proc/${pid}/stat" 2>/dev/null
  else
    ps -o lstart= -p "$pid" 2>/dev/null | awk '
      {
        sub(/^[[:space:]]+/, "")
        sub(/[[:space:]]+$/, "")
        print
      }
    '
  fi
}

preparation_lock_path() {
  local root="${DATA_ROOT%/}"
  [[ -n "$root" && "$root" == /* && "$root" != "/" ]] || return 1
  printf '%s/%s\n' "$root" "$PREPARATION_LOCK_BASENAME"
}

write_preparation_lock_marker() {
  local lock_dir="$1"
  local marker="$lock_dir/$PREPARATION_LOCK_OWNER_MARKER"
  local temporary_marker
  local owner_start_token
  [[ "$lock_dir" == "$(preparation_lock_path)" && \
     -d "$lock_dir" && ! -L "$lock_dir" ]] || return 1
  owner_start_token="$(
    process_start_token "$PREPARATION_OWNER_PID" 2>/dev/null || true
  )"
  owner_start_token="${owner_start_token:-unavailable}"
  temporary_marker="$(mktemp "$lock_dir/.lock-owner.XXXXXX")" || return 1
  if ! {
    echo "format_version=1"
    echo "session_id=$PREPARATION_SESSION_ID"
    echo "owner_pid=$PREPARATION_OWNER_PID"
    echo "owner_start_token=$owner_start_token"
    echo "directory_basename=$PREPARATION_LOCK_BASENAME"
  } > "$temporary_marker"; then
    rm -f -- "$temporary_marker"
    return 1
  fi
  if ! mv "$temporary_marker" "$marker"; then
    rm -f -- "$temporary_marker"
    return 1
  fi
}

preparation_lock_marker_matches() {
  local lock_dir="$1"
  local expected_session="${2:-}"
  local marker="$lock_dir/$PREPARATION_LOCK_OWNER_MARKER"
  local marker_session
  local marker_pid
  local marker_start_token
  [[ "$lock_dir" == "$(preparation_lock_path)" && \
     -d "$lock_dir" && ! -L "$lock_dir" ]] || return 1
  [[ -f "$marker" && ! -L "$marker" ]] || return 1
  [[ "$(manifest_value "$marker" format_version)" == "1" ]] || return 1
  [[ "$(manifest_value "$marker" directory_basename)" == \
      "$PREPARATION_LOCK_BASENAME" ]] || return 1
  marker_session="$(manifest_value "$marker" session_id)"
  marker_pid="$(manifest_value "$marker" owner_pid)"
  marker_start_token="$(manifest_value "$marker" owner_start_token)"
  [[ "$marker_session" =~ ^[A-Za-z0-9._-]+$ ]] || return 1
  [[ "$marker_pid" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ -n "$marker_start_token" ]] || return 1
  if [[ -n "$expected_session" ]]; then
    [[ "$marker_session" == "$expected_session" ]] || return 1
  fi
}

preparation_lock_owner_is_live() {
  local lock_dir="$1"
  local marker="$lock_dir/$PREPARATION_LOCK_OWNER_MARKER"
  local marker_pid
  local expected_start_token
  local observed_start_token
  marker_pid="$(manifest_value "$marker" owner_pid)"
  expected_start_token="$(manifest_value "$marker" owner_start_token)"
  if [[ "$expected_start_token" != "unavailable" ]]; then
    observed_start_token="$(
      process_start_token "$marker_pid" 2>/dev/null || true
    )"
    [[ -n "$observed_start_token" && \
       "$observed_start_token" == "$expected_start_token" ]]
    return
  fi
  kill -0 "$marker_pid" 2>/dev/null
}

acquire_preparation_lock() {
  local root="${DATA_ROOT%/}"
  local lock_dir
  [[ -n "$root" && "$root" == /* && "$root" != "/" && \
     -d "$root" && ! -L "$root" ]] || {
    echo "DATA_ROOT must be an existing non-symlink absolute directory." >&2
    return 1
  }
  [[ "$PREPARATION_LOCK_HELD" == "0" ]] || return 1
  lock_dir="$(preparation_lock_path)" || return 1
  if ! mkdir "$lock_dir" 2>/dev/null; then
    if ! preparation_lock_marker_matches "$lock_dir"; then
      echo "Preparation lock is unowned or malformed; refusing to remove it: $lock_dir" >&2
      return 1
    fi
    if preparation_lock_owner_is_live "$lock_dir"; then
      echo "Another live preparation owns $lock_dir; refusing concurrent preparation." >&2
    else
      echo "A stale preparation lock remains at $lock_dir; refusing automatic removal." >&2
    fi
    return 1
  fi
  if ! write_preparation_lock_marker "$lock_dir"; then
    rmdir "$lock_dir" 2>/dev/null || true
    return 1
  fi
  PREPARATION_LOCK_HELD=1
}

require_preparation_lock() {
  local lock_dir
  [[ "$PREPARATION_LOCK_HELD" == "1" ]] || {
    echo "Preparation mutation requires the DATA_ROOT lock." >&2
    return 1
  }
  lock_dir="$(preparation_lock_path)" || return 1
  preparation_lock_marker_matches \
    "$lock_dir" "$PREPARATION_SESSION_ID" || {
    echo "Preparation lock ownership changed; refusing mutation." >&2
    return 1
  }
}

release_preparation_lock() {
  local lock_dir
  local marker
  local unexpected
  [[ "$PREPARATION_LOCK_HELD" == "1" ]] || return 0
  lock_dir="$(preparation_lock_path)" || return 1
  preparation_lock_marker_matches \
    "$lock_dir" "$PREPARATION_SESSION_ID" || return 1
  marker="$lock_dir/$PREPARATION_LOCK_OWNER_MARKER"
  unexpected="$(
    find "$lock_dir" -mindepth 1 -maxdepth 1 \
      ! -name "$PREPARATION_LOCK_OWNER_MARKER" -print -quit
  )" || return 1
  [[ -z "$unexpected" ]] || {
    echo "Preparation lock contains unexpected state; retaining it: $unexpected" >&2
    return 1
  }
  rm -f -- "$marker" || return 1
  rmdir "$lock_dir" || return 1
  PREPARATION_LOCK_HELD=0
}

staging_path_scale() {
  local kind="$1"
  local staging_dir="$2"
  local expected_root
  local basename
  expected_root=""
  case "$kind" in
    incoming)
      expected_root="${DATA_ROOT%/}"
      ;;
    raw)
      expected_root="${RAW_LINEORDER_STAGING_ROOT%/}"
      ;;
    *)
      return 1
      ;;
  esac
  [[ -n "$expected_root" && "$expected_root" == /* && \
     "$expected_root" != "/" ]] || return 1
  [[ "${staging_dir%/*}" == "$expected_root" ]] || return 1
  basename="${staging_dir##*/}"
  if [[ "$kind" == "incoming" && \
        "$basename" =~ ^\.sf([1-9][0-9]*)\.incoming\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return
  fi
  if [[ "$kind" == "raw" && \
        "$basename" =~ ^\.sf([1-9][0-9]*)\.raw-lineorder\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return
  fi
  return 1
}

write_preparation_owner_marker() {
  local kind="$1"
  local sf="$2"
  local staging_dir="$3"
  local marker="$staging_dir/$PREPARATION_OWNER_MARKER"
  local temporary_marker
  local owner_start_token
  [[ "$(staging_path_scale "$kind" "$staging_dir")" == "$sf" ]] || \
    return 1
  owner_start_token="$(
    process_start_token "$PREPARATION_OWNER_PID" 2>/dev/null || true
  )"
  # Some restricted macOS environments deny ps inspection.  PID liveness is
  # still conservative there (a reused PID delays cleanup rather than risking
  # deletion); Linux records /proc's process-start tick to reject PID reuse.
  owner_start_token="${owner_start_token:-unavailable}"
  temporary_marker="$(mktemp "$staging_dir/.owner-marker.XXXXXX")" || \
    return 1
  if ! {
    echo "format_version=1"
    echo "session_id=$PREPARATION_SESSION_ID"
    echo "owner_pid=$PREPARATION_OWNER_PID"
    echo "owner_start_token=$owner_start_token"
    echo "kind=$kind"
    echo "scale_factor=$sf"
    echo "directory_basename=${staging_dir##*/}"
  } > "$temporary_marker"; then
    rm -f -- "$temporary_marker"
    return 1
  fi
  mv "$temporary_marker" "$marker" || return 1
}

owned_staging_marker_matches() {
  local kind="$1"
  local sf="$2"
  local staging_dir="$3"
  local expected_session="${4:-}"
  local marker="$staging_dir/$PREPARATION_OWNER_MARKER"
  [[ -d "$staging_dir" && ! -L "$staging_dir" ]] || return 1
  [[ "$(staging_path_scale "$kind" "$staging_dir")" == "$sf" ]] || \
    return 1
  [[ -f "$marker" && ! -L "$marker" ]] || return 1
  [[ "$(manifest_value "$marker" format_version)" == "1" ]] || return 1
  [[ "$(manifest_value "$marker" kind)" == "$kind" ]] || return 1
  [[ "$(manifest_value "$marker" scale_factor)" == "$sf" ]] || return 1
  [[ "$(manifest_value "$marker" directory_basename)" == \
      "${staging_dir##*/}" ]] || return 1
  local marker_session
  local marker_pid
  local marker_start_token
  marker_session="$(manifest_value "$marker" session_id)"
  marker_pid="$(manifest_value "$marker" owner_pid)"
  marker_start_token="$(manifest_value "$marker" owner_start_token)"
  [[ "$marker_session" =~ ^[A-Za-z0-9._-]+$ ]] || return 1
  [[ "$marker_pid" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ -n "$marker_start_token" ]] || return 1
  if [[ -n "$expected_session" ]]; then
    [[ "$marker_session" == "$expected_session" ]] || return 1
  fi
}

owned_staging_owner_is_live() {
  local staging_dir="$1"
  local marker="$staging_dir/$PREPARATION_OWNER_MARKER"
  local marker_pid
  local expected_start_token
  local observed_start_token
  marker_pid="$(manifest_value "$marker" owner_pid)"
  expected_start_token="$(manifest_value "$marker" owner_start_token)"
  if [[ "$expected_start_token" != "unavailable" ]]; then
    observed_start_token="$(
      process_start_token "$marker_pid" 2>/dev/null || true
    )"
    [[ -n "$observed_start_token" && \
       "$observed_start_token" == "$expected_start_token" ]]
    return
  fi
  # When start-token inspection is unavailable, kill -0 is conservative for
  # the same-user preparation processes that can own this dataset root.
  kill -0 "$marker_pid" 2>/dev/null
}

discard_owned_staging_dir() {
  local kind="$1"
  local sf="$2"
  local staging_dir="$3"
  local expected_session="${4:-$PREPARATION_SESSION_ID}"
  require_preparation_lock || return 1
  if ! owned_staging_marker_matches \
      "$kind" "$sf" "$staging_dir" "$expected_session"; then
    echo "Refusing to remove unowned $kind staging path: $staging_dir" >&2
    return 1
  fi
  rm -rf -- "$staging_dir" || return 1
}

register_active_staging_dir() {
  local kind="$1"
  local sf="$2"
  local staging_dir="$3"
  require_preparation_lock || return 1
  owned_staging_marker_matches \
    "$kind" "$sf" "$staging_dir" "$PREPARATION_SESSION_ID" || return 1
  if [[ "$kind" == "incoming" ]]; then
    PREPARATION_ACTIVE_INCOMING="$staging_dir"
    PREPARATION_ACTIVE_INCOMING_SF="$sf"
  else
    PREPARATION_ACTIVE_RAW="$staging_dir"
    PREPARATION_ACTIVE_RAW_SF="$sf"
  fi
}

clear_active_staging_dir() {
  local kind="$1"
  local staging_dir="$2"
  if [[ "$kind" == "incoming" && \
        "$PREPARATION_ACTIVE_INCOMING" == "$staging_dir" ]]; then
    PREPARATION_ACTIVE_INCOMING=""
    PREPARATION_ACTIVE_INCOMING_SF=""
  elif [[ "$kind" == "raw" && \
          "$PREPARATION_ACTIVE_RAW" == "$staging_dir" ]]; then
    PREPARATION_ACTIVE_RAW=""
    PREPARATION_ACTIVE_RAW_SF=""
  fi
}

cleanup_stale_owned_staging_root() {
  local kind="$1"
  local root="$2"
  local candidate
  local sf
  local marker_session
  local status=0
  [[ -n "$root" ]] || return 0
  [[ -d "$root" && ! -L "$root" ]] || return 0
  while IFS= read -r -d '' candidate; do
    sf="$(staging_path_scale "$kind" "$candidate" 2>/dev/null || true)"
    [[ -n "$sf" ]] || continue
    if ! owned_staging_marker_matches "$kind" "$sf" "$candidate"; then
      echo "Leaving unowned stale-looking staging path untouched: $candidate" >&2
      continue
    fi
    marker_session="$(
      manifest_value "$candidate/$PREPARATION_OWNER_MARKER" session_id
    )"
    if owned_staging_owner_is_live "$candidate"; then
      echo "Active preparation session owns $candidate; refusing concurrent preparation." >&2
      status=1
      continue
    fi
    echo "Removing stale owned $kind staging path: $candidate"
    discard_owned_staging_dir \
      "$kind" "$sf" "$candidate" "$marker_session" || status=1
  done < <(find "$root" -mindepth 1 -maxdepth 1 -type d -print0)
  return "$status"
}

cleanup_stale_owned_staging() {
  require_preparation_lock || return 1
  cleanup_stale_owned_staging_root incoming "${DATA_ROOT%/}" || return 1
  if [[ -n "$RAW_LINEORDER_STAGING_ROOT" ]]; then
    cleanup_stale_owned_staging_root \
      raw "${RAW_LINEORDER_STAGING_ROOT%/}" || return 1
  fi
}

cleanup_current_preparation_attempt() {
  local status=0
  if [[ "$PREPARATION_CLEANUP_RUNNING" == "1" ]]; then
    return 0
  fi
  PREPARATION_CLEANUP_RUNNING=1
  if [[ -n "$PREPARATION_ACTIVE_RAW" && \
        -e "$PREPARATION_ACTIVE_RAW" ]]; then
    discard_owned_staging_dir \
      raw "$PREPARATION_ACTIVE_RAW_SF" "$PREPARATION_ACTIVE_RAW" || \
      status=1
  fi
  PREPARATION_ACTIVE_RAW=""
  PREPARATION_ACTIVE_RAW_SF=""
  if [[ -n "$PREPARATION_ACTIVE_INCOMING" && \
        -e "$PREPARATION_ACTIVE_INCOMING" ]]; then
    discard_owned_staging_dir \
      incoming "$PREPARATION_ACTIVE_INCOMING_SF" \
      "$PREPARATION_ACTIVE_INCOMING" || status=1
  fi
  PREPARATION_ACTIVE_INCOMING=""
  PREPARATION_ACTIVE_INCOMING_SF=""
  if [[ -n "$PREPARATION_ACTIVE_ROLLBACK" && \
        -e "$PREPARATION_ACTIVE_ROLLBACK" ]]; then
    resolve_interrupted_rollback \
      "$PREPARATION_ACTIVE_ROLLBACK_SF" \
      "$PREPARATION_ACTIVE_ROLLBACK" \
      "$PREPARATION_ACTIVE_ROLLBACK_DESTINATION" || status=1
  fi
  PREPARATION_ACTIVE_ROLLBACK=""
  PREPARATION_ACTIVE_ROLLBACK_SF=""
  PREPARATION_ACTIVE_ROLLBACK_DESTINATION=""
  if ! release_preparation_lock; then
    status=1
  fi
  PREPARATION_CLEANUP_RUNNING=0
  return "$status"
}

preparation_exit_handler() {
  local status="$?"
  trap - EXIT HUP INT TERM
  if ! cleanup_current_preparation_attempt; then
    echo "Preparation cleanup was incomplete; owned markers were retained for safe startup recovery." >&2
    if [[ "$status" == "0" ]]; then
      status=1
    fi
  fi
  exit "$status"
}

preparation_signal_handler() {
  local status="$1"
  local signal_name="$2"
  echo "Received $signal_name; cleaning the current unpublished preparation attempt." >&2
  exit "$status"
}

install_preparation_cleanup_traps() {
  trap preparation_exit_handler EXIT
  trap 'preparation_signal_handler 129 HUP' HUP
  trap 'preparation_signal_handler 130 INT' INT
  trap 'preparation_signal_handler 143 TERM' TERM
}

write_manifest() {
  local sf="$1"
  local data_dir="$2"
  local source_manifest_sha256="${3:-}"
  local manifest="$data_dir/manifest.txt"
  local temporary_manifest
  local table
  [[ -d "$data_dir" && ! -L "$data_dir" ]] || return 1
  for table in "${TABLES[@]}"; do
    [[ -s "$data_dir/${table}.tbl" && \
       ! -L "$data_dir/${table}.tbl" ]] || return 1
  done
  temporary_manifest="$(mktemp "$data_dir/.manifest.txt.XXXXXX")" || return 1
  if ! {
    if [[ "$PHYSICAL_ORDER" == "date_pk" ]]; then
      echo "format_version=3"
    else
      echo "format_version=2"
    fi
    echo "generator=ssb-dbgen"
    echo "dbgen_commit=$DBGEN_COMMIT"
    echo "scale_factor=$sf"
    echo "dataset_id=$DATASET_ID"
    echo "physical_order=$PHYSICAL_ORDER"
    echo "source_preordered_layout=$EXPECTED_SOURCE_PREORDERED_LAYOUT"
    if [[ -n "$source_manifest_sha256" ]]; then
      echo "source_manifest_sha256=$source_manifest_sha256"
    fi
    for table in "${TABLES[@]}"; do
      local table_path="$data_dir/${table}.tbl"
      echo "${table}_rows=$(table_row_count "$table_path")"
      echo "${table}_bytes=$(file_byte_count "$table_path")"
      echo "${table}_sha256=$(file_sha256 "$table_path")"
    done
  } > "$temporary_manifest"; then
    return 1
  fi
  mv "$temporary_manifest" "$manifest" || return 1
}

validate_snapshot() {
  local sf="$1"
  local data_dir="$2"
  local expected_physical_order="$3"
  local expected_dataset_id="${4:-}"
  local manifest="${5:-$data_dir/manifest.txt}"
  [[ -d "$data_dir" && ! -L "$data_dir" ]] || return 1
  [[ -f "$manifest" && ! -L "$manifest" && -s "$manifest" ]] || return 1
  local expected_format_version=2
  local expected_preordered_layout=0
  if [[ "$expected_physical_order" == "date_pk" ]]; then
    expected_format_version=3
    expected_preordered_layout=1
  fi
  [[ "$(manifest_value "$manifest" format_version)" == \
      "$expected_format_version" ]] || return 1
  manifest_structure_is_valid \
    "$manifest" "$expected_format_version" || return 1
  [[ "$(manifest_value "$manifest" generator)" == "ssb-dbgen" ]] || return 1
  [[ "$(manifest_value "$manifest" dbgen_commit)" == "$DBGEN_COMMIT" ]] || return 1
  [[ "$(manifest_value "$manifest" scale_factor)" == "$sf" ]] || return 1
  local manifest_dataset_id
  manifest_dataset_id="$(manifest_value "$manifest" dataset_id)"
  if [[ "$expected_physical_order" == "date_pk" && \
        -n "$expected_dataset_id" ]]; then
    [[ -n "$manifest_dataset_id" && \
       "$manifest_dataset_id" == "$expected_dataset_id" ]] || return 1
  elif [[ -n "$manifest_dataset_id" && -n "$expected_dataset_id" ]]; then
    [[ "$manifest_dataset_id" == "$expected_dataset_id" ]] || return 1
  fi
  local manifest_physical_order
  manifest_physical_order="$(manifest_value "$manifest" physical_order)"
  if [[ "$expected_physical_order" == "standard" ]]; then
    # Format-2 manifests created before physical_order was recorded describe
    # the standard dbgen order and remain valid for compatibility.
    [[ -z "$manifest_physical_order" || \
       "$manifest_physical_order" == "standard" ]] || return 1
  else
    [[ "$manifest_physical_order" == "$expected_physical_order" ]] || return 1
  fi
  local manifest_preordered_layout
  manifest_preordered_layout="$(
    manifest_value "$manifest" source_preordered_layout
  )"
  if [[ "$expected_physical_order" == "date_pk" ]]; then
    [[ "$manifest_preordered_layout" == \
       "$expected_preordered_layout" ]] || return 1
  elif [[ -n "$manifest_preordered_layout" ]]; then
    [[ "$manifest_preordered_layout" == \
       "$expected_preordered_layout" ]] || return 1
  fi
  local source_manifest_sha256
  source_manifest_sha256="$(
    manifest_value "$manifest" source_manifest_sha256
  )"
  if [[ -n "$source_manifest_sha256" ]]; then
    [[ "$source_manifest_sha256" =~ ^[0-9a-f]{64}$ ]] || return 1
  fi
  for table in "${TABLES[@]}"; do
    local expected
    local observed
    local table_path="$data_dir/${table}.tbl"
    [[ -s "$table_path" && ! -L "$table_path" ]] || return 1
    expected="$(manifest_value "$manifest" "${table}_rows")"
    [[ "$expected" =~ ^[1-9][0-9]*$ ]] || return 1
    observed="$(table_row_count "$table_path")"
    [[ "$observed" == "$expected" ]] || return 1
    expected="$(manifest_value "$manifest" "${table}_bytes")"
    [[ "$expected" =~ ^[1-9][0-9]*$ ]] || return 1
    observed="$(file_byte_count "$table_path")"
    [[ "$observed" == "$expected" ]] || return 1
    expected="$(manifest_value "$manifest" "${table}_sha256")"
    [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || return 1
    observed="$(file_sha256 "$table_path")"
    [[ "$observed" == "$expected" ]] || return 1
  done
}

validate_manifest() {
  validate_snapshot "$1" "$2" "$PHYSICAL_ORDER" "$DATASET_ID"
}

sort_lineorder_range_partitions() {
  local data_dir="$1"
  local input="$data_dir/lineorder.tbl"
  local output="$data_dir/lineorder.tbl.sorted"
  local prefix="$data_dir/.lineorder.date-range"
  local bounds_file
  local min_key
  local max_key
  local partition
  local partition_path
  local sorted_partition
  local input_bytes
  local input_storage
  local bucket_bytes
  local sorted_bucket_bytes
  local partition_bytes_total=0
  local max_bucket_bytes=0
  local occupied_buckets=0
  local staging_tree_peak_base_bytes
  local all_storage_peak_base_bytes

  bounds_file="$(mktemp "$data_dir/.lineorder.bounds.XXXXXX")" || return 1
  if ! LC_ALL=C awk -F'|' '
      BEGIN { seen = 0; invalid = 0 }
      $6 !~ /^[0-9]+$/ { invalid = 1; exit }
      {
        key = $6 + 0
        if (!seen || key < min_key) min_key = key
        if (!seen || key > max_key) max_key = key
        seen = 1
      }
      END {
        if (invalid || !seen) exit 2
        printf "%.0f %.0f\n", min_key, max_key
      }
    ' "$input" > "$bounds_file"; then
    rm -f -- "$bounds_file"
    echo "LINEORDER contains an invalid DATE key." >&2
    return 1
  fi
  if ! read -r min_key max_key < "$bounds_file"; then
    rm -f -- "$bounds_file"
    return 1
  fi
  rm -f -- "$bounds_file" || return 1

  echo "Partitioning LINEORDER DATE range ${min_key}..${max_key} into ${LINEORDER_SORT_PARTITIONS} bounded sorts"
  if ! LC_ALL=C awk -F'|' \
      -v min_key="$min_key" \
      -v max_key="$max_key" \
      -v partitions="$LINEORDER_SORT_PARTITIONS" \
      -v prefix="$prefix" '
      $6 !~ /^[0-9]+$/ { exit 2 }
      {
        key = $6 + 0
        if (key < min_key || key > max_key) exit 2
        bucket = int((key - min_key) * partitions / (max_key - min_key + 1))
        if (bucket < 0) bucket = 0
        if (bucket >= partitions) bucket = partitions - 1
        path = sprintf("%s.%03d", prefix, bucket)
        print $0 > path
      }
      END {
        for (bucket = 0; bucket < partitions; bucket++) {
          close(sprintf("%s.%03d", prefix, bucket))
        }
      }
    ' "$input"; then
    echo "Unable to partition LINEORDER by its DATE key range." >&2
    return 1
  fi

  input_bytes="$(file_byte_count "$input")" || return 1
  for ((partition = 0; partition < LINEORDER_SORT_PARTITIONS; partition++)); do
    printf -v partition_path '%s.%03d' "$prefix" "$partition"
    [[ -e "$partition_path" ]] || continue
    bucket_bytes="$(file_byte_count "$partition_path")" || return 1
    occupied_buckets="$((occupied_buckets + 1))"
    partition_bytes_total="$((partition_bytes_total + bucket_bytes))"
    if (( bucket_bytes > max_bucket_bytes )); then
      max_bucket_bytes="$bucket_bytes"
    fi
  done
  if [[ "$partition_bytes_total" != "$input_bytes" ]]; then
    echo "Range partition bytes differ from dereferenced LINEORDER bytes." >&2
    return 1
  fi
  staging_tree_peak_base_bytes="$((partition_bytes_total + max_bucket_bytes))"
  if [[ -L "$input" ]]; then
    input_storage=symlink_target
  else
    input_storage=data_root_regular_file
    staging_tree_peak_base_bytes="$((
      staging_tree_peak_base_bytes + input_bytes
    ))"
  fi
  all_storage_peak_base_bytes="$((
    input_bytes + partition_bytes_total + max_bucket_bytes
  ))"
  echo "LINEORDER range-partition occupancy:" \
    "input_bytes=$input_bytes" \
    "input_storage=$input_storage" \
    "partition_bytes_total=$partition_bytes_total" \
    "occupied_buckets=$occupied_buckets/$LINEORDER_SORT_PARTITIONS" \
    "max_bucket_bytes=$max_bucket_bytes" \
    "staging_tree_peak_base_bytes=$staging_tree_peak_base_bytes" \
    "all_storage_peak_base_bytes=$all_storage_peak_base_bytes" \
    "peak_contract=original_input+partition_or_output+current_sorted_bucket+sort_temp"

  : > "$output" || return 1
  for ((partition = 0; partition < LINEORDER_SORT_PARTITIONS; partition++)); do
    printf -v partition_path '%s.%03d' "$prefix" "$partition"
    [[ -e "$partition_path" ]] || continue
    sorted_partition="${partition_path}.sorted"
    LC_ALL=C sort \
      --field-separator='|' \
      --key=6,6n --key=1,1n --key=2,2n \
      --parallel="$(processor_count)" \
      --buffer-size="$SORT_BUFFER_SIZE" \
      --temporary-directory="$data_dir" \
      "$partition_path" \
      --output="$sorted_partition" || return 1
    bucket_bytes="$(file_byte_count "$partition_path")" || return 1
    sorted_bucket_bytes="$(file_byte_count "$sorted_partition")" || return 1
    if [[ "$sorted_bucket_bytes" != "$bucket_bytes" ]]; then
      echo "Sorted range-partition bytes differ from their source bucket." >&2
      return 1
    fi
    # Once a complete byte-equivalent sorted copy exists, dropping the
    # unsorted bucket before append keeps partition-or-output bytes invariant.
    rm -f -- "$partition_path" || return 1
    cat "$sorted_partition" >> "$output" || return 1
    rm -f -- "$sorted_partition" || return 1
  done
  [[ -s "$output" ]]
}

apply_physical_order() {
  local data_dir="$1"
  [[ "$PHYSICAL_ORDER" == "date_pk" ]] || return 0

  echo "Ordering LINEORDER by date, order key, and line number"
  if (( LINEORDER_SORT_PARTITIONS > 1 )); then
    sort_lineorder_range_partitions "$data_dir" || return 1
  else
    LC_ALL=C sort \
      --field-separator='|' \
      --key=6,6n --key=1,1n --key=2,2n \
      --parallel="$(processor_count)" \
      --buffer-size="$SORT_BUFFER_SIZE" \
      --temporary-directory="$data_dir" \
      "$data_dir/lineorder.tbl" \
      --output="$data_dir/lineorder.tbl.sorted" || return 1
  fi
  mv "$data_dir/lineorder.tbl.sorted" "$data_dir/lineorder.tbl" || return 1

  for table in date customer supplier part; do
    LC_ALL=C sort \
      --field-separator='|' --key=1,1n \
      --parallel="$(processor_count)" \
      --buffer-size="$SORT_BUFFER_SIZE" \
      --temporary-directory="$data_dir" \
      "$data_dir/${table}.tbl" \
      --output="$data_dir/${table}.tbl.sorted" || return 1
    mv "$data_dir/${table}.tbl.sorted" \
      "$data_dir/${table}.tbl" || return 1
  done
}

materialize_preordered_scale() {
  local sf="$1"
  local data_dir="$2"
  [[ "$PHYSICAL_ORDER" == "date_pk" ]] || return 1
  [[ -n "$PREORDER_SOURCE_ROOT" ]] || return 1

  local source_dir="$PREORDER_SOURCE_ROOT/sf${sf}"
  local source_manifest="$source_dir/manifest.txt"
  [[ -s "$source_manifest" ]] || return 1

  # Copy into the unpublished staging directory, including the manifest
  # first. Then validate the copied bytes against that frozen manifest before
  # sorting or blessing the derived snapshot. This closes the source-copy
  # race as well as rejecting a stale or merely present source manifest.
  local copied_source_manifest="$data_dir/.source-manifest.txt"
  cp "$source_manifest" "$copied_source_manifest" || return 1
  for table in "${TABLES[@]}"; do
    cp "$source_dir/${table}.tbl" "$data_dir/${table}.tbl" || return 1
  done
  validate_snapshot \
    "$sf" "$data_dir" standard "" "$copied_source_manifest" || return 1
  local source_manifest_sha256
  source_manifest_sha256="$(file_sha256 "$copied_source_manifest")"
  rm -f "$copied_source_manifest" || return 1

  echo "Materializing date+PK SF${sf} from $source_dir"
  apply_physical_order "$data_dir" || return 1
  write_manifest "$sf" "$data_dir" "$source_manifest_sha256" || return 1
}

download_scale() {
  local sf="$1"
  local data_dir="$2"
  [[ -n "$GCS_ROOT" ]] || return 1
  gcloud storage cp \
    "$GCS_ROOT/sf${sf}/manifest.txt" "$data_dir/manifest.txt" || return 1
  for table in "${TABLES[@]}"; do
    gcloud storage cp \
      "$GCS_ROOT/sf${sf}/${table}.tbl" "$data_dir/${table}.tbl" || return 1
  done
}

generate_scale() {
  local sf="$1"
  local data_dir="$2"
  local raw_lineorder_staging_dir=""
  if [[ "$PHYSICAL_ORDER" == "date_pk" && \
        -n "$PREORDER_SOURCE_ROOT" ]]; then
    if ! materialize_preordered_scale "$sf" "$data_dir"; then
      echo "PREORDER_SOURCE_ROOT SF${sf} failed full manifest verification." >&2
      return 1
    fi
    return
  fi
  ensure_generator || return 1
  if [[ -n "$RAW_LINEORDER_STAGING_ROOT" ]]; then
    raw_lineorder_staging_dir="$(
      new_raw_lineorder_staging_dir "$sf"
    )" || return 1
    if ! register_active_staging_dir \
        raw "$sf" "$raw_lineorder_staging_dir"; then
      discard_raw_lineorder_staging_dir \
        "$sf" "$raw_lineorder_staging_dir" || true
      return 1
    fi
    if ! ln -s "$raw_lineorder_staging_dir/lineorder.tbl" \
        "$data_dir/lineorder.tbl"; then
      discard_raw_lineorder_staging_dir \
        "$sf" "$raw_lineorder_staging_dir" || true
      return 1
    fi
  fi
  if ! cp "$DBGEN_DIR/dists.dss" "$data_dir/dists.dss"; then
    if [[ -n "$raw_lineorder_staging_dir" ]]; then
      discard_raw_lineorder_staging_dir \
        "$sf" "$raw_lineorder_staging_dir" || true
    fi
    return 1
  fi
  if ! (
    cd "$data_dir"
    "$DBGEN_DIR/build/dbgen" -s "$sf" -f
  ); then
    if [[ -n "$raw_lineorder_staging_dir" ]]; then
      discard_raw_lineorder_staging_dir \
        "$sf" "$raw_lineorder_staging_dir" || true
    fi
    return 1
  fi
  if ! rm -f "$data_dir/dists.dss"; then
    if [[ -n "$raw_lineorder_staging_dir" ]]; then
      discard_raw_lineorder_staging_dir \
        "$sf" "$raw_lineorder_staging_dir" || true
    fi
    return 1
  fi
  if ! apply_physical_order "$data_dir"; then
    if [[ -n "$raw_lineorder_staging_dir" ]]; then
      discard_raw_lineorder_staging_dir \
        "$sf" "$raw_lineorder_staging_dir" || true
    fi
    return 1
  fi
  if [[ -n "$raw_lineorder_staging_dir" ]]; then
    if [[ -L "$data_dir/lineorder.tbl" || \
          ! -s "$data_dir/lineorder.tbl" ]]; then
      echo "Sorted LINEORDER did not replace its raw staging symlink." >&2
      discard_raw_lineorder_staging_dir \
        "$sf" "$raw_lineorder_staging_dir" || true
      return 1
    fi
    discard_raw_lineorder_staging_dir \
      "$sf" "$raw_lineorder_staging_dir" || return 1
  fi
  write_manifest "$sf" "$data_dir" || return 1
}

new_staging_dir() {
  local sf="$1"
  local staging_dir
  require_preparation_lock || return 1
  staging_dir="$(mktemp -d "$DATA_ROOT/.sf${sf}.incoming.XXXXXX")" || \
    return 1
  if ! write_preparation_owner_marker incoming "$sf" "$staging_dir"; then
    rm -rf -- "$staging_dir"
    return 1
  fi
  printf '%s\n' "$staging_dir"
}

new_raw_lineorder_staging_dir() {
  local sf="$1"
  local staging_dir
  require_preparation_lock || return 1
  [[ -n "$RAW_LINEORDER_STAGING_ROOT" ]] || return 1
  mkdir -p "$RAW_LINEORDER_STAGING_ROOT" || return 1
  staging_dir="$(mktemp -d \
    "${RAW_LINEORDER_STAGING_ROOT%/}/.sf${sf}.raw-lineorder.XXXXXX")" || \
    return 1
  if ! write_preparation_owner_marker raw "$sf" "$staging_dir"; then
    rm -rf -- "$staging_dir"
    return 1
  fi
  printf '%s\n' "$staging_dir"
}

discard_raw_lineorder_staging_dir() {
  local sf="$1"
  local staging_dir="$2"
  local expected_session="${3:-$PREPARATION_SESSION_ID}"
  require_preparation_lock || return 1
  discard_owned_staging_dir \
    raw "$sf" "$staging_dir" "$expected_session" || return 1
  clear_active_staging_dir raw "$staging_dir"
}

discard_staging_dir() {
  local sf="$1"
  local staging_dir="$2"
  local expected_session="${3:-$PREPARATION_SESSION_ID}"
  require_preparation_lock || return 1
  discard_owned_staging_dir \
    incoming "$sf" "$staging_dir" "$expected_session" || return 1
  clear_active_staging_dir incoming "$staging_dir"
}

recovery_candidate_kind() {
  local sf="$1"
  local candidate="$2"
  local root="${DATA_ROOT%/}"
  local basename
  [[ -n "$root" && "$root" == /* && "$root" != "/" ]] || return 1
  [[ "${candidate%/*}" == "$root" ]] || return 1
  basename="${candidate##*/}"
  if [[ "$basename" == ".sf${sf}.superseded" ]]; then
    echo superseded
    return
  fi
  if [[ "$basename" =~ ^\.sf${sf}\.superseded\.[0-9]{8}T[0-9]{6}Z\.[1-9][0-9]*$ ]]; then
    echo superseded
    return
  fi
  if [[ "$basename" =~ ^\.sf${sf}\.rollback\.[A-Za-z0-9._-]+$ ]]; then
    echo rollback
    return
  fi
  return 1
}

recovery_candidate_rank() {
  local sf="$1"
  local candidate="$2"
  local table
  recovery_candidate_kind "$sf" "$candidate" >/dev/null || return 1
  [[ -d "$candidate" && ! -L "$candidate" ]] || return 1
  if [[ -L "$candidate/manifest.txt" ]]; then
    echo 0
    return
  fi
  if validate_manifest "$sf" "$candidate"; then
    echo 2
    return
  fi
  for table in "${TABLES[@]}"; do
    [[ -s "$candidate/${table}.tbl" && \
       ! -L "$candidate/${table}.tbl" ]] || {
      echo 0
      return
    }
  done
  # A complete set of source tables can be inspected or have its manifest
  # rebuilt manually, so it is a bounded recovery point even when invalid.
  echo 1
}

recovery_candidate_generation() {
  local sf="$1"
  local candidate="$2"
  local marker="$candidate/$PREPARATION_RECOVERY_MARKER"
  local generation
  [[ -f "$marker" && ! -L "$marker" ]] || {
    echo 0
    return
  }
  [[ "$(manifest_value "$marker" format_version)" == "1" && \
     "$(manifest_value "$marker" scale_factor)" == "$sf" ]] || return 1
  generation="$(manifest_value "$marker" generation)"
  [[ "$generation" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ "$(manifest_value "$marker" session_id)" =~ \
     ^[A-Za-z0-9._-]+$ ]] || return 1
  echo "$generation"
}

next_recovery_generation() {
  local sf="$1"
  local data_dir="$2"
  local candidate
  local generation
  local max_generation=0
  while IFS= read -r -d '' candidate; do
    recovery_candidate_kind "$sf" "$candidate" >/dev/null || continue
    generation="$(recovery_candidate_generation "$sf" "$candidate")" || \
      return 1
    if (( generation > max_generation )); then
      max_generation="$generation"
    fi
  done < <(find "${DATA_ROOT%/}" -mindepth 1 -maxdepth 1 -type d -print0)
  if [[ -d "$data_dir" && ! -L "$data_dir" ]]; then
    generation="$(recovery_candidate_generation "$sf" "$data_dir")" || \
      return 1
    if (( generation > max_generation )); then
      max_generation="$generation"
    fi
  fi
  echo "$((max_generation + 1))"
}

write_recovery_candidate_marker() {
  local sf="$1"
  local candidate="$2"
  local generation="$3"
  local marker="$candidate/$PREPARATION_RECOVERY_MARKER"
  local temporary_marker
  require_preparation_lock || return 1
  [[ "$generation" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ -d "$candidate" && ! -L "$candidate" ]] || return 1
  if [[ "$candidate" != "${DATA_ROOT%/}/sf${sf}" ]]; then
    recovery_candidate_kind "$sf" "$candidate" >/dev/null || return 1
  fi
  temporary_marker="$(mktemp "$candidate/.recovery-marker.XXXXXX")" || \
    return 1
  if ! {
    echo "format_version=1"
    echo "scale_factor=$sf"
    echo "generation=$generation"
    echo "session_id=$PREPARATION_SESSION_ID"
  } > "$temporary_marker"; then
    rm -f -- "$temporary_marker"
    return 1
  fi
  mv "$temporary_marker" "$marker" || return 1
}

clear_recovery_candidate_marker() {
  local sf="$1"
  local candidate="$2"
  local marker="$candidate/$PREPARATION_RECOVERY_MARKER"
  [[ -e "$marker" || -L "$marker" ]] || return 0
  [[ ! -L "$marker" ]] || return 1
  recovery_candidate_generation "$sf" "$candidate" >/dev/null || return 1
  rm -f -- "$marker"
}

discard_recovery_candidate() {
  local sf="$1"
  local candidate="$2"
  local generation
  local rank
  require_preparation_lock || return 1
  recovery_candidate_kind "$sf" "$candidate" >/dev/null || {
    echo "Refusing to remove unexpected recovery path: $candidate" >&2
    return 1
  }
  [[ -d "$candidate" && ! -L "$candidate" ]] || return 1
  generation="$(recovery_candidate_generation "$sf" "$candidate")" || \
    return 1
  rank="$(recovery_candidate_rank "$sf" "$candidate")" || return 1
  if (( generation < 1 || rank < 1 )); then
    echo "Refusing to remove unmanaged or incomplete recovery path: $candidate" >&2
    return 1
  fi
  rm -rf -- "$candidate" || return 1
}

normalize_superseded_scale() {
  local sf="$1"
  local preferred_candidate="${2:-}"
  local fixed_backup="${DATA_ROOT%/}/.sf${sf}.superseded"
  local candidate
  local candidate_kind
  local rank
  local generation
  local best_candidate=""
  local best_kind=""
  local best_rank=-1
  local best_generation=-1
  # Bash 3.2 with nounset treats an empty-array expansion as unbound.  The
  # sentinel keeps this portable; it is skipped in the retention loop.
  local -a candidates=("")

  require_preparation_lock || return 1
  if [[ -L "$fixed_backup" ]]; then
    echo "Refusing symlink recovery path: $fixed_backup" >&2
    return 1
  fi
  if [[ -e "$fixed_backup" && ! -d "$fixed_backup" ]]; then
    echo "Refusing non-directory recovery path: $fixed_backup" >&2
    return 1
  fi

  if [[ -n "$preferred_candidate" ]]; then
    recovery_candidate_kind "$sf" "$preferred_candidate" >/dev/null || \
      return 1
  fi

  while IFS= read -r -d '' candidate; do
    candidate_kind="$(recovery_candidate_kind "$sf" "$candidate")" || \
      continue
    rank="$(recovery_candidate_rank "$sf" "$candidate")" || return 1
    generation="$(recovery_candidate_generation "$sf" "$candidate")" || \
      return 1
    if (( rank < 1 || generation < 1 )); then
      if [[ "$candidate" == "$fixed_backup" ]]; then
        echo "Unmanaged or incomplete fixed recovery path blocks mutation: $fixed_backup" >&2
        return 1
      fi
      echo "Leaving unmanaged or incomplete recovery lookalike untouched: $candidate" >&2
      continue
    fi
    candidates+=("$candidate")
    if (( rank > best_rank )) || \
        { (( rank == best_rank )) && \
          [[ "$candidate" == "$preferred_candidate" && \
             "$best_candidate" != "$preferred_candidate" ]]; } || \
        { (( rank == best_rank )) && \
          [[ "$best_candidate" != "$preferred_candidate" ]] && \
          (( generation > best_generation )); } || \
        { (( rank == best_rank && generation == best_generation )) && \
          [[ "$best_candidate" != "$preferred_candidate" && \
             "$candidate_kind" == "rollback" && \
             "$best_kind" != "rollback" ]]; } || \
        { (( rank == best_rank && generation == best_generation )) && \
          [[ "$best_candidate" != "$preferred_candidate" && \
             "$candidate_kind" == "$best_kind" && \
             "$candidate" > "$best_candidate" ]]; }; then
      best_candidate="$candidate"
      best_kind="$candidate_kind"
      best_rank="$rank"
      best_generation="$generation"
    fi
  done < <(find "${DATA_ROOT%/}" -mindepth 1 -maxdepth 1 -type d -print0)

  if (( best_rank < 1 )); then
    best_candidate=""
  elif [[ "$best_candidate" != "$fixed_backup" ]]; then
    if [[ -e "$fixed_backup" || -L "$fixed_backup" ]]; then
      discard_recovery_candidate "$sf" "$fixed_backup" || return 1
    fi
    mv "$best_candidate" "$fixed_backup" || return 1
    best_candidate="$fixed_backup"
  fi

  for candidate in "${candidates[@]}"; do
    [[ -n "$candidate" ]] || continue
    [[ "$candidate" == "$best_candidate" ]] && continue
    [[ -e "$candidate" || -L "$candidate" ]] || continue
    discard_recovery_candidate "$sf" "$candidate" || return 1
  done
}

recover_interrupted_scale() {
  local sf="$1"
  local data_dir="$2"
  local fixed_backup="${DATA_ROOT%/}/.sf${sf}.superseded"
  local fixed_generation
  local fixed_rank
  require_preparation_lock || return 1
  [[ "$data_dir" == "${DATA_ROOT%/}/sf${sf}" ]] || {
    echo "Refusing recovery outside the canonical SF${sf} destination: $data_dir" >&2
    return 1
  }
  if [[ -L "$data_dir" || -L "$fixed_backup" ]]; then
    echo "Refusing symlink canonical or recovery destination for SF${sf}." >&2
    return 1
  fi
  if [[ -e "$data_dir" && ! -d "$data_dir" ]]; then
    echo "Refusing non-directory canonical destination for SF${sf}." >&2
    return 1
  fi
  normalize_superseded_scale "$sf" || return 1
  if [[ ! -e "$data_dir" && ! -L "$data_dir" && \
        -d "$fixed_backup" ]]; then
    fixed_generation="$(
      recovery_candidate_generation "$sf" "$fixed_backup"
    )" || return 1
    fixed_rank="$(recovery_candidate_rank "$sf" "$fixed_backup")" || \
      return 1
    if (( fixed_generation > 0 && fixed_rank > 0 )); then
      echo "Restoring interrupted SF${sf} destination from $fixed_backup"
      mv "$fixed_backup" "$data_dir" || return 1
    else
      echo "Leaving unmanaged recovery lookalike untouched: $fixed_backup" >&2
    fi
  fi
  if [[ -d "$data_dir" && ! -L "$data_dir" ]]; then
    clear_recovery_candidate_marker "$sf" "$data_dir" || return 1
  fi
}

clear_published_owner_marker() {
  local sf="$1"
  local data_dir="$2"
  local expected_session="$3"
  local marker="$data_dir/$PREPARATION_OWNER_MARKER"
  local recorded_basename
  [[ -f "$marker" && ! -L "$marker" ]] || return 0
  [[ "$(manifest_value "$marker" format_version)" == "1" && \
     "$(manifest_value "$marker" kind)" == "incoming" && \
     "$(manifest_value "$marker" scale_factor)" == "$sf" && \
     "$(manifest_value "$marker" session_id)" == "$expected_session" ]] || \
    return 1
  recorded_basename="$(manifest_value "$marker" directory_basename)"
  [[ "$recorded_basename" =~ ^\.sf${sf}\.incoming\.[A-Za-z0-9]+$ ]] || \
    return 1
  rm -f -- "$marker"
}

resolve_interrupted_rollback() {
  local sf="$1"
  local rollback_dir="$2"
  local data_dir="$3"
  require_preparation_lock || return 1
  [[ "$data_dir" == "${DATA_ROOT%/}/sf${sf}" && ! -L "$data_dir" ]] || \
    return 1
  recovery_candidate_kind "$sf" "$rollback_dir" >/dev/null || return 1
  if [[ ! -e "$data_dir" && ! -L "$data_dir" ]]; then
    echo "Restoring interrupted SF${sf} promotion from $rollback_dir" >&2
    mv "$rollback_dir" "$data_dir" || return 1
  else
    normalize_superseded_scale "$sf" "$rollback_dir" || return 1
  fi
  clear_recovery_candidate_marker "$sf" "$data_dir" || return 1
  clear_published_owner_marker \
    "$sf" "$data_dir" "$PREPARATION_SESSION_ID" || return 1
}

install_prepared_scale() {
  local sf="$1"
  local staging_dir="$2"
  local data_dir="$3"
  require_preparation_lock || return 1
  [[ "$data_dir" == "${DATA_ROOT%/}/sf${sf}" && ! -L "$data_dir" ]] || {
    echo "Refusing install outside a non-symlink canonical SF${sf} destination." >&2
    return 1
  }
  owned_staging_marker_matches \
    incoming "$sf" "$staging_dir" "$PREPARATION_SESSION_ID" || {
    echo "Refusing install from an unowned or unexpected staging path." >&2
    return 1
  }
  if ! validate_manifest "$sf" "$staging_dir"; then
    echo "Refusing to install unverified SF${sf} staging data." >&2
    return 1
  fi

  if [[ -e "$data_dir" || -L "$data_dir" ]]; then
    local rollback_dir
    local recovery_generation
    normalize_superseded_scale "$sf" || return 1
    rollback_dir="${DATA_ROOT%/}/.sf${sf}.rollback.${PREPARATION_SESSION_ID}"
    recovery_candidate_kind "$sf" "$rollback_dir" >/dev/null || return 1
    [[ ! -e "$rollback_dir" && ! -L "$rollback_dir" ]] || return 1
    recovery_generation="$(next_recovery_generation "$sf" "$data_dir")" || \
      return 1
    PREPARATION_ACTIVE_ROLLBACK="$rollback_dir"
    PREPARATION_ACTIVE_ROLLBACK_SF="$sf"
    PREPARATION_ACTIVE_ROLLBACK_DESTINATION="$data_dir"
    write_recovery_candidate_marker \
      "$sf" "$data_dir" "$recovery_generation" || return 1
    mv "$data_dir" "$rollback_dir" || return 1
    if ! mv "$staging_dir" "$data_dir"; then
      mv "$rollback_dir" "$data_dir" || true
      clear_recovery_candidate_marker "$sf" "$data_dir" || true
      PREPARATION_ACTIVE_ROLLBACK=""
      PREPARATION_ACTIVE_ROLLBACK_SF=""
      PREPARATION_ACTIVE_ROLLBACK_DESTINATION=""
      echo "Unable to atomically install verified SF${sf}." >&2
      return 1
    fi
    clear_published_owner_marker \
      "$sf" "$data_dir" "$PREPARATION_SESSION_ID" || return 1
    clear_active_staging_dir incoming "$staging_dir"
    normalize_superseded_scale "$sf" "$rollback_dir" || return 1
    PREPARATION_ACTIVE_ROLLBACK=""
    PREPARATION_ACTIVE_ROLLBACK_SF=""
    PREPARATION_ACTIVE_ROLLBACK_DESTINATION=""
    if [[ -d "${DATA_ROOT%/}/.sf${sf}.superseded" ]]; then
      echo "Retained one recoverable previous SF${sf} at ${DATA_ROOT%/}/.sf${sf}.superseded"
    fi
  else
    mv "$staging_dir" "$data_dir" || return 1
    clear_published_owner_marker \
      "$sf" "$data_dir" "$PREPARATION_SESSION_ID" || return 1
    clear_active_staging_dir incoming "$staging_dir"
  fi
}

prepare_scale() {
  local sf="$1"
  local data_dir="$DATA_ROOT/sf${sf}"
  local generated=0

  require_preparation_lock || return 1

  recover_interrupted_scale "$sf" "$data_dir" || return 1

  if validate_manifest "$sf" "$data_dir"; then
    echo "Reusing verified ${DATASET_ID} SF${sf} at $data_dir"
  else
    local staging_dir=""
    local prepared=0
    if [[ -n "$GCS_ROOT" ]]; then
      echo "Fetching ${DATASET_ID} SF${sf} from $GCS_ROOT"
      staging_dir="$(new_staging_dir "$sf")" || return 1
      if ! register_active_staging_dir incoming "$sf" "$staging_dir"; then
        discard_staging_dir "$sf" "$staging_dir" || true
        return 1
      fi
      if download_scale "$sf" "$staging_dir" && \
          install_prepared_scale "$sf" "$staging_dir" "$data_dir"; then
        prepared=1
        staging_dir=""
      else
        echo "Downloaded SF${sf} failed complete manifest verification." >&2
        if [[ -d "$staging_dir" ]]; then
          discard_staging_dir "$sf" "$staging_dir"
        fi
        staging_dir=""
      fi
    else
      echo "No GCS_ROOT configured for ${DATASET_ID} SF${sf}"
    fi
    if [[ "$prepared" != "1" ]]; then
      if [[ "$ALLOW_DBGEN_FALLBACK" != "1" ]]; then
        echo "Unable to obtain a verified SF${sf} dataset" >&2
        echo "Set ALLOW_DBGEN_FALLBACK=1 to regenerate the pinned dataset." >&2
        return 1
      fi
      staging_dir="$(new_staging_dir "$sf")" || return 1
      if ! register_active_staging_dir incoming "$sf" "$staging_dir"; then
        discard_staging_dir "$sf" "$staging_dir" || true
        return 1
      fi
      echo "Preparing ${PHYSICAL_ORDER} SSB SF${sf} at pinned commit $DBGEN_COMMIT"
      if ! generate_scale "$sf" "$staging_dir" || \
          ! install_prepared_scale "$sf" "$staging_dir" "$data_dir"; then
        echo "Generated SF${sf} failed complete manifest verification." >&2
        if [[ -d "$staging_dir" ]]; then
          discard_staging_dir "$sf" "$staging_dir"
        fi
        return 1
      fi
      staging_dir=""
      generated=1
    fi
  fi

  if [[ "$PUBLISH_GENERATED_DATA" == "1" && "$generated" == "1" ]]; then
    if [[ -z "$GCS_ROOT" ]]; then
      echo "PUBLISH_GENERATED_DATA=1 requires GCS_ROOT." >&2
      return 1
    fi
    for table in "${TABLES[@]}"; do
      gcloud storage cp \
        "$data_dir/${table}.tbl" "$GCS_ROOT/sf${sf}/${table}.tbl"
    done
    # Publish last: a manifest means the complete dataset is available.
    gcloud storage cp \
      "$data_dir/manifest.txt" "$GCS_ROOT/sf${sf}/manifest.txt"
  fi
  for table in "${TABLES[@]}"; do
    printf "%12s %s\n" \
      "$(manifest_value "$data_dir/manifest.txt" "${table}_rows")" \
      "$data_dir/${table}.tbl"
  done
}

main() {
  mkdir -p "$DATA_ROOT"
  install_preparation_cleanup_traps
  acquire_preparation_lock
  cleanup_stale_owned_staging
  for sf in $SCALE_FACTORS; do
    prepare_scale "$sf"
  done
  echo "Verified frozen datasets under $DATA_ROOT"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
