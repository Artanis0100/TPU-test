#!/usr/bin/env bash
# Disconnect-safe launcher. Only small console/status files are persisted.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
python_path="${SECTILE_PYTHON:-$HOME/test/sectile-env/bin/python}"
for tool in nohup flock mktemp; do
    command -v "$tool" >/dev/null || { printf 'Required command missing: %s\n' "$tool" >&2; exit 1; }
done
[[ -x "$python_path" ]] || { printf 'Existing Python environment not found: %s\n' "$python_path" >&2; exit 1; }
[[ -f "$script_dir/run_benchmark.py" ]] || { printf 'run_benchmark.py is missing.\n' >&2; exit 1; }

# The background wrapper inherits descriptor 9 and retains the lock while
# waiting for Python. The kernel releases it even if the processes crash.
exec 9>"$script_dir/.benchmark.lock"
flock -n 9 || { printf 'A background experiment in this folder is still running.\n' >&2; exit 1; }

results="$script_dir/results"
mkdir -p -- "$results"
if [[ -e "$results/latest" && ! -L "$results/latest" ]]; then
    printf 'Refusing to replace the existing non-symlink results/latest.\n' >&2
    exit 1
fi
run_dir="$(mktemp -d "$results/run-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
ln -sfn -- "${run_dir##*/}" "$results/latest"

# No activation or terminal input is required. A wrapper (not exec Python)
# records the exit status on both success and ordinary benchmark failure.
nohup bash -c '
    exit_file=$1
    shift
    status=0
    "$@" || status=$?
    printf "%s\n" "$status" > "$exit_file"
    exit "$status"
' sectile-background "$run_dir/exitcode" \
    "$python_path" -u -B "$script_dir/run_benchmark.py" "$@" \
    < /dev/null > "$run_dir/result.txt" 2>&1 &
wrapper_pid=$!
printf '%s\n' "$wrapper_pid" > "$run_dir/pid"
printf 'Started background wrapper PID %s\nResults: %s/result.txt\n' "$wrapper_pid" "$run_dir"
printf 'View: tail -f "%s/latest/result.txt"\n' "$results"
printf 'Completion: cat "%s/latest/exitcode" (0 = success)\n' "$results"
