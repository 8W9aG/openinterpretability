#!/usr/bin/env bash
# Run a command under a memory cap, and report its peak memory (#4592).
#
#   scripts/capped.sh CAP WARN -- COMMAND [ARG...]
#
# CAP and WARN are sizes such as 13G. CAP may be "infinity", which only measures: no
# cap and no swap limit. CAPPED_SWAP (default 0) is the swap that the command may use
# on top of CAP. Otherwise the command runs in a transient systemd scope
# with MemoryMax=CAP and MemorySwapMax=CAPPED_SWAP. Then the kernel kills only the command when it goes
# above the cap, not the GitHub runner agent. A killed agent reports nothing, and the
# job waits about 45 minutes before it fails with no reason in its log.
#
# The scope's cgroup counts the command and every process that it forks, so the
# peak is the total of the build and its workers. After the command ends, the script:
# - prints the peak, also to the step summary on GitHub;
# - writes a warning when the peak is above WARN;
# - writes an error and exits 137 when the kernel killed a process for memory. A killed
#   worker usually makes the build exit 1 (BrokenProcessPool), so the script reads the
#   cgroup's oom_kill count, not the exit code.
#
# Without systemd-run or passwordless sudo (macOS, a container), the command runs
# with no cap and no report.
set -uo pipefail

if [ $# -lt 4 ] || [ "$3" != -- ]; then
  echo "usage: $0 CAP WARN -- COMMAND [ARG...]" >&2
  exit 2
fi
cap=$1 warn=$2
shift 3

if ! command -v systemd-run >/dev/null 2>&1 || ! sudo -n true 2>/dev/null; then
  echo "capped.sh: no systemd-run or sudo here, so the command runs with no memory cap." >&2
  exec "$@"
fi

report=$(mktemp)
# Runs inside the scope: the command, then the scope's own cgroup counters. The single
# quotes are on purpose: the inner shell expands these, not this one.
# shellcheck disable=SC2016
inner='
"$@"
status=$?
cgroup=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
peak=$(cat "$cgroup/memory.peak" 2>/dev/null || echo 0)
swap=$(cat "$cgroup/memory.swap.peak" 2>/dev/null || echo 0)
kills=$(awk "/^oom_kill /{print \$2}" "$cgroup/memory.events" 2>/dev/null)
echo "$status $peak ${kills:-0} $swap" > "$CAPPED_REPORT"
exit "$status"'

limits=(-p MemoryMax="$cap" -p MemorySwapMax="${CAPPED_SWAP:-0}")
[ "$cap" = infinity ] && limits=()
sudo --preserve-env systemd-run --scope --quiet "${limits[@]}" \
  --uid "$(id -u)" --gid "$(id -g)" \
  env "PATH=$PATH" "HOME=$HOME" "CAPPED_REPORT=$report" \
  bash -c "$inner" capped "$@"
status=$?

read -r inner_status peak kills swap < "$report" 2>/dev/null || true
rm -f "$report"
gib() { awk -v b="$1" 'BEGIN { printf "%.1f GiB", b / 1073741824 }'; }
summary() { [ -n "${GITHUB_STEP_SUMMARY:-}" ] && echo "$1" >> "$GITHUB_STEP_SUMMARY"; }

if [ -z "${peak:-}" ]; then
  # The report is missing: the scope went above its cap, and systemd stopped all of it,
  # the inner shell too. A scope stops on an out-of-memory kill (OOMPolicy=stop).
  echo "::error::The command ran out of memory: it went above its $cap cap, and systemd stopped it (exit $status)."
  summary "**Out of memory:** the command went above its $cap cap."
  exit 137
fi

line="Peak memory: $(gib "$peak") (the command and its workers), cap $cap."
[ "${swap:-0}" -gt 0 ] && line="$line Peak swap: $(gib "$swap")."
echo "$line"
summary "$line"
if [ "${kills:-0}" -gt 0 ]; then
  echo "::error::The command ran out of memory: it went above its $cap cap, and the kernel killed $kills process(es)."
  summary "**Out of memory:** the command went above its $cap cap."
  exit 137
fi
if [ "$peak" -gt "$(numfmt --from=iec "$warn")" ]; then
  echo "::warning::Peak memory $(gib "$peak") is above $warn, close to the $cap cap."
fi
exit "${inner_status:-$status}"
