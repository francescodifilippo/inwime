#!/bin/sh
# Runs inside the experiment scope: execute the worker, then record the scope's
# kernel counters before the scope disappears (also after an OOM kill).
out="$1"
shift
"$@"
rc=$?
cg="/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)"
{
  echo "== cgroup.path"; echo "$cg"
  echo "== worker.exit"; echo "$rc"
  for f in memory.peak memory.current memory.max memory.swap.max memory.swap.current \
           memory.swap.peak memory.events memory.events.local memory.swap.events memory.stat; do
    echo "== $f"; cat "$cg/$f"
  done
} > "$out"
exit "$rc"
