#!/bin/sh
# Increment-10 campaign steps, one invocation per step, never repeated automatically.
#
#   tools/campaign10_sequence.sh STEP [STEP...]
#
# Steps: R1 R2 R3 R4 F-sync F-pipeline N6 O14-stock_fp32 O14-full64 O14-stock_fp64 integrity;
# later attempts after a DEFER or a fixed probe: F-<mode>-aN, O14-<contract>-aN, O7-stock_fp64-aN
#
# Before each step the host is polled read-only (tools/hostmem10.py, profile of the step,
# campaign baseline) every WAIT_S seconds, at most WAIT_MAX_S: this is the preregistered
# DEFER = postpone. Then tools/campaign10.py performs the official start check; a step
# whose official check is not GO writes blocked-<run>.json and is NOT executed. A step
# that already has a record (run or blocked file) is refused by campaign10 itself.
# Nothing on the host or in WSL is changed (no drop_caches, sysctl, swap, .wslconfig).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 1
C=results/increment10/campaign
A=artifacts/increment10
CK=checkpoints/qwen2.5-14b-97e1e76
BASE=$C/hostmem-baseline.json
PY=${PY:-python3}
TOKPY=${TOKPY:?TOKPY must name a Python with tokenizers==0.22.1 (optional text interface)}
ORACLEPY=${ORACLEPY:-$ROOT/.venv-oracle7/bin/python}
WAIT_S=${WAIT_S:-60}
WAIT_MAX_S=${WAIT_MAX_S:-1800}
SEQ=785,6722,315,9625,374,12095,13,576
SHARDS=""
for s in "$ROOT/$CK"/model-0000?-of-00008.safetensors; do SHARDS="$SHARDS --file $s"; done
mkdir -p "$C" "$A"

wait_go() {  # profile
  waited=0
  while :; do
    out=$($PY -m tools.hostmem10 --checkpoint-volume "$CK" --backing-volume /mnt/c --profile "$1" \
          --interval 30 --baseline "$BASE" 2>/dev/null | $PY -c 'import json,sys; print(json.load(sys.stdin)["verdict"])')
    echo "$(date -u +%FT%TZ) host $1: $out (waited ${waited}s)" >> "$C/host-wait.log"
    [ "$out" = GO ] && return 0
    [ "$waited" -ge "$WAIT_MAX_S" ] && return 0      # campaign10 records the official DEFER as blocked-*.json
    sleep "$WAIT_S"; waited=$((waited + WAIT_S))
  done
}

runtime() {  # run mode as_mib cache condition capture purpose
  run=$1 mode=$2 mib=$3 cache=$4 cond=$5 cap=$6 purpose=$7
  capture=""; [ "$cap" = full ] && capture="--capture $A/$run-capture"
  files=""; [ "$cond" = evicted ] && files=$SHARDS
  state=evicted-under-scope; [ "$cond" = as-is ] && state=warm-as-is-after-cold-run
  wait_go runtime
  # shellcheck disable=SC2086
  $PY -m tools.campaign10 --run "$run" --label "m10-$run" --kind runtime --profile runtime --cache "$cache" \
    --purpose "$purpose" --condition "$cond" --memory-max 2147483648 $files --output-dir "$C" --baseline "$BASE" \
    --checkpoint-volume "$CK" --produces "$C/$run-forward.json" -- \
    "$TOKPY" -m tools.benchmark_increment9 "$CK" --family qwen2 --mode "$mode" --as-mib "$mib" \
    --output "$C/$run-forward.json" $capture --cache-state "$state"
}

oracle() {  # contract [attempt suffix, e.g. a2]
  run=O14-$1${2:+-$2}
  wait_go oracle
  $PY -m tools.campaign10 --run "$run" --label "m10-$run" --kind oracle --profile oracle --cache as-is \
    --purpose "oracle10 14B ($1) for parity with R1" --condition as-is --memory-max 6442450944 \
    --output-dir "$C" --baseline "$BASE" --checkpoint-volume "$CK" \
    --produces "$A/oracle10-14b-$1.npz" --produces "$A/oracle10-14b-$1.json" -- \
    "$ORACLEPY" -m tools.oracle10 "$CK" --contract "$1" --sequence $SEQ --prompt-length 5 --chunks 5,1,1,1 \
    --memory-budget 5368709120 --scope-memory-max 6442450944 --threads 2 --output "$A/oracle10-14b-$1.npz"
}

for step in "$@"; do
  echo "$(date -u +%FT%TZ) step $step start" >> "$C/sequence.log"
  case $step in
    R1) runtime R1 sync 256 C evicted full "primary: cold guest, sync, full captures" ;;
    R2) runtime R2 pipeline 384 C evicted digest "cold guest, pipeline, byte identity with R1" ;;
    R3) runtime R3 sync 256 C evicted digest "cold guest, sync, repetition" ;;
    R4) runtime R4 sync 256 W as-is digest "warm as-is right after R3, informative only" ;;
    F-sync|F-pipeline|F-sync-a*|F-pipeline-a*)
      mode=$(echo "$step" | cut -d- -f2); wait_go runtime
      $PY -m tools.campaign10 --run "$step" --label "m10-$step" --kind fault --profile runtime --cache as-is \
        --purpose "faults N1-N4 ($mode)" --condition as-is --memory-max 2147483648 --output-dir "$C" \
        --baseline "$BASE" --checkpoint-volume "$CK" --produces "$C/$step-faults.json" -- \
        "$PY" -m tools.fault10 "$CK" --mode "$mode" --output "$C/$step-faults.json" ;;
    N6)
      wait_go runtime
      $PY -m tools.campaign10 --run N6 --label m10-N6 --kind fault --profile runtime --cache as-is \
        --purpose "N6: planned OOM of SmolLM2-1.7B in a 32 MiB scope (expected kernel OOM kill, FAIL expected)" \
        --condition as-is --memory-max 33554432 --output-dir "$C" --baseline "$BASE" \
        --checkpoint-volume checkpoints/smollm2-1.7b-d4c18cc -- \
        "$TOKPY" -m tools.benchmark_increment9 checkpoints/smollm2-1.7b-d4c18cc --family llama --mode sync \
        --as-mib 256 --output "$C/N6-forward.json" --cache-state as-is ;;
    O14-*-a*) c=${step#O14-}; oracle "${c%-a*}" "a${c##*-a}" ;;
    O14-*) oracle "${step#O14-}" ;;
    O7-stock_fp64-a*)
      wait_go oracle
      $PY -m tools.campaign10 --run "$step" --label "m10-$step" --kind oracle --profile oracle --cache as-is \
        --purpose "oracle10 7B end-to-end equivalence (stock_fp64, diagnostic) vs M9 oracle9 capture" \
        --condition as-is --memory-max 6442450944 --output-dir "$C" --baseline "$BASE" \
        --checkpoint-volume checkpoints/qwen2.5-7b-d149729 \
        --produces "$A/oracle10-7b-stock_fp64.npz" --produces "$A/oracle10-7b-stock_fp64.json" -- \
        "$ORACLEPY" -m tools.oracle10 checkpoints/qwen2.5-7b-d149729 --contract stock_fp64 \
        --sequence 785,6722,315,9625,374,12095,13,1084 --prompt-length 5 --chunks 8 \
        --memory-budget 5368709120 --scope-memory-max 6442450944 --threads 2 \
        --output "$A/oracle10-7b-stock_fp64.npz" ;;
    integrity)
      $PY -m tools.integrity9 --output "$C/integrity-final.json" integrity "$CK" ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
  esac
  echo "$(date -u +%FT%TZ) step $step exit $?" >> "$C/sequence.log"
done
