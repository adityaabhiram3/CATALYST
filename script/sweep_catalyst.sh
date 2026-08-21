#!/bin/bash
# Sweep CATALYST tree workloads x pattern policies -- run on NODE 0.
# Run sweep_catalyst_other.sh on the other machine at the same time.
#
#   cp ../script/run_catalyst*.sh ../script/sweep_catalyst*.sh \
#      ../script/catalyst_workloads.sh .
#   ./sweep_catalyst.sh
#
# Both nodes iterate the same matrix in the same order, and every iteration is a
# complete newbench run with its own barriers, so the two stay in lockstep as
# long as neither run is skipped. If one node dies mid-sweep, stop both and
# restart -- a half-finished run leaves memcached barrier keys behind, which is
# what wedges the next startup (run_catalyst.sh flushes them for this reason).
#
#   WORKLOADS="point_zipf scan_zipf" ./sweep_catalyst.sh   # subset
#   POLICIES="funnel model0" ./sweep_catalyst.sh           # subset
#   BULK=50 RUNNUM=50 ./sweep_catalyst.sh                  # larger runs

set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "${HERE}/catalyst_workloads.sh"

WORKLOADS=${WORKLOADS:-$ALL_WORKLOADS}
POLICIES=${POLICIES:-$ALL_POLICIES}
OUTDIR=${OUTDIR:-sweep-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUTDIR"

# NOTE: the p99 column stays "-" unless LATENCY is enabled. newbench.cpp line 28
# has `// #define LATENCY 1` commented out, so no percentile is ever printed.
# Uncomment it and rebuild if you need the tail-latency numbers of Fig. 12.
SUMMARY="${OUTDIR}/summary.tsv"
printf 'workload\tpolicy\texpect\tthroughput_mops\tp99_us\thit_pct\tadmit\treject\tswitches\tsmo_fallback\tcursor_kb\n' > "$SUMMARY"

total=0
for w in $WORKLOADS; do for p in $POLICIES; do total=$((total+1)); done; done
echo "=== sweep: ${total} runs -> ${OUTDIR}/"

n=0
for w in $WORKLOADS; do
  if ! declare -f "workload_${w}" > /dev/null; then
    echo "!!! unknown workload '${w}', skipping"; continue
  fi
  "workload_${w}"                 # sets READ/INSERT/.../SCAN_LEN/EXPECT
  for pol in $POLICIES; do
    n=$((n+1))
    log="${OUTDIR}/${w}__${pol}.log"
    echo "--- [${n}/${total}] ${w} / ${pol}"

    env $(policy_env "$pol") \
        INDEX=3 \
        READ=$READ INSERT=$INSERT UPDATE=$UPDATE DELETE=$DELETE RANGE=$RANGE \
        UNIFORM=$UNIFORM ZIPF=$ZIPF SCAN_LEN=$SCAN_LEN \
        "${HERE}/run_catalyst.sh" > "$log" 2>&1

    # Pull the headline numbers out of the log. Missing fields become "-" so a
    # crashed run still produces a row rather than silently vanishing.
    tp=$(grep -oE 'cluster throughput[^0-9]*[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+$')
    p99=$(grep -oE 'p99[^0-9]*[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+$')
    stat=$(grep -m1 '^CATALYST: probes=' "$log")
    hit=$(echo "$stat"  | grep -oE 'hit=[0-9.]+' | cut -d= -f2)
    smo=$(echo "$stat"  | grep -oE 'smo_fallback=[0-9]+' | cut -d= -f2)
    theta=$(grep -m1 '^CATALYST: pattern=' "$log")
    adm=$(echo "$theta" | grep -oE 'admit=[0-9]+' | cut -d= -f2)
    rej=$(echo "$theta" | grep -oE 'reject=[0-9]+' | cut -d= -f2)
    sw=$(echo "$theta"  | grep -oE 'switches=[0-9]+' | cut -d= -f2)
    kb=$(grep -m1 'cursor table' "$log" | grep -oE '[0-9]+ KB' | grep -oE '[0-9]+')

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$w" "$pol" "$EXPECT" "${tp:--}" "${p99:--}" "${hit:--}" \
      "${adm:--}" "${rej:--}" "${sw:--}" "${smo:--}" "${kb:--}" >> "$SUMMARY"

    sleep 3   # let the previous run's QPs drain before the next barrier
  done
done

echo
echo "=== summary: ${SUMMARY}"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
