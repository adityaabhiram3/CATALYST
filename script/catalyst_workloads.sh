#!/bin/bash
# Tree workload definitions for CATALYST, sourced by sweep_catalyst*.sh.
#
# Each workload sets the newbench knobs for one row of the paper's evaluation
# (Sec. 6 / Table 1) and names the pattern from Sec. 3.4 it is meant to exercise.
# Kept in one file so node 0 and node 1 provably run the same matrix -- the two
# processes must agree on every parameter or they will not meet at the barrier.
#
# Insert-heavy rows are deliberately absent: for INDEX=3 an insert that splits a
# leaf still falls back to the DEX path, so those numbers would measure the
# fallback rather than CATALYST. They come back with server-side SMO.

# workload_<name> sets: READ INSERT UPDATE DELETE RANGE UNIFORM ZIPF SCAN_LEN
# and EXPECT, the pattern that should win if Sec. 3.4 is right.

workload_point_zipf() {          # Case 1a: skew narrows reuse to a deep band
  READ=100 INSERT=0 UPDATE=0 DELETE=0 RANGE=0
  UNIFORM=0 ZIPF=0.99 SCAN_LEN=100
  EXPECT=funnel
}

workload_point_uniform() {       # Case 1b: reuse spreads, forcing wider bands
  READ=100 INSERT=0 UPDATE=0 DELETE=0 RANGE=0
  UNIFORM=1 ZIPF=0.99 SCAN_LEN=100
  EXPECT=funnel
}

workload_read_intensive() {      # Table 1: 95/5 read/update
  READ=95 INSERT=0 UPDATE=5 DELETE=0 RANGE=0
  UNIFORM=0 ZIPF=0.99 SCAN_LEN=100
  EXPECT=funnel
}

workload_write_intensive() {     # Table 1: 50/50 read/update, no SMO
  READ=50 INSERT=0 UPDATE=50 DELETE=0 RANGE=0
  UNIFORM=0 ZIPF=0.99 SCAN_LEN=100
  EXPECT=funnel
}

workload_range_zipf() {          # Case 2a: neighbouring windows overlap
  READ=0 INSERT=0 UPDATE=0 DELETE=0 RANGE=100
  UNIFORM=0 ZIPF=0.99 SCAN_LEN=100
  EXPECT=interval
}

workload_range_uniform() {
  READ=0 INSERT=0 UPDATE=0 DELETE=0 RANGE=100
  UNIFORM=1 ZIPF=0.99 SCAN_LEN=100
  EXPECT=interval
}

workload_scan_zipf() {           # Case 2b: value migrates to the leaf frontier
  READ=0 INSERT=0 UPDATE=0 DELETE=0 RANGE=100
  UNIFORM=0 ZIPF=0.99 SCAN_LEN=10000
  EXPECT=interval
}

workload_scan_uniform() {
  READ=0 INSERT=0 UPDATE=0 DELETE=0 RANGE=100
  UNIFORM=1 ZIPF=0.99 SCAN_LEN=10000
  EXPECT=interval
}

workload_mixed() {               # Sec. 6.4's non-insert mix: 60/30/10
  READ=60 INSERT=0 UPDATE=0 DELETE=0 RANGE=40
  UNIFORM=0 ZIPF=0.99 SCAN_LEN=100
  EXPECT=auto
}

ALL_WORKLOADS="point_zipf point_uniform read_intensive write_intensive \
range_zipf range_uniform scan_zipf scan_uniform mixed"

# funnel/interval/branch are the Sec. 5 patterns; auto lets the control loop
# choose; model0 disables the loop entirely and uses the fixed Sec. 4 envelope,
# which is the baseline for "does the control loop earn its place".
ALL_POLICIES="funnel interval branch auto model0"

# Translate a policy name into the two environment variables that set it.
policy_env() {
  case "$1" in
    model0) echo "MODEL=0 PATTERN=funnel" ;;
    *)      echo "MODEL=1 PATTERN=$1" ;;
  esac
}
