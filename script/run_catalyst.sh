#!/bin/bash
# CATALYST / DEX benchmark driver -- run this on NODE 0.
# Run script/run_catalyst_other.sh on the other machine right after.
#
# Copy into the build directory alongside newbench and restartMemc.sh:
#   cp ../script/run_catalyst*.sh .
#
# Every knob is an environment variable, so a DEX-vs-CATALYST A/B is just:
#   INDEX=0 ./run_catalyst.sh     # DEX, path cache
#   INDEX=3 ./run_catalyst.sh     # CATALYST, cursor table
# Both build the same remote tree with the same partitioning, so the only
# difference is how the compute side reaches a leaf.

# ---- index under test -------------------------------------------------------
# 0 = DEX (path caching)   1 = Sherman   2 = SMART   3 = CATALYST (cursors)
INDEX=${INDEX:-3}

# ---- cluster shape ----------------------------------------------------------
# Each machine is both a compute server and a memory server, as in DEX's setup.
NODES=${NODES:-2}              # machineNR: number of physical machines
KMAXTHREAD=${KMAXTHREAD:-36}   # worker threads per compute node (<= MAX_APP_THREAD)
# CNodeCount = ceil(THREADS / KMAXTHREAD). To make BOTH machines act as compute
# nodes you need THREADS > KMAXTHREAD; at THREADS <= KMAXTHREAD only node 0
# runs the workload and node 1 serves memory only.
THREADS=${THREADS:-72}         # total worker threads across the cluster
MEMTHREADS=${MEMTHREADS:-4}    # directory threads per memory node

# ---- workload ---------------------------------------------------------------
# read insert update delete range, must sum to 100.
# Insert is NOT production-ready for INDEX=3 yet (it falls back to the DEX split
# path); stick to read/update mixes until server-side SMO lands.
READ=${READ:-100}; INSERT=${INSERT:-0}; UPDATE=${UPDATE:-0}
DELETE=${DELETE:-0}; RANGE=${RANGE:-0}

UNIFORM=${UNIFORM:-0}          # 0 = zipfian, 1 = uniform
ZIPF=${ZIPF:-0.99}

# Millions of keys / operations. Defaults are a fast smoke configuration --
# bulk load is single-threaded on node 0, so 50M takes a while. For numbers
# comparable to the paper use BULK=200 WARMUP=10 RUNNUM=200.
BULK=${BULK:-5}
WARMUP=${WARMUP:-1}
RUNNUM=${RUNNUM:-5}

# ---- misc -------------------------------------------------------------------
# argv[15]; inert unless CHECK_CORRECTNESS is #defined in newbench.cpp.
CORRECT=${CORRECT:-0}
TIMEBASE=${TIMEBASE:-1}
EARLY=${EARLY:-1}
# argv[19]/argv[20]: DEX's pushdown and admission rates. CATALYST ignores both
# -- it has no admission probability and always uses two-sided RPC.
RPC=${RPC:-1}
ADMIT=${ADMIT:-0.1}
TUNE=${TUNE:-0}
# argv[9]. For INDEX=3 this sizes the DEX *fallback* cache used for bulk load
# and leaf splits -- it is NOT CATALYST's cache. The cursor table is fixed at
# ~1.9MB and prints its real footprint at startup.
CACHE=${CACHE:-256}

echo "=== index=$INDEX nodes=$NODES threads=$THREADS (kMaxThread=$KMAXTHREAD)"
echo "=== compute nodes = ceil($THREADS/$KMAXTHREAD) = $(( (THREADS + KMAXTHREAD - 1) / KMAXTHREAD ))"
echo "=== mix r/i/u/d/s = $READ/$INSERT/$UPDATE/$DELETE/$RANGE  bulk=${BULK}M ops=${RUNNUM}M"

# Wipe stale memcached state before restartMemc.sh re-initialises the counters.
#
# DSMKeeper::barrier() has node 0 reset barrier-<name> to 0 and every node then
# increment it, so a leftover value from an aborted run desynchronises startup:
# if barrier-DSM-init still holds 1, node 1 increments to 2, sees the target and
# proceeds, while node 0 resets to 0, reaches 1, and waits forever. restartMemc
# normally avoids this by relaunching memcached, but when the ssh to the
# memcached host is refused it only resets serverNum/clientNum -- every
# barrier-* key survives. Flushing first is what makes a run repeatable.
# Must run BEFORE restartMemc.sh, which sets serverNum/clientNum afterwards.
MEMC_ADDR=$(head -1 ../memcached.conf)
MEMC_PORT=$(awk 'NR==2{print}' ../memcached.conf)
echo "=== flushing memcached at ${MEMC_ADDR}:${MEMC_PORT}"
printf 'flush_all\r\nquit\r\n' | nc "${MEMC_ADDR}" "${MEMC_PORT}"

./restartMemc.sh

sudo ./newbench $NODES $READ $INSERT $UPDATE $DELETE $RANGE \
     $THREADS $MEMTHREADS $CACHE $UNIFORM $ZIPF \
     $BULK $WARMUP $RUNNUM \
     $CORRECT $TIMEBASE $EARLY $INDEX $RPC $ADMIT $TUNE $KMAXTHREAD
