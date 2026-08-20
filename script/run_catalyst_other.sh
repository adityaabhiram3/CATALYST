#!/bin/bash
# CATALYST / DEX benchmark driver -- run this on every node EXCEPT node 0.
# Start node 0 first, then run this within a few seconds (there is a barrier).
#
# Copy into the build directory alongside newbench and restartMemc.sh:
#   cp ../script/run_catalyst*.sh .
#
# Every knob is an environment variable, so a DEX-vs-CATALYST A/B is just:
#   INDEX=0 ./run_catalyst_other.sh     # DEX, path cache
#   INDEX=3 ./run_catalyst_other.sh     # CATALYST, cursor table
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

# Node 0 owns memcached init (flush + counter reset); this node must not touch
# it. Start node 0 first and wait until it prints "dir 0 launch!" before
# launching this -- node 0 resets barrier-DSM-init just after that point, and
# incrementing the barrier before the reset is what wedges startup.
sleep 5

sudo ./newbench $NODES $READ $INSERT $UPDATE $DELETE $RANGE \
     $THREADS $MEMTHREADS $CACHE $UNIFORM $ZIPF \
     $BULK $WARMUP $RUNNUM \
     $CORRECT $TIMEBASE $EARLY $INDEX $RPC $ADMIT $TUNE $KMAXTHREAD
