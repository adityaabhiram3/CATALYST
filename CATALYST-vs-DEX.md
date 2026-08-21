# DEX vs CATALYST: how each one talks to remote memory

Both indexes in this repo store **the same B+-tree, in the same layout, on the
same memory nodes**. Neither changes the node format, the fanout, or the
partitioning. What differs is entirely the compute side: how a query gets from a
key to a leaf.

| | DEX (`cachepush::BTree`) | CATALYST (`catalyst::BTree`) |
|---|---|---|
| Source | [`include/tree/leanstore_tree.h`](include/tree/leanstore_tree.h) | [`include/catalyst/catalyst_tree.h`](include/catalyst/catalyst_tree.h) |
| Compute-side state | page cache of full 1KB nodes | cursor table of range-tagged pointers |
| Footprint | 256–512 MB (up to 4 GB) | **~1.86 MB** |
| Steady-state verbs | one-sided `READ` per uncached level | **one two-sided RPC per operation** |
| Bytes per level | 1024 B (a whole page) | 0 — no page ever crosses |
| On a miss | fetch page, admit into cache, evict something | resume deeper next time; nothing replicated |
| Benchmark index | `argv[18] = 0` | `argv[18] = 3` |

---

> Code below is quoted from the tree, abridged for clarity: lock-coupling and
> restart handling are elided where they are not the point. File references
> point at the full text.

## 1. The traversal loop

### DEX: one round trip per tree level

`lookup()` walks the tree on the **compute side**. Each level that is not
resident costs a network round trip, because the compute node must hold the node
in order to read the child pointer out of it.

```cpp
// include/tree/leanstore_tree.h  — lookup(), inner loop
while (cur_node->type == PageType::BTreeInner) {
  auto inner = static_cast<BTreeInner<Key> *>(cur_node);
  auto idx   = inner->lowerBound(k);

  // Resident? follow a swizzled pointer locally. Not resident? cur_node
  // comes back nullptr and we have to go to the network.
  cur_node = new_get_mem_node(inner->children[idx], inner, idx,
                              versionNode, needRestart, refresh, false);

  if (cur_node == nullptr) {
    // Fetch the missing node and admit it into the cache.
    remote_flag = cache.cold_to_hot(inner->children[idx],
                                    reinterpret_cast<void **>(&cur_node),
                                    inner, idx, refresh);
    ...
    new_swizzling(inner->children[idx], inner, idx, cur_node);  // cache it
  }
}
```

`cold_to_hot` ends in a one-sided READ of a full page:

```cpp
// include/cache/node_wr.h  — the non-shared path
global_dsm_->read_sync(page_buffer, global_node, pageSize, nullptr);   // 1024 B
```

For a node that **crosses a partition boundary** (shared between compute nodes)
DEX cannot trust a plain read, so it runs the three-READ optimistic protocol:

```cpp
// include/cache/node_wr.h — opt_remote_read_by_sibling()
global_dsm_->read_sync(page_buffer, global_node, 16, nullptr);        // 1. version
uint64_t org_version = mem_node->front_version;

global_dsm_->read_sync(page_buffer, global_node, pageSize, nullptr);  // 2. content

global_dsm_->read_sync(new_page_buffer, global_node, 16, nullptr);    // 3. re-verify
if (new_mem_node->front_version != org_version) return nullptr;       //    retry
```

So a cold shared level is **3 round trips**, not one.

### CATALYST: one round trip per operation, regardless of depth

The compute side never dereferences a tree node. It probes a local table for a
*position*, then ships the whole operation to the memory node.

```cpp
// include/catalyst/catalyst_tree.h — run()
Cursor<Key> c = cursors_.probe_point(k);          // ~73 ns, purely local
GlobalAddress start = c.valid() ? GlobalAddress(c.node)   // resume mid-tree
                                : current_root();         // or start at the root

Msg msg;
msg.start = start;  msg.k = k;  msg.v = v_in;  msg.op = op;

dsm_->catalyst_traverse(msg);                     // ONE two-sided RPC

install_path(msg, k, c, resumed_depth);           // capture cursors from the reply
```

There is no loop over levels. The `for (hop...)` wrapper in `run()` iterates only
for two exceptional cases — a stale cursor (`kStale`) or a subtree that lives on
a different memory node (`kCrossNode`) — not once per level.

---

## 2. Where the descent actually happens

DEX's memory node is passive for reads: an RDMA NIC serves `READ`s with no CPU
involvement. CATALYST's memory node runs the descent itself.

```cpp
// include/cache/btree_rpc.h — catalyst_traverse(), runs ON the memory node
NodeBase *mem_node = reinterpret_cast<NodeBase *>(dsm_base + start.offset);

// The cursor asserts only "a descent for k may resume here". Validate it
// against the authoritative node, under the same latch an SMO would take.
if (mem_node->obsolete) return kStale;
if (!inner->rangeValid(k))  return kStale;        // fence-key check

while (mem_node->type == PageType::BTreeInner) {
  path[npath] = {node, mem_node->min_limit_, mem_node->max_limit_,
                 mem_node->level};                 // record for the reply
  ++npath;
  node = inner->children[inner->lowerBound(k)];    // local pointer chase
  mem_node = reinterpret_cast<NodeBase *>(dsm_base + node.offset);
}
```

Two consequences:

* **Pointer chasing is local.** Each level is a DRAM dereference on the memory
  node, not a network hop.
* **Cursor validation costs no extra round trip.** The fence-key check happens
  where the index lives, so it cannot interleave with a concurrent split. A
  misaligned cursor returns `kStale` in the same reply that would have carried
  the result.

---

## 3. Verb accounting, per point lookup

| | DEX | CATALYST |
|---|---|---|
| One-sided `READ` | 1 per uncached level (3 if the node is shared) | **0** |
| Two-sided `SEND`/`RECV` | 0 | **1 request + 1 reply** |
| Bytes to memory node | ~8 B per READ request | 30 B (`offsetof(TraverseMsg, status)`) |
| Bytes back | 1024 B per level | 51 + 25·`npath` B (≈101 B for a 2-level path) |
| Memory-node CPU | none (NIC serves the READ) | one directory thread, one descent |

DEX's own paper (Table 2, read-only, 144 threads) measures **0.33 one-sided
reads and 333.9 B per operation** — low precisely *because* a 256 MB cache is
absorbing most traversals; the uncached baselines in the same table are Sherman
at 3.02 reads / 1064.7 B and SMART at 1.44 reads / 997.0 B. DEX buys its 0.33 by
spending the footprint. CATALYST targets one round trip and no page transfer at
all, from 1.86 MB — that is the trade the comparison is meant to expose.

---

## 4. Where CATALYST *does* still use one-sided RDMA

Only three places, none of them on a point-query path:

```cpp
// range_scan(): the records have to cross the wire regardless, and they do
// not fit in a message slot.
dsm_->read_sync(buf, leaf, cachepush::pageSize, nullptr);
```

* **`range_scan`** — one RPC to locate each leaf, then a one-sided READ of that
  leaf's payload.
* **`bulk_load` / `get_newest_root` / `flush_all`** — delegated to the embedded
  `cachepush::BTree`, at setup and between phases.
* **`insert` that splits a leaf** — falls back to `dex_->insert()`, which owns
  splits. Counted as `smo_fallback` in the statistics line.

For read-only, read-intensive and write-intensive (read+update) runs,
`smo_fallback=0` confirms **zero one-sided traffic occurred**.

---

## 5. Wire format

DEX reuses the 33-byte `RawMessage` for its RPCs. CATALYST needs to carry a path
back, so it adds a second struct at the same header offsets — the directory
thread reads `m->type` before it knows which message it has.

```cpp
// include/RawMessageConnection.h
struct PathEntry {          // 25 B
  GlobalAddress addr;
  uint64_t lo, hi;          // the node's fence keys == the cursor's band
  uint8_t  level;
} __attribute__((packed));

struct TraverseMsg {        // 351 B
  RpcType type; uint16_t node_id, app_id;   // must mirror RawMessage
  GlobalAddress start; uint64_t k, v; Op op;         // request
  int32_t status; uint64_t value; GlobalAddress leaf;// response
  uint8_t npath; PathEntry path[kMaxPathLen];
} __attribute__((packed));
```

`MESSAGE_SIZE` was raised 96 → 512 to fit this after the 40-byte UD GRH. That is
the *slot* size in the message pool, **not** wire bytes: `sendRawMessage`
transmits `sizeof(the struct)`, so DEX's RPCs still put 33 B on the wire and are
unaffected.

The reply is trimmed to what was actually filled:

```cpp
// src/Directory.cpp
send_len = offsetof(catalyst_wire::TraverseMsg, path) +
           npath * sizeof(catalyst_wire::PathEntry);
```

---

## 6. Why the cursor table stays small

DEX must hold **every node on a path** to elide that path — with fanout *f* and
height *h*, covering the top ℓ levels costs Θ(*f*^ℓ) nodes. A cursor stores a
*position*, so two queries whose keys fall in the same separator interval share
one entry no matter how large the subtree beneath them is.

Measured on the target cluster (Xeon Gold 6242R, AVX-512):

```
   cursors      table        probe        slots
                 (KB)   (ns/probe)      scanned
      4096        123        106.8          512
     65536       1863         73.5          512      <- 1.86 MB, constant cost
```

Probe cost is set by `slots_per_bucket + overflow_slots`, not by capacity.

---

## 7. Running the comparison

Both indexes build an identical remote tree with identical partitioning, so the
only variable is navigation:

```bash
INDEX=0 THREADS=72 ./run_catalyst.sh     # DEX, path cache
INDEX=3 THREADS=72 ./run_catalyst.sh     # CATALYST, cursor table
```

CATALYST prints its real footprint and per-run behaviour:

```
CATALYST: cursor table 1863 KB (65792 slots, 512 scanned per probe)
CATALYST: probes=N hit=X% stale=N cross_node=N smo_fallback=N resident=N/N
```

* `hit=` should climb through warmup.
* `smo_fallback=` must be 0 for read/update mixes — nonzero means the DEX split
  path ran and one-sided traffic occurred.
* `stale=` counts cursors invalidated by a concurrent split.

**Note on `cacheSize` (`argv[9]`).** For `INDEX=3` this sizes the *DEX fallback
cache* used for bulk load and splits — it is **not** CATALYST's cache. The number
that belongs in a footprint comparison is the `cursor table ... KB` line.

---

## 8. Current limitations

* **Insert-heavy workloads are not ready.** A leaf split still falls back to the
  DEX compute-side path, followed by `flush_all()` so no dirty cached page can
  overwrite a server-side mutation. Correct but slow. Moving splits onto the
  memory node removes this path entirely.

