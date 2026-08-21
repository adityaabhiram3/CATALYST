#pragma once

/* CATALYST index: navigation instead of caching (paper Sec. 4.2).
 *
 * The remote data structure is DEX's B+-tree, unmodified -- "Catalyst overlays
 * cursors on these layouts without modifying the index implementations"
 * (Sec. 2.1). What changes is how the compute side reaches a leaf:
 *
 *   DEX      probe page cache -> on miss, one-sided READ each node down the
 *            path, admit pages into a 512MB cache.
 *   CATALYST probe a ~2MB cursor table -> one two-sided RPC that resumes the
 *            descent at the cursor's node and returns the walked path, from
 *            which new cursors are installed. Nothing is replicated.
 *
 * The embedded cachepush::BTree is reused for three things only: building the
 * tree during bulk load, holding the root pointer, and the insert path when a
 * leaf split is required. It is deliberately given a small cache, since
 * CATALYST's steady-state reads never touch it.
 *
 * Coherence between the two. CATALYST mutates leaves on the memory node while
 * the embedded DEX cache may hold copies, so the two must never be live at the
 * same time for the same page. Reads, updates and deletes go exclusively
 * through the RPC and never populate the DEX cache, so they are safe by
 * construction. The one exception is a leaf split, which DEX performs from the
 * compute side; flush_dex_cache() is called immediately afterwards so remote
 * memory is authoritative again and no dirty copy can be written back later on
 * top of a server-side mutation. See the note on smo_fallbacks() below --
 * moving splits onto the memory node (Sec. 4.4, "all mutation happens on the
 * memory node") removes this path entirely and is the next milestone.
 */

#include "../DSM.h"
#include "../tree/leanstore_tree.h"
#include "../tree_api.h"
#include "bucketed_cursor_table.h"
#include "pattern.h"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace catalyst {

/* Placement envelope (Sec. 4.1): explicit bounds on the three coordinates of
 * Sec. 3.3. Levels use DEX's convention, where 0 is the leaf and larger is
 * closer to the root -- the inverse of the paper's figures, which count depth
 * downward from the root. The default is deep-narrow: cursors at the leaf and
 * the two levels above it, which is where point lookups concentrate. */
struct Envelope {
  uint8_t min_level = 0;
  uint8_t max_level = 2;
  uint64_t max_width = std::numeric_limits<uint64_t>::max();

  bool admits(const catalyst_wire::PathEntry &p) const {
    if (p.level < min_level || p.level > max_level) return false;
    if (p.hi - p.lo > max_width) return false;
    return true;
  }
};

struct CursorConfig {
  int bucket_bits = 8;
  size_t slots_per_bucket = 256;
  size_t overflow_slots = 256;
  int max_span = 4;
  Envelope envelope{};        // only used when the control loop is disabled
  ControlConfig control{};    // Sec. 5 pattern model
  bool use_pattern_model = true;

  /* Read from the environment so a workload sweep needs no rebuild and no
   * change to newbench's fixed 22-argument list:
   *   CATALYST_PATTERN = funnel | interval | branch | spatial | auto
   *   CATALYST_TAU     = admission margin (default 0)
   *   CATALYST_MODEL   = 0 to fall back to the fixed Sec. 4 envelope
   */
  static CursorConfig from_env() {
    CursorConfig c;
    if (const char *p = std::getenv("CATALYST_PATTERN")) {
      c.control.pattern = pattern_from_string(p);
    }
    if (const char *t = std::getenv("CATALYST_TAU")) {
      c.control.tau = std::atof(t);
    }
    if (const char *m = std::getenv("CATALYST_MODEL")) {
      c.use_pattern_model = (std::atoi(m) != 0);
    }
    return c;
  }
};

template <class Key, class Value> class BTree : public tree_api<Key, Value> {
public:
  using Msg = catalyst_wire::TraverseMsg;
  using Op = catalyst_wire::Op;

  /* dex_cache_mb sizes the fallback cache only. It is not CATALYST's cache:
   * cursor_footprint_bytes() is the number that belongs in a footprint
   * comparison against DEX's 512MB. */
  BTree(DSM *dsm, uint64_t tree_id, uint64_t dex_cache_mb,
        std::vector<Key> &partition, int num_partitions,
        const CursorConfig &cfg = CursorConfig())
      : dsm_(dsm), cfg_(cfg), env_(cfg.envelope),
        cursors_(cfg.bucket_bits, cfg.slots_per_bucket, cfg.overflow_slots,
                 cfg.max_span),
        ctrls_(MAX_APP_THREAD, Slot{Controller(cfg.control)}) {
    // rpc_rate 0 / admission 1: the embedded tree is only a builder and a
    // split path, so its own pushdown heuristics must stay out of the way.
    dex_ = new cachepush::BTree<Key, Value>(dsm, tree_id, dex_cache_mb, 0.0, 1.0,
                                            partition, num_partitions);
    std::cout << "CATALYST: cursor table " << cursors_.footprint_bytes() / 1024
              << " KB (" << cursors_.capacity() << " slots, "
              << cursors_.probe_slots() << " scanned per probe), envelope "
              << "levels [" << (int)env_.min_level << ", "
              << (int)env_.max_level << "]" << std::endl;
    std::cout << "CATALYST: DEX fallback cache " << dex_cache_mb << " MB"
              << std::endl;
    if (cfg_.use_pattern_model) {
      std::cout << "CATALYST: pattern model ON, pattern="
                << pattern_name(cfg_.control.pattern)
                << " tau=" << cfg_.control.tau << std::endl;
    } else {
      std::cout << "CATALYST: pattern model OFF (fixed Sec. 4 envelope)"
                << std::endl;
    }
  }

  ~BTree() { delete dex_; }

  /* ---------------- point operations ---------------- */

  bool lookup(Key k, Value &result) override {
    Value v = 0;
    GlobalAddress leaf;
    const int st = run(k, 0, Op::Lookup, v, leaf);
    if (st == catalyst_wire::kFound) {
      result = v;
      return true;
    }
    return false;
  }

  bool update(Key k, Value v) override {
    Value out = 0;
    GlobalAddress leaf;
    return run(k, v, Op::Update, out, leaf) == catalyst_wire::kFound;
  }

  bool remove(Key k) override {
    Value out = 0;
    GlobalAddress leaf;
    return run(k, 0, Op::Delete, out, leaf) == catalyst_wire::kFound;
  }

  bool insert(Key k, Value v) override {
    Value out = 0;
    GlobalAddress leaf;
    const int st = run(k, v, Op::Insert, out, leaf);
    if (st == catalyst_wire::kNeedsSMO) {
      // The leaf is full and a split propagates upward, which offloading must
      // not attempt (Sec. 6, "fall back to the normal path"). DEX owns splits.
      ++smo_fallbacks_;
      const bool ok = dex_->insert(k, v);
      flush_dex_cache();
      return ok;
    }
    return st == catalyst_wire::kInserted || st == catalyst_wire::kFound;
  }

  /* ---------------- range scan ---------------- */

  /* Walk leaves left to right, re-entering the tree once per leaf.
   *
   * A cursor seeds the first descent; subsequent leaves are reached by
   * traversing for the key just past the previous leaf's high fence, which is
   * how DEX's own scan advances (it keeps no sibling links). Leaf payloads are
   * pulled with a one-sided READ, because the records have to cross the wire
   * regardless and they do not fit in a message slot -- consistent with
   * Fig. 14, where the data payload dominates CATALYST's per-query bytes. */
  int range_scan(Key k, uint32_t num, std::pair<Key, Value> *&result) override {
    uint32_t collected = 0;
    Key cur = k;

    while (collected < num) {
      Value v = 0;
      GlobalAddress leaf;
      const int st = run(cur, 0, Op::Lookup, v, leaf);
      if (st != catalyst_wire::kFound && st != catalyst_wire::kAbsent) break;
      if (leaf == GlobalAddress::Null()) break;

      auto *buf = (dsm_->get_rbuf(0)).get_page_buffer();
      dsm_->read_sync(buf, leaf, cachepush::pageSize, nullptr);
      auto *page = reinterpret_cast<cachepush::BTreeLeaf<Key, Value> *>(buf);

      unsigned pos = page->lowerBound(cur);
      for (; pos < page->count && collected < num; ++pos) {
        result[collected++] = page->data[pos];
      }

      const Key high = page->max_limit_;
      if (high == std::numeric_limits<Key>::max()) break; // last leaf
      cur = high + 1;
    }
    return (int)collected;
  }

  /* ---------------- build / lifecycle ---------------- */

  void bulk_load(Key *bulk_array, uint64_t bulk_load_num) override {
    dex_->bulk_load(bulk_array, bulk_load_num);
    // Push everything the builder cached out to the memory pool, so the
    // memory node is authoritative before any cursor is issued.
    flush_dex_cache();
  }

  void set_shared(std::vector<Key> &bound) override { dex_->set_shared(bound); }
  void set_bound(Key left, Key right) override { dex_->set_bound(left, right); }
  void get_newest_root() override { dex_->get_newest_root(); }
  void get_basic() override { dex_->get_basic(); }
  void validate() override { dex_->validate(); }

  /* Between benchmark phases the tree may have been rebuilt or re-partitioned,
   * so every cursor is suspect. Dropping them is cheap and the table refills
   * within the warmup phase. */
  void reset_buffer_pool(bool flush_dirty) override {
    dex_->reset_buffer_pool(flush_dirty);
    cursors_.clear();
    // The descriptor and the confidence sketch describe a table that no longer
    // exists, so they have to go too or the controller spends the next phase
    // acting on evidence about evicted cursors.
    for (auto &s : ctrls_) s.c.reset();
    hits_ = misses_ = stale_ = cross_ = 0;
  }

  void clear_statistic() override {
    hits_ = misses_ = stale_ = cross_ = smo_fallbacks_ = 0;
  }

  void get_statistic() override {
    const uint64_t probes = hits_ + misses_;
    std::printf("CATALYST: probes=%lu hit=%.1f%% stale=%lu cross_node=%lu "
                "smo_fallback=%lu resident=%zu/%zu\n",
                probes,
                probes ? 100.0 * (double)hits_ / (double)probes : 0.0, stale_,
                cross_, smo_fallbacks_, cursors_.size(), cursors_.capacity());
    if (cfg_.use_pattern_model) {
      const Controller &c = ctrls_[0].c;
      const auto &d = c.descriptor();
      const auto &f = c.feedback();
      std::printf("CATALYST: pattern=%s theta{funnel[%u,%u] anchor=%u "
                  "pivot=%u/h%u b=%u} ewma{hit=%.2f skip=%.2f residual=%.2f} "
                  "admit=%lu reject=%lu switches=%u root_level=%u\n",
                  pattern_name(c.kind()), d.d_start, d.d_end, d.d_anchor,
                  d.d_pivot, d.branch_height, d.breadth, f.hit.get(),
                  f.skipped.get(), f.residual.get(), f.admits, f.rejects,
                  c.pattern_switches(), c.root_level());
    }
  }

  // CATALYST's actual compute-side footprint, for Sec. 6.3 comparisons.
  size_t cursor_footprint_bytes() const { return cursors_.footprint_bytes(); }
  uint64_t smo_fallbacks() const { return smo_fallbacks_; }

  void set_rpc_ratio(double) override {}
  void set_admission_ratio(double) override {}

private:
  // The RPC may bounce once per memory node crossing, plus one retry after a
  // stale cursor; a handful of hops is plenty for a tree of this height.
  static constexpr int kMaxHops = 8;

  // Cursor depth must increase with distance from the root so that is_better
  // prefers the cursor that elides more hops, but DEX numbers levels upward
  // from the leaf. Invert into a small positive range.
  static constexpr uint8_t kDepthBias = 32;
  static uint8_t depth_of(uint8_t level) {
    return (uint8_t)(kDepthBias - (level < kDepthBias ? level : kDepthBias));
  }
  // Inverse, so a resident cursor's stored depth can be turned back into a DEX
  // level and then into the controller's paper depth.
  static uint8_t level_of(uint8_t stored_depth) {
    return (uint8_t)(stored_depth <= kDepthBias ? kDepthBias - stored_depth : 0);
  }

  /* One index operation: probe, traverse, capture (Fig. 8). */
  int run(Key k, Value v_in, Op op, Value &v_out, GlobalAddress &leaf_out) {
    Cursor<Key> c = cursors_.probe_point(k);
    bool holding_cursor = c.valid();
    holding_cursor ? ++hits_ : ++misses_;
    if (cfg_.use_pattern_model) ctrl().note_probe(holding_cursor);

    // How far down the tree this query gets to start, in the controller's
    // depth convention -- this is exactly the "levels skipped" feedback signal.
    const uint8_t resumed_depth =
        holding_cursor ? ctrl().to_depth(level_of(c.depth)) : 0;

    GlobalAddress start =
        holding_cursor ? GlobalAddress(c.node) : current_root();

    Msg msg;
    for (int hop = 0; hop < kMaxHops; ++hop) {
      msg.start = start;
      msg.k = k;
      msg.v = v_in;
      msg.op = op;
      msg.status = 0;
      msg.npath = 0;
      dsm_->catalyst_traverse(msg);

      if (msg.status == catalyst_wire::kStale) {
        ++stale_;
        if (holding_cursor) {
          // Lazy discovery: the split or merge that invalidated this band is
          // repaired by the one query that noticed, not by a broadcast.
          cursors_.retire(c);
          holding_cursor = false;
          start = current_root();
        } else {
          // Already starting from a root that the memory node rejected, so our
          // root pointer is the stale thing.
          dex_->get_newest_root();
          start = current_root();
        }
        continue;
      }

      // Partial paths are still valid cursor candidates, so capture before
      // following the operation onto the next memory node.
      install_path(msg, k, c, holding_cursor ? resumed_depth : 0);

      if (msg.status == catalyst_wire::kCrossNode) {
        ++cross_;
        start = msg.leaf;
        continue;
      }

      v_out = msg.value;
      leaf_out = msg.leaf;
      return msg.status;
    }
    return catalyst_wire::kStale;
  }

  /* Capture (Sec. 4.1) plus admission.
   *
   * With the pattern model on this is the Sec. 5 loop: feed the returned path
   * into the confidence sketch, let the controller move theta, pick the single
   * best candidate inside the admissible region, and admit it only if it beats
   * the resident cursor by tau. Admitting one cursor per query rather than the
   * whole path is what keeps a bounded table selective.
   *
   * With the model off this degrades to the fixed envelope of Sec. 4, which is
   * the A/B baseline for "does the control loop actually earn its place". */
  void install_path(const Msg &msg, Key k, const Cursor<Key> &resident,
                    uint8_t resumed_depth) {
    const uint8_t n = msg.npath < catalyst_wire::kMaxPathLen
                          ? msg.npath
                          : catalyst_wire::kMaxPathLen;
    if (n == 0) return;

    if (!cfg_.use_pattern_model) {
      for (uint8_t i = 0; i < n; ++i) {
        const auto &p = msg.path[i];
        if (p.addr == GlobalAddress::Null()) continue;
        if (!env_.admits(p)) continue;
        if (p.lo > p.hi) continue;
        cursors_.install(p.lo, p.hi, p.addr.val, 0, depth_of(p.level));
      }
      return;
    }

    Controller &c = ctrl();
    c.observe(msg.path, n, resumed_depth);

    catalyst_wire::PathEntry best{};
    uint8_t best_depth = 0;
    const size_t occ = cursors_.bucket_occupancy(cursors_.bucket_of(k));
    if (!c.select(msg.path, n, occ, best, best_depth)) return;
    if (best.addr == GlobalAddress::Null() || best.lo > best.hi) return;
    const uint64_t res_node = resident.valid() ? resident.node : kNoNode;
    const uint8_t res_depth =
        resident.valid() ? c.to_depth(level_of(resident.depth)) : 0;
    if (!c.admit(best.addr.val, best_depth, res_node, res_depth)) return;

    if (cursors_.install(best.lo, best.hi, best.addr.val, 0,
                         depth_of(best.level))) {
      c.note_admitted_width(best.hi - best.lo);
    }
  }

  Controller &ctrl() {
    const int id = dsm_->getMyThreadID();
    return ctrls_[(id >= 0 && id < (int)ctrls_.size()) ? id : 0].c;
  }

  GlobalAddress current_root() const { return dex_->root; }

  // Make remote memory authoritative and leave no dirty page behind that could
  // later be written back over a server-side mutation.
  void flush_dex_cache() { dex_->flush_all(); }

  // One controller per worker thread, cache-line separated. The cursor table
  // is shared; the descriptor and feedback are per-thread estimates of that
  // thread's query mix, which matches DEX's per-thread key partitioning.
  struct alignas(64) Slot {
    Controller c;
  };

  DSM *dsm_;
  cachepush::BTree<Key, Value> *dex_;
  CursorConfig cfg_;
  Envelope env_;
  BucketedCursorTable<Key> cursors_;
  std::vector<Slot> ctrls_;

  uint64_t hits_ = 0, misses_ = 0, stale_ = 0, cross_ = 0, smo_fallbacks_ = 0;
};

} // namespace catalyst
