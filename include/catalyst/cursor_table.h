#pragma once

/* CATALYST cursor table (paper Sec. 4.3).
 *
 * A cursor is a range-tagged smart pointer: <key band [lo,hi], remote node,
 * epoch>. It asserts only *where a descent may resume*, never what the node
 * contains, so a stale cursor costs a repair and never correctness (Sec. 4.4).
 *
 * Layout is columnar. The probe compares only the lo_/hi_ arrays, which is why
 * they are kept separate from the payload: a scan touches 2 x 8B per entry
 * instead of 29B. node_/epoch_/depth_ are gathered only for matching lanes.
 *

 *
 * Concurrency: probes are lock-free and tolerate races. Bands are retired
 * before the payload is rewritten and published after, so a concurrent reader
 * sees either the old band or the new one; the worst case is a band paired
 * with a node it no longer describes, which the memory-side fence-key check
 * rejects exactly as it rejects a cursor invalidated by a split (Sec. 4.4).
 * Every field is <= 8B and naturally aligned, so no field is ever torn.
 */

#include "simd.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>

namespace catalyst {

// Matches GlobalAddress::Null(); kept as a raw u64 so this header does not
// pull in the RDMA stack and can be unit-tested standalone.
static constexpr uint64_t kNoNode = 0;
static constexpr uint8_t kNoDepth = 0xFF;

template <typename KeyT = uint64_t> struct Cursor {
  uint64_t node;  // GlobalAddress::val of the node to resume from
  KeyT lo;        // fence keys of that node; the band the cursor covers
  KeyT hi;
  uint32_t epoch; // node generation, checked memory-side (Sec. 4.4)
  uint8_t depth;  // tree level, the D1 coordinate

  bool valid() const { return node != kNoNode; }
  KeyT width() const { return hi - lo; }

  static Cursor none() {
    return Cursor{kNoNode, KeyT(0), KeyT(0), 0u, kNoDepth};
  }
};

/* Ranking used everywhere a cursor is chosen: deeper elides more hops, and at
 * equal depth the narrower band is the more specific landing point. */
template <typename KeyT>
inline bool is_better(const Cursor<KeyT> &a, const Cursor<KeyT> &b) {
  if (a.depth != b.depth) return a.depth > b.depth;
  return a.width() < b.width();
}

// Fold a candidate into a running best, treating an invalid best as worst.
// The explicit validity guard matters: an empty cursor carries kNoDepth
// (0xFF), which would otherwise outrank every real cursor.
template <typename KeyT>
inline void merge_best(const Cursor<KeyT> &c, Cursor<KeyT> &best) {
  if (!c.valid()) return;
  if (!best.valid() || is_better(c, best)) best = c;
}

template <typename KeyT = uint64_t, int VecBytes = simd::kNativeBytes>
class CursorTable {
public:
  using Vec = typename simd::GccVec<KeyT, VecBytes>::type;
  static constexpr int kLanes = simd::GccVec<KeyT, VecBytes>::kLanes;

  static constexpr KeyT kKeyMin = std::numeric_limits<KeyT>::min();
  static constexpr KeyT kKeyMax = std::numeric_limits<KeyT>::max();

  // Capacity is rounded up to a whole number of vectors so scans need no
  // scalar tail (Fig. 9 asserts the same invariant).
  explicit CursorTable(size_t capacity)
      : capacity_(round_up(capacity, kLanes)) {
    lo_ = alloc<KeyT>(capacity_);
    hi_ = alloc<KeyT>(capacity_);
    node_ = alloc<uint64_t>(capacity_);
    epoch_ = alloc<uint32_t>(capacity_);
    depth_ = alloc<uint8_t>(capacity_);
    clear();
  }

  ~CursorTable() {
    std::free(lo_);
    std::free(hi_);
    std::free(node_);
    std::free(epoch_);
    std::free(depth_);
  }

  CursorTable(const CursorTable &) = delete;
  CursorTable &operator=(const CursorTable &) = delete;

  size_t capacity() const { return capacity_; }

  static size_t bytes_per_cursor() {
    return 2 * sizeof(KeyT) + sizeof(uint64_t) + sizeof(uint32_t) +
           sizeof(uint8_t);
  }

  // Bytes actually resident, for the footprint numbers in Sec. 6.3.
  size_t footprint_bytes() const { return capacity_ * bytes_per_cursor(); }

  void clear() {
    for (size_t i = 0; i < capacity_; ++i) retire(i);
  }

  /* ---------------- region-limited scans ---------------- */

  /* SIMD-scan [base, base+count) for bands containing k, folding the deepest
   * into `best`. This is Fig. 9's table_scan restricted to one region. */
  void scan_point(size_t base, size_t count, KeyT k, Cursor<KeyT> &best) const {
    assert(base % kLanes == 0 && count % kLanes == 0);
    assert(base + count <= capacity_);
    const Vec key = splat(k);
    for (size_t i = base; i < base + count; i += kLanes) {
      const Vec lo = simd::load<Vec>(lo_ + i);
      const Vec hi = simd::load<Vec>(hi_ + i);
      // Contained iff lo <= k <= hi. Empty slots hold lo=MAX, hi=MIN, so they
      // fail for every k and need no separate validity bitmap.
      const auto m = (lo <= key) & (key <= hi);
      uint32_t bits = simd::lane_bits(m, kLanes);
      while (bits) {
        const size_t s = i + simd::next_lane(bits);
        if (node_[s] == kNoNode) continue;
        merge_best(read(s), best);
      }
    }
  }

  /* SIMD-scan [base, base+count) for bands overlapping [start, end].
   *
   * Appends into out[0..kept), displacing the shallowest kept match once the
   * buffer is full, and accumulates the total number seen into `found`.
   * Keeping the deepest rather than the first max_out encountered matters:
   * the policy layer (Sec. 5) trades depth for reach and cannot do so if
   * overflow hands it an arbitrary subset determined by slot order.
   */
  template <typename Accept>
  void scan_range(size_t base, size_t count, KeyT start, KeyT end,
                  Cursor<KeyT> *out, int max_out, int &kept, int &found,
                  Accept accept) const {
    assert(base % kLanes == 0 && count % kLanes == 0);
    assert(base + count <= capacity_);
    assert(start <= end);
    const Vec vstart = splat(start);
    const Vec vend = splat(end);

    for (size_t i = base; i < base + count; i += kLanes) {
      const Vec lo = simd::load<Vec>(lo_ + i);
      const Vec hi = simd::load<Vec>(hi_ + i);
      // Overlap is the negation of the disjoint test (lo > end || hi < start),
      // which is the form Sec. 4.3 describes.
      const auto m = (lo <= vend) & (hi >= vstart);
      uint32_t bits = simd::lane_bits(m, kLanes);
      while (bits) {
        const size_t s = i + simd::next_lane(bits);
        // A full-span probe would otherwise match empty slots, whose sentinel
        // band is inverted rather than disjoint. Cheap, and off the hot path.
        if (node_[s] == kNoNode) continue;
        const Cursor<KeyT> c = read(s);
        // Lets a partitioned caller count a replicated band exactly once.
        if (!accept(c)) continue;
        ++found;
        if (kept < max_out) {
          out[kept++] = c;
        } else if (max_out > 0) {
          int worst = 0;
          for (int j = 1; j < kept; ++j) {
            if (is_better(out[worst], out[j])) worst = j;
          }
          if (is_better(c, out[worst])) out[worst] = c;
        }
      }
    }
  }

  void scan_range(size_t base, size_t count, KeyT start, KeyT end,
                  Cursor<KeyT> *out, int max_out, int &kept, int &found) const {
    scan_range(base, count, start, end, out, max_out, kept, found,
               [](const Cursor<KeyT> &) { return true; });
  }

  /* ---------------- whole-table probes ---------------- */

  // Deepest cursor whose band contains k.
  Cursor<KeyT> probe_point(KeyT k) const {
    Cursor<KeyT> best = Cursor<KeyT>::none();
    scan_point(0, capacity_, k, best);
    return best;
  }

  // Deepest min(found, max_out) cursors overlapping [start, end], written
  // deepest-first; returns the total number found.
  int probe_range(KeyT start, KeyT end, Cursor<KeyT> *out, int max_out) const {
    int kept = 0, found = 0;
    scan_range(0, capacity_, start, end, out, max_out, kept, found);
    std::sort(out, out + kept, is_better<KeyT>);
    return found;
  }

  /* ---------------- maintenance ---------------- */

  // Publish a cursor into a slot chosen by the policy layer.
  void install_at(size_t slot, KeyT lo, KeyT hi, uint64_t node, uint32_t epoch,
                  uint8_t depth) {
    assert(slot < capacity_);
    assert(lo <= hi);
    assert(node != kNoNode);

    retire(slot);           // band first: no probe can match a stale payload
    compiler_barrier();
    node_[slot] = node;
    epoch_[slot] = epoch;
    depth_[slot] = depth;
    compiler_barrier();
    hi_[slot] = hi;         // publish: band becomes matchable last
    lo_[slot] = lo;
  }

  // Make a slot unmatchable without disturbing its neighbours.
  void retire(size_t slot) {
    assert(slot < capacity_);
    lo_[slot] = kKeyMax;
    hi_[slot] = kKeyMin;
    compiler_barrier();
    node_[slot] = kNoNode;
    epoch_[slot] = 0;
    depth_[slot] = kNoDepth;
  }

  /* Narrow a cursor in place after a split (Sec. 4.4, Table 3 case ii).
   *
   * The repair is arithmetic on the band: a split moves a boundary rather than
   * invalidating the entry, which is the property that lets cursors survive
   * mutation where address-keyed caches cannot.
   */
  bool narrow(size_t slot, KeyT new_hi) {
    assert(slot < capacity_);
    if (node_[slot] == kNoNode || new_hi < lo_[slot] || new_hi >= hi_[slot]) {
      return false;
    }
    hi_[slot] = new_hi;
    return true;
  }

  bool occupied(size_t slot) const { return node_[slot] != kNoNode; }

  Cursor<KeyT> read(size_t slot) const {
    return Cursor<KeyT>{node_[slot], lo_[slot], hi_[slot], epoch_[slot],
                        depth_[slot]};
  }

  // First free slot in a region, or base+count if it is full.
  size_t find_free_slot_in(size_t base, size_t count) const {
    for (size_t i = base; i < base + count; ++i) {
      if (node_[i] == kNoNode) return i;
    }
    return base + count;
  }

  size_t find_free_slot() const { return find_free_slot_in(0, capacity_); }

  // Shallowest occupied slot in a region, or base+count if empty. The eviction
  // candidate a bucket offers when the policy layer needs a free slot.
  size_t shallowest_in(size_t base, size_t count) const {
    size_t victim = base + count;
    for (size_t i = base; i < base + count; ++i) {
      if (node_[i] == kNoNode) continue;
      if (victim == base + count || is_better(read(victim), read(i))) victim = i;
    }
    return victim;
  }

  size_t size_in(size_t base, size_t count) const {
    size_t n = 0;
    for (size_t i = base; i < base + count; ++i) n += (node_[i] != kNoNode);
    return n;
  }

  size_t size() const { return size_in(0, capacity_); }

private:
  static size_t round_up(size_t n, size_t m) {
    if (n == 0) return m;
    return ((n + m - 1) / m) * m;
  }

  template <typename T> static T *alloc(size_t n) {
    void *p = nullptr;
    // 64B alignment: vector loads stay within a cache line boundary.
    if (posix_memalign(&p, 64, n * sizeof(T)) != 0 || p == nullptr) {
      throw std::bad_alloc();
    }
    return static_cast<T *>(p);
  }

  static Vec splat(KeyT k) {
    Vec v;
    for (int i = 0; i < kLanes; ++i) v[i] = k;
    return v;
  }

  static void compiler_barrier() { asm volatile("" ::: "memory"); }

  size_t capacity_;
  KeyT *lo_;       // hot: scanned by every probe
  KeyT *hi_;       // hot
  uint64_t *node_; // cold: gathered on match only
  uint32_t *epoch_;
  uint8_t *depth_;
};

} // namespace catalyst
