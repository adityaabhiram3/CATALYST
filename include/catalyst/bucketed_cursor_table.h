#pragma once

/* partitioned CATALYST cursor table.
 *
 * Rationale. Fig. 9's table_scan() sweeps every row, so a probe costs
 * Theta(table size). Measured on an i9-9900K (AVX2, 64-bit keys) that is
 * ~100ns at 256 cursors but ~68us at 145k cursors -- so the sub-100ns probe of
 * Fig. 8 and the 2-4MB / 250k-cursor footprint of Sec. 6.3 cannot both hold
 * under a flat scan. Partitioning reconciles them.
 *
 * Design. Slots are split into 2^bucket_bits buckets indexed by the key's high
 * bits, carved out of one flat arena so the SIMD kernel is reused unchanged --
 * a bucket is just a region handed to CursorTable::scan_*. A point probe scans
 * exactly one bucket plus a small overflow region, so its cost is set by
 * slots_per_bucket rather than by total capacity.
 *
 * Wide bands. A band is replicated into every bucket it overlaps, which is
 * fine for the deep-narrow envelope (Table 2) where a cursor sits below one
 * separator interval and touches one bucket. Bands spanning more than
 * max_span buckets -- the shallow-wide envelope -- go to the overflow region
 * instead, which every probe scans. Overflow is therefore sized for the few
 * shallow cursors a workload keeps, not for the bulk.
 *
 * Cost. Point probe: (slots_per_bucket + overflow_slots) slots regardless of
 * capacity. Range probe: one bucket per bucket the query range spans, so a
 * scan across the whole key space degrades to the flat cost -- correct, and
 * the reason Sec. 3.4 Case 2 places scan cursors at the leaf frontier.
 */

#include "cursor_table.h"

#include <atomic>
#include <memory>

namespace catalyst {

/* Writers take a per-bucket spinlock; probes never lock.
 *
 * Only installs and retires contend, and a bucket is picked by key, so worker
 * threads working disjoint key ranges (which DEX's logical partitioning gives
 * us) rarely collide. Readers stay lock-free because a racing reader can at
 * worst pair a band with a node it no longer describes, which the memory-side
 * fence check rejects -- the same repair a split already triggers (Sec. 4.4).
 */
class BucketLock {
public:
  void lock() {
    while (flag_.test_and_set(std::memory_order_acquire)) {
      __builtin_ia32_pause();
    }
  }
  void unlock() { flag_.clear(std::memory_order_release); }

private:
  std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
  char pad_[64 - sizeof(std::atomic_flag)];
};

class BucketGuard {
public:
  explicit BucketGuard(BucketLock &l) : l_(l) { l_.lock(); }
  ~BucketGuard() { l_.unlock(); }

private:
  BucketLock &l_;
};

template <typename KeyT = uint64_t, int VecBytes = simd::kNativeBytes>
class BucketedCursorTable {
public:
  using Table = CursorTable<KeyT, VecBytes>;
  static constexpr int kLanes = Table::kLanes;
  static constexpr int kKeyBits = (int)(sizeof(KeyT) * 8);

  /* Defaults target the paper's ~2MB footprint with a ~100ns probe:
   * 256 buckets x 256 slots = 65536 cursors, plus 256 overflow slots.
   * A probe scans 512 slots => 8KB of hot lo_/hi_ data. */
  explicit BucketedCursorTable(int bucket_bits = 8,
                               size_t slots_per_bucket = 256,
                               size_t overflow_slots = 256, int max_span = 4)
      : bucket_bits_(bucket_bits),
        num_buckets_(size_t(1) << bucket_bits),
        slots_per_bucket_(round_up(slots_per_bucket, kLanes)),
        // 0 is a meaningful setting here -- no overflow region, so over-wide
        // bands are rejected outright -- so it must not round up to one vector.
        overflow_slots_(overflow_slots == 0 ? 0
                                            : round_up(overflow_slots, kLanes)),
        max_span_(max_span < 1 ? 1 : max_span),
        arena_(num_buckets_ * slots_per_bucket_ + overflow_slots_) {
    assert(bucket_bits_ >= 1 && bucket_bits_ < kKeyBits);
    overflow_base_ = num_buckets_ * slots_per_bucket_;
    // One extra lock guards the overflow region.
    locks_.reset(new BucketLock[num_buckets_ + 1]);
  }

  size_t num_buckets() const { return num_buckets_; }
  size_t slots_per_bucket() const { return slots_per_bucket_; }
  size_t overflow_slots() const { return overflow_slots_; }
  size_t capacity() const { return arena_.capacity(); }
  size_t footprint_bytes() const { return arena_.footprint_bytes(); }
  size_t size() const { return arena_.size(); }

  // Slots touched by a point probe -- what actually sets probe latency.
  size_t probe_slots() const { return slots_per_bucket_ + overflow_slots_; }

  size_t bucket_of(KeyT k) const {
    return (size_t)(k >> (kKeyBits - bucket_bits_));
  }

  void clear() { arena_.clear(); }

  /* ---------------- probes ---------------- */

  Cursor<KeyT> probe_point(KeyT k) const {
    Cursor<KeyT> best = Cursor<KeyT>::none();
    arena_.scan_point(bucket_base(bucket_of(k)), slots_per_bucket_, k, best);
    if (overflow_slots_) {
      arena_.scan_point(overflow_base_, overflow_slots_, k, best);
    }
    return best;
  }

  int probe_range(KeyT start, KeyT end, Cursor<KeyT> *out, int max_out) const {
    assert(start <= end);
    const size_t bs = bucket_of(start), be = bucket_of(end);
    int kept = 0, found = 0;
    for (size_t b = bs; b <= be; ++b) {
      /* A band replicated across buckets would otherwise be reported once per
       * bucket the query spans. Count it only at its canonical owner, the
       * first bucket it occupies -- or at the first scanned bucket when the
       * band starts to the left of the query range. Exactly one replica of
       * every distinct band therefore passes. */
      arena_.scan_range(bucket_base(b), slots_per_bucket_, start, end, out,
                        max_out, kept, found,
                        [&](const Cursor<KeyT> &c) {
                          const size_t owner = bucket_of(c.lo);
                          return owner >= bs ? (owner == b) : (b == bs);
                        });
    }
    if (overflow_slots_) {
      arena_.scan_range(overflow_base_, overflow_slots_, start, end, out,
                        max_out, kept, found);
    }
    std::sort(out, out + kept, is_better<KeyT>);
    return found;
  }

  /* ---------------- maintenance ---------------- */

  /* Install a cursor, replicating it across the buckets its band overlaps.
   * All-or-nothing: if any target bucket is full nothing is written, so the
   * policy layer sees a clean failure and can evict first. */
  bool install(KeyT lo, KeyT hi, uint64_t node, uint32_t epoch, uint8_t depth) {
    if (node == kNoNode || lo > hi) return false;

    if (goes_to_overflow(lo, hi)) {
      if (!overflow_slots_) return false;
      BucketGuard g(locks_[num_buckets_]);
      const size_t s = arena_.find_free_slot_in(overflow_base_, overflow_slots_);
      if (s == overflow_base_ + overflow_slots_) return false;
      arena_.install_at(s, lo, hi, node, epoch, depth);
      return true;
    }

    const size_t bs = bucket_of(lo), be = bucket_of(hi);
    // Locks are taken in ascending bucket order here and everywhere else that
    // spans buckets, so replicated writes cannot deadlock against each other.
    for (size_t b = bs; b <= be; ++b) locks_[b].lock();
    bool ok = true;
    for (size_t b = bs; b <= be && ok; ++b) {
      if (arena_.find_free_slot_in(bucket_base(b), slots_per_bucket_) ==
          bucket_base(b) + slots_per_bucket_) {
        ok = false; // pre-check, so replication never half-commits
      }
    }
    if (ok) {
      for (size_t b = bs; b <= be; ++b) {
        const size_t s =
            arena_.find_free_slot_in(bucket_base(b), slots_per_bucket_);
        arena_.install_at(s, lo, hi, node, epoch, depth);
      }
    }
    for (size_t b = be + 1; b-- > bs;) locks_[b].unlock();
    return ok;
  }

  /* Remove every replica of a cursor. Identified by value rather than slot,
   * since replicas live at unrelated offsets in different buckets; returns
   * how many were removed. */
  int retire(const Cursor<KeyT> &c) {
    if (!c.valid()) return 0;
    int removed = 0;
    if (goes_to_overflow(c.lo, c.hi)) {
      BucketGuard g(locks_[num_buckets_]);
      removed += retire_matching(overflow_base_, overflow_slots_, c);
    } else {
      const size_t bs = bucket_of(c.lo), be = bucket_of(c.hi);
      for (size_t b = bs; b <= be; ++b) locks_[b].lock();
      for (size_t b = bs; b <= be; ++b) {
        removed += retire_matching(bucket_base(b), slots_per_bucket_, c);
      }
      for (size_t b = be + 1; b-- > bs;) locks_[b].unlock();
    }
    return removed;
  }

  /* Narrow every replica after a split (Sec. 4.4, Table 3 case ii).
   *
   * When the new band still occupies the same buckets -- the common case for a
   * deep-narrow cursor sitting inside one separator interval -- this is an
   * in-place edit of hi_. When the split drops the band into fewer buckets the
   * replicas must be re-placed, which is a retire + install. */
  bool narrow(const Cursor<KeyT> &c, KeyT new_hi) {
    if (!c.valid() || new_hi < c.lo || new_hi >= c.hi) return false;

    const bool was_overflow = goes_to_overflow(c.lo, c.hi);
    const bool now_overflow = goes_to_overflow(c.lo, new_hi);

    if (was_overflow == now_overflow &&
        (now_overflow || bucket_of(new_hi) == bucket_of(c.hi))) {
      // Placement unchanged: edit the band where it already lives.
      const size_t base = was_overflow ? overflow_base_ : bucket_base(bucket_of(c.lo));
      const size_t count = was_overflow ? overflow_slots_ : slots_per_bucket_;
      int touched = 0;
      if (was_overflow) {
        BucketGuard g(locks_[num_buckets_]);
        touched += narrow_matching(base, count, c, new_hi);
      } else {
        const size_t bs = bucket_of(c.lo), be = bucket_of(c.hi);
        for (size_t b = bs; b <= be; ++b) locks_[b].lock();
        for (size_t b = bs; b <= be; ++b) {
          touched += narrow_matching(bucket_base(b), slots_per_bucket_, c, new_hi);
        }
        for (size_t b = be + 1; b-- > bs;) locks_[b].unlock();
      }
      return touched > 0;
    }

    // Placement changed: re-place. The buckets needed afterwards are a subset
    // of those just freed unless the band moved out of overflow, so the
    // reinstall can only fail in that one case -- where dropping the cursor is
    // a miss, never a correctness problem.
    if (retire(c) == 0) return false;
    return install(c.lo, new_hi, c.node, c.epoch, c.depth);
  }

  /* ---------------- introspection (for the Sec. 5 policy layer) ---------- */

  size_t bucket_occupancy(size_t b) const {
    return arena_.size_in(bucket_base(b), slots_per_bucket_);
  }

  size_t overflow_occupancy() const {
    return overflow_slots_ ? arena_.size_in(overflow_base_, overflow_slots_) : 0;
  }

  // Shallowest cursor in the bucket a key maps to: the eviction candidate an
  // admission test weighs its candidate against.
  Cursor<KeyT> eviction_candidate(KeyT k) const {
    const size_t base = bucket_base(bucket_of(k));
    const size_t s = arena_.shallowest_in(base, slots_per_bucket_);
    if (s == base + slots_per_bucket_) return Cursor<KeyT>::none();
    return arena_.read(s);
  }

  const Table &arena() const { return arena_; }

private:
  static size_t round_up(size_t n, size_t m) {
    if (n == 0) return m;
    return ((n + m - 1) / m) * m;
  }

  size_t bucket_base(size_t b) const { return b * slots_per_bucket_; }

  bool goes_to_overflow(KeyT lo, KeyT hi) const {
    return (bucket_of(hi) - bucket_of(lo) + 1) > (size_t)max_span_;
  }

  static bool same_cursor(const Cursor<KeyT> &a, const Cursor<KeyT> &b) {
    return a.node == b.node && a.lo == b.lo && a.hi == b.hi;
  }

  int retire_matching(size_t base, size_t count, const Cursor<KeyT> &c) {
    int n = 0;
    for (size_t i = base; i < base + count; ++i) {
      if (!arena_.occupied(i)) continue;
      if (same_cursor(arena_.read(i), c)) {
        arena_.retire(i);
        ++n;
      }
    }
    return n;
  }

  int narrow_matching(size_t base, size_t count, const Cursor<KeyT> &c,
                      KeyT new_hi) {
    int n = 0;
    for (size_t i = base; i < base + count; ++i) {
      if (!arena_.occupied(i)) continue;
      if (same_cursor(arena_.read(i), c) && arena_.narrow(i, new_hi)) ++n;
    }
    return n;
  }

  int bucket_bits_;
  size_t num_buckets_;
  size_t slots_per_bucket_;
  size_t overflow_slots_;
  int max_span_;
  Table arena_;
  size_t overflow_base_;
  std::unique_ptr<BucketLock[]> locks_; // num_buckets_ + 1 (last = overflow)
};

} // namespace catalyst
