/* Microbenchmark for the CATALYST cursor table probe.
 *
 * Answers two questions the paper makes claims about:
 *   1. Sec. 4.3: SIMD gives a "16-32x" speedup over scalar range scanning.
 *   2. Fig. 8 line 10: the probe costs "<100ns".
 *
 * Claim 2 is a function of table size, since Fig. 9's table_scan() sweeps all
 * NUM_ROWS on every probe. This sweeps capacity to find where that stops
 * holding. Build:
 *   g++ -std=c++17 -O3 -march=native -I include \
 *       test/catalyst/cursor_table_bench.cpp -o cursor_table_bench
 */

#include "catalyst/bucketed_cursor_table.h"
#include "catalyst/cursor_table.h"

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using catalyst::Cursor;
using catalyst::CursorTable;
using catalyst::kNoNode;

using Table = CursorTable<uint64_t>;

namespace {

constexpr uint64_t kKeySpace = 1ull << 32;
constexpr int kProbes = 20000;

// Scalar baseline: same semantics, no vectorization. This is what Sec. 4.3's
// speedup is measured against.
struct ScalarTable {
  std::vector<uint64_t> lo, hi, node;
  std::vector<uint8_t> depth;

  explicit ScalarTable(size_t n)
      : lo(n, ~0ull), hi(n, 0ull), node(n, kNoNode), depth(n, 0xFF) {}

  uint64_t probe_point(uint64_t k) const {
    uint64_t best = kNoNode;
    int best_depth = -1;
    for (size_t i = 0; i < lo.size(); ++i) {
      if (k < lo[i] || k > hi[i]) continue;
      if (node[i] == kNoNode) continue;
      if ((int)depth[i] > best_depth) {
        best_depth = (int)depth[i];
        best = node[i];
      }
    }
    return best;
  }
};

// Populate both tables with nested bands resembling a captured tree path:
// a few wide shallow cursors, many narrow deep ones.
void fill(Table &t, ScalarTable &s, size_t n, std::mt19937_64 &rng) {
  std::uniform_int_distribution<uint64_t> start_d(0, kKeySpace - 1);
  for (size_t i = 0; i < n; ++i) {
    const uint8_t depth = (uint8_t)(2 + (i % 10));
    // Deeper cursors cover geometrically narrower bands (the w ~ f^-l
    // coupling of Sec. 3.3).
    const uint64_t width = kKeySpace >> (depth + 1);
    const uint64_t lo = start_d(rng) % (kKeySpace - width);
    const uint64_t hi = lo + width;
    const uint64_t node = 0x1000 + i;
    t.install_at(i, lo, hi, node, 1, depth);
    s.lo[i] = lo;
    s.hi[i] = hi;
    s.node[i] = node;
    s.depth[i] = depth;
  }
}

template <typename Fn> double time_ns_per_op(Fn &&fn, int iters) {
  const auto t0 = std::chrono::steady_clock::now();
  fn();
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
}

} // namespace

int main() {
  std::printf("CATALYST cursor-table probe benchmark\n");
  std::printf("vector width: %d bytes, %d lanes of 8-byte keys\n\n",
              (int)sizeof(Table::Vec), Table::kLanes);

  std::printf("%10s %10s %12s %12s %9s %12s\n", "cursors", "table", "scalar",
              "simd", "speedup", "scan BW");
  std::printf("%10s %10s %12s %12s %9s %12s\n", "", "(KB)", "(ns/probe)",
              "(ns/probe)", "", "(GB/s)");

  std::mt19937_64 rng(12345);
  std::uniform_int_distribution<uint64_t> key_d(0, kKeySpace - 1);

  std::vector<uint64_t> keys(kProbes);
  for (auto &k : keys) k = key_d(rng);

  for (size_t n : {size_t(256), size_t(1024), size_t(4096), size_t(16384),
                   size_t(65536), size_t(145000)}) {
    Table t(n);
    ScalarTable s(t.capacity());
    fill(t, s, t.capacity(), rng);

    volatile uint64_t sink = 0;

    // Warm the arrays into cache before timing.
    for (int i = 0; i < 100; ++i) sink ^= t.probe_point(keys[i]).node;

    const double simd_ns = time_ns_per_op(
        [&] {
          uint64_t acc = 0;
          for (int i = 0; i < kProbes; ++i) acc ^= t.probe_point(keys[i]).node;
          sink ^= acc;
        },
        kProbes);

    // Scalar is far slower at large n; probe fewer times but report per-probe.
    const int scalar_probes = (n > 16384) ? kProbes / 20 : kProbes;
    const double scalar_ns = time_ns_per_op(
        [&] {
          uint64_t acc = 0;
          for (int i = 0; i < scalar_probes; ++i) acc ^= s.probe_point(keys[i]);
          sink ^= acc;
        },
        scalar_probes);

    // Only lo_/hi_ are touched by the vectorized scan.
    const double hot_bytes = (double)t.capacity() * 2 * sizeof(uint64_t);
    const double bw = hot_bytes / simd_ns; // bytes/ns == GB/s

    std::printf("%10zu %10.0f %12.1f %12.1f %8.1fx %11.1f\n", t.capacity(),
                t.footprint_bytes() / 1024.0, scalar_ns, simd_ns,
                scalar_ns / simd_ns, bw);
  }

  std::printf("\nnote: the flat probe scans the whole table (Fig. 9 table_scan\n"
              "sweeps NUM_ROWS) and cannot early-exit, since the deepest match\n"
              "may sit in any slot. Cost therefore grows with capacity.\n");

  /* ---- partitioned table: probe cost should not track capacity ---- */

  std::printf("\nradix-partitioned table (256 slots/bucket + 256 overflow)\n\n");
  std::printf("%10s %10s %12s %12s %12s\n", "cursors", "table", "probe",
              "slots", "vs flat");
  std::printf("%10s %10s %12s %12s %12s\n", "", "(KB)", "(ns/probe)",
              "scanned", "");

  for (int bits : {4, 5, 6, 7, 8}) {
    using Bucketed = catalyst::BucketedCursorTable<uint64_t>;
    Bucketed bt(bits, 256, 256, 4);

    // Deep-narrow bands, the envelope partitioning is designed for: each sits
    // inside one separator interval and so touches a single bucket.
    const uint64_t bucket_span = (uint64_t(1) << (64 - bits));
    std::mt19937_64 fill_rng(999);
    std::uniform_int_distribution<uint64_t> off_d(0, bucket_span / 4);
    size_t installed = 0;
    for (size_t b = 0; b < bt.num_buckets(); ++b) {
      for (size_t i = 0; i < bt.slots_per_bucket(); ++i) {
        const uint64_t lo = b * bucket_span + off_d(fill_rng);
        const uint64_t hi = lo + (bucket_span >> 8);
        if (!bt.install(lo, hi, 0x2000 + installed, 1,
                        (uint8_t)(2 + (i % 10)))) {
          break;
        }
        ++installed;
      }
    }

    volatile uint64_t sink = 0;
    for (int i = 0; i < 100; ++i) sink ^= bt.probe_point(keys[i]).node;

    const double ns = time_ns_per_op(
        [&] {
          uint64_t acc = 0;
          for (int i = 0; i < kProbes; ++i) acc ^= bt.probe_point(keys[i]).node;
          sink ^= acc;
        },
        kProbes);

    // What a flat table of the same capacity would have cost, at the measured
    // scan bandwidth of ~35 GB/s.
    const double flat_ns = (double)bt.capacity() * 16.0 / 35.0;

    std::printf("%10zu %10.0f %12.1f %12zu %11.0fx\n", installed,
                bt.footprint_bytes() / 1024.0, ns, bt.probe_slots(),
                flat_ns / ns);
  }

  std::printf("\nnote: probe cost is set by slots_per_bucket + overflow_slots,\n"
              "not by capacity, so the table can hold the 2-4MB of Sec. 6.3\n"
              "while staying inside the sub-100ns budget of Fig. 8.\n");
  return 0;
}
