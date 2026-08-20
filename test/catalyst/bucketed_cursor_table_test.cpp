/* Correctness tests for the radix-partitioned CATALYST cursor table.
 *
 * The load-bearing test is the randomized cross-check: the bucketed table must
 * return exactly what the flat table returns for the same cursor set, since
 * the flat table is itself cross-checked against a scalar reference in
 * cursor_table_test.cpp. Replication across buckets is the part most likely to
 * diverge, so band widths here deliberately straddle the overflow threshold.
 */

#include "catalyst/bucketed_cursor_table.h"
#include "catalyst/cursor_table.h"

#include <cstdio>
#include <random>
#include <vector>

using catalyst::BucketedCursorTable;
using catalyst::Cursor;
using catalyst::CursorTable;

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    ++g_checks;                                                                \
    if (!(cond)) {                                                             \
      std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);            \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

using Bucketed = BucketedCursorTable<uint64_t>;
using Flat = CursorTable<uint64_t>;

// bucket_bits=4 -> 16 buckets, each spanning 2^60 keys.
static constexpr int kBits = 4;
static constexpr uint64_t kBucketSpan = uint64_t(1) << (64 - kBits);

static void test_geometry() {
  std::printf("geometry and probe cost\n");
  Bucketed t(kBits, 32, 32, 4);
  CHECK(t.num_buckets() == 16);
  CHECK(t.capacity() == 16 * t.slots_per_bucket() + t.overflow_slots());

  // The point of partitioning: probe cost is independent of capacity.
  Bucketed big(8, 256, 256, 4);
  CHECK(big.probe_slots() == 512);
  CHECK(big.capacity() == 256 * 256 + 256);
  CHECK(big.probe_slots() < big.capacity() / 100);

  CHECK(t.bucket_of(0) == 0);
  CHECK(t.bucket_of(kBucketSpan) == 1);
  CHECK(t.bucket_of(~uint64_t(0)) == 15);
}

static void test_single_bucket_band() {
  std::printf("band inside one bucket\n");
  Bucketed t(kBits, 32, 32, 4);
  const uint64_t lo = kBucketSpan * 3 + 100;
  const uint64_t hi = kBucketSpan * 3 + 200;
  CHECK(t.install(lo, hi, 0xA, 1, 9));
  CHECK(t.size() == 1); // one bucket, one replica

  CHECK(t.probe_point(lo).node == 0xA);
  CHECK(t.probe_point(hi).node == 0xA);
  CHECK(!t.probe_point(lo - 1).valid());
  CHECK(!t.probe_point(hi + 1).valid());
  CHECK(t.bucket_occupancy(3) == 1);
  CHECK(t.bucket_occupancy(4) == 0);
}

static void test_multi_bucket_band_replicates() {
  std::printf("band spanning buckets is replicated\n");
  Bucketed t(kBits, 32, 32, 4);
  // Spans buckets 2, 3, 4.
  const uint64_t lo = kBucketSpan * 2 + 5;
  const uint64_t hi = kBucketSpan * 4 + 5;
  CHECK(t.install(lo, hi, 0xB, 1, 6));
  CHECK(t.size() == 3);
  CHECK(t.bucket_occupancy(2) == 1);
  CHECK(t.bucket_occupancy(3) == 1);
  CHECK(t.bucket_occupancy(4) == 1);
  CHECK(t.overflow_occupancy() == 0);

  // Reachable from every bucket it spans, including the interior one where
  // neither endpoint lives.
  CHECK(t.probe_point(lo).node == 0xB);
  CHECK(t.probe_point(kBucketSpan * 3 + 999).node == 0xB);
  CHECK(t.probe_point(hi).node == 0xB);
  CHECK(!t.probe_point(kBucketSpan * 5).valid());
}

static void test_wide_band_goes_to_overflow() {
  std::printf("band wider than max_span goes to overflow\n");
  Bucketed t(kBits, 32, 32, /*max_span=*/4);
  // Spans buckets 1..10, well over max_span.
  const uint64_t lo = kBucketSpan * 1;
  const uint64_t hi = kBucketSpan * 10;
  CHECK(t.install(lo, hi, 0xC, 1, 2));
  CHECK(t.overflow_occupancy() == 1);
  CHECK(t.size() == 1); // single copy, not replicated
  for (size_t b = 0; b < t.num_buckets(); ++b) CHECK(t.bucket_occupancy(b) == 0);

  // Still found, because every probe also scans overflow.
  CHECK(t.probe_point(kBucketSpan * 5).node == 0xC);
  CHECK(!t.probe_point(kBucketSpan * 11).valid());
}

static void test_deepest_across_bucket_and_overflow() {
  std::printf("deepest wins across bucket and overflow\n");
  Bucketed t(kBits, 32, 32, 4);
  const uint64_t k = kBucketSpan * 5 + 77;
  CHECK(t.install(0, ~uint64_t(0), 0x1, 1, 2));                 // overflow, shallow
  CHECK(t.install(kBucketSpan * 5, kBucketSpan * 5 + 1000, 0x2, 1, 10)); // bucket, deep
  CHECK(t.probe_point(k).node == 0x2);

  // Remove the deep one and the shallow overflow cursor takes over.
  CHECK(t.retire(t.probe_point(k)) == 1);
  CHECK(t.probe_point(k).node == 0x1);
}

static void test_range_counts_each_band_once() {
  std::printf("range probe counts a replicated band once\n");
  Bucketed t(kBits, 32, 32, 4);
  const uint64_t lo = kBucketSpan * 2 + 5;
  const uint64_t hi = kBucketSpan * 4 + 5;
  CHECK(t.install(lo, hi, 0xB, 1, 6)); // 3 replicas
  CHECK(t.size() == 3);

  Cursor<uint64_t> out[16];
  // A query spanning all three buckets must still report one cursor.
  CHECK(t.probe_range(kBucketSpan * 2, kBucketSpan * 5, out, 16) == 1);
  // And so must a query starting to the left of the band.
  CHECK(t.probe_range(0, kBucketSpan * 5, out, 16) == 1);
  // And one entirely inside the band's interior bucket.
  CHECK(t.probe_range(kBucketSpan * 3, kBucketSpan * 3 + 10, out, 16) == 1);
  CHECK(out[0].node == 0xB);
}

static void test_retire_removes_all_replicas() {
  std::printf("retire removes every replica\n");
  Bucketed t(kBits, 32, 32, 4);
  const uint64_t lo = kBucketSpan * 2 + 5;
  const uint64_t hi = kBucketSpan * 4 + 5;
  CHECK(t.install(lo, hi, 0xB, 1, 6));
  CHECK(t.size() == 3);

  const auto c = t.probe_point(lo);
  CHECK(t.retire(c) == 3);
  CHECK(t.size() == 0);
  CHECK(!t.probe_point(kBucketSpan * 3).valid());
}

static void test_narrow_in_place_and_across_buckets() {
  std::printf("narrow, in place and across buckets\n");
  {
    // Same bucket before and after: in-place edit.
    Bucketed t(kBits, 32, 32, 4);
    const uint64_t lo = kBucketSpan * 3 + 100;
    CHECK(t.install(lo, lo + 500, 0xA, 1, 9));
    CHECK(t.narrow(t.probe_point(lo), lo + 200));
    CHECK(t.size() == 1);
    CHECK(t.probe_point(lo + 150).valid());
    CHECK(!t.probe_point(lo + 300).valid());
  }
  {
    // Split drops the band from three buckets into one: replicas re-placed.
    Bucketed t(kBits, 32, 32, 4);
    const uint64_t lo = kBucketSpan * 2 + 5;
    const uint64_t hi = kBucketSpan * 4 + 5;
    CHECK(t.install(lo, hi, 0xB, 1, 6));
    CHECK(t.size() == 3);
    CHECK(t.narrow(t.probe_point(lo), lo + 10));
    CHECK(t.size() == 1); // only bucket 2 now
    CHECK(t.bucket_occupancy(2) == 1);
    CHECK(t.bucket_occupancy(3) == 0);
    CHECK(t.bucket_occupancy(4) == 0);
    CHECK(t.probe_point(lo + 5).valid());
    CHECK(!t.probe_point(kBucketSpan * 3).valid());
  }
  {
    // Overflow band narrowed until it fits the bucket array.
    Bucketed t(kBits, 32, 32, 4);
    CHECK(t.install(0, kBucketSpan * 10, 0xC, 1, 2));
    CHECK(t.overflow_occupancy() == 1);
    CHECK(t.narrow(t.probe_point(kBucketSpan * 5), 100));
    CHECK(t.overflow_occupancy() == 0);
    CHECK(t.bucket_occupancy(0) == 1);
    CHECK(t.probe_point(50).node == 0xC);
    CHECK(!t.probe_point(kBucketSpan * 5).valid());
  }
}

static void test_install_all_or_nothing() {
  std::printf("install is all-or-nothing when a bucket is full\n");
  // One slot per bucket (rounded up to a vector), no overflow.
  Bucketed t(kBits, 1, 0, 4);
  const size_t per = t.slots_per_bucket();

  // Fill bucket 3 completely with bands local to it.
  for (size_t i = 0; i < per; ++i) {
    CHECK(t.install(kBucketSpan * 3 + i * 10, kBucketSpan * 3 + i * 10 + 1,
                    0x100 + i, 1, 5));
  }
  CHECK(t.bucket_occupancy(3) == per);

  // A band spanning buckets 2..4 must now fail, leaving 2 and 4 untouched.
  const size_t before2 = t.bucket_occupancy(2);
  const size_t before4 = t.bucket_occupancy(4);
  CHECK(!t.install(kBucketSpan * 2 + 1, kBucketSpan * 4 + 1, 0xDEAD, 1, 6));
  CHECK(t.bucket_occupancy(2) == before2);
  CHECK(t.bucket_occupancy(4) == before4);

  // With no overflow region, an over-wide band cannot be installed at all.
  CHECK(!t.install(0, ~uint64_t(0), 0xBEEF, 1, 1));
}

static void test_randomized_matches_flat_table() {
  std::printf("randomized cross-check vs flat table\n");
  std::mt19937_64 rng(0xB0CCE7ED);

  Bucketed bt(kBits, 64, 64, 4);
  Flat flat(bt.capacity());
  std::vector<Cursor<uint64_t>> live;

  std::uniform_int_distribution<uint64_t> key_d(0, ~uint64_t(0));
  std::uniform_int_distribution<int> depth_d(1, 11);
  std::uniform_int_distribution<int> wshift_d(40, 62); // straddles overflow
  std::uniform_int_distribution<int> op_d(0, 99);

  int point_mismatch = 0, range_mismatch = 0;
  uint64_t uniq = 0;

  for (int iter = 0; iter < 30000; ++iter) {
    const int op = op_d(rng);

    if (op < 45) { // install into both, or neither
      const uint64_t lo = key_d(rng);
      // Unique widths keep is_better a total order, so "best" is unambiguous
      // and the two tables cannot legitimately disagree on ties.
      const uint64_t w = (uint64_t(1) << wshift_d(rng)) + (++uniq);
      if (lo > ~uint64_t(0) - w) continue;
      const uint64_t hi = lo + w;
      const uint64_t node = 0x1000 + iter;
      const uint8_t d = (uint8_t)depth_d(rng);

      if (bt.install(lo, hi, node, 1, d)) {
        const size_t s = flat.find_free_slot();
        if (s < flat.capacity()) {
          flat.install_at(s, lo, hi, node, 1, d);
          live.push_back(Cursor<uint64_t>{node, lo, hi, 1, d});
        } else {
          bt.retire(Cursor<uint64_t>{node, lo, hi, 1, d}); // keep in lockstep
        }
      }
    } else if (op < 60 && !live.empty()) { // retire from both
      std::uniform_int_distribution<size_t> pick(0, live.size() - 1);
      const size_t i = pick(rng);
      const Cursor<uint64_t> c = live[i];
      bt.retire(c);
      for (size_t s = 0; s < flat.capacity(); ++s) {
        if (flat.occupied(s)) {
          const auto f = flat.read(s);
          if (f.node == c.node && f.lo == c.lo && f.hi == c.hi) flat.retire(s);
        }
      }
      live[i] = live.back();
      live.pop_back();
    } else if (op < 85) { // point probe
      const uint64_t k = key_d(rng);
      const auto a = bt.probe_point(k);
      const auto b = flat.probe_point(k);
      if (a.node != b.node || a.lo != b.lo || a.hi != b.hi ||
          a.depth != b.depth) {
        ++point_mismatch;
      }
    } else { // range probe
      uint64_t s = key_d(rng), e = key_d(rng);
      if (s > e) std::swap(s, e);
      Cursor<uint64_t> oa[32], ob[32];
      const int na = bt.probe_range(s, e, oa, 32);
      const int nb = flat.probe_range(s, e, ob, 32);
      if (na != nb) {
        ++range_mismatch;
      } else {
        const int k = na < 32 ? na : 32;
        for (int j = 0; j < k; ++j) {
          if (oa[j].node != ob[j].node) { ++range_mismatch; break; }
        }
      }
    }
  }

  std::printf("  (%zu cursors live at end, %zu slots occupied)\n", live.size(),
              bt.size());
  CHECK(point_mismatch == 0);
  CHECK(range_mismatch == 0);
  if (point_mismatch || range_mismatch) {
    std::printf("  point mismatches=%d range mismatches=%d\n", point_mismatch,
                range_mismatch);
  }
}

int main() {
  std::printf("bucketed cursor table: %d lanes/vector\n\n", Bucketed::kLanes);

  test_geometry();
  test_single_bucket_band();
  test_multi_bucket_band_replicates();
  test_wide_band_goes_to_overflow();
  test_deepest_across_bucket_and_overflow();
  test_range_counts_each_band_once();
  test_retire_removes_all_replicas();
  test_narrow_in_place_and_across_buckets();
  test_install_all_or_nothing();
  test_randomized_matches_flat_table();

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
