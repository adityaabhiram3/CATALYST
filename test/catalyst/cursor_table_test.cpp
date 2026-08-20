/* Correctness tests for the CATALYST cursor table.
 *
 * Standalone on purpose: the table has no RDMA dependency, so this builds and
 * runs without libmemcached/cityhash/ibverbs. Build via the catalyst_tests
 * target, or directly:
 *   g++ -std=c++17 -O3 -march=native -I include \
 *       test/catalyst/cursor_table_test.cpp -o cursor_table_test
 */

#include "catalyst/cursor_table.h"

#include <cstdio>
#include <random>
#include <vector>

using catalyst::Cursor;
using catalyst::CursorTable;
using catalyst::kNoNode;

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

using Table = CursorTable<uint64_t>;

/* Scalar reference implementation. The randomized tests cross-check the
 * vectorized probe against this, which is the only way to catch a lane-mask or
 * empty-slot-sentinel bug that happens to be invisible on hand-built cases. */
struct RefEntry {
  uint64_t lo, hi, node;
  uint32_t epoch;
  uint8_t depth;
};

static Cursor<uint64_t> ref_probe_point(const std::vector<RefEntry> &ents,
                                        uint64_t k) {
  Cursor<uint64_t> best = Cursor<uint64_t>::none();
  int best_depth = -1;
  for (const auto &e : ents) {
    if (e.node == kNoNode) continue;
    if (k < e.lo || k > e.hi) continue;
    const int d = (int)e.depth;
    const uint64_t w = e.hi - e.lo;
    if (d < best_depth) continue;
    if (d == best_depth && w >= best.width()) continue;
    best = Cursor<uint64_t>{e.node, e.lo, e.hi, e.epoch, e.depth};
    best_depth = d;
  }
  return best;
}

static int ref_probe_range(const std::vector<RefEntry> &ents, uint64_t s,
                           uint64_t e_) {
  int n = 0;
  for (const auto &e : ents) {
    if (e.node == kNoNode) continue;
    if (e.lo <= e_ && e.hi >= s) ++n;
  }
  return n;
}

static void test_empty_table() {
  std::printf("empty table\n");
  Table t(64);
  CHECK(t.size() == 0);
  CHECK(!t.probe_point(0).valid());
  CHECK(!t.probe_point(12345).valid());
  CHECK(!t.probe_point(Table::kKeyMax).valid());

  // A full-span range probe must not surface the inverted empty sentinel.
  Cursor<uint64_t> out[8];
  CHECK(t.probe_range(Table::kKeyMin, Table::kKeyMax, out, 8) == 0);
}

static void test_basic_containment() {
  std::printf("basic containment\n");
  Table t(32);
  t.install_at(0, 100, 200, 0xAAAA, 7, 9);

  CHECK(t.size() == 1);
  CHECK(!t.probe_point(99).valid());
  CHECK(t.probe_point(100).valid());  // inclusive lower bound
  CHECK(t.probe_point(150).valid());
  CHECK(t.probe_point(200).valid());  // inclusive upper bound
  CHECK(!t.probe_point(201).valid());

  const auto c = t.probe_point(150);
  CHECK(c.node == 0xAAAA);
  CHECK(c.lo == 100 && c.hi == 200);
  CHECK(c.epoch == 7);
  CHECK(c.depth == 9);
}

static void test_deepest_wins() {
  std::printf("deepest match wins\n");
  Table t(32);
  // Nested bands, as produced by capturing a whole root-to-leaf path.
  t.install_at(0, 0, 1000, 0x1, 1, 2);   // shallow, wide
  t.install_at(1, 400, 600, 0x2, 1, 6);  // mid
  t.install_at(2, 480, 520, 0x3, 1, 10); // deep, narrow

  CHECK(t.probe_point(500).node == 0x3); // all three contain it
  CHECK(t.probe_point(550).node == 0x2); // deep band misses
  CHECK(t.probe_point(900).node == 0x1); // only the wide one
}

static void test_depth_tie_breaks_on_width() {
  std::printf("equal depth breaks toward narrower band\n");
  Table t(32);
  t.install_at(0, 0, 1000, 0x1, 1, 5);
  t.install_at(1, 400, 600, 0x2, 1, 5); // same depth, narrower
  CHECK(t.probe_point(500).node == 0x2);
}

static void test_range_overlap_boundaries() {
  std::printf("range overlap boundaries\n");
  Table t(32);
  t.install_at(0, 100, 200, 0xA, 1, 5);
  Cursor<uint64_t> out[8];

  CHECK(t.probe_range(0, 99, out, 8) == 0);    // strictly left
  CHECK(t.probe_range(0, 100, out, 8) == 1);   // touches lower bound
  CHECK(t.probe_range(200, 999, out, 8) == 1); // touches upper bound
  CHECK(t.probe_range(201, 999, out, 8) == 0); // strictly right
  CHECK(t.probe_range(120, 130, out, 8) == 1); // contained
  CHECK(t.probe_range(0, 999, out, 8) == 1);   // contains
}

static void test_range_returns_deepest_first() {
  std::printf("range results ordered deepest first\n");
  Table t(32);
  t.install_at(0, 0, 1000, 0x1, 1, 2);
  t.install_at(1, 400, 600, 0x2, 1, 6);
  t.install_at(2, 480, 520, 0x3, 1, 10);

  Cursor<uint64_t> out[8];
  const int n = t.probe_range(490, 510, out, 8);
  CHECK(n == 3);
  CHECK(out[0].depth == 10);
  CHECK(out[1].depth == 6);
  CHECK(out[2].depth == 2);
}

static void test_range_reports_overflow() {
  std::printf("range reports total found beyond max_out\n");
  Table t(32);
  for (int i = 0; i < 5; ++i) {
    t.install_at(i, 0, 1000, 0x10 + i, 1, (uint8_t)(i + 1));
  }
  Cursor<uint64_t> out[2];
  const int n = t.probe_range(500, 500, out, 2);
  CHECK(n == 5);            // total found
  CHECK(out[0].depth == 5); // deepest two written
  CHECK(out[1].depth == 4);
}

static void test_retire_and_reuse() {
  std::printf("retire and reuse\n");
  Table t(32);
  t.install_at(3, 100, 200, 0xA, 1, 5);
  CHECK(t.probe_point(150).valid());
  CHECK(t.find_free_slot() == 0);

  t.retire(3);
  CHECK(!t.probe_point(150).valid());
  CHECK(t.size() == 0);
  CHECK(!t.occupied(3));

  t.install_at(3, 300, 400, 0xB, 2, 6);
  CHECK(t.probe_point(350).node == 0xB);
  CHECK(!t.probe_point(150).valid());
}

static void test_narrow_after_split() {
  std::printf("narrow after split (Table 3, case ii)\n");
  Table t(32);
  t.install_at(0, 100, 200, 0xA, 1, 5);

  CHECK(t.narrow(0, 150));
  CHECK(t.probe_point(120).valid());  // still covered
  CHECK(!t.probe_point(180).valid()); // moved to the new sibling
  CHECK(t.probe_point(120).node == 0xA);
  CHECK(t.probe_point(150).hi == 150);

  CHECK(!t.narrow(0, 200)); // widening is not a narrow
  CHECK(!t.narrow(0, 50));  // below lo
  CHECK(!t.narrow(1, 100)); // empty slot
}

static void test_key_extremes() {
  std::printf("key-space extremes\n");
  Table t(32);
  t.install_at(0, Table::kKeyMin, Table::kKeyMin, 0xA, 1, 5);
  t.install_at(1, Table::kKeyMax, Table::kKeyMax, 0xB, 1, 5);

  CHECK(t.probe_point(Table::kKeyMin).node == 0xA);
  CHECK(t.probe_point(Table::kKeyMax).node == 0xB);
  CHECK(!t.probe_point(1).valid());

  // The sentinel is lo=MAX/hi=MIN; a real band at those keys must still work.
  Cursor<uint64_t> out[8];
  CHECK(t.probe_range(Table::kKeyMin, Table::kKeyMax, out, 8) == 2);
}

static void test_capacity_rounding() {
  std::printf("capacity rounds up to whole vectors\n");
  Table t(5);
  CHECK(t.capacity() % Table::kLanes == 0);
  CHECK(t.capacity() >= 5);
  // Every rounded-up slot must be probeable and start empty.
  for (size_t i = 0; i < t.capacity(); ++i) CHECK(!t.occupied(i));
  t.install_at(t.capacity() - 1, 10, 20, 0xC, 1, 3);
  CHECK(t.probe_point(15).node == 0xC);
}

static void test_randomized_against_reference() {
  std::printf("randomized cross-check vs scalar reference\n");
  std::mt19937_64 rng(0xCA7A1751);
  const size_t kCap = 512;
  Table t(kCap);
  std::vector<RefEntry> ref(t.capacity(),
                            RefEntry{0, 0, kNoNode, 0, catalyst::kNoDepth});

  std::uniform_int_distribution<uint64_t> key_d(0, 100000);
  std::uniform_int_distribution<size_t> slot_d(0, t.capacity() - 1);
  std::uniform_int_distribution<int> depth_d(1, 11);
  std::uniform_int_distribution<int> op_d(0, 99);

  int point_mismatch = 0, range_mismatch = 0;

  for (int iter = 0; iter < 20000; ++iter) {
    const int op = op_d(rng);
    const size_t s = slot_d(rng);

    if (op < 55) { // install
      uint64_t a = key_d(rng), b = key_d(rng);
      if (a > b) std::swap(a, b);
      const uint64_t node = 0x1000 + iter;
      const uint32_t ep = (uint32_t)(iter & 0xFFFF);
      const uint8_t d = (uint8_t)depth_d(rng);
      t.install_at(s, a, b, node, ep, d);
      ref[s] = RefEntry{a, b, node, ep, d};
    } else if (op < 65) { // retire
      t.retire(s);
      ref[s] = RefEntry{0, 0, kNoNode, 0, catalyst::kNoDepth};
    } else if (op < 70) { // narrow
      if (ref[s].node != kNoNode && ref[s].hi > ref[s].lo) {
        const uint64_t nh = ref[s].lo + (ref[s].hi - ref[s].lo) / 2;
        if (t.narrow(s, nh)) ref[s].hi = nh;
      }
    } else if (op < 90) { // point probe
      const uint64_t k = key_d(rng);
      const auto got = t.probe_point(k);
      const auto want = ref_probe_point(ref, k);
      if (got.node != want.node || got.lo != want.lo || got.hi != want.hi ||
          got.depth != want.depth) {
        ++point_mismatch;
      }
    } else { // range probe
      uint64_t a = key_d(rng), b = key_d(rng);
      if (a > b) std::swap(a, b);
      Cursor<uint64_t> out[64];
      const int got = t.probe_range(a, b, out, 64);
      const int want = ref_probe_range(ref, a, b);
      if (got != want) ++range_mismatch;
    }
  }

  CHECK(point_mismatch == 0);
  CHECK(range_mismatch == 0);
  if (point_mismatch || range_mismatch) {
    std::printf("  point mismatches=%d range mismatches=%d\n", point_mismatch,
                range_mismatch);
  }
}

static void test_footprint_reporting() {
  std::printf("footprint accounting\n");
  Table t(1024);
  // 8(lo)+8(hi)+8(node)+4(epoch)+1(depth) = 29B per cursor with 64-bit keys.
  CHECK(t.footprint_bytes() == t.capacity() * 29);
}

int main() {
  std::printf("cursor table: %d-byte vectors, %d lanes of %zu-byte keys\n\n",
              (int)sizeof(Table::Vec), Table::kLanes, sizeof(uint64_t));

  test_empty_table();
  test_basic_containment();
  test_deepest_wins();
  test_depth_tie_breaks_on_width();
  test_range_overlap_boundaries();
  test_range_returns_deepest_first();
  test_range_reports_overflow();
  test_retire_and_reuse();
  test_narrow_after_split();
  test_key_extremes();
  test_capacity_rounding();
  test_randomized_against_reference();
  test_footprint_reporting();

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
