/* Unit test for the memory-side CATALYST traversal (cachepush::catalyst_traverse).
 *
 * No RDMA involved: the handler only does pointer arithmetic over a DSM base
 * address, so a tree built in ordinary local memory exercises exactly the code
 * that runs on the memory node. The tree is assembled with DEX's own node
 * primitives (leaf insert/split, inner insert) so the fence-key and separator
 * invariants are the real ones rather than something this test made up.
 *
 * Built by the existing test/ glob, so it links the DEX library like newbench.
 */

#include "cache/btree_rpc.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using cachepush::BTreeInner;
using cachepush::BTreeLeaf;
using cachepush::pageSize;
using namespace catalyst_wire;

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

namespace {

using Leaf = BTreeLeaf<Key, Value>;
using Inner = BTreeInner<Key>;

// A two-level tree laid out in a local buffer that stands in for a memory
// node's registered region.
struct Fixture {
  char *arena;
  uint64_t base;
  GlobalAddress root_addr, leafA_addr, leafB_addr;
  Inner *root;
  Leaf *leafA, *leafB;
  Key sep;

  Fixture() {
    // 8 slots is plenty; offset 0 is the root so a null GlobalAddress is never
    // a legal node here.
    if (posix_memalign((void **)&arena, 4096, 8 * pageSize) != 0) std::abort();
    memset(arena, 0, 8 * pageSize);
    base = (uint64_t)arena;

    root_addr = GlobalAddress(0, 0 * pageSize);
    leafA_addr = GlobalAddress(0, 1 * pageSize);
    leafB_addr = GlobalAddress(0, 2 * pageSize);

    leafA = new (arena + 1 * pageSize) Leaf(leafA_addr);
    leafB = new (arena + 2 * pageSize) Leaf(leafB_addr);

    // Fill A, then split it exactly the way DEX does, so the fence keys and
    // the separator are produced by the real code path.
    for (uint64_t i = 1; i <= Leaf::maxEntries; ++i) {
      leafA->insert(i * 10, i * 100);
    }
    leafA->split(sep, leafB, leafB_addr);

    /* Populate the root directly rather than through Inner::insert().
     *
     * Inner::lowerBound() on a count==0 node returns 1, not 0 (its do/while
     * probes keys[0] before testing the loop condition), so inserting the
     * first separator into an empty inner node lands it at index 1 and leaves
     * children[1] null. DEX never hits this because it fills a freshly created
     * root's slots by hand -- see the partitioned BTree constructor -- so this
     * mirrors what the real code does. */
    root = new (arena + 0 * pageSize) Inner(1, root_addr);
    root->keys[0] = sep;
    root->children[0] = leafA_addr;
    root->children[1] = leafB_addr;
    root->count = 1;
  }

  ~Fixture() { free(arena); }

  // Any key that lands in the left leaf / right leaf.
  Key key_in_A() const { return 10; }
  Key key_in_B() const { return sep + 10; }
};

int traverse(Fixture &f, GlobalAddress start, Key k, Op op, Value v_in,
             Value &v_out, GlobalAddress &leaf_out, PathEntry *path,
             uint8_t &npath) {
  return cachepush::catalyst_traverse(start, f.base, k, op, v_in, v_out,
                                      leaf_out, path, npath);
}

} // namespace

static void test_full_descent_records_path() {
  std::printf("descent from root records the whole path\n");
  Fixture f;
  Value v = 0;
  GlobalAddress leaf;
  PathEntry path[kMaxPathLen];
  uint8_t n = 0;

  const int st = traverse(f, f.root_addr, f.key_in_A(), Op::Lookup, 0, v, leaf,
                          path, n);
  CHECK(st == kFound);
  CHECK(v == 100); // key 10 -> value 100
  CHECK(leaf == f.leafA_addr);

  // Root then leaf, in descent order, with DEX's level convention.
  CHECK(n == 2);
  CHECK(path[0].addr == f.root_addr);
  CHECK(path[0].level == 1);
  CHECK(path[1].addr == f.leafA_addr);
  CHECK(path[1].level == 0);

  // Each entry carries the node's own fence keys: this is what makes a cursor
  // band free to compute (Sec. 4.1).
  CHECK(path[1].lo == f.leafA->min_limit_);
  CHECK(path[1].hi == f.leafA->max_limit_);
  CHECK(path[1].hi == f.sep);
}

static void test_cursor_short_circuits() {
  std::printf("resuming at a leaf cursor skips the inner levels\n");
  Fixture f;
  Value v = 0;
  GlobalAddress leaf;
  PathEntry path[kMaxPathLen];
  uint8_t n = 0;

  const int st = traverse(f, f.leafA_addr, f.key_in_A(), Op::Lookup, 0, v, leaf,
                          path, n);
  CHECK(st == kFound);
  CHECK(v == 100);
  CHECK(leaf == f.leafA_addr);
  // The whole point: one node visited instead of two.
  CHECK(n == 1);
  CHECK(path[0].addr == f.leafA_addr);
  CHECK(path[0].level == 0);
}

static void test_misaligned_cursor_is_rejected() {
  std::printf("cursor whose band no longer covers the key is rejected\n");
  Fixture f;
  Value v = 0;
  GlobalAddress leaf;
  PathEntry path[kMaxPathLen];
  uint8_t n = 0;

  // Key belongs in B, but the cursor names A -- exactly the state a split
  // leaves behind. The fence check must catch it rather than return a wrong
  // answer from the wrong leaf.
  const int st = traverse(f, f.leafA_addr, f.key_in_B(), Op::Lookup, 0, v, leaf,
                          path, n);
  CHECK(st == kStale);
  CHECK(n == 0);

  // And the same key from the root resolves correctly, which is the repair.
  const int st2 = traverse(f, f.root_addr, f.key_in_B(), Op::Lookup, 0, v, leaf,
                           path, n);
  CHECK(st2 == kFound || st2 == kAbsent);
  CHECK(leaf == f.leafB_addr);
}

static void test_absent_key() {
  std::printf("absent key reports the leaf it should have been in\n");
  Fixture f;
  Value v = 0;
  GlobalAddress leaf;
  PathEntry path[kMaxPathLen];
  uint8_t n = 0;

  const int st =
      traverse(f, f.root_addr, 15, Op::Lookup, 0, v, leaf, path, n); // 10,20,...
  CHECK(st == kAbsent);
  CHECK(leaf == f.leafA_addr); // still tells us where to put it
  CHECK(n == 2);
}

static void test_update_and_delete() {
  std::printf("update and delete apply on the memory node\n");
  Fixture f;
  Value v = 0;
  GlobalAddress leaf;
  PathEntry path[kMaxPathLen];
  uint8_t n = 0;

  CHECK(traverse(f, f.root_addr, 10, Op::Update, 777, v, leaf, path, n) == kFound);
  CHECK(traverse(f, f.root_addr, 10, Op::Lookup, 0, v, leaf, path, n) == kFound);
  CHECK(v == 777);
  CHECK(f.leafA->dirty);

  // Updating a key that is not there must not invent one.
  CHECK(traverse(f, f.root_addr, 15, Op::Update, 1, v, leaf, path, n) == kAbsent);

  CHECK(traverse(f, f.root_addr, 10, Op::Delete, 0, v, leaf, path, n) == kFound);
  CHECK(traverse(f, f.root_addr, 10, Op::Lookup, 0, v, leaf, path, n) == kAbsent);
  CHECK(traverse(f, f.root_addr, 10, Op::Delete, 0, v, leaf, path, n) == kAbsent);
}

static void test_insert_and_smo_bail() {
  std::printf("insert applies in place, and bails when a split is needed\n");
  Fixture f;
  Value v = 0;
  GlobalAddress leaf;
  PathEntry path[kMaxPathLen];
  uint8_t n = 0;

  // After the split leafA is about half full, so this fits.
  const uint8_t before = f.leafA->count;
  CHECK(traverse(f, f.root_addr, 15, Op::Insert, 1515, v, leaf, path, n) ==
        kInserted);
  CHECK(f.leafA->count == before + 1);
  CHECK(traverse(f, f.root_addr, 15, Op::Lookup, 0, v, leaf, path, n) == kFound);
  CHECK(v == 1515);

  // Inserting an existing key is an upsert and never structural.
  CHECK(traverse(f, f.root_addr, 15, Op::Insert, 99, v, leaf, path, n) == kFound);
  CHECK(traverse(f, f.root_addr, 15, Op::Lookup, 0, v, leaf, path, n) == kFound);
  CHECK(v == 99);

  // Fill the leaf to capacity, then the next new key must refuse to split.
  Key k = 11;
  while (f.leafA->count < Leaf::maxEntries) {
    if (k > f.sep) break;
    traverse(f, f.root_addr, k, Op::Insert, k, v, leaf, path, n);
    ++k;
  }
  CHECK(f.leafA->count == Leaf::maxEntries);
  int st = kInserted;
  while (k <= f.sep && st != kNeedsSMO) {
    st = traverse(f, f.root_addr, k, Op::Insert, k, v, leaf, path, n);
    ++k;
  }
  CHECK(st == kNeedsSMO);
  // A refused insert must leave the leaf untouched for the fallback path.
  CHECK(f.leafA->count == Leaf::maxEntries);
}

static void test_obsolete_node_rejected() {
  std::printf("cursor to an obsolete node is rejected\n");
  Fixture f;
  Value v = 0;
  GlobalAddress leaf;
  PathEntry path[kMaxPathLen];
  uint8_t n = 0;

  f.leafA->obsolete = true;
  CHECK(traverse(f, f.leafA_addr, f.key_in_A(), Op::Lookup, 0, v, leaf, path,
                 n) == kStale);
}

int main() {
  std::printf("catalyst_traverse: leaf fanout=%lu, inner fanout=%lu, "
              "TraverseMsg=%zuB (slot %d, GRH 40)\n\n",
              Leaf::maxEntries, Inner::maxEntries, sizeof(TraverseMsg),
              MESSAGE_SIZE);

  test_full_descent_records_path();
  test_cursor_short_circuits();
  test_misaligned_cursor_is_rejected();
  test_absent_key();
  test_update_and_delete();
  test_insert_and_smo_bail();
  test_obsolete_node_rejected();

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
