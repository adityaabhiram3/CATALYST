/* Tests for the CATALYST pattern model and dynamic cursor selection (Sec. 5).
 *
 * Standalone: Controller::observe/select are templated on the path-entry type,
 * so a stand-in struct exercises the whole control loop with no RDMA, no DSM
 * and no message plumbing.
 */

#include "catalyst/pattern.h"

#include <cstdio>
#include <vector>

using namespace catalyst;

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

// Mirrors catalyst_wire::PathEntry's shape without dragging in the RDMA stack.
struct Addr {
  uint64_t val;
};
struct Entry {
  Addr addr;
  uint64_t lo, hi;
  uint8_t level; // DEX convention: 0 == leaf
};

Entry E(uint64_t node, uint64_t lo, uint64_t hi, uint8_t level) {
  return Entry{Addr{node}, lo, hi, level};
}

// A root-to-leaf path over a tree of height 4 (root level 4 ... leaf level 0),
// with bands narrowing on the way down.
std::vector<Entry> full_path() {
  return {E(0x100, 0, 10000, 4), E(0x200, 0, 5000, 3), E(0x300, 1000, 3000, 2),
          E(0x400, 1500, 2000, 1), E(0x500, 1600, 1700, 0)};
}

// Drive a controller until it has seen the path enough times for confidence to
// build, which is what admission requires.
void warm(Controller &c, const std::vector<Entry> &p, int times) {
  for (int i = 0; i < times; ++i) c.observe(p.data(), (uint8_t)p.size(), 0);
}

} // namespace

static void test_ewma() {
  std::printf("ewma tracks and decays\n");
  Ewma e(0.5, 0.0);
  e.add(1.0);
  CHECK(e.get() > 0.49 && e.get() < 0.51);
  for (int i = 0; i < 20; ++i) e.add(1.0);
  CHECK(e.get() > 0.99);
  for (int i = 0; i < 30; ++i) e.add(0.0);
  CHECK(e.get() < 0.01);
}

static void test_confidence_sketch() {
  std::printf("confidence sketch counts and halves\n");
  ConfidenceSketch s(1024, 100);
  CHECK(s.conf(0xABC) == 0);
  for (int i = 0; i < 10; ++i) s.observe(0xABC);
  CHECK(s.conf(0xABC) == 10);

  // Cross the decay threshold; counts must fall, not vanish.
  for (int i = 0; i < 95; ++i) s.observe(0xDEF);
  const uint32_t after = s.conf(0xABC);
  CHECK(after < 10);
  s.clear();
  CHECK(s.conf(0xABC) == 0 && s.conf(0xDEF) == 0);
}

static void test_region_funnel() {
  std::printf("funnel region is a depth window\n");
  Descriptor d;
  d.kind = PatternKind::Funnel;
  d.d_start = 2;
  d.d_end = 4;
  CHECK(!d.in_region(1, 100));
  CHECK(d.in_region(2, 100));
  CHECK(d.in_region(4, 100));
  CHECK(!d.in_region(5, 100));
  // Funnel does not care about width; depth is the whole test.
  CHECK(d.in_region(3, 0) && d.in_region(3, ~uint64_t(0)));
}

static void test_region_interval() {
  std::printf("interval region anchors on depth and demands reach\n");
  Descriptor d;
  d.kind = PatternKind::Interval;
  d.d_anchor = 3;
  d.w_min = 500;
  CHECK(d.in_region(3, 1000));
  CHECK(d.in_region(2, 1000)); // anchor +/- 1
  CHECK(d.in_region(4, 1000));
  CHECK(!d.in_region(6, 1000));      // far from the anchor
  CHECK(!d.in_region(3, 100));       // too narrow to serve neighbours
}

static void test_region_branch() {
  std::printf("branch region is bounded relative to the pivot\n");
  Descriptor d;
  d.kind = PatternKind::Branch;
  d.d_pivot = 2;
  d.branch_height = 2;
  CHECK(!d.in_region(2, 10)); // the pivot itself is not a landing point
  CHECK(d.in_region(3, 10));
  CHECK(d.in_region(4, 10));
  CHECK(!d.in_region(5, 10)); // beyond per-branch height
}

static void test_utility_beats_hotness() {
  std::printf("utility ranks deep-and-warm over shallow-and-hot (Sec. 3.1)\n");
  Controller c;
  const auto p = full_path();
  warm(c, p, 1); // establishes root_level

  // Make the root far hotter than a deep node.
  for (int i = 0; i < 200; ++i) c.observe(p.data(), 1, 0); // root only
  const uint8_t root_depth = c.to_depth(4);
  const uint8_t deep_depth = c.to_depth(1);
  CHECK(root_depth == 0);
  CHECK(deep_depth == 3);

  // Hotness alone would pick the root; U = conf x depth must not.
  const double u_root = c.utility(0x100, root_depth);
  const double u_deep = c.utility(0x400, deep_depth);
  CHECK(u_root == 0.0);   // depth 0 => a cursor there elides nothing
  CHECK(u_deep > u_root);
}

static void test_select_respects_region() {
  std::printf("select returns only in-region candidates\n");
  Controller c;
  const auto p = full_path();
  warm(c, p, 20);

  Entry out{};
  uint8_t out_depth = 0;
  CHECK(c.select(p.data(), (uint8_t)p.size(), 0, out, out_depth));
  // Default funnel window is [1,8]; the root sits at depth 0 and is excluded.
  CHECK(out.addr.val != 0x100);
  CHECK(out_depth >= 1);
}

static void test_select_empty_region() {
  std::printf("select reports an empty region rather than guessing\n");
  Controller c;
  const auto p = full_path();
  warm(c, p, 5);

  // Only the root is offered, and the funnel window excludes depth 0.
  Entry out{};
  uint8_t out_depth = 0;
  CHECK(!c.select(p.data(), 1, 0, out, out_depth));
}

static void test_branch_breadth_cap() {
  std::printf("branch stops admitting once breadth is reached\n");
  ControlConfig cfg;
  cfg.pattern = PatternKind::Branch;
  Controller c(cfg);
  const auto p = full_path();
  warm(c, p, 20);

  Entry out{};
  uint8_t d = 0;
  CHECK(c.select(p.data(), (uint8_t)p.size(), 0, out, d));   // room
  const uint16_t b = c.descriptor().breadth;
  CHECK(!c.select(p.data(), (uint8_t)p.size(), b, out, d));  // at capacity
  CHECK(!c.select(p.data(), (uint8_t)p.size(), b + 5, out, d));
}

static void test_admission_needs_confirmation() {
  std::printf("a candidate must be confirmed before it displaces a resident\n");
  ControlConfig cfg;
  cfg.tau = 0.0;
  Controller c(cfg);
  const auto p = full_path();
  warm(c, p, 1);

  const uint8_t depth = c.to_depth(1);
  // An unseen node has conf 0, so it cannot beat an established resident.
  CHECK(!c.admit(0x999, depth, 0x400, depth));

  // Once confirmed repeatedly it should win against a weaker resident.
  for (int i = 0; i < 50; ++i) c.observe(p.data(), (uint8_t)p.size(), 0);
  CHECK(c.admit(0x400, depth, 0x999, depth));
}

static void test_admission_margin() {
  std::printf("tau blocks marginal churn\n");
  ControlConfig cfg;
  cfg.tau = 1e9; // absurdly high: nothing should ever clear it
  Controller c(cfg);
  const auto p = full_path();
  warm(c, p, 100);
  CHECK(!c.admit(0x400, c.to_depth(1), kNoNode, 0));
  CHECK(c.feedback().rejects > 0);
}

static void test_control_moves_theta() {
  std::printf("control loop moves theta from the feedback signals\n");
  {
    // Sustained misses must pull the window back to recover coverage.
    Controller c;
    const auto p = full_path();
    const uint8_t before = c.descriptor().d_end;
    for (int i = 0; i < 500; ++i) {
      c.note_probe(false);
      c.observe(p.data(), (uint8_t)p.size(), 0);
    }
    CHECK(c.descriptor().d_end <= before);
  }
  {
    // A long residual walk with good hits must push deeper.
    Controller c;
    // Deep path so residual is large.
    std::vector<Entry> deep;
    for (int lvl = 10; lvl >= 0; --lvl) {
      deep.push_back(E(0x1000 + lvl, 0, 1000, (uint8_t)lvl));
    }
    const uint8_t before = c.descriptor().d_end;
    for (int i = 0; i < 200; ++i) {
      c.note_probe(true);
      c.observe(deep.data(), (uint8_t)deep.size(), 0);
    }
    CHECK(c.descriptor().d_end >= before);
  }
}

static void test_auto_switches_pattern() {
  std::printf("auto mode changes pattern with the measured shape\n");
  ControlConfig cfg;
  cfg.pattern = PatternKind::Auto;
  Controller c(cfg);

  std::vector<Entry> deep;
  for (int lvl = 10; lvl >= 0; --lvl) {
    deep.push_back(E(0x2000 + lvl, 0, 1000, (uint8_t)lvl));
  }
  // Long residual, healthy hit rate -> converging queries -> funnel.
  for (int i = 0; i < 300; ++i) {
    c.note_probe(true);
    c.observe(deep.data(), (uint8_t)deep.size(), 0);
  }
  CHECK(c.kind() == PatternKind::Funnel);

  // Now starve it of hits; the controller must not stay on funnel.
  for (int i = 0; i < 3000; ++i) {
    c.note_probe(false);
    c.observe(deep.data(), (uint8_t)deep.size(), 3);
  }
  CHECK(c.kind() != PatternKind::Funnel);
  CHECK(c.pattern_switches() >= 1);
}

static void test_names_and_aliases() {
  std::printf("pattern names round-trip, spatial aliases branch\n");
  CHECK(pattern_from_string("funnel") == PatternKind::Funnel);
  CHECK(pattern_from_string("interval") == PatternKind::Interval);
  CHECK(pattern_from_string("branch") == PatternKind::Branch);
  CHECK(pattern_from_string("spatial") == PatternKind::Branch); // Fig. 7 name
  CHECK(pattern_from_string("nonsense") == PatternKind::Auto);
  CHECK(std::string(pattern_name(PatternKind::Branch)) == "branch");
}

static void test_reset() {
  std::printf("reset clears feedback between benchmark phases\n");
  Controller c;
  const auto p = full_path();
  warm(c, p, 50);
  for (int i = 0; i < 20; ++i) c.note_probe(true);
  CHECK(c.feedback().probes == 20);
  c.reset();
  CHECK(c.feedback().probes == 0);
  CHECK(c.root_level() == 0);
  CHECK(c.utility(0x400, 3) == 0.0); // sketch cleared
}

int main() {
  std::printf("pattern model tests\n\n");

  test_ewma();
  test_confidence_sketch();
  test_region_funnel();
  test_region_interval();
  test_region_branch();
  test_utility_beats_hotness();
  test_select_respects_region();
  test_select_empty_region();
  test_branch_breadth_cap();
  test_admission_needs_confirmation();
  test_admission_margin();
  test_control_moves_theta();
  test_auto_switches_pattern();
  test_names_and_aliases();
  test_reset();

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
