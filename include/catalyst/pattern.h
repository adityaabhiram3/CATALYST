#pragma once

/* CATALYST pattern model and dynamic cursor selection (paper Sec. 5).
 *
 * Sec. 4 fixed a placement envelope by hand. This replaces it with the control
 * loop of Sec. 5: a pattern *descriptor* bounds which returned nodes are even
 * eligible, per-query *feedback records* summarise how well the table is
 * serving the workload, and a *control policy* moves the descriptor and decides
 * admission by marginal utility.
 *
 * Three patterns, one per reuse shape identified in Sec. 3.4:
 *
 *   Funnel    theta = (d_start, d_end)     narrow convergence just above the
 *                                          leaves; skewed point lookups and
 *                                          hub-dominated graph algorithms.
 *   Interval  theta = (d_anchor, w_min)    a contiguous key span at mid-level;
 *                                          range queries and scans, where reach
 *                                          matters more than depth.
 *   Branch    theta = (d_pivot, b, h)      several disjoint sub-branches under
 *                                          a common pivot; scattered multi-path
 *                                          workloads such as BFS frontiers.
 *
 * Naming note: the paper calls the third pattern "Branch" in Sec. 5, "Spatial"
 * in Fig. 7, and "pivot-bounded" in Table 2. Branch is used here because Sec. 5
 * is where its parameters are defined; Spatial is accepted as an alias.
 *
 * Depth convention. Everything here uses the paper's depth: 0 at the root,
 * larger deeper. DEX numbers levels the other way (0 at the leaf), so the
 * conversion happens once, at the boundary, in Controller::observe().
 *
 * Threading. One Controller per worker thread. The cursor table it feeds is
 * shared, but the descriptor, the feedback records and the confidence sketch
 * are thread-local estimates of *that thread's* query mix -- which is the right
 * granularity, since DEX's logical partitioning already gives each thread its
 * own key range. Nothing here needs a lock.
 */

#include "cursor_table.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace catalyst {

enum class PatternKind : uint8_t { Funnel, Interval, Branch, Auto };

inline const char *pattern_name(PatternKind k) {
  switch (k) {
  case PatternKind::Funnel: return "funnel";
  case PatternKind::Interval: return "interval";
  case PatternKind::Branch: return "branch";
  case PatternKind::Auto: return "auto";
  }
  return "?";
}

// "spatial" is Fig. 7's name for Branch; accepted so either vocabulary works.
inline PatternKind pattern_from_string(const std::string &s) {
  if (s == "funnel") return PatternKind::Funnel;
  if (s == "interval") return PatternKind::Interval;
  if (s == "branch" || s == "spatial") return PatternKind::Branch;
  return PatternKind::Auto;
}

/* Exponentially decayed per-query signal: x_hat(t) = (1-a) x_hat(t-1) + a x(t).
 * Sec. 5 keeps every feedback metric in this form so a record is O(1) state. */
class Ewma {
public:
  explicit Ewma(double alpha = 0.05, double init = 0.0)
      : alpha_(alpha), v_(init) {}
  void add(double x) { v_ = (1.0 - alpha_) * v_ + alpha_ * x; }
  double get() const { return v_; }
  void reset(double v = 0.0) { v_ = v; }

private:
  double alpha_;
  double v_;
};

/* conf(c): the decayed count of recent queries that saw node c on their path.
 *
 * A brand-new candidate has to be confirmed by several queries before its
 * utility can beat a resident cursor, which is what stops a bounded table from
 * churning on one-off traversals. Kept as a small direct-mapped sketch rather
 * than a per-cursor column so the table's 29 bytes/cursor is unchanged;
 * collisions only over-credit a candidate, and the admission margin absorbs it.
 */
class ConfidenceSketch {
public:
  explicit ConfidenceSketch(size_t slots = 4096, uint64_t halve_every = 1 << 14)
      : mask_(round_pow2(slots) - 1), halve_every_(halve_every),
        c_(round_pow2(slots), 0) {}

  void observe(uint64_t node) {
    uint16_t &s = c_[idx(node)];
    if (s < 0xFFFF) ++s;
    if (++seen_ >= halve_every_) decay();
  }

  uint32_t conf(uint64_t node) const { return c_[idx(node)]; }

  void clear() {
    std::fill(c_.begin(), c_.end(), 0);
    seen_ = 0;
  }

private:
  static size_t round_pow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
  }
  size_t idx(uint64_t node) const {
    // Node addresses are page-aligned, so mix before masking.
    uint64_t h = node * 0x9E3779B97F4A7C15ull;
    return (size_t)((h >> 32) & mask_);
  }
  void decay() {
    for (auto &v : c_) v >>= 1;
    seen_ = 0;
  }

  uint64_t mask_;
  uint64_t halve_every_;
  uint64_t seen_ = 0;
  std::vector<uint16_t> c_;
};

/* Descriptor state: theta for whichever pattern is active, plus the region test
 * it induces. Sec. 5: "the pattern specifies the reuse shape, while the
 * descriptor binds that shape to concrete key regions". */
struct Descriptor {
  PatternKind kind = PatternKind::Funnel;

  // Funnel: active depth window, in paper depth (larger = deeper).
  uint8_t d_start = 1;
  uint8_t d_end = 8;

  // Interval: anchor depth and the minimum band width worth admitting.
  uint8_t d_anchor = 3;
  uint64_t w_min = 0;

  // Branch: pivot depth, how many sibling branches to keep, and how far below
  // the pivot a cursor may sit.
  uint8_t d_pivot = 2;
  uint16_t breadth = 8;
  uint8_t branch_height = 2;

  // Is a candidate at this depth/width inside the admissible region?
  bool in_region(uint8_t depth, uint64_t width) const {
    switch (kind) {
    case PatternKind::Funnel:
      return depth >= d_start && depth <= d_end;
    case PatternKind::Interval:
      // Bias toward reach: an anchor band, wide enough to serve neighbours.
      return depth >= (d_anchor > 1 ? d_anchor - 1 : 0) &&
             depth <= d_anchor + 1 && width >= w_min;
    case PatternKind::Branch:
      // Bounded relative to the pivot, not the root, so the table cannot fill
      // with one deep path.
      return depth > d_pivot && depth <= (uint8_t)(d_pivot + branch_height);
    case PatternKind::Auto:
      return true;
    }
    return false;
  }
};

/* Per-query feedback records (Sec. 5). Two records per query: a structural one
 * (levels skipped, residual depth) and a table one (hit/miss, covered width). */
struct Feedback {
  Ewma hit{0.02, 0.0};       // table hit rate
  Ewma skipped{0.05, 0.0};   // levels a cursor elided
  Ewma residual{0.05, 0.0};  // remaining walk after the cursor
  Ewma width{0.05, 0.0};     // admitted band width, normalised
  uint64_t probes = 0, hits = 0, admits = 0, rejects = 0;
};

struct ControlConfig {
  PatternKind pattern = PatternKind::Funnel;
  double tau = 0.0;          // admission margin; must strictly improve
  size_t sketch_slots = 4096;
  uint32_t stable_windows = 8; // windows of low miss before widening reach
};

/* The control loop: observe a returned path, move theta, pick at most one
 * candidate, and admit it only if it beats the resident cursor by tau. */
class Controller {
public:
  explicit Controller(const ControlConfig &cfg = ControlConfig())
      : cfg_(cfg), sketch_(cfg.sketch_slots) {
    desc_.kind = cfg.pattern == PatternKind::Auto ? PatternKind::Funnel
                                                  : cfg.pattern;
    auto_ = (cfg.pattern == PatternKind::Auto);
  }

  const Descriptor &descriptor() const { return desc_; }
  const Feedback &feedback() const { return fb_; }
  PatternKind kind() const { return desc_.kind; }

  // Highest DEX level seen so far stands in for the root level, which is what
  // converts a level into a paper depth and into "hops a cursor would save".
  uint8_t root_level() const { return root_level_; }

  void note_probe(bool hit) {
    ++fb_.probes;
    fb_.hits += hit ? 1 : 0;
    fb_.hit.add(hit ? 1.0 : 0.0);
  }

  /* Record everything the reply told us, and update theta.
   *
   * `entries` are the visited nodes in descent order. `resumed_depth` is the
   * depth the query actually started from (0 when it started at the root), so
   * skipped = resumed_depth and residual = deepest - resumed_depth.
   */
  template <typename PathEntryT>
  void observe(const PathEntryT *entries, uint8_t n, uint8_t resumed_depth) {
    if (n == 0) return;
    for (uint8_t i = 0; i < n; ++i) {
      if (entries[i].level > root_level_) root_level_ = entries[i].level;
      sketch_.observe(entries[i].addr.val);
    }
    const uint8_t deepest = to_depth(entries[n - 1].level);
    fb_.skipped.add(resumed_depth);
    fb_.residual.add(deepest > resumed_depth ? deepest - resumed_depth : 0);
    retune();
  }

  uint8_t to_depth(uint8_t level) const {
    return (uint8_t)(root_level_ >= level ? root_level_ - level : 0);
  }

  /* U(c) = conf(c) x skip(c): how often this landing point has been confirmed,
   * times how much work starting there would elide. Sec. 3.1's point is that
   * hotness alone over-values shallow nodes, which is why depth is a factor. */
  double utility(uint64_t node, uint8_t depth) const {
    return (double)sketch_.conf(node) * (double)depth;
  }

  /* Pick c*: the best candidate inside the region, ranked by utility and then
   * by covered width, so reach breaks ties in favour of the cursor that will
   * serve more future queries (Sec. 5). Returns false if the region is empty. */
  template <typename PathEntryT>
  bool select(const PathEntryT *entries, uint8_t n, size_t bucket_occupancy,
              PathEntryT &out, uint8_t &out_depth) const {
    bool found = false;
    double best_u = -1.0;
    uint64_t best_w = 0;

    // Branch caps how many sibling cursors a pivot region may hold.
    if (desc_.kind == PatternKind::Branch &&
        bucket_occupancy >= desc_.breadth) {
      return false;
    }

    for (uint8_t i = 0; i < n; ++i) {
      const auto &p = entries[i];
      if (p.lo > p.hi) continue;
      const uint8_t d = to_depth(p.level);
      const uint64_t w = p.hi - p.lo;
      if (!desc_.in_region(d, w)) continue;
      const double u = utility(p.addr.val, d);
      if (!found || u > best_u || (u == best_u && w > best_w)) {
        found = true;
        best_u = u;
        best_w = w;
        out = p;
        out_depth = d;
      }
    }
    return found;
  }

  /* Admission: delta(c) = U(c) - max U(e) over overlapping residents, admit if
   * delta > tau. The deepest resident matching the search key is the
   * overlapping entry the probe already found, so this costs no extra work. */
  bool admit(uint64_t cand_node, uint8_t cand_depth, uint64_t resident_node,
             uint8_t resident_depth) {
    const double u_c = utility(cand_node, cand_depth);
    const double u_e =
        resident_node != kNoNode ? utility(resident_node, resident_depth) : 0.0;
    const bool ok = (u_c - u_e) > cfg_.tau;
    ok ? ++fb_.admits : ++fb_.rejects;
    return ok;
  }

  void note_admitted_width(uint64_t width) {
    fb_.width.add((double)(width >> 32)); // scaled; only trends matter
  }

  void reset() {
    fb_ = Feedback{};
    sketch_.clear();
    stable_ = 0;
    root_level_ = 0;
  }

private:
  /* Move theta from the feedback signals (Sec. 5, "Control Update").
   *
   *   high residual        -> push deeper, there is still walk left to elide
   *   high miss rate       -> back off depth to recover coverage
   *   stable low miss rate -> widen reach, and after several stable windows
   *                           stop bothering with the upper levels
   */
  void retune() {
    const double miss = 1.0 - fb_.hit.get();
    const double residual = fb_.residual.get();

    if (miss > kMissHigh) {
      stable_ = 0;
      // Losing coverage: pull the window back toward the root.
      if (desc_.d_end > desc_.d_start + 1) --desc_.d_end;
      if (desc_.d_anchor > 1) --desc_.d_anchor;
      if (desc_.w_min > 0) desc_.w_min >>= 1; // accept narrower bands again
    } else if (miss < kMissLow) {
      if (++stable_ >= cfg_.stable_windows) {
        stable_ = 0;
        // Comfortable: ignore upper levels, they are redundant under a
        // consecutively cached path.
        if (desc_.d_start < desc_.d_end) ++desc_.d_start;
        if (desc_.d_pivot < kMaxDepth - desc_.branch_height) ++desc_.d_pivot;
      }
    }

    if (residual > kResidualHigh && desc_.d_end < kMaxDepth) {
      ++desc_.d_end; // still walking after the cursor: go deeper
      if (desc_.d_anchor < kMaxDepth) ++desc_.d_anchor;
    }

    if (auto_) pick_pattern();
  }

  /* Auto mode: let the measured shape choose the pattern.
   *
   * A long residual with a high hit rate means queries converge on a narrow
   * deep band -- funnel. Wide admitted bands with misses on neighbouring keys
   * mean reach is what is missing -- interval. Neither, with a poor hit rate,
   * means the profitable set keeps moving -- branch, whose cursors are expected
   * to age out rather than persist. */
  void pick_pattern() {
    const double miss = 1.0 - fb_.hit.get();
    PatternKind want = desc_.kind;
    if (fb_.residual.get() >= kResidualHigh && miss < kMissHigh) {
      want = PatternKind::Funnel;
    } else if (miss >= kMissHigh && fb_.skipped.get() > 0.5) {
      want = PatternKind::Interval;
    } else if (miss >= kMissVeryHigh) {
      want = PatternKind::Branch;
    }
    if (want != desc_.kind) {
      desc_.kind = want;
      ++switches_;
    }
  }

  static constexpr double kMissHigh = 0.5;
  static constexpr double kMissVeryHigh = 0.8;
  static constexpr double kMissLow = 0.2;
  static constexpr double kResidualHigh = 1.5;
  static constexpr uint8_t kMaxDepth = 24;

  ControlConfig cfg_;
  Descriptor desc_;
  Feedback fb_;
  ConfidenceSketch sketch_;
  bool auto_ = false;
  uint32_t stable_ = 0;
  uint32_t switches_ = 0;
  uint8_t root_level_ = 0;

public:
  uint32_t pattern_switches() const { return switches_; }
};

} // namespace catalyst
