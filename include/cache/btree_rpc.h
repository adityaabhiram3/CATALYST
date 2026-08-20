#pragma once

#include "../GlobalAddress.h"
#include "../RawMessageConnection.h" // catalyst_wire message layout
#include "btree_node.h"

namespace cachepush {

// -1 means lookup failure, 0 means next global_address_ptr, 1 means find value
// result, 2 means find nothing
inline int lookup(GlobalAddress root, uint64_t dsm_base, Key k, Value &v_result,
           GlobalAddress &g_result) {
  auto node_id = root.nodeID;
  auto node = root;
  NodeBase *mem_node = reinterpret_cast<NodeBase *>(dsm_base + root.offset);
  while (mem_node->type == PageType::BTreeInner) {
    auto inner = static_cast<BTreeInner<Key> *>(mem_node);
    node = inner->children[inner->lowerBound(k)];
    if (node.nodeID != node_id) {
      g_result = node;
      return 0;
    } else {
      mem_node = reinterpret_cast<NodeBase *>(dsm_base + node.offset);
    }
  }

  BTreeLeaf<Key, Value> *leaf = static_cast<BTreeLeaf<Key, Value> *>(mem_node);
  if (!leaf->rangeValid(k)) {
    return -1;
  }

  unsigned pos = leaf->lowerBound(k);
  int ret = 2;
  if ((pos < leaf->count) && (leaf->data[pos].first == k)) {
    ret = 1;
    v_result = leaf->data[pos].second;
  }

  return ret;
}

// -1 means update failure because enterring wrong leaf node, 0 means failure
// because not enter into a leaf node, 1 means update value succeeds, 2 means
// update nothing
inline int update(GlobalAddress &root, uint64_t dsm_base, Key k, Value v_result) {
  auto node_id = root.nodeID;
  auto node = root;
  NodeBase *mem_node = reinterpret_cast<NodeBase *>(dsm_base + root.offset);
  while (mem_node->type == PageType::BTreeInner) {
    auto inner = static_cast<BTreeInner<Key> *>(mem_node);
    node = inner->children[inner->lowerBound(k)];
    if (node.nodeID != node_id) {
      return 0;
    } else {
      mem_node = reinterpret_cast<NodeBase *>(dsm_base + node.offset);
    }
  }

  BTreeLeaf<Key, Value> *leaf = static_cast<BTreeLeaf<Key, Value> *>(mem_node);
  if (!leaf->rangeValid(k)) {
    return -1;
  }

  unsigned pos = leaf->lowerBound(k);
  int ret = 2;
  if ((pos < leaf->count) && (leaf->data[pos].first == k)) {
    ret = 1;
    leaf->data[pos].second = v_result;
    root = leaf->remote_address;
  }

  return ret;
}

// -1 means insert failure because enterring wrong leaf node (-1 also means it
// needs to SMO), 0 means failure because not enter into a leaf node, 1 means
// insert value succeeds, 2 means update a existing value;
inline int insert(GlobalAddress &root, uint64_t dsm_base, Key k, Value v_result) {
  auto node_id = root.nodeID;
  auto node = root;
  NodeBase *mem_node = reinterpret_cast<NodeBase *>(dsm_base + root.offset);
  while (mem_node->type == PageType::BTreeInner) {
    auto inner = static_cast<BTreeInner<Key> *>(mem_node);
    node = inner->children[inner->lowerBound(k)];
    if (node.nodeID != node_id) {
      return 0;
    } else {
      mem_node = reinterpret_cast<NodeBase *>(dsm_base + node.offset);
    }
  }

  BTreeLeaf<Key, Value> *leaf = static_cast<BTreeLeaf<Key, Value> *>(mem_node);
  if (!leaf->rangeValid(k) || (leaf->count == leaf->maxEntries)) {
    return -1;
  }

  root = leaf->remote_address;
  bool insert_success = leaf->insert(k, v_result);
  if (insert_success) {
    return 1;
  }

  return 2;
}

// -1 means remove failure because enterring wrong leaf node, 0 means failure
// because not enter into a leaf node, 1 means remove value succeeds, 2 means
// remove nothing
inline int remove(GlobalAddress &root, uint64_t dsm_base, Key k) {
  auto node_id = root.nodeID;
  auto node = root;
  NodeBase *mem_node = reinterpret_cast<NodeBase *>(dsm_base + root.offset);
  while (mem_node->type == PageType::BTreeInner) {
    auto inner = static_cast<BTreeInner<Key> *>(mem_node);
    node = inner->children[inner->lowerBound(k)];
    if (node.nodeID != node_id) {
      return 0;
    } else {
      mem_node = reinterpret_cast<NodeBase *>(dsm_base + node.offset);
    }
  }

  BTreeLeaf<Key, Value> *leaf = static_cast<BTreeLeaf<Key, Value> *>(mem_node);
  if (!leaf->rangeValid(k)) {
    return -1;
  }

  auto flag = leaf->remove(k);
  int ret = flag ? 1 : 2;
  root = leaf->remote_address;
  return ret;
}


/* CATALYST: traverse from a cursor target, recording the path (Sec. 4.2).
 *
 * Differs from the DEX pushdown functions above in three ways:
 *   1. It may start part-way down the tree, at whatever node a cursor named,
 *      so it first validates that node against the search key.
 *   2. It records every node it visits, giving the compute side up to h-l
 *      cursor candidates for free -- "capture is a by-product of a miss".
 *   3. Validation is the standard B-link fence-key check, performed here on
 *      the memory node where the index actually lives, so it can never
 *      interleave with a concurrent structural change (Sec. 4.4).
 *
 * Safety of resuming from a compute-supplied pointer: DEX never frees or
 * merges tree nodes (there is no dsm_->free() anywhere in leanstore_tree.h),
 * so a node named by a cursor is always still an allocated node of the tree.
 * A stale cursor can therefore be wrong about *which* keys the node covers,
 * which the fence check catches, but never dangling.
 */
inline int catalyst_traverse(GlobalAddress start, uint64_t dsm_base, Key k,
                             catalyst_wire::Op op, Value v_in, Value &v_out,
                             GlobalAddress &leaf_out,
                             catalyst_wire::PathEntry *path, uint8_t &npath) {
  using namespace catalyst_wire;
  npath = 0;
  leaf_out = GlobalAddress::Null();

  const auto node_id = start.nodeID;
  GlobalAddress node = start;
  NodeBase *mem_node = reinterpret_cast<NodeBase *>(dsm_base + start.offset);

  // The cursor asserts only that a descent for k may resume here. If the node
  // has been restructured out from under it, say so and let the caller retry
  // from the root; the cost is bounded by the descent the cursor replaced.
  if (mem_node->obsolete) return kStale;
  if (mem_node->type == PageType::BTreeInner) {
    auto inner = static_cast<BTreeInner<Key> *>(mem_node);
    if (!inner->rangeValid(k)) return kStale;
  } else {
    auto leaf = static_cast<BTreeLeaf<Key, Value> *>(mem_node);
    if (!leaf->rangeValid(k)) return kStale;
  }

  /* Bound the descent. This handler runs on a directory thread serving every
   * compute node, so it must not be possible to wedge it: a torn read of a
   * node being restructured, or a corrupt child pointer, would otherwise spin
   * here forever. Any real path is at most the tree height. */
  int hops = 0;
  while (mem_node->type == PageType::BTreeInner) {
    if (++hops > kMaxDescent) return kStale;
    if (npath < kMaxPathLen) {
      path[npath].addr = node;
      path[npath].lo = mem_node->min_limit_;
      path[npath].hi = mem_node->max_limit_;
      path[npath].level = mem_node->level;
      ++npath;
    }
    auto inner = static_cast<BTreeInner<Key> *>(mem_node);
    node = inner->children[inner->lowerBound(k)];
    // Offset 0 is where the tree's own root pointer lives, never a child, so a
    // null child means we read a node mid-update. Let the caller retry.
    if (node == GlobalAddress::Null()) return kStale;
    if (node.nodeID != node_id) {
      // Subtrees below level M live on one memory server, so this is rare;
      // hand the address back and let the caller continue there.
      leaf_out = node;
      return kCrossNode;
    }
    mem_node = reinterpret_cast<NodeBase *>(dsm_base + node.offset);
  }

  auto leaf = static_cast<BTreeLeaf<Key, Value> *>(mem_node);
  if (!leaf->rangeValid(k)) return kStale;

  if (npath < kMaxPathLen) {
    path[npath].addr = node;
    path[npath].lo = leaf->min_limit_;
    path[npath].hi = leaf->max_limit_;
    path[npath].level = 0;
    ++npath;
  }
  leaf_out = node;

  unsigned pos = leaf->lowerBound(k);
  const bool present = (pos < leaf->count) && (leaf->data[pos].first == k);

  switch (op) {
  case Op::Lookup:
    if (present) {
      v_out = leaf->data[pos].second;
      return kFound;
    }
    return kAbsent;

  case Op::Update:
    if (present) {
      leaf->data[pos].second = v_in;
      leaf->dirty = true;
      return kFound;
    }
    return kAbsent;

  case Op::Insert:
    if (present) { // upsert, no structural change
      leaf->data[pos].second = v_in;
      leaf->dirty = true;
      return kFound;
    }
    // A split would propagate upward, which offloading must not attempt;
    // the compute side falls back to the normal DEX insert path.
    if (leaf->count == leaf->maxEntries) return kNeedsSMO;
    leaf->insert(k, v_in);
    leaf->dirty = true;
    return kInserted;

  case Op::Delete:
    if (present) {
      leaf->remove(k);
      leaf->dirty = true;
      return kFound;
    }
    return kAbsent;
  }
  return kAbsent;
}

} // namespace cachepush
