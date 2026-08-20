#ifndef __RAWMESSAGECONNECTION_H__
#define __RAWMESSAGECONNECTION_H__

#include "AbstractMessageConnection.h"
#include "GlobalAddress.h"

#include <thread>

enum RpcType : uint8_t {
  MALLOC,
  FREE,
  NEW_ROOT,
  NOP,
  LOOKUP,
  INSERT,
  UPDATE,
  DELETE,
  SMO,
  CATALYST_TRAVERSE
};

struct RawMessage {
  RpcType type; // operation type

  uint16_t node_id; // source node ID
  uint16_t app_id; // source thread ID, so the receiver can send the message for
                   // reply

  GlobalAddress addr; // for malloc and for root_address of B+-Tree RPC
  int level;          // the return value of pushdown op
  uint64_t k;         // key for B+-Tree RPC
  uint64_t v;         // value for B+-Tree RPC
  // Updated node
} __attribute__((packed));

/* CATALYST traverse RPC (paper Sec. 4.2, Fig. 8).
 *
 * One round trip carries the operation *and* returns the path the memory-side
 * handler walked, so cursor candidates are a by-product of a miss rather than
 * something a profiling pass has to collect (Sec. 4.1).
 *
 * The struct deliberately repeats RawMessage's first three fields at the same
 * offsets: the directory thread reads m->type off the receive buffer before it
 * knows which message this is, so the discriminant must live at offset 0.
 */
namespace catalyst_wire {

// A DEX tree over 1KB pages is ~5 levels deep at 200M keys; 12 also covers the
// 128B-node configuration of the paper's Lesson 2.
static constexpr int kMaxPathLen = 12;

// Hard ceiling on how far a memory-side traversal will descend before giving
// up. Only a corrupt or concurrently-restructured node can exceed a real tree
// height; the bound keeps a directory thread from spinning on one.
static constexpr int kMaxDescent = 32;

enum class Op : uint8_t { Lookup, Insert, Update, Delete };

// Status codes returned in TraverseMsg::status.
enum Status : int32_t {
  kFound = 1,       // key present; value returned
  kAbsent = 2,      // traversal reached the right leaf, key not there
  kInserted = 3,    // insert placed a new key
  kStale = -1,      // cursor no longer covers k: retire it and retry from root
  kCrossNode = -2,  // next child lives on another memory node; resume there
  kNeedsSMO = -3,   // leaf is full / would restructure: fall back to DEX path
};

// One visited node: its address plus the fence keys that define the band a
// cursor to it would cover. Sec. 4.1 notes D2 costs no extra state because the
// index already stores these to bound the subtree.
struct PathEntry {
  GlobalAddress addr;
  uint64_t lo;
  uint64_t hi;
  uint8_t level; // DEX convention: 0 == leaf, larger == closer to the root
} __attribute__((packed));

struct TraverseMsg {
  // --- must mirror RawMessage's header ---
  RpcType type;
  uint16_t node_id;
  uint16_t app_id;

  // --- request ---
  GlobalAddress start; // cursor target, or the root when the probe missed
  uint64_t k;
  uint64_t v;
  Op op;

  // --- response ---
  int32_t status;
  uint64_t value;
  GlobalAddress leaf; // leaf reached, or the next node on kCrossNode
  uint8_t npath;
  PathEntry path[kMaxPathLen];
} __attribute__((packed));

// 40 bytes of the receive slot are the UD global routing header.
static_assert(sizeof(TraverseMsg) + 40 <= MESSAGE_SIZE,
              "TraverseMsg must fit a message slot after the UD GRH");

} // namespace catalyst_wire

class RawMessageConnection : public AbstractMessageConnection {

public:
  RawMessageConnection(RdmaContext &ctx, ibv_cq *cq, uint32_t messageNR);

  void initSend();
  // len defaults to a plain RawMessage so every existing caller is unchanged;
  // CATALYST passes sizeof(catalyst_wire::TraverseMsg).
  void sendRawMessage(RawMessage *m, uint32_t remoteQPN, ibv_ah *ah,
                      size_t len = sizeof(RawMessage));
};

#endif /* __RAWMESSAGECONNECTION_H__ */
