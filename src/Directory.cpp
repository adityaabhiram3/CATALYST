#include "Directory.h"
#include "Common.h"

#include "Connection.h"
#include "cache/btree_rpc.h"

#include <gperftools/profiler.h>

GlobalAddress g_root_ptr = GlobalAddress::Null();
int g_root_level = -1;
bool enable_cache;

Directory::Directory(DirectoryConnection *dCon, RemoteConnection *remoteInfo,
                     uint32_t machineNR, uint16_t dirID, uint16_t nodeID,
                     int memThreadCount)
    : dCon(dCon), remoteInfo(remoteInfo), machineNR(machineNR), dirID(dirID),
      nodeID(nodeID), dirTh(nullptr) {

  { // chunck alloctor
    GlobalAddress dsm_start;
    uint64_t per_directory_dsm_size = dCon->dsmSize / memThreadCount;
    dsm_start.nodeID = nodeID;
    dsm_start.offset = per_directory_dsm_size * dirID;
    // std::cout << "Per directory DM size (MB) = "
    //           << per_directory_dsm_size / define::MB << std::endl;
    chunckAlloc = new GlobalAllocator(dsm_start, per_directory_dsm_size);
  }

  dirTh = new std::thread(&Directory::dirThread, this);
}

Directory::~Directory() { delete chunckAlloc; }

void Directory::dirThread() {
  // bindCore((19 - dirID) * 2);
  bindCore(39 - dirID);
  Debug::notifyInfo("dir %d launch!\n", dirID);

  while (true) {
    struct ibv_wc wc;
    pollWithCQ(dCon->cq, 1, &wc);
    switch (int(wc.opcode)) {
    case IBV_WC_RECV: // control message
    {
      // printf("Dir receives a mesage\n");
      auto *m = (RawMessage *)dCon->message->getMessage();

      process_message(m);

      break;
    }
    case IBV_WC_RDMA_WRITE: {
      break;
    }
    case IBV_WC_RECV_RDMA_WITH_IMM: {

      break;
    }
    default:
      assert(false);
    }
  }
}

void Directory::process_message(const RawMessage *m) {
  RawMessage *send = nullptr;
  // CATALYST replies are larger than a RawMessage and variable-length, so the
  // reply size travels alongside the reply pointer.
  size_t send_len = sizeof(RawMessage);
  switch (m->type) {

  case RpcType::LOOKUP: {
    auto addr = m->addr;
    Value v_result;
    GlobalAddress g_result;
    auto ret = cachepush::lookup(addr, remoteInfo[addr.nodeID].dsmBase, m->k,
                                 v_result, g_result);
    send = (RawMessage *)dCon->message->getSendPool();
    send->level = ret;
    if (ret == 1) {
      send->addr.val = v_result;
    } else if (ret == 2) {
      send->addr = g_result;
    }

    break;
  }

  case RpcType::UPDATE: {
    auto addr = m->addr;
    auto ret =
        cachepush::update(addr, remoteInfo[addr.nodeID].dsmBase, m->k, m->v);
    send = (RawMessage *)dCon->message->getSendPool();
    send->level = ret;
    send->addr = addr;
    break;
  }

  case RpcType::INSERT: {
    auto addr = m->addr;
    auto ret =
        cachepush::insert(addr, remoteInfo[addr.nodeID].dsmBase, m->k, m->v);
    send = (RawMessage *)dCon->message->getSendPool();
    send->level = ret;
    send->addr = addr;
    break;
  }

  case RpcType::DELETE: {
    auto addr = m->addr;
    auto ret = cachepush::remove(addr, remoteInfo[addr.nodeID].dsmBase, m->k);
    send = (RawMessage *)dCon->message->getSendPool();
    send->level = ret;
    send->addr = addr;
    break;
  }

  case RpcType::MALLOC: {
    // printf("DIR has received a MALLOC msg\n");
    send = (RawMessage *)dCon->message->getSendPool();
    send->addr = chunckAlloc->alloc_chunck();
    break;
  }

  case RpcType::NEW_ROOT: {

    if (g_root_level < m->level) {
      g_root_ptr = m->addr;
      g_root_level = m->level;
      if (g_root_level >= 3) {
        enable_cache = true;
      }
    }

    break;
  }

  /* CATALYST: run the operation from wherever the cursor pointed and hand the
   * walked path back, so the compute side can install cursors without a
   * profiling pass or a second round trip (Sec. 4.1, Fig. 8 lines 23-27). */
  case RpcType::CATALYST_TRAVERSE: {
    auto *req = reinterpret_cast<const catalyst_wire::TraverseMsg *>(m);
    auto *resp = reinterpret_cast<catalyst_wire::TraverseMsg *>(
        dCon->message->getSendPool());
    resp->type = RpcType::CATALYST_TRAVERSE;

    Value v_out = 0;
    GlobalAddress leaf = GlobalAddress::Null();
    uint8_t npath = 0;
    resp->status = cachepush::catalyst_traverse(
        req->start, remoteInfo[req->start.nodeID].dsmBase, req->k, req->op,
        req->v, v_out, leaf, resp->path, npath);
    resp->value = v_out;
    resp->leaf = leaf;
    resp->npath = npath;

    send = reinterpret_cast<RawMessage *>(resp);
    // Only transmit the path slots actually written; a shallow resume from a
    // deep cursor sends almost nothing back.
    send_len = offsetof(catalyst_wire::TraverseMsg, path) +
               npath * sizeof(catalyst_wire::PathEntry);
    break;
  }

  default:
    assert(false);
  }

  if (send) {
    // printf("Send back the message to node %d, app %d\n", m->node_id,
    // m->app_id);
    dCon->sendMessage2App(send, m->node_id, m->app_id, send_len);
  }
}