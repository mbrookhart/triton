/*
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/RegionUtils.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <memory>

#define GEN_PASS_CLASSES
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define DEBUG_TYPE "nvws-lower-aref"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
using namespace triton::nvidia_gpu;
using namespace triton::nvws;

struct CarriedValue {
  Value val;
  int retIdx = -1;
};

struct ArefMetadata {
  // Metadata to help track phase through the execution of the program
  ArefMetadata() = default; // This needed to satisfy MapVector
  ArefMetadata(int depth, Value emptyMbars, Value fullMbars)
      : depth(depth), emptyMbars(emptyMbars), fullMbars(fullMbars), phase{} {
    OpBuilder b(fullMbars.getContext());
    b.setInsertionPoint(fullMbars.getDefiningOp());
    Value zero = b.create<arith::ConstantIntOp>(fullMbars.getLoc(), 0, 32);
    phase[fullMbars.getParentRegion()] = CarriedValue{zero};
  }

  Value getPhase(Operation *op) {
    return phase[op->getParentRegion()].val;
  }

  int getPhaseRetIndex(Region *region) { return phase[region].retIdx; }

  void setPhase(Operation *op, Value newPhase) {
    Value &oldPhase = phase[op->getParentRegion()].val;
    replaceAllUsesAfter(oldPhase, newPhase, newPhase.getDefiningOp());
    oldPhase = newPhase;
  }

  Value incrementPhase(OpBuilder &b, Operation *op, Value phase) {
    auto loc = op->getLoc();
    b.setInsertionPointAfter(op);
    Value one = b.template create<arith::ConstantIntOp>(loc, 1, 32);
    Value nextPhase = b.template create<arith::XOrIOp>(loc, phase, one);
    setPhase(op, nextPhase);
    return nextPhase;
  }

  void returnPhase(Operation *op) {
    CarriedValue &parentPhase = phase[op->getParentRegion()];
    CarriedValue &currPhase = phase[&op->getRegions().front()];
    Value retPhase;
    if (isa<scf::IfOp, scf::ForOp>(op))
      retPhase = op->getResult(currPhase.retIdx);
    else
      retPhase = currPhase.val;
    replaceAllUsesAfter(parentPhase.val, retPhase, op);
    parentPhase.val = retPhase;
  }

  void replaceAllUsesAfter(Value from, Value to, Operation *op) {
    from.replaceUsesWithIf(to, [&](OpOperand &use) -> bool {
      if (op->getParentRegion()->isAncestor(
              use.getOwner()->getParentRegion())) {
        if (auto ancestor =
                op->getBlock()->findAncestorOpInBlock(*use.getOwner())) {
          return op->isBeforeInBlock(ancestor);
        }
      }
      return false;
    });
  }
  int depth = 0;
  Value emptyMbars;
  Value fullMbars;
  // HACK HACK HACK
  SmallVector<Value> updatedValues;
  int putPredIdx = -1;
  DenseMap<Region *, CarriedValue> phase;
};

using ArefMap = llvm::MapVector<Value, ArefMetadata>;

struct ArefUseNode {
  ArefUseNode *parent;
  Operation *op;
  SmallVector<ArefUseNode *> subOps;
  bool containsAsync = false;
};

struct ArefUseGraph {
  llvm::MapVector<Operation *, ArefUseNode> nodes;
  ArefMap arefs;
};

void initializePhase(Operation *op, ArefMap &arefs) {
  Region *parentRegion = op->getParentRegion();
  for (auto [aref, _] : arefs) {
    CarriedValue &parentPhase = arefs[aref].phase[op->getParentRegion()];
    for (Region &region : op->getRegions()) {
      if (arefs[aref].phase.contains(&region))
        continue;
      arefs[aref].phase[&region] = CarriedValue{parentPhase.val, -1};
    }
  }
}

void initializePhase(scf::IfOp &op, ArefMap &arefs) {
  Region *parentRegion = op->getParentRegion();
  OpBuilder b(op->getContext());
  b.setInsertionPoint(op);
  auto initPhases = llvm::map_to_vector(
      arefs, [&](auto &aref) { return aref.second.phase[parentRegion].val; });
  auto phaseTypes = llvm::map_to_vector(arefs, [&](auto &aref) {
    return aref.second.phase[parentRegion].val.getType();
  });
  auto newIfOp = replaceIfOpWithNewSignature(b, op, phaseTypes);
  auto init = initPhases.begin();
  for (auto [aref, _] : arefs) {
    auto yield = newIfOp.thenYield();
    int idx = yield.getNumOperands();
    yield->insertOperands(idx, *init);
    yield = newIfOp.elseYield();
    yield->insertOperands(idx, *init);
    arefs[aref].phase[&newIfOp.getThenRegion()] = CarriedValue{*init, idx};
    arefs[aref].phase[&newIfOp.getElseRegion()] = CarriedValue{*init, idx};
    init++;
  }
  op.erase();
  op = newIfOp;
}

void initializePhase(scf::ForOp &op, ArefMap &arefs) {
  Region *parentRegion = op->getParentRegion();
  auto initPhases = llvm::map_to_vector(
      arefs, [&](auto &aref) { return aref.second.phase[parentRegion].val; });
  OpBuilder b(op->getContext());
  b.setInsertionPoint(op);
  auto newPhases = addIterArgsToLoop(b, op, initPhases);
  auto yield = cast<scf::YieldOp>(op.getBody()->getTerminator());

  auto init = newPhases.begin();
  for (auto [aref, _] : arefs) {
    yield->insertOperands(yield.getNumOperands(), *init);
    arefs[aref].phase[&op.getRegion()] =
        CarriedValue{*init, (int)init->getArgNumber() - 1};
    init++;
  }
}

Attribute getPartition(Operation *op) {
  if (!op->hasAttr(kPartitionAttrName))
    return Attribute();
  return op->getAttr(kPartitionAttrName);
}

class PartitionOpBuilder : public OpBuilder {
public:
  PartitionOpBuilder(MLIRContext *ctx, Attribute partition)
      : OpBuilder(ctx), partition(partition) {}
  // Create an operation inside a partition.
  template <typename OpT, typename... Args> OpT create(Args &&...args) {
    auto op = OpBuilder::create<OpT>(std::forward<Args>(args)...);
    if (partition)
      op->setAttr(kPartitionAttrName, partition);
    return op;
  }
  void replaceAllUsesExcept(Value from, Value to, Operation *exceptedUser) {
    return from.replaceUsesWithIf(to, [&](OpOperand &use) {
      Operation *user = use.getOwner();
      return user != exceptedUser;
    });
  }

protected:
  Attribute partition;
};

// Recursively find the Warp Specialized Loop that
// this op belongs to
scf::ForOp findWSLoop(Operation *op) {
  if (auto loop = dyn_cast<scf::ForOp>(op))
    if (loop->hasAttr(kPartitionStagesAttrName))
      return loop;
  auto parent = op->getParentOfType<scf::ForOp>();
  if (!parent) // bail if we've run out of parent loops
    return parent;
  return findWSLoop(parent);
}

scf::ForOp findWSLoop(Value val) {
  if (auto arg = dyn_cast<BlockArgument>(val))
    return findWSLoop(arg.getOwner()->getParentOp());
  return findWSLoop(val.getDefiningOp());
}

template <typename T> Value getIndex(T op) {
  return op.getIndexes().size() > 0 ? op.getIndexes()[0] : Value();
}

// Template to get specialization for automatically writing in partitions
template <typename bT> Value sliceBarrier(bT &b, Value barrier, Value index) {
  if (index)
    return createSingleBufferView(b, barrier, index);
  return barrier;
}

Type getBarrierType(OpBuilder &b, int depth) {
  auto ctx = b.getContext();
  Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
  auto barrierCTALayout = CTALayoutAttr::get(ctx, {1}, {1}, {0});
  auto barrierEncoding =
      SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1, {0}, barrierCTALayout);
  return MemDescType::get(SmallVector<int64_t>{depth}, b.getI64Type(),
                          barrierEncoding, sharedMemorySpace, true);
}

LogicalResult LowerArefCreate(ArefCreateOp op, ArefUseGraph &graph) {
  auto loc = op.getLoc();
  auto ctx = op.getContext();
  OpBuilder b(ctx);
  auto uses = op.getResult().getUses();
  auto numBatches = op.getResult().getType().getNumBatchAxes();

  int rank = 0;
  if (numBatches)
    rank = *numBatches;
  if (rank > 1)
    return op.emitError("TODO: Implement multi-axis slicing");

  int depth = 1;
  if (rank == 1) {
    if (auto mType = dyn_cast<MemDescType>(op.getOperand(0).getType()))
      depth = mType.getShape()[0];
    if (auto rType = dyn_cast<RankedTensorType>(op.getOperand(0).getType()))
      depth = rType.getShape()[0];
  }

  b.setInsertionPointAfter(op);
  // Create two aref_empty_mbarriers
  auto barrierType = getBarrierType(b, depth);
  auto emptyMbars = b.create<LocalAllocOp>(loc, barrierType, Value());
  emptyMbars->setAttr("aref_empty_mbarriers", b.getUnitAttr());

  auto fullMbars = b.create<LocalAllocOp>(loc, barrierType, Value());
  fullMbars->setAttr("aref_full_mbarriers", b.getUnitAttr());

  auto zero = b.create<arith::ConstantIntOp>(loc, 0, 32);
  auto ub = b.create<arith::ConstantIntOp>(loc, depth, 32);
  auto one = b.create<arith::ConstantIntOp>(loc, 1, 32);
  auto dLoop = b.create<scf::ForOp>(loc, zero, ub, one);
  b.setInsertionPointToStart(dLoop.getBody());

  for (int i = 0; i < 2; ++i) {
    auto singleBarrier = createSingleBufferView(
        b, i == 0 ? emptyMbars.getResult() : fullMbars.getResult(),
        dLoop.getInductionVar());
    int count = i == 0 ? 0 : 1;
    b.create<InitBarrierOp>(loc, singleBarrier, count);
  }
  Value aref = op.getResult();
  // Initialize the metadata for tracking
  graph.arefs.insert({aref, ArefMetadata(depth, emptyMbars.getResult(),
                                         fullMbars.getResult())});
  return success();
}

// This function needs more work, currently not actually parallel
LogicalResult ParallelizeDescriptorStore(DescriptorStoreOp op) {
  MLIRContext *ctx = op.getContext();
  OpBuilder b(ctx);
  Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
  auto loc = op.getLoc();
  auto tensorType = op.getSrc().getType();
  auto order = getOrder(tensorType);
  auto ctaLayout = getCTALayout(tensorType.getEncoding());
  auto m = op->getParentOfType<ModuleOp>();
  auto numWarps =
      mlir::cast<mlir::IntegerAttr>(m->getAttr("ttg.num-warps")).getInt();
  Attribute encoding = SwizzledSharedEncodingAttr::get(
      tensorType.getContext(), 1, 1, 1, order, ctaLayout);
  if (tensorType.getRank() > 1) {
    encoding = NVMMASharedEncodingAttr::get(
        tensorType.getContext(), tensorType.getShape(), order, ctaLayout,
        tensorType.getElementType(), false);
  }
  MemDescType memDescType =
      MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                       encoding, sharedMemorySpace,
                       /*mutableMemory=*/true);
  Value alloc = b.create<LocalAllocOp>(loc, memDescType, op.getSrc());
  b.create<triton::nvidia_gpu::FenceAsyncSharedOp>(loc, false);
  // use id 2 for named barrier
  auto barId = b.create<arith::ConstantIntOp>(op.getLoc(), 2, 32);
  auto numThreads =
      b.create<arith::ConstantIntOp>(op.getLoc(), numWarps * 32, 32);
  b.create<NVVM::BarrierOp>(op.getLoc(), barId, numThreads);
  auto tensorOp =
      dyn_cast<ReinterpretTensorDescOp>(op.getDesc().getDefiningOp());
  b.create<triton::nvidia_gpu::AsyncTMACopyLocalToGlobalOp>(
      loc, tensorOp.getRawDesc(), op.getIndices(), alloc);
  b.create<triton::nvidia_gpu::TMAStoreWaitOp>(loc, 0);
  // Ensure all threads arrive at this point to avoid race conditions between
  // two TMA stores in Blackwell tests with sub-tiling enabled. Without this
  // barrier, TMAStoreWaitOp might be executed by another warp that is slightly
  // ahead of the warp issuing AsyncTMACopyLocalToGlobal. The barrier ensures
  // that all warps proceed simultaneously after the data is copied.
  b.create<NVVM::BarrierOp>(op.getLoc(), barId, numThreads);
  op.erase();
  return success();
}

//  TODO: Extend to descriptor gather
LogicalResult ParallelizeDescriptorLoad(DescriptorLoadOp op,
                                        ArefUseGraph &graph) {

  auto ctx = op.getContext();
  auto loc = op.getLoc();
  PartitionOpBuilder b(ctx, getPartition(op));

  SmallVector<Operation *> users(op->user_begin(), op->user_end());
  if (users.size() != 1)
    return failure();
  if (!isa<LocalStoreOp>(users[0]))
    return failure();

  auto putOp = dyn_cast<ArefPutOp>(op->getBlock()->getParentOp());
  if (!putOp)
    return failure();

  auto aref = putOp.getOperand();
  if (!graph.arefs.contains(aref))
    return failure();

  auto loop = findWSLoop(op.getOperation());
  if (!loop)
    return failure();
  auto index = getIndex(putOp);
  auto &arefMetadata = graph.arefs[aref];
  Value phase = arefMetadata.getPhase(op);

  b.setInsertionPointToStart(putOp.getBody());
  Value tmaPtr;
  if (isa<BlockArgument>(op.getDesc())) {
    tmaPtr = b.create<TensorDescToTMAPtrOp>(op.getLoc(), op.getDesc());
  } else if (auto descOp = dyn_cast<ReinterpretTensorDescOp>(
                 op.getDesc().getDefiningOp())) {
    tmaPtr = descOp.getRawDesc();
  } else {
    return op.emitWarning("Unknown source for Aref DescriptorLoadOp");
  }

  auto fullBarrier = sliceBarrier(b, arefMetadata.fullMbars, index);

  auto buf = users[0]->getOperand(1);
  Value pred = b.create<arith::ConstantIntOp>(loc, 1, 1);
  b.create<triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp>(
      loc, tmaPtr, op.getIndices(), fullBarrier, buf, pred);
  int sizeInBytes = 0;
  auto memDesc = cast<MemDescType>(buf.getType());
  auto bufShape = memDesc.getShape();
  auto elementType = memDesc.getElementType();
  SmallVector<int64_t> tensorShape(bufShape.begin() + 1, bufShape.end());
  sizeInBytes += product(tensorShape) * elementType.getIntOrFloatBitWidth() / 8;
  // Need to limit to one of these per put op
  b.create<triton::nvidia_gpu::BarrierExpectOp>(loc, fullBarrier, sizeInBytes,
                                                pred);
  graph.nodes[putOp].containsAsync = true;
  return success();
}

template <typename T>
LogicalResult ParallelizeTCGen5MMA(T op, ArefUseGraph &graph) {
  auto ctx = op.getContext();
  auto loc = op.getLoc();

  auto getOp = dyn_cast<ArefGetOp>(op->getBlock()->getParentOp());
  if (!getOp)
    return failure();

  auto putOp = getOp->template getParentOfType<ArefPutOp>();
  if (!putOp)
    return failure();

  auto putAref = putOp.getOperand();
  auto getAref = getOp.getOperand();

  if (!graph.arefs.contains(putAref) || !graph.arefs.contains(getAref))
    return failure();

  auto &getArefMetadata = graph.arefs[getAref];
  auto &putArefMetadata = graph.arefs[putAref];

  auto loop = findWSLoop(putOp);
  if (!loop)
    return failure();

  if (op.getBarriers().size() > 0)
    return failure(); // Can't parallelize an already parallel mma

  if (putAref.getType().getBaseType().size() != 1)
    return failure(
        "TODO: support putting into more than one tmem_buffer at a time");
  if (putOp.getBody()->getArguments()[0] != op.getAccumulator())
    return failure("This mma isn't accumulating to the put argument");

  PartitionOpBuilder b(ctx, getPartition(op));
  Value getIdx = getIndex(getOp);
  Value putIdx = getIndex(putOp);

  Value putPhase = putArefMetadata.getPhase(op);
  Value getPhase = getArefMetadata.getPhase(op);

  // Find the Reuse argument in the loop for the put op
  auto pred = putOp.getReuse();
  if (auto arg = dyn_cast<BlockArgument>(pred))
    putArefMetadata.putPredIdx = arg.getArgNumber() - 1;

  Value nextPred;
  if (putArefMetadata.putPredIdx >= 0) {
    // loop carried variable
    nextPred =
        loop.getBody()->getTerminator()->getOperand(putArefMetadata.putPredIdx);
  } else if (pred.getDefiningOp()->getParentRegion()->isProperAncestor(
                 putOp->getParentRegion())) {
    // defined above the loop
    nextPred = pred;
  } else if (isa<arith::ConstantOp>(pred.getDefiningOp())) {
    // constant
    nextPred = pred;
  } else {
    return op.emitError("ArefPut predicate is neither a loop carried variable "
                        "or defined above the WS loop.");
  }

  b.setInsertionPoint(op);

  // This assumes inputs to an mma are both from a single aref get. Need to
  // relax that to allow nested aref gets for attention
  auto getEmptyBarrier = sliceBarrier(b, getArefMetadata.emptyMbars, getIdx);
  op.addCompletionBarrier(getEmptyBarrier,
                          b.create<arith::ConstantIntOp>(loc, 1, 1));
  b.setInsertionPointAfter(op);
  T newOp = cast<T>(b.clone(*op));
  b.setInsertionPoint(newOp);
  auto putFullBarrier = sliceBarrier(b, putArefMetadata.fullMbars, getIdx);
  Value trueVal = b.create<arith::ConstantOp>(loc, b.getBoolAttr(true));
  newOp.addCompletionBarrier(putFullBarrier, b.create<arith::ConstantIntOp>(loc, 1, 1));
  newOp.getPredMutable().assign(b.create<arith::XOrIOp>(loc, pred, trueVal));

  return success();
}

LogicalResult ParallelizeWarpGroupDot(WarpGroupDotOp op, ArefUseGraph &graph) {
  auto ctx = op.getContext();
  auto loc = op.getLoc();
  PartitionOpBuilder b(ctx, getPartition(op));

  auto getOp = dyn_cast<ArefGetOp>(op->getBlock()->getParentOp());
  if (!getOp)
    return failure();

  auto kLoop = dyn_cast<scf::ForOp>(getOp->getBlock()->getParentOp());
  if (!kLoop)
    return failure();

  auto getAref = getOp.getOperand();
  if (!graph.arefs.contains(getAref))
    return failure();

  auto getArefValue = graph.arefs[getAref];

  SmallVector<Operation *> users(op->user_begin(), op->user_end());
  if (users.size() != 1)
    return failure();

  auto wait = dyn_cast<WarpGroupDotWaitOp>(users[0]);
  if (!wait)
    return failure();

  if (wait.getPendings() == (getArefValue.depth - 1))
    return failure(); // we've already processed this

  wait.setPendings(getArefValue.depth - 1);
  b.setInsertionPointAfter(wait);

  // This assumes inputs to an mma are both from a single aref get. Need to
  // relax that to allow nested aref gets for attention
  Value getIdx = getIndex(getOp);
  getOp.getIndexes().size();
  auto depthVal =
      b.create<arith::ConstantIntOp>(loc, getArefValue.depth - 1, 32);
  Value idx = b.create<arith::SubIOp>(loc, getIdx, depthVal);
  auto zero = b.create<arith::ConstantIntOp>(loc, 0, 32);
  Value cond =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, idx, zero);
  auto ifOp = b.create<scf::IfOp>(loc, SmallVector<Type>(), cond);
  b.setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());
  auto bar = sliceBarrier(b, getArefValue.emptyMbars, idx);
  b.create<ArriveBarrierOp>(loc, bar, 1);
  b.create<scf::YieldOp>(loc);

  // Find the return and wait on it after the loop
  auto &retOp = getOp.getRegion().front().back();
  auto getRetIdx =
      llvm::find(retOp.getOperands(), wait.getResult(0)).getIndex();

  auto &yieldOp = kLoop.getRegion().front().back();
  auto yieldIdx =
      llvm::find(yieldOp.getOperands(), getOp.getResult(getRetIdx)).getIndex();

  Value acc = kLoop.getResult(yieldIdx);

  b.setInsertionPointAfter(kLoop);

  Value seqAcc = b.create<WarpGroupDotWaitOp>(loc, acc, 0).getResult(0);
  b.replaceAllUsesExcept(acc, seqAcc, seqAcc.getDefiningOp());
  // Since we manually signaled the that the barriers are empty,
  // we don't need to automatically do it
  graph.nodes[getOp].containsAsync = true;
  return success();
}

// Pre-declare to meet C++ rules
LogicalResult LowerArefGet(ArefGetOp op, ArefUseGraph &graph);
LogicalResult LowerArefPut(ArefPutOp op, ArefUseGraph &graph);

// Make this function a template to propogate type information
// to initializePhase template selection
template <typename T> void LowerAref(T &currOp, ArefUseGraph &graph) {

  // Only initialize phase for loops and ifs for the WS loop
  if (findWSLoop(currOp) == currOp)
    initializePhase(currOp, graph.arefs);

  SmallVector<Operation *> opsToErase;
  for (Region &currRegion : currOp->getRegions()) {
    for (auto &op : llvm::make_early_inc_range(currRegion.getOps())) {
      mlir::TypeSwitch<Operation *>(&op)
          .template Case<ArefCreateOp>([&](auto op) {
            if (!failed(LowerArefCreate(op, graph)))
              opsToErase.push_back(op.getOperation());
          })
          .template Case<TCGen5MMAOp, TCGen5MMAScaledOp>(
              [&](auto op) { auto _ = ParallelizeTCGen5MMA(op, graph); })
          .template Case<WarpGroupDotOp>(
              [&](auto op) { auto _ = ParallelizeWarpGroupDot(op, graph); })
          .template Case<DescriptorLoadOp>([&](auto op) {
            if (!failed(ParallelizeDescriptorLoad(op, graph))) {
              opsToErase.push_back(op.getOperation());
              opsToErase.push_back(*op->getUsers().begin());
            }
          })
          .template Case<DescriptorStoreOp>(
              [&](auto op) { auto _ = ParallelizeDescriptorStore(op); })
          // Aref Get/Put have pre-order lowering steps (barrier waits)
          // But also post-order requirements (they need to stick around
          // until we have parellized their contents).
          // Thus, we don't explicitly recurse here, but do it inside
          // LowerArefPut/LowerArefGet
          .template Case<ArefPutOp>([&](auto op) {
            initializePhase(op.getOperation(), graph.arefs);
            auto _ = LowerArefPut(op, graph);
          })
          .template Case<ArefGetOp>([&](auto op) {
            initializePhase(op.getOperation(), graph.arefs);
            auto _ = LowerArefGet(op, graph);
          })
          .template Case<scf::IfOp, scf::ForOp>(
              [&](auto op) { LowerAref(op, graph); });
    }
  }
  if (findWSLoop(currOp) == currOp)
    for (auto &aref : graph.arefs)
      aref.second.returnPhase(currOp);

  for (auto op : llvm::reverse(opsToErase))
    op->erase();
}

LogicalResult LowerArefGet(ArefGetOp op, ArefUseGraph &graph) {
  auto ctx = op.getContext();
  auto loc = op.getLoc();
  auto aref = op.getOperand();

  if (!graph.arefs.contains(aref))
    return failure();
  auto &arefMetadata = graph.arefs[aref];

  PartitionOpBuilder b(ctx, getPartition(op));
  Value phase = arefMetadata.getPhase(op);
  Value idx = getIndex(op);
  scf::ForOp loop = findWSLoop(op);

  Operation* parent = nullptr;
  if (loop) {
    parent = loop.getBody()->findAncestorOpInBlock(*op.getOperation());
  }
  if (parent) {
    if (isa<scf::ForOp>(parent))
      return op.emitError("Don't yet support nested loops with phase tracking");
    if (isa<scf::IfOp>(parent))
      phase = arefMetadata.getPhase(parent);
  }

  b.setInsertionPoint(op);
  auto fullBar = sliceBarrier(b, arefMetadata.fullMbars, idx);
  b.create<WaitBarrierOp>(loc, fullBar, phase);
  // Now that we have waited on full, lower the body
  LowerAref(op, graph);

  // slice aref values
  SmallVector<Value> views;
  if (op.getIndexes().size() != 0) {
    for (auto value :
         aref.template getDefiningOp<ArefCreateOp>().getOperands()) {
      if (auto mType = dyn_cast<MemDescType>(value.getType())) {
        views.push_back(createSingleBufferView(b, value, idx));
      } else if (auto rType = dyn_cast<RankedTensorType>(value.getType())) {
        return op.emitError("FIXME: In-register Tensors not yet supported");
      } else {
        return op.emitError("Aref input type not supported for slicing");
      }
    }
  } else {
    if (graph.arefs[aref].updatedValues.size() > 0) {
      views.append(graph.arefs[aref].updatedValues.begin(),
                   graph.arefs[aref].updatedValues.end());
    } else {
      for (auto value :
           aref.template getDefiningOp<ArefCreateOp>().getOperands())
        views.push_back(value);
    }
  }

  // Move body out of op
  auto opBody = op.getBody();
  if (opBody->getArguments().size() != views.size())
    return op.emitError("number of views and arguments mismatch.");

  for (auto [arg, view] : zip(opBody->getArguments(), views))
    arg.replaceAllUsesWith(view);

  for (auto [ret, val] :
       llvm::zip(op.getResults(), opBody->back().getOperands())) {
    ret.replaceAllUsesWith(val);
  }

  for (auto &bodyOp : llvm::make_early_inc_range(opBody->without_terminator()))
    bodyOp.moveBefore(op);

  // Incrementing the phase needs to happen in all partitions, so
  // use a generic op builder
  OpBuilder builder(ctx);
  builder.setInsertionPointAfter(op);
  if (parent) {
    if (auto ifParent = dyn_cast<scf::IfOp>(parent)) {
      builder.setInsertionPointAfter(parent);
      scf::IfOp newIf = builder.create<scf::IfOp>(loc, TypeRange{}, ifParent.getCondition());
      initializePhase(newIf, graph.arefs);
      arefMetadata.incrementPhase(builder, newIf, phase);
      for (auto &aref : graph.arefs)
        aref.second.returnPhase(newIf.getOperation());
    } else {
      arefMetadata.incrementPhase(builder, op, phase);
    }
  }

  // wait if needed
  b.setInsertionPointAfter(op);
  if (!graph.nodes[op].containsAsync) {
    auto bar = sliceBarrier(b, arefMetadata.emptyMbars, idx);
    ArriveBarrierOp arrive = b.create<ArriveBarrierOp>(loc, bar, 1);
  }

  op->erase();
  return success();
}

LogicalResult LowerArefPut(ArefPutOp op, ArefUseGraph &graph) {
  auto ctx = op.getContext();
  auto loc = op.getLoc();
  auto aref = op.getOperand();

  if (!graph.arefs.contains(aref))
    return failure();
  auto &arefMetadata = graph.arefs[aref];

  PartitionOpBuilder b(ctx, getPartition(op));
  Value phase = arefMetadata.getPhase(op);

  Value idx = getIndex(op);

  auto loop = findWSLoop(op);

  // Find the Reuse argument in the loop
  auto pred = op.getReuse();
  if (auto arg = dyn_cast<BlockArgument>(pred))
    arefMetadata.putPredIdx = arg.getArgNumber() - 1;

  Value nextPred;
  if (arefMetadata.putPredIdx >= 0) {
    // loop carried variable
    nextPred =
        loop.getBody()->getTerminator()->getOperand(arefMetadata.putPredIdx);
  } else if (pred.getDefiningOp()->getParentRegion()->isProperAncestor(
                 op->getParentRegion())) {
    // defined above the loop
    nextPred = pred;
  } else if (isa<arith::ConstantOp>(pred.getDefiningOp())) {
    // constant
    nextPred = pred;
  } else {
    return op.emitError("ArefPut predicate is neither a loop carried variable "
                        "or defined above the WS loop.");
  }

  b.setInsertionPoint(op);
  auto emptyBar = sliceBarrier(b, arefMetadata.emptyMbars, idx);

  Value trueVal = b.create<arith::ConstantOp>(loc, b.getBoolAttr(true));
  Value cond = b.create<arith::XOrIOp>(loc, pred, trueVal);

  b.create<WaitBarrierOp>(loc, emptyBar, phase, cond);

  // Now that we have waited on empty, lower the body
  LowerAref(op, graph);

  // slice aref values
  SmallVector<Value> views;
  if (op.getIndexes().size() != 0) {
    for (auto value :
         aref.template getDefiningOp<ArefCreateOp>().getOperands()) {
      if (auto mType = dyn_cast<MemDescType>(value.getType())) {
        Value singleBuffer = createSingleBufferView(b, value, idx);
        views.push_back(singleBuffer);
      } else if (auto rType = dyn_cast<RankedTensorType>(value.getType())) {
        return op.emitError("FIXME: In-register Tensors not yet supported");
      } else {
        return op.emitError("Aref input type not supported for slicing");
      }
    }
  } else {
    if (graph.arefs[aref].updatedValues.size() > 0) {
      views.append(graph.arefs[aref].updatedValues.begin(),
                   graph.arefs[aref].updatedValues.end());
    } else {
      for (auto value :
           aref.template getDefiningOp<ArefCreateOp>().getOperands())
        views.push_back(value);
    }
  }

  // Move body to parent region
  auto opBody = op.getBody();
  if (opBody->getArguments().size() != views.size())
    return op.emitError("number of views and arguments mismatch.");
  for (auto [arg, view] : zip(opBody->getArguments(), views))
    arg.replaceAllUsesWith(view);

  for (auto [ret, val] :
       llvm::zip(op.getResults(), opBody->back().getOperands())) {
    ret.replaceAllUsesWith(val);
  }

  for (auto result : opBody->back().getOperands())
    if (isa<RankedTensorType>(result.getType()))
      graph.arefs[op.getOperand()].updatedValues.push_back(result);

  for (auto &bodyOp : llvm::make_early_inc_range(opBody->without_terminator()))
    bodyOp.moveBefore(op);

  // Insert arrives as needed
  if (!graph.nodes[op].containsAsync) {
    if (arefMetadata.putPredIdx >= 0)
      b.setInsertionPointAfter(op->isBeforeInBlock(nextPred.getDefiningOp())
                                   ? nextPred.getDefiningOp()
                                   : op);
    else
      b.setInsertionPointAfter(op);
    auto bar = sliceBarrier(b, arefMetadata.fullMbars, idx);

    Value trueVal = b.create<arith::ConstantOp>(loc, b.getBoolAttr(true));
    Value cond = b.create<arith::XOrIOp>(loc, nextPred, trueVal);
    ArriveBarrierOp arrive = b.create<ArriveBarrierOp>(loc, bar, 1, cond);
  }

  // The body may have updated the phase for any arefs actions contained within.
  // Thus, update the phase for all arefs
  for (auto &aref : graph.arefs)
    aref.second.returnPhase(op.getOperation());

  op->erase();
  return success();
}

static ArefUseGraph analyzeArefUseDef(Operation *op) {
  ArefUseGraph graph;
  DenseSet<Operation *> seen;

  ArefUseNode *parent = nullptr;
  std::function<void(Operation * op)> createGraph;
  // Psuedo recursion to get the dependency graph
  createGraph = [&](Operation *op) {
    if (seen.contains(op))
      return;
    ArefUseNode node;
    node.parent = parent;
    node.op = op;
    if (isa<ArefCreateOp>(op)) {
      graph.nodes[op] = node;
    } else if (auto get = dyn_cast<ArefGetOp>(op)) {
      graph.nodes[op] = node;
      auto old_parent = parent;
      parent = &node;
      get.getRegion().walk(createGraph);
      parent = old_parent;
    } else if (auto put = dyn_cast<ArefPutOp>(op)) {
      graph.nodes[op] = node;
      auto old_parent = parent;
      parent = &graph.nodes[op];
      put.getRegion().walk(createGraph);
      parent = old_parent;
    } else {
      return;
    }
    if (parent)
      graph.nodes[parent->op].subOps.push_back(&graph.nodes[op]);
    seen.insert(op);
  };

  op->walk(createGraph);

  return graph;
}

class NVWSLowerAref : public NVWSLowerArefBase<NVWSLowerAref> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::ModuleOp m = getOperation();
    SmallVector<triton::FuncOp> funcs;
    m.walk([&](triton::FuncOp func) { funcs.push_back(func); });

    for (auto func : funcs) {
      ArefUseGraph graph = analyzeArefUseDef(func);
      LowerAref(func, graph);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createNVWSLowerArefPass() {
  return std::make_unique<NVWSLowerAref>();
}
