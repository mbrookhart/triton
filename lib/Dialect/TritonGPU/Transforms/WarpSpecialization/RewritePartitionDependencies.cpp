#include "PartitionBuilder.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/WarpSpecialization.h"
#include "llvm/ADT/SCCIterator.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
namespace ttng = triton::nvidia_gpu;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Get the earliest user of a value, assuming all users are in the same block.
static Operation *getEarliestUser(ArrayRef<OpOperand *> uses) {
  OpOperand *use = *llvm::min_element(uses, [](OpOperand *lhs, OpOperand *rhs) {
    return lhs->getOwner()->isBeforeInBlock(rhs->getOwner());
  });
  return use->getOwner();
}

//===----------------------------------------------------------------------===//
// UseInfo
//===----------------------------------------------------------------------===//

namespace {
// Use information for a partition SSA output.
struct UseInfo {
  // Get the maximum distance to a use, according for stage and iteration,
  // given the partition where the value is defined.
  int getMaxUseDistance(const Partition &partitition);

  // Map from partition and distance to the uses in that partition.
  llvm::MapVector<std::pair<Partition *, unsigned>, SmallVector<OpOperand *>>
      consumers;
  llvm::MapVector<Partition *, unsigned> multiplicitySize;
};

} // namespace

int UseInfo::getMaxUseDistance(const Partition &partition) {
  int maxDistance = 0;
  for (auto [usePartition, distance] : llvm::make_first_range(consumers)) {
    int dist = usePartition->getStage() - partition.getStage() + distance;
    assert(dist > 0 && "expected verifier to check schedule validity");
    maxDistance = std::max(maxDistance, dist);
  }
  return maxDistance;
}
//===----------------------------------------------------------------------===//
// AsyncRef
//===----------------------------------------------------------------------===//

namespace {
struct AsyncRef {
  Value getValueView(ImplicitLocOpBuilder &b, Value idx) const {
    Value zero = b.create<arith::ConstantOp>(b.getI32IntegerAttr(0));
    SmallVector<Value> offsets(allocType.getRank(), zero);
    offsets.front() = idx;
    return b.create<MemDescSubviewOp>(viewType, alloc, offsets);
  }
  Value getReadyView(ImplicitLocOpBuilder &b, Value idx) const {
    return createSingleBufferView(b, readyBars, idx);
  }
  Value getEmptyView(ImplicitLocOpBuilder &b, Value idx) const {
    return createSingleBufferView(b, emptyBars, idx);
  }
  auto getView(ImplicitLocOpBuilder &b, Value idx) const {
    auto valView = getValueView(b, idx);
    auto readyView = getReadyView(b, idx);
    auto emptyView = getEmptyView(b, idx);
    return std::make_tuple(valView, readyView, emptyView);
  }

  unsigned multiplicitySize;
  unsigned maxDistance;
  Value alloc;
  Value readyBars;
  Value emptyBars;

  MemDescType allocType;
  MemDescType viewType;
};

//===----------------------------------------------------------------------===//
// DependencyRewriter
//===----------------------------------------------------------------------===//

// Helper class for dependency rewriting.
class DependencyRewriter {
public:
  DependencyRewriter(WarpSchedule &schedule, scf::ForOp &loop)
      : schedule(schedule), loop(loop), b(loop.getLoc(), loop),
        endBuilder(loop.getLoc(), loop->getNextNode()) {}

  // Partition the loop.
  LogicalResult run();

private:
  void resolveOutputMultiplicity(llvm::MapVector<OpResult, UseInfo> &useInfo,
                                 const Partition &partition);
  AsyncRef allocateAsyncValue(RankedTensorType tensorType, unsigned maxDistance,
                              unsigned multiplicitySize);
  void initializeBarriers(int index, const AsyncRef &aref,
                          unsigned numConsumers, Value init);
  std::pair<Value, Value> createAndGetAsyncIndex(
      const AsyncRef &aref,
      function_ref<void(int &, Value &, Value)> extraCondition = {});

  // The schedule to apply.
  WarpSchedule &schedule;
  // The loop to partition.
  scf::ForOp &loop;
  // The builders to use.
  PartitionBuilder b, endBuilder;
};
} // namespace

void DependencyRewriter::resolveOutputMultiplicity(
    llvm::MapVector<OpResult, UseInfo> &useInfo, const Partition &partition) {
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  for (UseInfo &info : llvm::make_second_range(useInfo)) {
    for (auto [key, uses] : info.consumers) {
      auto [usePartition, distance] = key;
      assert(usePartition != &partition && "unexpected self-recursion");
      if (distance == 0) {
        // This is a use of the output in the current iteration.
        continue;
      }
      // We have uses of a value in a future iteration.
      if (info.multiplicitySize.contains(usePartition)) {
        info.multiplicitySize[usePartition] += distance;
      } else {
        info.multiplicitySize.insert({usePartition, distance});
      }
    }
  }
}

AsyncRef DependencyRewriter::allocateAsyncValue(RankedTensorType tensorType,
                                                unsigned maxDistance,
                                                unsigned multiplicitySize) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(loop);
  unsigned numBars = multiplicitySize + maxDistance;
  Value alloc = createAlloc(loop, tensorType, b.getLoc(),
                            getSharedEncoding(tensorType), numBars);
  Value readyBars = createScalarAlloc(b, b.getI64Type(), numBars);
  Value emptyBars = createScalarAlloc(b, b.getI64Type(), numBars);
  auto allocType = cast<MemDescType>(alloc.getType());
  return AsyncRef{multiplicitySize,
                  maxDistance,
                  alloc,
                  readyBars,
                  emptyBars,
                  allocType,
                  getBufferViewType(allocType)};
}

// Initialize the barriers for a particular buffer. If there is an initial
// value for the buffer, store it and mark the buffer as ready to be consumed.
void DependencyRewriter::initializeBarriers(int index, const AsyncRef &aref,
                                            unsigned numConsumers, Value init) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(loop);

  Value idx = b.intCst(index);
  if (init) {
    Value view = aref.getValueView(b, idx);
    b.create<LocalStoreOp>(init, view);
  }

  Value readyView = aref.getReadyView(b, idx);
  Value emptyView = aref.getEmptyView(b, idx);
  b.create<ttng::InitBarrierOp>(readyView, 1);
  b.create<ttng::InitBarrierOp>(emptyView, numConsumers);
  if (init)
    b.create<ttng::ArriveBarrierOp>(readyView, 1);
  else
    b.create<ttng::ArriveBarrierOp>(emptyView, numConsumers);

  endBuilder.create<ttng::InvalBarrierOp>(readyView);
  endBuilder.create<ttng::InvalBarrierOp>(emptyView);
}

std::pair<Value, Value> DependencyRewriter::createAndGetAsyncIndex(
    const AsyncRef &aref,
    function_ref<void(int &, Value &, Value)> extraCondition) {
  Block *body = loop.getBody();
  Value one = b.intCst(1);

  // Thread the phase and buffer index through the loop. The index is
  // pre-incremented.
  Value idx = body->addArgument(b.getI32Type(), b.getLoc());
  Value phase = body->addArgument(b.getI32Type(), b.getLoc());
  idx = b.create<arith::AddIOp>(idx, one);
  Value nextPhase = b.create<arith::XOrIOp>(phase, one);
  Value cnd = b.create<arith::CmpIOp>(
      arith::CmpIPredicate::eq, idx,
      b.intCst(aref.multiplicitySize + aref.maxDistance));
  // The phase flips when we reach the end of all buffers.
  phase = b.create<arith::SelectOp>(cnd, nextPhase, phase);

  int startIdx = aref.multiplicitySize;
  if (extraCondition)
    extraCondition(startIdx, cnd, idx);

  idx = b.create<arith::SelectOp>(cnd, b.intCst(aref.multiplicitySize), idx);

  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  yield.getResultsMutable().append({idx, phase});
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(loop);
  // The index is preincremented so subtract 1 from the start.
  loop.getInitArgsMutable().append({b.intCst(startIdx - 1), b.intCst(0)});
  return {idx, phase};
}

LogicalResult DependencyRewriter::run() {
  SmallVector<llvm::MapVector<OpResult, UseInfo>> partitionUseInfo;
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());

  for (const Partition &partition : schedule.getPartitions()) {
    // Find all consumers of all outputs of this partition, tracking the
    // specific partition and distance of each use.
    auto &useInfo = partitionUseInfo.emplace_back();
    auto callback = [&](OpResult output, OpOperand &use, unsigned distance) {
      Operation *user = loop.getBody()->findAncestorOpInBlock(*use.getOwner());
      Partition *usePartition = schedule.getPartition(user);
      // Ignore uses in the same partition in the future.
      if (usePartition == &partition) {
        assert(distance > 0 && "self-recursion must occur in the future");
        return;
      }
      UseInfo &info = useInfo[output];
      info.consumers[{usePartition, distance}].push_back(&use);
    };
    schedule.iterateUses(loop, &partition, callback);
    resolveOutputMultiplicity(useInfo, partition);
  }

  // Cut all SSA dependencies by passing outputs through shared memory.
  for (auto [partition, useInfo] :
       llvm::zip(schedule.getPartitions(), partitionUseInfo)) {
    // The amount of buffering is based on the longest distance to a user.
    for (auto &[output, info] : useInfo) {
      // FIXME: No IR support for passing simple scalars through shared
      // memory.
      auto tensorType = dyn_cast<RankedTensorType>(output.getType());
      if (!tensorType) {
        return mlir::emitWarning(output.getLoc(),
                                 "FIXME: only tensor SSA dependencies between "
                                 "partitions are supported");
      }

      // Buffer the value based on the greatest distance to a consumer
      // partition.
      int maxDistance = info.getMaxUseDistance(partition);
      // Number the branches of the multiplicity tree. The total number of
      // required buffers includes the lengths of all branches.
      int multiplicitySize = info.multiplicitySize[&partition];

      // Allocate buffers for the value and its associated barriers.
      b.setLoc(output.getLoc());
      ImplicitLocOpBuilder endBuilder(b.getLoc(), loop->getNextNode());
      AsyncRef aref =
          allocateAsyncValue(tensorType, maxDistance, multiplicitySize);

      // Initialize the initial values of the loop-carried dependencies.
      unsigned numConsumers = info.consumers.size();

      // Initialize the buffers.
      for (auto i : llvm::seq(maxDistance + multiplicitySize)) {
        initializeBarriers(i * multiplicitySize, aref, numConsumers,
                           /*init=*/Value());
      }
      // Deallocate shared memory after the buffers are deinitialized.
      endBuilder.create<LocalDeallocOp>(aref.readyBars);
      endBuilder.create<LocalDeallocOp>(aref.emptyBars);

      for (auto &[key, uses] : info.consumers) {
        assert(!uses.empty() && "expected at least one use");
        Operation *earliestUser = getEarliestUser(uses);
        b.setInsertionPoint(
            loop.getBody()->findAncestorOpInBlock(*earliestUser));

        auto [usePartition, distance] = key;
        auto [idx, phase] = createAndGetAsyncIndex(aref);

        // Wait for the value to be available.
        auto [view, readyView, emptyView] = aref.getView(b, idx);
        StageCluster sinkSrcCluster = getStageCluster(earliestUser);
        b.createInto<ttng::WaitBarrierOp>(*usePartition, sinkSrcCluster,
                                          readyView, phase);
        // Load the value at the current index and replace uses in this
        // partition with it.
        Value value = b.createInto<LocalLoadOp>(*usePartition, sinkSrcCluster,
                                                tensorType, view);
        for (OpOperand *use : uses)
          use->set(value);
        // Mark the buffer as ready.
        b.createInto<ttng::ArriveBarrierOp>(*usePartition, sinkSrcCluster,
                                            emptyView, 1);
      }

      // Set up production of the value.
      b.setInsertionPointAfter(output.getDefiningOp());
      auto [idx, phase] = createAndGetAsyncIndex(aref);
      auto [view, readyView, emptyView] = aref.getView(b, idx);
      StageCluster srcStageCluster = getStageCluster(output.getDefiningOp());
      b.createInto<ttng::WaitBarrierOp>(partition, srcStageCluster, emptyView,
                                        phase);
      b.createInto<LocalStoreOp>(partition, srcStageCluster, output, view);
      b.createInto<ttng::ArriveBarrierOp>(partition, srcStageCluster, readyView,
                                          1);
    }
  }

  // Rewrite the loop to add the new results. Calling this function with no
  // indices set will just resize the results.
  eraseLoopCarriedValues(loop, {});

  // Update the schedule.
  schedule.serialize(loop);
  return success();
}

//===----------------------------------------------------------------------===//
// rewritePartitionDependenies
//===----------------------------------------------------------------------===//

LogicalResult triton::gpu::rewritePartitionDependencies(scf::ForOp &loop) {
  FailureOr<WarpSchedule> scheduleOr = WarpSchedule::deserialize(loop);
  if (failed(scheduleOr))
    return failure();
  WarpSchedule schedule = std::move(*scheduleOr);
  if (failed(schedule.verify(loop)))
    return failure();
  DependencyRewriter rewriter(schedule, loop);
  if (failed(rewriter.run()))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUREWRITEPARTITIONDEPENDENCIES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct RewritePartitionDependencies
    : triton::gpu::impl::TritonGPURewritePartitionDependenciesBase<
          RewritePartitionDependencies> {
  using TritonGPURewritePartitionDependenciesBase::
      TritonGPURewritePartitionDependenciesBase;

  void runOnOperation() override;
};
} // namespace

void RewritePartitionDependencies::runOnOperation() {
  // Collect for loops to warp specialize. This pass expects the loop to
  // already be scheduled.
  SmallVector<scf::ForOp> loops;
  getOperation().walk([&](scf::ForOp loop) {
    if (loop->hasAttrOfType<ArrayAttr>(kPartitionStagesAttrName))
      loops.push_back(loop);
  });

  for (scf::ForOp loop : loops) {
    if (failed(rewritePartitionDependencies(loop)))
      return signalPassFailure();
  }
}
