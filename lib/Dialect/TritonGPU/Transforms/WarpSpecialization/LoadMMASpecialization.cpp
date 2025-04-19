#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/WarpSpecialization.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
namespace ttng = triton::nvidia_gpu;

using Partition = WarpSchedule::Partition;

//===----------------------------------------------------------------------===//
// specializeLoadMMADependencies
//===----------------------------------------------------------------------===//

// Pattern match a simple `tma_load -> ... -> tl.dot` single-user chain. This
// ensures there are extraneous users of the load or intermediate values and
// that a valid partition schedule can be formed.
//
// TODO: Expand partioning scheme to support arbitrary DAG of loads and MMAs.
static LogicalResult findSingleChainToLoad(scf::ForOp loop, Value value,
                                           SmallVectorImpl<Operation *> &ops) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp || !value.hasOneUse() || defOp->getParentOp() != loop)
    return failure();

  // This only works on TMA loads because they directly use the mbarrier
  // mechanism. Since async groups are per-thread, commit groups cannot be used
  // to synchronize across warp groups. We have to wait on the async group in
  // the same partition as the loads and arrive an mbarrier to synchronize with
  // the MMA partition, and then software pipeline the load partition.
  //
  // Triple-buffered example:
  //
  //   cp.async %a_ptrs[0], %a_buf[0]
  //   cp.async %b_ptrs[0], %b_buf[0]
  //   cp.async.commit_group
  //
  //   cp.async %a_ptrs[1], %a_buf[1]
  //   cp.async %b_ptrs[1], %b_buf[1]
  //   cp.async.commit_group
  //
  //   for i in range(2, N+2):
  //     @i<N mbarrier.wait %empty_mbars[i%3]
  //     @i<N cp.async %a_ptrs[i], %a_buf[i%3]
  //     @i<N cp.async %b_ptrs[i], %b_buf[i%3]
  //     @i<N cp.async.commit_group
  //
  //     cp.async.wait_group 2 # the i-2 load group is complete
  //     mbarrier.arrive %load_mbars[(i-2)%3]
  if (isa<DescriptorLoadOp, DescriptorGatherOp>(defOp)) {
    ops.push_back(defOp);
    return success();
  }

  // See through allocations and layout conversions.
  if (isa<ttng::TMEMAllocOp, LocalAllocOp, MemDescTransOp, ConvertLayoutOp>(
          defOp)) {
    assert(llvm::is_contained({0, 1}, defOp->getNumOperands()));
    // Alloc ops have an optional source operand.
    if (defOp->getNumOperands() != 1)
      return failure();
    ops.push_back(defOp);
    return findSingleChainToLoad(loop, defOp->getOperand(0), ops);
  }

  return failure();
}
static BlockArgument addIndex(ImplicitLocOpBuilder &b, scf::ForOp &loop,
                              unsigned numStages, Value epilogue = {}) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(loop);
  auto intCst = [&](int value) {
    return b.create<arith::ConstantIntOp>(value, 32);
  };

  // Index starts at 0.
  unsigned curArgIdx = loop.getNumRegionIterArgs();
  auto newArgs = addIterArgsToLoop(b, loop, {intCst(0)});
  BlockArgument index = newArgs[0];

  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  b.setInsertionPoint(yield);

  Value nextIndex = b.create<arith::AddIOp>(index, intCst(1));

  Value rollover = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, nextIndex,
                                           intCst(numStages));
  nextIndex = b.create<arith::SelectOp>(rollover, intCst(0), nextIndex);
  // Post-increment the index and phase.
  if (epilogue) {
    nextIndex = b.create<arith::SelectOp>(epilogue, nextIndex, index);
  }

  yield->insertOperands(yield.getNumOperands(), {nextIndex});

  return index;
}

// Create an operation inside a partition.
template <typename OpT, typename... Args>
static auto createInPartition(ImplicitLocOpBuilder &b, Partition *partition,
                              Args &&...args) {
  auto op = b.create<OpT>(std::forward<Args>(args)...);
  if (partition != nullptr)
    partition->insert(op);
  return op;
}

static std::pair<Value, Operation *>
getUserPrecondition(ImplicitLocOpBuilder &b, scf::ForOp loop, Operation *domOp,
                    Value initialValue = {}) {
  // If the use is inside a loop besides the actual loop being pipelined, we
  // have to hoist the use up to that loop, otherwise the barriers will be
  // inserted in the loop.
  for (Operation *userLoop;
       loop != (userLoop = domOp->getParentOfType<LoopLikeOpInterface>());)
    domOp = userLoop;
  assert(loop->isProperAncestor(domOp));

  Value trueVal = b.create<arith::ConstantOp>(b.getBoolAttr(true));
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(loop.getBody()->findAncestorOpInBlock(*domOp));

  Value precondition = initialValue ? initialValue : trueVal;
  Operation *parentOp = domOp;
  while (loop != (parentOp = parentOp->getParentOp())) {
    assert(!isa<LoopLikeOpInterface>(parentOp));
    auto ifOp = dyn_cast<scf::IfOp>(parentOp);
    if (!ifOp) {
      llvm::report_fatal_error(
          "FIXME: unsupported parent operation for MMA user");
    }
    Value cond = ifOp.getCondition();
    if (domOp->getParentRegion() == &ifOp.getElseRegion())
      cond = b.create<arith::XOrIOp>(cond, trueVal);
    precondition = b.create<arith::AndIOp>(precondition, cond);
  }

  return {precondition, domOp};
}

template <bool mut, typename T>
static void populateArefOp(ImplicitLocOpBuilder &b, T op) {
  Block *block = &op.getRegion().emplaceBlock();
  for (auto input : op.getOperand().getDefiningOp()->getOperands()) {
    block->addArgument(createViewType(input.getType(), mut), b.getLoc());
  }
  b.setInsertionPointToStart(block);
}

static nvws::ArefPutOp createArefPutOpInPartition(ImplicitLocOpBuilder &b,
                                                  Partition *partition,
                                                  TypeRange types, Value aref,
                                                  ValueRange indexes,
                                                  Value reset = Value()) {
  if (!reset)
    reset = b.create<arith::ConstantOp>(b.getBoolAttr(false));
  auto op = createInPartition<nvws::ArefPutOp>(b, partition, types, aref,
                                               indexes, reset);
  populateArefOp<true>(b, op);
  return op;
}

static nvws::ArefGetOp createArefGetOpInPartition(ImplicitLocOpBuilder &b,
                                                  Partition *partition,
                                                  TypeRange types, Value aref,
                                                  ValueRange indexes) {
  auto op =
      createInPartition<nvws::ArefGetOp>(b, partition, types, aref, indexes);
  populateArefOp<false>(b, op);
  return op;
}

LogicalResult triton::gpu::specializeLoadMMADependencies(scf::ForOp &loop,
                                                         int defaultNumStages) {
  auto ops = llvm::to_vector(loop.getOps<ttng::MMAv5OpInterface>());
  if (ops.empty())
    return success();
  // Support only 1 MMA op.
  if (ops.size() > 1) {
    return mlir::emitWarning(
        loop.getLoc(),
        "failed to warp specialize: more than one `tt.dot` found in the loop");
  }
  ttng::MMAv5OpInterface mmaOp = ops.front();
  auto dot = cast<DotOpInterface>(*mmaOp);

  // Look for the loads that feed the A and B operands.
  SmallVector<Operation *> aChain, bChain;
  if (failed(findSingleChainToLoad(loop, dot.getA(), aChain)) ||
      failed(findSingleChainToLoad(loop, dot.getB(), bChain))) {
    return mlir::emitWarning(loop.getLoc(),
                             "failed to warp specialize: could not find TMA "
                             "loads for `tt.dot` operands");
  }

  ttng::TMEMAllocOp oldAccAlloc =
      mmaOp.getAccumulator().getDefiningOp<ttng::TMEMAllocOp>();
  if (!oldAccAlloc)
    return mlir::emitWarning(mmaOp.getLoc(), "accumulator is not a TMEM alloc");
  auto accUsersInLoop = llvm::to_vector(
      llvm::make_filter_range(oldAccAlloc->getUsers(), [&](Operation *user) {
        return loop.getBody()->findAncestorOpInBlock(*user);
      }));

  // Determine if the MMA accumulator can be multibuffered.
  auto isLoadPipelineable = [&](Operation *op) {
    return llvm::is_contained({aChain.back(), bChain.back()}, op);
  };
  bool accIsMultiBuffered =
      // All operand feeds are pipelineable.
      ttng::mmaHasPipelineableOperands(mmaOp, loop, isLoadPipelineable) &&
      // MMAs in subsequent iterations can be overlapped.
      !ttng::hasAccReadModifyWrite(mmaOp, loop) &&
      // The accumulator is reset at some point, thus allowing multibuffering.
      ttng::isAccMultibufferingPossible(mmaOp, loop) &&
      // The user didn't disable it with a flag.
      !getDisallowAccMultiBuffer(loop);

  // Uses of the accumulator inside the loop must occur after the MMA op as they
  // will be placed in a user partition.
  // TODO: We can support uses prior to the MMA op by rotating the user loop.
  DominanceInfo domInfo(loop);
  for (Operation *user : accUsersInLoop) {
    if (domInfo.dominates(mmaOp, user))
      continue;
    return mlir::emitWarning(loop.getLoc(),
                             "failed to warp specialize: accumulator user does "
                             "not occur after the `tt.dot`");
  }

  ImplicitLocOpBuilder b(mmaOp.getLoc(), loop);
  auto intCst = [&](int value, unsigned width = 32) {
    return b.create<arith::ConstantIntOp>(value, width);
  };

  // Collect a condition that fires whenever the accumulator value is reset in
  // the loop.
  Value overridePred;
  for (Operation *user : accUsersInLoop) {
    if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(user)) {
      Value flag = mmaOp.useAccumulator();
      if (!matchPattern(flag, m_One())) {
        if (auto arg = dyn_cast<BlockArgument>(flag)) {
          auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
          overridePred = yield.getOperand(arg.getArgNumber() - 1);
          b.setInsertionPoint(yield);
          overridePred = b.create<arith::XOrIOp>(overridePred, intCst(true, 1));
        } else {
          return mlir::emitWarning(flag.getLoc(), "acc use flag is not an arg");
        }
      }
    } else if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(user)) {
      if (!matchPattern(storeOp.getPred(), m_Zero()))
        overridePred = storeOp.getPred();
    } else if (!isa<ttng::TMEMLoadOp>(user)) {
      return mlir::emitWarning(user->getLoc(), "unexpected accumulator user");
    }
  }

  // Pattern match succeeded. Now rewrite the loads and MMA ops to pass tensor
  // values through buffers.
  int numStages = getNumStagesOrDefault(loop, defaultNumStages);
  int numMmaStages = 1 + accIsMultiBuffered;
  WarpSchedule schedule;
  Partition *loadPartition = schedule.addPartition(0);
  Partition *mmaPartition = schedule.addPartition(numStages);

  // Create the Load Aref
  auto loadIndex = addIndex(b, loop, numStages);

  Operation *aLoad = aChain.back();
  Operation *bLoad = bChain.back();
  auto aType = cast<RankedTensorType>(aLoad->getResult(0).getType());
  auto bType = cast<RankedTensorType>(bLoad->getResult(0).getType());
  SharedEncodingTrait aEnc = getSharedEncoding(aChain.back());
  SharedEncodingTrait bEnc = getSharedEncoding(bChain.back());
  Value aAlloc = createAlloc(loop, aType, aLoad->getLoc(), aEnc, numStages);
  Value bAlloc = createAlloc(loop, bType, bLoad->getLoc(), bEnc, numStages);

  auto tmaTypes = nvws::TypeArrayAttr::get(
      loop.getContext(), {aAlloc.getType(), bAlloc.getType()});
  auto tmaArefType = nvws::ArefType::get(loop.getContext(), tmaTypes, 1);

  b.setInsertionPoint(loop);
  Value tmaAref = b.create<nvws::ArefCreateOp>(
      tmaArefType, SmallVector<Value>{aAlloc, bAlloc});

  b.setInsertionPoint(oldAccAlloc);
  ttng::TMEMAllocOp accAlloc =
      createTMemAlloc(b, oldAccAlloc, /*multiBuffered=*/true, numMmaStages);
  // If the accumulator is multibuffered, the buffer changes when the
  // accumulator is reset.
  auto accIndex = addIndex(b, loop, numMmaStages, overridePred);
  // Create Accumulator Aref
  auto mmaArefType = nvws::ArefType::get(
      loop.getContext(),
      nvws::TypeArrayAttr::get(loop.getContext(), {accAlloc.getType()}), 1);
  Value accAref =
      b.create<nvws::ArefCreateOp>(mmaArefType, SmallVector<Value>{accAlloc});

  // Copy the loads into the aref put value;
  b.setInsertionPoint(aLoad->isBeforeInBlock(bLoad) ? aLoad : bLoad);
  auto loadPut = createArefPutOpInPartition(b, loadPartition, TypeRange{},
                                            tmaAref, loadIndex);
  for (auto [i, load] :
       llvm::enumerate(SmallVector<Operation *>{aLoad, bLoad})) {
    auto store = createInPartition<LocalStoreOp>(
        b, loadPartition, TypeRange{}, load->getResult(0),
        loadPut.getRegion().getArgument(i));
    load->moveBefore(store);
    loadPartition->insert(load);
  }
  createInPartition<nvws::ArefReturnOp>(b, loadPartition, ValueRange{});

  // Now handle the accumulator, which is the tricky bit. The accumulator value
  // may be conditionally reset in the MMA partition before the MMA op, and it
  // may be conditionally used in a user partition.
  b.setInsertionPointAfter(loadPut);
  // Copy the acc into the aref get value;
  auto mmaPut = createArefPutOpInPartition(
      b, mmaPartition, TypeRange{}, accAref, accIndex,
      b.create<arith::XOrIOp>(mmaOp.useAccumulator(), intCst(true, 1)));
  auto mmaReturn =
      createInPartition<nvws::ArefReturnOp>(b, mmaPartition, ValueRange{});

  // Copy the loads into the aref get value;
  b.setInsertionPoint(mmaReturn);
  auto tmaGet = createArefGetOpInPartition(b, mmaPartition, TypeRange{},
                                           tmaAref, {accIndex});
  auto getReturn =
      createInPartition<nvws::ArefReturnOp>(b, mmaPartition, ValueRange{});
  b.setInsertionPoint(getReturn);

  // Place users in the MMA partition, dropping the loads themselves
  aChain.pop_back();
  bChain.pop_back();
  auto allUsers = llvm::to_vector(llvm::concat<Operation *>(aChain, bChain));
  // Move users and mma to aref
  for (Operation *user : llvm::reverse(allUsers)) {
    user->moveBefore(getReturn);
  }
  mmaOp->moveBefore(getReturn);
  mmaPartition->insert(mmaOp);

  // Use the views for the results of the loads and the acc
  for (auto [i, load] :
       llvm::enumerate(SmallVector<Operation *>{aLoad, bLoad})) {
    mlir::replaceAllUsesInRegionWith(load->getResult(0),
                                     tmaGet.getRegion().getArgument(i),
                                     tmaGet.getRegion());
  }
  mlir::replaceAllUsesInRegionWith(oldAccAlloc.getResult(),
                                   mmaPut.getRegion().getArgument(0),
                                   mmaPut.getRegion());

  // assign Users to partition and remove unneeded Local Alloc ops
  for (Operation *user : llvm::reverse(allUsers)) {
    if (isa<LocalAllocOp>(user)) {
      replaceUsesAndPropagateType(b, user, user->getOperand(0));
      user->erase();
    } else {
      mmaPartition->insert(user);
    }
  }

  // Replace uses of the original accumulator with the right subview before,
  // inside, and after the loop.
  SmallVector<Operation *> loadsInLoop;
  for (OpOperand &use : llvm::make_early_inc_range(oldAccAlloc->getUses())) {
    Operation *user = use.getOwner();
    b.setInsertionPoint(user);
    Value bufIdx;
    if (auto store = dyn_cast<ttng::TMEMStoreOp>(user)) {
      Partition *partition = nullptr;
      if (loop->isAncestor(store)) {
        bufIdx = b.create<arith::AddIOp>(accIndex, intCst(numMmaStages - 1));
        bufIdx = b.create<arith::RemUIOp>(bufIdx, intCst(numMmaStages));
        partition = mmaPartition;
      } else {
        if (!store->isBeforeInBlock(loop))
          return mlir::emitWarning(store.getLoc(), "store not before loop?");
        bufIdx = intCst(0);
      }
      Value trueVal = b.create<arith::ConstantOp>(b.getBoolAttr(true));
      auto mmaClear = createArefPutOpInPartition(b, partition, TypeRange{},
                                                 accAref, {bufIdx}, trueVal);
      auto mmaReturn =
          createInPartition<nvws::ArefReturnOp>(b, partition, ValueRange{});
      store->moveBefore(mmaReturn);
      store.setOperand(0, mmaClear.getRegion().getArgument(0));
      if (partition)
        partition->insert(store);
      b.setInsertionPointAfter(mmaClear);
    } else if (auto load = dyn_cast<ttng::TMEMLoadOp>(user)) {
      if (loop->isAncestor(load)) {
        loadsInLoop.push_back(load);
      } else {
        // TODO: Support non-flattened persistent epilogues
        if (!loop->isBeforeInBlock(load))
          return mlir::emitWarning(load.getLoc(), "load not after loop?");
        bufIdx = loop.getResult(accIndex.getArgNumber() - 1);
        auto arefGet = createArefGetOpInPartition(
            b, nullptr, TypeRange{load.getResult().getType()}, accAref,
            {bufIdx});
        load.replaceAllUsesWith(arefGet.getResult(0));
        auto arefReturn =
            b.create<nvws::ArefReturnOp>(ValueRange{load.getResult()});
        load->moveBefore(arefReturn);
        load.setOperand(arefGet.getRegion().getArgument(0));
      }
    } else if (user == mmaOp) {
    } else {
      return mlir::emitWarning(user->getLoc(), "unknown acc user");
    }
  }

  b.setInsertionPointAfter(mmaPut);
  OpBuilder::InsertPoint donePt = b.saveInsertionPoint();

  // Replace uses of the accumulator inside the loop with a value loaded from
  // the buffer. Place these in a new user partition.
  if (!loadsInLoop.empty()) {
    // Because the accumulator reset occurs after the MMA op, we have to place
    // the wait on the empty barrier after the MMA op as well. This is OK since
    // we know all buffers are empty upon entry to the loop. However, this means
    // the last mbarrier is guarding the first buffer. Thus, initialize all but
    // the last mbarrier.
    Operation *domOp = findNearestCommonDominator(loadsInLoop, domInfo);
    assert(domOp && "could not find common dominator for accumulator uses");
    b.restoreInsertionPoint(donePt);

    b.setInsertionPoint(domOp);
    Partition *userPartition = schedule.addPartition(numStages + numMmaStages);
    // Acquire and get the accumulator result. Normally, we want to acquire the
    // accumulator for as small of a critical section as possible to unblock
    // dependents, but if the most dominating user is inside a conditional,
    // acquire the accumulator for the whole branch. This will improve
    // instruction scheduling and interleaving of the TMEM load.
    if (auto ifOp = dyn_cast<scf::IfOp>(domOp->getParentOp()))
      for (Operation &op : ifOp.getBody(0)->getOperations())
        userPartition->insert(&op);
    auto arefGet = createArefGetOpInPartition(
        b, userPartition, TypeRange{domOp->getResult(0).getType()}, accAref,
        {accIndex});
    domOp->replaceAllUsesWith(ValueRange{arefGet.getResult(0)});
    auto arefReturn = createInPartition<nvws::ArefReturnOp>(
        b, userPartition, ValueRange{domOp->getResult(0)});
    domOp->moveBefore(arefReturn);
    domOp->setOperand(0, arefGet.getRegion().getArgument(0));
    userPartition->insert(domOp);

    // Propagate the partition to transitive users. If this happens to create a
    // cycle, subsequent warp specialization steps will fail.
    SmallVector<Operation *> transitiveUsers(loadsInLoop.begin(),
                                             loadsInLoop.end());
    while (!transitiveUsers.empty()) {
      Operation *op = transitiveUsers.pop_back_val();
      if (isa<scf::YieldOp>(op))
        continue;
      op = loop.getBody()->findAncestorOpInBlock(*op);
      userPartition->insert(op);
      llvm::append_range(transitiveUsers, op->getUsers());
    }
    // Place the epilogue partition in the default warpgroup. The MMA and load
    // partitions shouldn't have tensor computations in them, which means they
    // will get assigned just 1 warp each.
    schedule.reorderPartitions({2, 1, 0});
  }
  oldAccAlloc->erase();

  schedule.serialize(loop);

  auto mod = loop->getParentOfType<ModuleOp>();
  if (failed(mod.verify()))
    return mod.emitError("failed to verify");
  llvm::errs() << mod << '\n';
  //  HACK: Set this attribute so that LowerLoops will multi-buffer TMA
  //  descriptors.
  loop->setAttr(kScheduledMaxStageAttrName, b.getI32IntegerAttr(numStages));
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPULOADMMASPECIALIZATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct LoadMMASpecialization
    : triton::gpu::impl::TritonGPULoadMMASpecializationBase<
          LoadMMASpecialization> {
  using TritonGPULoadMMASpecializationBase::TritonGPULoadMMASpecializationBase;

  void runOnOperation() override;
};
} // namespace

void LoadMMASpecialization::runOnOperation() {
  SmallVector<scf::ForOp> loops;
  getOperation().walk([&](scf::ForOp loop) {
    if (loop->hasAttr(kWarpSpecializeAttrName))
      loops.push_back(loop);
  });
  for (scf::ForOp loop : loops) {
    if (failed(specializeLoadMMADependencies(loop, numStages)))
      continue;
  }
}
