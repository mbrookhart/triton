// RUN: triton-opt %s --tritongpu-hoist-tmem-alloc --tritongpu-partition-scheduling -allow-unregistered-dialect | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#load_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_T = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>

#smem = #ttg.shared_memory
#tmem_acc = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
#tmem_lhs = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = false>

#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = false>
#tmem2 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @attention_forward
tt.func public @attention_forward(
  %Q_shared: !ttg.memdesc<256x64xf16, #shared, #smem>,
  %K_desc: !tt.tensordesc<tensor<64x64xf16, #shared>>,
  %V_desc: !tt.tensordesc<tensor<64x64xf16, #shared>>,
  %qk_scale: f32,
  %n_tiles: i32
) {
  %true = arith.constant true
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32

  %neg_inf = arith.constant dense<0xFF800000> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  %zero = arith.constant dense<0.0> : tensor<256x64xf32, #blocked>
  %one = arith.constant dense<1.0> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>

  %QK_tmem, %QK_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<256x64xf32, #tmem_acc, #ttng.tensor_memory, mutable>, !ttg.async.token)

  %loop_outs:4 = scf.for %i = %c0_i32 to %n_tiles step %c64_i32 iter_args(
    %l_i = %one,
    %acc = %zero,
    %m_i = %neg_inf,
    %e_i = %one
  ) -> (
    tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>,
    tensor<256x64xf32, #blocked>,
    tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>,
    tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  ) : i32 {

    %K = tt.descriptor_load %K_desc[%i, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #load_blocked>
    %K_shared = ttg.local_alloc %K : (tensor<64x64xf16, #load_blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>

    %K_trans = ttg.memdesc_trans %K_shared {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared_T, #smem>
    %QK_mma_tok = ttng.tc_gen5_mma %Q_shared, %K_trans, %QK_tmem[%QK_tok], %false, %true : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared_T, #smem>, !ttg.memdesc<256x64xf32, #tmem_acc, #ttng.tensor_memory, mutable>

    %QK, %QK_load_tok = ttng.tmem_load %QK_tmem[%QK_mma_tok] : !ttg.memdesc<256x64xf32, #tmem_acc, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>
    %row_max = "compute_row_max"(%QK, %qk_scale) : (tensor<256x64xf32, #blocked>, f32) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %QK_adj = "sub_row_max"(%QK, %row_max, %qk_scale) : (tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, f32) -> tensor<256x64xf32, #blocked>
    // CHECK: [[SOFTMAX:%.*]] = math.exp2 {{.*}} {ttg.partition = 0 : i32} : tensor<256x64xf32
    %softmax = math.exp2 %QK_adj : tensor<256x64xf32, #blocked>
    %diff = arith.subf %m_i, %row_max : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %alpha = math.exp2 %diff : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>

    %l_ij = "tt.reduce"(%softmax) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %68 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %68 : f32
    }) : (tensor<256x64xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %l_i_scaled = arith.mulf %l_i, %alpha : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %next_l_i = arith.addf %l_i_scaled, %l_ij : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>

    %alpha_0 = tt.expand_dims %alpha {axis = 1 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
    %alpha_1 = tt.broadcast %alpha_0 : tensor<256x1xf32, #blocked> -> tensor<256x64xf32, #blocked>

    %acc_corrected = arith.mulf %acc, %alpha_1 : tensor<256x64xf32, #blocked>

    // CHECK: [[X:%.*]] = arith.addf [[SOFTMAX]], [[SOFTMAX]] {ttg.partition = 0 : i32}
    %x = arith.addf %softmax, %softmax : tensor<256x64xf32, #blocked>
    // CHECK-NEXT: [[ACC_X:%.*]] = arith.addf %{{.*}}, [[X]] {ttg.partition = 3 : i32}
    %acc_x = arith.addf %acc, %x : tensor<256x64xf32, #blocked>
    %e = "sum"(%acc_x) : (tensor<256x64xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %next_e_i = arith.addf %e_i, %e : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>

    %V = tt.descriptor_load %V_desc[%i, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #load_blocked>
    %V_shared = ttg.local_alloc %V : (tensor<64x64xf16, #load_blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %P = arith.truncf %softmax : tensor<256x64xf32, #blocked> to tensor<256x64xf16, #blocked>

    %P_tmem = ttng.tmem_alloc %P : (tensor<256x64xf16, #blocked>) -> !ttg.memdesc<256x64xf16, #tmem_lhs, #ttng.tensor_memory>
    %acc_tmem, %acc_tok = ttng.tmem_alloc %acc_corrected : (tensor<256x64xf32, #blocked>) -> (!ttg.memdesc<256x64xf32, #tmem_acc, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %PV_mma_tok = ttng.tc_gen5_mma %P_tmem, %V_shared, %acc_tmem[%acc_tok], %true, %true : !ttg.memdesc<256x64xf16, #tmem_lhs, #ttng.tensor_memory>, !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<256x64xf32, #tmem_acc, #ttng.tensor_memory, mutable>
    %O, %O_tok = ttng.tmem_load %acc_tmem[%PV_mma_tok] : !ttg.memdesc<256x64xf32, #tmem_acc, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>

    scf.yield %next_l_i, %O, %row_max, %next_e_i : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  } {tt.warp_specialize}

  "use"(%loop_outs#0, %loop_outs#1, %loop_outs#2) : (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> ()

  tt.return
}
tt.func public @matmul_kernel_descriptor_persistent(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
  %false = arith.constant false
  %true = arith.constant true
  %c148_i32 = arith.constant 148 : i32
  %c8_i32 = arith.constant 8 : i32
  %c128_i32 = arith.constant 128 : i32
  %c64_i32 = arith.constant 64 : i32
  %c1_i64 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c127_i32 = arith.constant 127 : i32
  %c63_i32 = arith.constant 63 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.addi %arg3, %c127_i32 : i32
  %2 = arith.divsi %1, %c128_i32 : i32
  %3 = arith.addi %arg4, %c127_i32 : i32
  %4 = arith.divsi %3, %c128_i32 : i32
  %5 = arith.addi %arg5, %c63_i32 : i32
  %6 = arith.divsi %5, %c64_i32 : i32
  %7 = arith.muli %2, %4 : i32
  %8 = arith.extsi %arg5 : i32 to i64
  %9 = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%8, %c1_i64] : <f16>, <tensor<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>
  %10 = tt.make_tensor_descriptor %arg1, [%arg4, %arg5], [%8, %c1_i64] : <f16>, <tensor<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>
  %11 = arith.extsi %arg4 : i32 to i64
  %12 = tt.make_tensor_descriptor %arg2, [%arg3, %arg4], [%11, %c1_i64] : <f16>, <tensor<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>
  %13 = arith.subi %0, %c148_i32 : i32
  %14 = arith.muli %4, %c8_i32 : i32
  %15 = scf.for %arg6 = %0 to %7 step %c148_i32 iter_args(%arg7 = %13) -> (i32)  : i32 {
    %16 = arith.divsi %arg6, %14 : i32
    %17 = arith.muli %16, %c8_i32 : i32
    %18 = arith.subi %2, %17 : i32
    %19 = arith.minsi %18, %c8_i32 : i32
    %20 = arith.remsi %arg6, %19 : i32
    %21 = arith.addi %17, %20 : i32
    %22 = arith.remsi %arg6, %14 : i32
    %23 = arith.divsi %22, %19 : i32
    %24 = arith.muli %21, %c128_i32 : i32
    %25 = arith.muli %23, %c128_i32 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %26:2 = scf.for %arg8 = %c0_i32 to %6 step %c1_i32 iter_args(%arg9 = %false, %arg10 = %token) -> (i1, !ttg.async.token)  : i32 {
      %45 = arith.muli %arg8, %c64_i32 : i32
      %46 = tt.descriptor_load %9[%24, %45] : !tt.tensordesc<tensor<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
      %47 = ttg.local_alloc %46 : (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>) -> !ttg.memdesc<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
      %48 = tt.descriptor_load %10[%25, %45] : !tt.tensordesc<tensor<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
      %49 = ttg.local_alloc %48 : (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>) -> !ttg.memdesc<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
      %50 = ttg.memdesc_trans %49 {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory> -> !ttg.memdesc<64x128xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory>
      %51 = ttng.tc_gen5_mma %47, %50, %result[%arg10], %arg9, %true : !ttg.memdesc<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<64x128xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable>
      scf.yield %true, %51 : i1, !ttg.async.token
    }
    %27 = arith.addi %arg7, %c148_i32 : i32
    %28 = arith.divsi %27, %14 : i32
    %29 = arith.muli %28, %c8_i32 : i32
    %30 = arith.subi %2, %29 : i32
    %31 = arith.minsi %30, %c8_i32 : i32
    %32 = arith.remsi %27, %31 : i32
    %33 = arith.addi %29, %32 : i32
    %34 = arith.remsi %27, %14 : i32
    %35 = arith.divsi %34, %31 : i32
    %36 = arith.muli %33, %c128_i32 : i32
    %37 = arith.muli %35, %c128_i32 : i32
    %result_0, %token_1 = ttng.tmem_load %result[%26#1] : !ttg.memdesc<128x128xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>>
    %38 = tt.reshape %result_0 : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<128x2x64xf32, #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>>
    %39 = tt.trans %38 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>> -> tensor<128x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>>
    %outLHS, %outRHS = tt.split %39 : tensor<128x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>> -> tensor<128x64xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>>
    %40 = arith.truncf %outLHS : tensor<128x64xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>> to tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>>
    %41 = ttg.convert_layout %40 : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    tt.descriptor_store %12[%36, %37], %41 : !tt.tensordesc<tensor<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %42 = arith.truncf %outRHS : tensor<128x64xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>> to tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>>
    %43 = ttg.convert_layout %42 : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %44 = arith.addi %37, %c64_i32 : i32
    tt.descriptor_store %12[%36, %44], %43 : !tt.tensordesc<tensor<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    scf.yield %27 : i32
  } {tt.flatten, tt.warp_specialize}
  tt.return
}
}
