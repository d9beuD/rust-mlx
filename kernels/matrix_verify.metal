constexpr int BM = TILE_M;
constexpr int BN = 32;
constexpr int BK = TILE_K;
const uint tid = thread_position_in_threadgroup.x;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int n0 = int(threadgroup_position_in_grid.y) * BN;
constexpr int CHUNK = K_SIZE / SPLITS;
const int begin = int(sg) * CHUNK;
threadgroup T weight_tile[SPLITS][BK * BN];
threadgroup float partial[SPLITS][BM * BN];

constexpr auto descriptor = matmul2d_descriptor(
    BM, BN, BK, false, false, false,
    matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<descriptor, metal::execution_simdgroup> op;
tensor<device T, dextents<int, 2>, tensor_inline> A(
    (device T*)x, dextents<int, 2>{K_SIZE, BM}, array<int, 2>{1, K_SIZE});
tensor<threadgroup T, extents<int, BN, BK>, tensor_inline> B(weight_tile[sg], extents<int, BN, BK>{});
tensor<threadgroup float, extents<int, BN, BM>, tensor_inline> C(partial[sg], extents<int, BN, BM>{});
using LeftRegister = typename matmul2d<descriptor, metal::execution_simdgroup>::template cooperative_tensor_left_input_t<T,T,float>;
using RightRegister = typename matmul2d<descriptor, metal::execution_simdgroup>::template cooperative_tensor_right_input_t<T,T,float>;
using LeftOperand = conditional_t<REGISTER_INPUT && !HYBRID_INPUT, LeftRegister,
    tensor<device T, extents<int, BK, BM>, tensor_inline>>;
using RightOperand = conditional_t<REGISTER_INPUT, RightRegister,
    tensor<threadgroup T, extents<int, BN, BK>, tensor_inline>>;
auto accumulator = op.template get_destination_cooperative_tensor<LeftOperand, RightOperand, float>();
#pragma clang loop unroll(full)
for (uint16_t i = 0; i < accumulator.get_capacity(); ++i) {
    accumulator[i] = 0.0f;
}

for (int k0 = begin; k0 < begin + CHUNK; k0 += BK) {
    if constexpr (REGISTER_INPUT) {
        // macOS27 cooperative inputs: affine unpack/dequantize stays in
        // thread registers; no threadgroup weight stores or barriers.
        auto right = op.template get_right_input_cooperative_tensor<T, T, float>();
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < right.get_capacity(); ++i) {
            // Physical register storage can exceed the logical operand tile.
            right[i] = T(0);
            if (right.is_valid_element(i)) {
                const auto p = right.get_multidimensional_index(i);
                right[i] = matrix_weight<T>(w, scales, biases,
                    n0 + p[0], k0 + p[1], K_SIZE, BITS, GROUP_SIZE);
            }
        }
        if constexpr (HYBRID_INPUT) {
            auto left = A.template slice<BK, BM>(k0, 0);
            op.run(left, right, accumulator);
        } else {
            auto left = op.template get_left_input_cooperative_tensor<T, T, float>();
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < left.get_capacity(); ++i) {
                left[i] = T(0);
                if (left.is_valid_element(i)) {
                    const auto p = left.get_multidimensional_index(i);
                    left[i] = p[1] < VERIFY_T ? x[p[1] * K_SIZE + k0 + p[0]] : T(0);
                }
            }
            op.run(left, right, accumulator);
        }
    } else {
#pragma clang loop unroll(full)
        for (int ki = 0; ki < BK; ++ki) {
            weight_tile[sg][ki * BN + lane] = matrix_weight<T>(
                w, scales, biases, n0 + int(lane), k0 + ki,
                K_SIZE, BITS, GROUP_SIZE);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto left = A.template slice<BK, BM>(k0, 0);
        op.run(left, B, accumulator);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
accumulator.store(C);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int off = int(tid); off < VERIFY_T * BN; off += SPLITS * 32) {
    float value = partial[0][off];
#pragma clang loop unroll(full)
    for (int s = 1; s < SPLITS; ++s) value += partial[s][off];
    y[(off / BN) * N_SIZE + n0 + off % BN] = T(value);
}
