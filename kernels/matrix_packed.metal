// Affine Q4 codes stay packed in device memory. Two original native QMV
// lane partitions are mapped into each token's pair of virtual M rows.
// M8/K32 avoids M16 padding and the register unpack of integer weight codes.
constexpr int BM = 8;
constexpr int BN = 32;
constexpr int BK = 32;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int n0 = int(threadgroup_position_in_grid.y) * BN;
threadgroup T input_tile[SPLITS][BM * BK];
threadgroup float partial[32][VERIFY_T * BN];
constexpr auto descriptor = matmul2d_descriptor(BM, BN, BK, false, true,
    false, matmul2d_descriptor::mode::multiply);
matmul2d<descriptor, metal::execution_simdgroup> op;
tensor<threadgroup T, extents<int, BK, BM>, tensor_inline> A(input_tile[sg], extents<int, BK, BM>{});
// Sub-byte inline tensors use tightly packed first-axis storage; K is the
// first axis here, exactly matching the checkpoint's row-contiguous words.
tensor<device uint4b_format, dextents<int, 2>, tensor_inline> B(
    (device uchar*)w, dextents<int, 2>{K_SIZE, N_SIZE});
auto dot = op.template get_destination_cooperative_tensor<
    tensor<threadgroup T, extents<int, BK, BM>, tensor_inline>,
    tensor<device uint4b_format, extents<int, BK, BN>, tensor_inline>, float>();
// Cooperative storage can be larger than the logical tile. Let TensorOps
// allocate its actual capacity rather than deriving it from M*N/32.
auto result = op.template get_destination_cooperative_tensor<
    tensor<threadgroup T, extents<int, BK, BM>, tensor_inline>,
    tensor<device uint4b_format, extents<int, BK, BN>, tensor_inline>, float>();
for (int pair = int(sg) * 2; pair < 32; pair += SPLITS * 2) {
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < dot.get_capacity(); ++i) result[i] = 0.0f;
    for (int k0 = pair * 16; k0 < K_SIZE; k0 += 512) {
#pragma clang loop unroll(full)
        for (int row = 0; row < BM; ++row) {
            const int token = row / 2;
            input_tile[sg][row * BK + lane] = token < VERIFY_T
                && int(lane) / 16 == row % 2 ? x[token * K_SIZE + k0 + lane] : T(0);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto right = B.template slice<BK, BN>(k0, n0);
        op.run(A, right, dot);
        simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < dot.get_capacity(); ++i) {
            if (dot.is_valid_element(i)) {
                const auto p = dot.get_multidimensional_index(i);
                const int token = p[1] / 2;
                if (token < VERIFY_T) {
                    const int base = k0 + (p[1] % 2) * 16;
                    const device T* values = x + token * K_SIZE + base;
                    float sum = 0.0f;
#pragma clang loop unroll(full)
                    for (int v = 0; v < 16; v += 4)
                        sum += values[v] + values[v + 1] + values[v + 2] + values[v + 3];
                    const size_t meta = size_t(n0 + p[0]) * (K_SIZE / GROUP_SIZE) + base / GROUP_SIZE;
                    result[i] += float(scales[meta]) * dot[i] + sum * float(biases[meta]);
                }
            }
        }
    }
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < dot.get_capacity(); ++i) {
        if (dot.is_valid_element(i)) {
            const auto p = dot.get_multidimensional_index(i);
            if (p[1] / 2 < VERIFY_T)
                partial[pair + p[1] % 2][(p[1] / 2) * BN + p[0]] = result[i];
        }
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int off = int(sg); off < VERIFY_T * BN; off += SPLITS) {
    float value = partial[lane][off];
    value = simd_sum(value);
    if (lane == 0) y[(off / BN) * N_SIZE + n0 + off % BN] = T(value);
}
