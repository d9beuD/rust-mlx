// Affine algebra without BF16 weight reconstruction. Thirty-two partitions
// groups reproduce the native QMV lane partitions and final simd reduction.
// TensorOps dot-product order is still measured rather than assumed exact.
constexpr int BM = TILE_M;
constexpr int BN = 32;
constexpr int BK = 16;
constexpr int VALUES = BITS == 4 || BITS == 5 ? 16 : 8;
const uint sg = simdgroup_index_in_threadgroup;
const int n0 = int(threadgroup_position_in_grid.y) * BN;
threadgroup float partial[32][VERIFY_T * BN];
constexpr auto descriptor = matmul2d_descriptor(BM, BN, BK, false, false,
    false, matmul2d_descriptor::mode::multiply);
matmul2d<descriptor, metal::execution_simdgroup> op;
auto left = op.template get_left_input_cooperative_tensor<T, T, float>();
auto right = op.template get_right_input_cooperative_tensor<T, T, float>();
using LeftOperand = typename matmul2d<descriptor, metal::execution_simdgroup>::template cooperative_tensor_left_input_t<T,T,float>;
using RightOperand = typename matmul2d<descriptor, metal::execution_simdgroup>::template cooperative_tensor_right_input_t<T,T,float>;
auto dot = op.template get_destination_cooperative_tensor<LeftOperand, RightOperand, float>();
auto result = op.template get_destination_cooperative_tensor<LeftOperand, RightOperand, float>();
for (int part = int(sg); part < 32; part += SPLITS) {
#pragma clang loop unroll(full)
for (uint16_t i = 0; i < dot.get_capacity(); ++i) result[i] = 0.0f;
for (int k0 = part * VALUES; k0 < K_SIZE; k0 += 32 * VALUES) {
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < left.get_capacity(); ++i) {
        left[i] = T(0);
        if (left.is_valid_element(i)) {
            const auto p = left.get_multidimensional_index(i);
            left[i] = p[1] < VERIFY_T && p[0] < VALUES ? x[p[1] * K_SIZE + k0 + p[0]] : T(0);
        }
    }
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < right.get_capacity(); ++i) {
        right[i] = T(0);
        if (right.is_valid_element(i)) {
            const auto p = right.get_multidimensional_index(i);
            right[i] = p[1] < VALUES ? T(matrix_code(w, n0 + p[0], k0 + p[1], K_SIZE, BITS)) : T(0);
        }
    }
    op.run(left, right, dot);
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < dot.get_capacity(); ++i) {
        if (dot.is_valid_element(i)) {
            const auto p = dot.get_multidimensional_index(i);
            if (p[1] < VERIFY_T) {
                const device T* values = x + p[1] * K_SIZE + k0;
                float sum = 0.0f;
#pragma clang loop unroll(full)
                for (int v = 0; v < VALUES; v += (BITS == 8 ? 1 : BITS == 5 ? 8 : 4)) {
                    if constexpr (BITS == 8) {
                        sum += values[v];
                    } else if constexpr (BITS == 5) {
                        sum += values[v] + values[v + 1] + values[v + 2] + values[v + 3]
                            + values[v + 4] + values[v + 5] + values[v + 6] + values[v + 7];
                    } else {
                        sum += values[v] + values[v + 1] + values[v + 2] + values[v + 3];
                    }
                }
                const size_t metadata = size_t(n0 + p[0]) * (K_SIZE / GROUP_SIZE) + k0 / GROUP_SIZE;
                result[i] += float(scales[metadata]) * dot[i] + sum * float(biases[metadata]);
            }
        }
    }
}
#pragma clang loop unroll(full)
for (uint16_t i = 0; i < dot.get_capacity(); ++i) {
    if (dot.is_valid_element(i)) {
        const auto p = dot.get_multidimensional_index(i);
        if (p[1] < VERIFY_T) partial[part][p[1] * BN + p[0]] = result[i];
    }
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);
// Each SIMD group reduces one output element with native simd_sum order.
for (int off = int(sg); off < VERIFY_T * BN; off += SPLITS) {
    float value = partial[thread_index_in_simdgroup][off];
    value = simd_sum(value);
    if (thread_index_in_simdgroup == 0) {
        y[(off / BN) * N_SIZE + n0 + off % BN] = T(value);
    }
}
