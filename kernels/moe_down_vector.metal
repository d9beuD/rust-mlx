// Additional adaptation: hoisted packed addresses and joint four-output uint4 load.
// MTPLX by Youssof Altoukhi, Apache-2.0, revision9882703f3105363ddc37eca9f97aa09a1d387112.
// Source mtplx/kernels/qwen4_m4_routed_down.py; third-party/MTPLX-APACHE-2.0 and NOTICE.
// Adaptation: affine g64; preserve scalar/reduction casts; bounded1-8-row dispatch.

    const uint output_tile = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint output_base =
        output_tile * OUTPUTS_PER_THREADGROUP
        + simd_group * OUTPUTS_PER_SIMD;

    bfloat pending[OUTPUTS_PER_SIMD];
    bfloat routed_value[OUTPUTS_PER_SIMD];

    for (uint order_index = 0; order_index < TOP_K; ++order_index) {
        const uint slot = SLOT_ORDER[order_index];
        const uint expert = expert_ids[row * TOP_K + slot];
        const device bfloat* x =
            routed_h + (row * TOP_K + slot) * K + lane * VALUES_PER_THREAD;
        // Prepared owning allocation: four adjacent output codes per uint4.
        // Every row base is aligned and K/8=80 uint4s per output tile.
        const device uint4* w = (const device uint4*)weights
            + (((size_t)expert*HIDDEN + output_base)/4)*(K/8) + lane;
        const device bfloat* scale = scales
            + ((size_t)expert * HIDDEN + output_base) * GROUPS_PER_ROW
            + lane / (GROUP_SIZE / VALUES_PER_THREAD);
        const device bfloat* bias = biases
            + ((size_t)expert * HIDDEN + output_base) * GROUPS_PER_ROW
            + lane / (GROUP_SIZE / VALUES_PER_THREAD);

        float result[OUTPUTS_PER_SIMD] = {0.0f};
        int k = 0;
        for (; k < int(K - BLOCK_SIZE); k += BLOCK_SIZE) {
            float x_thread[VALUES_PER_THREAD];
            float sum = load_q4_vector(x, x_thread);
            uint4 codes=*w;
            for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {

                const device bfloat* output_scale =
                    scale + out * GROUPS_PER_ROW;
                const device bfloat* output_bias =
                    bias + out * GROUPS_PER_ROW;
                result[out] += qdot_register(
                    codes[out],
                    x_thread,
                    float(output_scale[0]),
                    float(output_bias[0]),
                    sum, VALUES_PER_THREAD);
            }
            w += BLOCK_SIZE / 8;
            scale += BLOCK_SIZE / GROUP_SIZE;
            bias += BLOCK_SIZE / GROUP_SIZE;
            x += BLOCK_SIZE;
        }

        const int remaining = clamp(
            int(K) - k - int(lane * VALUES_PER_THREAD),
            0,
            int(VALUES_PER_THREAD));
        if (remaining > 0) {
            float x_thread[VALUES_PER_THREAD];
            float sum = load_q4_vector_safe(x, x_thread, remaining);
            uint4 codes=*w;
            for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {

                const device bfloat* output_scale =
                    scale + out * GROUPS_PER_ROW;
                const device bfloat* output_bias =
                    bias + out * GROUPS_PER_ROW;
                result[out] += qdot_register(
                    codes[out],
                    x_thread,
                    float(output_scale[0]),
                    float(output_bias[0]),
                    sum,
                    remaining);
            }
        }

        for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {
            result[out] = simd_sum(result[out]);
        }
        if (lane == 0) {
            for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {
                bfloat down_value = bfloat(result[out]);
                bfloat product = bfloat(
                    float(down_value) * float(route_scores[row * TOP_K + slot]));
                if (order_index == 0 || order_index == 2) {
                    pending[out] = product;
                } else if (order_index == 1) {
                    routed_value[out] = bfloat(float(pending[out]) + float(product));
                } else if (order_index == 3) {
                    bfloat second = bfloat(float(pending[out]) + float(product));
                    routed_value[out] = bfloat(float(second) + float(routed_value[out]));
                } else {
                    routed_value[out] = bfloat(float(product) + float(routed_value[out]));
                }
            }
        }
    }

    if (lane == 0) {
        for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {
            routed_down[row * HIDDEN + output_base + out] = routed_value[out];
        }
    }
