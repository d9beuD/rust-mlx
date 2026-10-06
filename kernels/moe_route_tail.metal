// Adapted from MTPLX by Youssof Altoukhi, Apache-2.0, revision9882703f3105363ddc37eca9f97aa09a1d387112.
// mtplx/kernels/qwen4_m4_route.py; see third-party/MTPLX-{APACHE-2.0,NOTICE}.
// Modification: native router/shared projections remain separate; tail accepts1–8 independent rows.

    constexpr int TNEXP = 512;
    constexpr int TTOPK = 10;
    constexpr int TNR = 4;
    constexpr int TNT = 128;
    constexpr int TNSG = TNT / 32;

    const uint gid = threadgroup_position_in_grid.x;      // verifier row
    const uint lid = thread_position_in_threadgroup.x;
    const uint slid = thread_index_in_simdgroup;
    const uint sgid = simdgroup_index_in_threadgroup;

    threadgroup float local_max[32];
    threadgroup float local_norm[32];
    threadgroup uint keys[TNEXP];
    threadgroup bfloat gate_values[TNEXP];
    threadgroup uint partial[TNSG];
    threadgroup uint selected[TTOPK];

    const device bfloat* in = logits + (size_t)gid * TNEXP + lid * TNR;

    // ---- softmax_single_row<bfloat16_t, float, 4>, layout-for-layout -----
    float ld[TNR];
    #pragma clang loop unroll(full)
    for (int i = 0; i < TNR; ++i) ld[i] = float(in[i]);

    if (sgid == 0) {
        local_max[slid] = -metal::numeric_limits<float>::infinity();
        local_norm[slid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Limits<AccT>::finite_min, exactly as softmax_single_row seeds it.
    float maxval = -metal::numeric_limits<float>::max();
    #pragma clang loop unroll(full)
    for (int i = 0; i < TNR; ++i) maxval = (maxval < ld[i]) ? ld[i] : maxval;
    maxval = simd_max(maxval);
    if (slid == 0) local_max[sgid] = maxval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        maxval = simd_max(local_max[slid]);
        if (slid == 0) local_max[0] = maxval;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maxval = local_max[0];

    float normalizer = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < TNR; ++i) {
        // softmax.h softmax_exp() is fast::exp, explicitly.
        const float e = fast::exp(ld[i] - maxval);
        ld[i] = e;
        normalizer += e;
    }
    normalizer = simd_sum(normalizer);
    if (slid == 0) local_norm[sgid] = normalizer;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        normalizer = simd_sum(local_norm[slid]);
        if (slid == 0) local_norm[0] = normalizer;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normalizer = 1.0f / local_norm[0];

    #pragma clang loop unroll(full)
    for (int i = 0; i < TNR; ++i) {
        const int idx = int(lid) * TNR + i;
        const bfloat g = bfloat(ld[i] * normalizer);
        gate_values[idx] = g;
        keys[idx] = route_sort_key(g, (uint)idx);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- top-10 under the stable ascending sort's (value, index) key ----
    // Slot 9 is written first: the stock path slices the LAST ten of an
    // ascending argsort, so slot j holds the (10 - j)-th largest.
    for (int s = 0; s < TTOPK; ++s) {
        uint best = 0u;
        #pragma clang loop unroll(full)
        for (int i = 0; i < TNR; ++i) {
            best = max(best, keys[lid * TNR + i]);
        }
        best = simd_max(best);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (slid == 0) partial[sgid] = best;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint winner = partial[0];
        for (int g = 1; g < TNSG; ++g) winner = max(winner, partial[g]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            selected[TTOPK - 1 - s] = winner;
            // Keys are unique in their low 16 bits, and every real key has the
            // order-map's sign bit set, so 0 can never win a later round.
            keys[winner & 0xFFFFu] = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- gather, bf16 renormalise, shared-gate sigmoid ------------------
    if (lid == 0) {
        bfloat picked[TTOPK];
        #pragma clang loop unroll(full)
        for (int j = 0; j < TTOPK; ++j) {
            const uint idx = selected[j] & 0xFFFFu;
            expert_ids[gid * TTOPK + j] = idx;
            picked[j] = gate_values[idx];
        }
        // row_reduce_small instantiates T == U == bfloat: Sum::init, then one
        // rounded bf16 add per element in ascending slot order.
        bfloat total = bfloat(0.0f);
        #pragma clang loop unroll(full)
        for (int j = 0; j < TTOPK; ++j) total = picked[j] + total;
        #pragma clang loop unroll(full)
        for (int j = 0; j < TTOPK; ++j) {
            route_scores[gid * TTOPK + j] = picked[j] / total;
        }
        // unary_ops.h Sigmoid at T = bfloat16_t, exactly as spelled there.
        const bfloat sx = shared_logits[gid];
        auto sigmoid_y = 1 / (1 + metal::exp(metal::abs(sx)));
        shared_factor[gid] = sx < bfloat(0.0f)
            ? bfloat(sigmoid_y)
            : bfloat(1 - sigmoid_y);
    }
