// Adapted from pinned oMLX hc_projection.py (Apache-2.0), with Apple MLX MIT arithmetic.
// See third-party/omlx-APACHE-2.0, third-party/MLX-MIT and NOTICE.
const uint input_row = threadgroup_position_in_grid.z;
const device T* xrow = x + (size_t)input_row * K;
device T* row_output = combined + (size_t)input_row * 324;
const uint tg = threadgroup_position_in_grid.y;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    constexpr int PF = hc_pack_factor<BITS>();
    constexpr int BP = hc_bytes_per_pack<BITS>();
    constexpr int GROUPS = K / GS;
    constexpr int ROW_BYTES = K * BP / PF;

    if (tg < 40) {
        constexpr int PPT = 2;
        constexpr int VPT = PF * PPT;
        constexpr int BLOCK = VPT * 32;
        constexpr int SCALE_STEP = GS / VPT;
        const int out_row = int(tg) * 8 + int(sg) * 4;
        const device uint8_t* wp = (const device uint8_t*)down_w
            + out_row * ROW_BYTES + int(lane) * PPT * BP;
        const device T* sp = down_s + out_row * GROUPS
            + int(lane) / SCALE_STEP;
        const device T* bp = down_b + out_row * GROUPS
            + int(lane) / SCALE_STEP;
        const device T* xp = xrow + int(lane) * VPT;
        float result[4] = {0.0f};
        float xv[VPT];
        for (int k = 0; k < K; k += BLOCK) {
            float sum = hc_load_vector<T, VPT, BITS>(xp, xv);
            for (int row = 0; row < 4; ++row) {
                result[row] += hc_qdot<VPT, BITS>(
                    wp + row * ROW_BYTES,
                    xv,
                    float(sp[row * GROUPS]),
                    float(bp[row * GROUPS]),
                    sum);
            }
            wp += BLOCK * BP / PF;
            sp += BLOCK / GS;
            bp += BLOCK / GS;
            xp += BLOCK;
        }
        for (int row = 0; row < 4; ++row) {
            result[row] = simd_sum(result[row]);
            if (lane == 0) row_output[out_row + row] = T(result[row]);
        }
        return;
    }

    if (sg != 0) return;
    constexpr int PPT = 1;
    constexpr int VPT = PF;
    constexpr int BLOCK = VPT * 32;
    constexpr int SCALE_STEP = GS / VPT;
    const device uint8_t* wp = (const device uint8_t*)inject_w
        + int(lane) * BP;
    const device T* sp = inject_s + int(lane) / SCALE_STEP;
    const device T* bp = inject_b + int(lane) / SCALE_STEP;
    const device T* xp = xrow + int(lane) * VPT;
    float result[4] = {0.0f};
    float xv[VPT];
    int k = 0;
    for (; k < K - BLOCK; k += BLOCK) {
        float sum = hc_load_vector<T, VPT, BITS>(xp, xv);
        for (int row = 0; row < 4; ++row) {
            result[row] += hc_qdot<VPT, BITS>(
                wp + row * ROW_BYTES,
                xv,
                float(sp[row * GROUPS]),
                float(bp[row * GROUPS]),
                sum);
        }
        wp += BLOCK * BP / PF;
        sp += BLOCK / GS;
        bp += BLOCK / GS;
        xp += BLOCK;
    }
    float sum = hc_load_vector<T, VPT, BITS>(xp, xv);
    for (int row = 0; row < 4; ++row) {
        result[row] += hc_qdot_safe<VPT, BITS>(
            wp + row * ROW_BYTES,
            xv,
            float(sp[row * GROUPS]),
            float(bp[row * GROUPS]),
            sum);
        result[row] = simd_sum(result[row]);
        if (lane == 0) row_output[320 + row] = T(result[row]);
    }
