// MTPLX by Youssof Altoukhi, Apache-2.0, revision9882703f3105363ddc37eca9f97aa09a1d387112.
// Source mtplx/kernels/qwen4_m4_routed_down.py; third-party/MTPLX-APACHE-2.0 and NOTICE.
// Adaptation: affine g64; preserve scalar/reduction casts; bounded1-8-row dispatch.

    #include <metal_simdgroup>
    #include <metal_stdlib>
    using namespace metal;

    constant constexpr uint ROWS = 4;
    constant constexpr uint TOP_K = 10;
    constant constexpr uint K = 640;
    constant constexpr uint HIDDEN = 2560;
    constant constexpr uint GROUP_SIZE = 64;
    constant constexpr uint VALUES_PER_THREAD = 8;
    constant constexpr uint BLOCK_SIZE = VALUES_PER_THREAD * 32;
    constant constexpr uint OUTPUTS_PER_SIMD = 4;
    constant constexpr uint OUTPUTS_PER_THREADGROUP = 8;
    constant constexpr uint WEIGHT_BYTES_PER_ROW = K / 2;
    constant constexpr uint GROUPS_PER_ROW = K / GROUP_SIZE;
    constant constexpr ushort SLOT_ORDER[TOP_K] = {0, 8, 1, 9, 2, 3, 4, 5, 6, 7};

    inline float load_q4_vector(
        const device bfloat* x,
        thread float* x_thread) {
        float sum = 0.0f;
        for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 16.0f;
            x_thread[i + 2] = x[i + 2] / 256.0f;
            x_thread[i + 3] = x[i + 3] / 4096.0f;
        }
        return sum;
    }

    inline float load_q4_vector_safe(
        const device bfloat* x,
        thread float* x_thread,
        int count) {
        float sum = 0.0f;
        for (int i = 0; i < count; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 16.0f;
            x_thread[i + 2] = x[i + 2] / 256.0f;
            x_thread[i + 3] = x[i + 3] / 4096.0f;
        }
        for (int i = count; i < VALUES_PER_THREAD; ++i) {
            x_thread[i] = 0.0f;
        }
        return sum;
    }


inline float qdot_register(uint codes,const thread float* x_thread,float scale,float bias,float sum,int count) {
    float accum=0.0f;
    for(int i=0;i<count/4;++i) {
        ushort ws=ushort(codes>>(16*i));
        accum += (x_thread[4*i]*(ws&0x000f) + x_thread[4*i+1]*(ws&0x00f0)
            + x_thread[4*i+2]*(ws&0x0f00) + x_thread[4*i+3]*(ws&0xf000));
    }
    return scale*accum+sum*bias;
}
