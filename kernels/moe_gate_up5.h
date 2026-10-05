// Ported from mlx-vlm 0.7.6 models/fast_ops.py, MIT; see third-party/mlx-vlm-MIT.

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

constant constexpr int SIMD_SIZE = 32;
constant constexpr int PACK_FACTOR = 8;
constant constexpr int BYTES_PER_PACK = 5;
constant constexpr int PACKS_PER_THREAD = 2;
constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
constant constexpr int SCALE_STEP_PER_THREAD =
    GROUP_SIZE / VALUES_PER_THREAD;
constant constexpr int RESULTS_PER_SIMDGROUP = 4;
constant constexpr int NUM_SIMDGROUPS = 2;
constant constexpr int ROWS_PER_TG =
    RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

template <typename T>
inline float load_affine5_vector_exact(
    const device T* x,
    thread float* x_thread) {
  float sum = 0.0f;
  for (int i = 0; i < VALUES_PER_THREAD; i += 8) {
    sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
        x[i + 6] + x[i + 7];
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 32.0f;
    x_thread[i + 2] = x[i + 2] / 4.0f;
    x_thread[i + 3] = x[i + 3] / 128.0f;
    x_thread[i + 4] = x[i + 4] / 16.0f;
    x_thread[i + 5] = x[i + 5] / 2.0f;
    x_thread[i + 6] = x[i + 6] / 64.0f;
    x_thread[i + 7] = x[i + 7] / 8.0f;
  }
  return sum;
}

inline float affine5_qdot_exact(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  float accum = 0.0f;
  for (int i = 0; i < VALUES_PER_THREAD / 8; ++i) {
    const thread float* xt = x_thread + 8 * i;
    const device uint8_t* wb = w + 5 * i;
    accum += (wb[0] & 0x1f) * xt[0];
    accum += (wb[0] & 0xe0) * xt[1];
    accum += (wb[1] & 0x03) * (xt[1] * 256.0f);
    accum += (wb[1] & 0x7c) * xt[2];
    accum += (wb[1] & 0x80) * xt[3];
    accum += (wb[2] & 0x0f) * (xt[3] * 256.0f);
    accum += (wb[2] & 0xf0) * xt[4];
    accum += (wb[3] & 0x01) * (xt[4] * 256.0f);
    accum += (wb[3] & 0x3e) * xt[5];
    accum += (wb[3] & 0xc0) * xt[6];
    accum += (wb[4] & 0x07) * (xt[6] * 256.0f);
    accum += (wb[4] & 0xf8) * xt[7];
  }
  return scale * accum + sum * bias;
}
