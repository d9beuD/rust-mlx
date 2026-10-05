// Ported from mlx-vlm 0.7.6 models/fast_ops.py, MIT; see third-party/mlx-vlm-MIT.

uint n_tile = threadgroup_position_in_grid.y;
uint route = threadgroup_position_in_grid.z;
uint simd_gid = simdgroup_index_in_threadgroup;
uint simd_lid = thread_index_in_simdgroup;

int out_row = int(n_tile) * ROWS_PER_TG +
    int(simd_gid) * RESULTS_PER_SIMDGROUP;
int token = int(route) / TOP_K;
int batch = token / VERIFY_T;
int time = token - batch * VERIFY_T;
int batch_route = batch * VERIFY_T * TOP_K;
int expert = int(indices[route]);
int paired_route = -1;
if ((time & 1) == 0 && time + 1 < VERIFY_T) {
  int next_route = batch_route + (time + 1) * TOP_K;
  for (int other = 0; other < TOP_K; ++other) {
    if (int(indices[next_route + other]) == expert) {
      paired_route = next_route + other;
      break;
    }
  }
} else if ((time & 1) != 0) {
  int previous_route = batch_route + (time - 1) * TOP_K;
  for (int other = 0; other < TOP_K; ++other) {
    if (int(indices[previous_route + other]) == expert) {
      return;
    }
  }
}
int pair_count = paired_route >= 0 ? 2 : 1;
constexpr int W_ROW_BYTES = K_SIZE * 4 / 8;
constexpr int W_EXPERT_BYTES = N_SIZE * W_ROW_BYTES;
constexpr int GROUPS = K_SIZE / GROUP_SIZE;
constexpr int S_EXPERT_SIZE = N_SIZE * GROUPS;

const device uint8_t* up_ws = (const device uint8_t*)up_w +
    expert * W_EXPERT_BYTES + out_row * W_ROW_BYTES +
    int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
const device T* up_sc = up_scales + expert * S_EXPERT_SIZE +
    out_row * GROUPS + int(simd_lid) / SCALE_STEP_PER_THREAD;
const device T* up_bs = up_biases + expert * S_EXPERT_SIZE +
    out_row * GROUPS + int(simd_lid) / SCALE_STEP_PER_THREAD;
const device uint8_t* gate_ws = (const device uint8_t*)gate_w +
    expert * W_EXPERT_BYTES + out_row * W_ROW_BYTES +
    int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
const device T* gate_sc = gate_scales + expert * S_EXPERT_SIZE +
    out_row * GROUPS + int(simd_lid) / SCALE_STEP_PER_THREAD;
const device T* gate_bs = gate_biases + expert * S_EXPERT_SIZE +
    out_row * GROUPS + int(simd_lid) / SCALE_STEP_PER_THREAD;
const device T* xk[2];
xk[0] = x + token * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;
int paired_token = paired_route >= 0 ? paired_route / TOP_K : token;
xk[1] = x + paired_token * K_SIZE +
    int(simd_lid) * VALUES_PER_THREAD;

float up_result[2][RESULTS_PER_SIMDGROUP] = {0.0f};
float gate_result[2][RESULTS_PER_SIMDGROUP] = {0.0f};
float x_thread[2][VALUES_PER_THREAD];

for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
  float sums[2];
  for (int pair = 0; pair < pair_count; ++pair) {
    sums[pair] = load_affine4_vector_exact<T>(xk[pair], x_thread[pair]);
  }
  for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
    const device uint8_t* uw = up_ws + row * W_ROW_BYTES;
    const device T* us = up_sc + row * GROUPS;
    const device T* ub = up_bs + row * GROUPS;
    const device uint8_t* gw = gate_ws + row * W_ROW_BYTES;
    const device T* gs = gate_sc + row * GROUPS;
    const device T* gb = gate_bs + row * GROUPS;
    for (int pair = 0; pair < pair_count; ++pair) {
      up_result[pair][row] += affine4_qdot_exact(
          uw, x_thread[pair], float(us[0]), float(ub[0]), sums[pair]);
      gate_result[pair][row] += affine4_qdot_exact(
          gw, x_thread[pair], float(gs[0]), float(gb[0]), sums[pair]);
    }
  }
  up_ws += BLOCK_SIZE * 4 / 8;
  up_sc += BLOCK_SIZE / GROUP_SIZE;
  up_bs += BLOCK_SIZE / GROUP_SIZE;
  gate_ws += BLOCK_SIZE * 4 / 8;
  gate_sc += BLOCK_SIZE / GROUP_SIZE;
  gate_bs += BLOCK_SIZE / GROUP_SIZE;
  xk[0] += BLOCK_SIZE;
  xk[1] += BLOCK_SIZE;
}

for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
  int n = out_row + row;
  for (int pair = 0; pair < pair_count; ++pair) {
    float up_value = simd_sum(up_result[pair][row]);
    float gate_value = simd_sum(gate_result[pair][row]);
    if (simd_lid == 0 && n < N_SIZE) {
      int output_route = pair == 0 ? int(route) : paired_route;
      up_y[output_route * N_SIZE + n] = T(up_value);
      gate_y[output_route * N_SIZE + n] = T(gate_value);
    }
  }
}
