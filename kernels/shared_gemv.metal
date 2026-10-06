// Adapted from Apple MLX 0.32.2 GEMVKernel, Copyright 2023-2024 Apple Inc.
// MIT license, third-party/MLX-MIT. Native BM4 BN1 SM1 SN32 TM4 TN4 tree.
// Independent input rows share matrix loads; each accumulator keeps native order.
int lane = int(thread_index_in_simdgroup);
int out_row = int(threadgroup_position_in_grid.x) * (4 * COLS)
            + int(simdgroup_index_in_threadgroup) * COLS;
float result[ROWS][COLS];
for (int r = 0; r < ROWS; ++r) {
  for (int m = 0; m < COLS; ++m) result[r][m] = 0.0f;
}
for (int k = 0; k < K_SIZE; k += 128) {
  float v[ROWS][4];
  for (int r = 0; r < ROWS; ++r) {
#pragma clang loop unroll(full)
    for (int c = 0; c < 4; ++c) v[r][c] = float(x[r * K_SIZE + k + lane * 4 + c]);
  }
#pragma clang loop unroll(full)
  for (int m = 0; m < COLS; ++m) {
    T inter[4];
#pragma clang loop unroll(full)
    for (int c = 0; c < 4; ++c) inter[c] = w[(out_row + m) * K_SIZE + k + lane * 4 + c];
    for (int r = 0; r < ROWS; ++r) {
#pragma clang loop unroll(full)
      for (int c = 0; c < 4; ++c) result[r][m] += inter[c] * v[r][c];
    }
  }
}
for (int r = 0; r < ROWS; ++r) {
#pragma clang loop unroll(full)
  for (int m = 0; m < COLS; ++m) {
#pragma clang loop unroll(full)
    for (ushort delta = 16; delta >= 1; delta >>= 1) {
      result[r][m] += simd_shuffle_down(result[r][m], delta);
    }
    if (lane == 0) y[r * N_SIZE + out_row + m] = T(result[r][m]);
  }
}
