// Ported from mlx-vlm 0.7.6 models/quantized_verifier.py, MIT; see third-party/mlx-vlm-MIT.

    uint n_tile = threadgroup_position_in_grid.y;
    uint b_idx = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K_SIZE / GS;

    const device uint8_t* ws_base =
        (const device uint8_t*)w + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* scales_base =
        scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* biases_base =
        biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* x_base =
        x + int(b_idx) * VERIFY_T * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;

    float result[VERIFY_T][RESULTS_PER_SIMDGROUP];
    float x_thread[VERIFY_T][VALUES_PER_THREAD];
    for (int t = 0; t < VERIFY_T; ++t) {
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        result[t][row] = 0.0f;
      }
    }

    for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
      // Recompute pointer offsets from the original buffers each iteration.
      // Avoid loop-carried device pointers under shader instrumentation.
      const device uint8_t* ws = ws_base + size_t(k) * BYTES_PER_PACK / PACK_FACTOR;
      const device T* sc = scales_base + size_t(k) / GS;
      const device T* bs = biases_base + size_t(k) / GS;
      const device T* xk = x_base + size_t(k);
      float sums[VERIFY_T];
      for (int t = 0; t < VERIFY_T; ++t) {
        sums[t] = load_vector_exact<T>(xk + t * K_SIZE, x_thread[t]);
      }

#pragma clang loop unroll_count(2)
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device T* sl = sc + row * in_vec_size_g;
        const device T* bl = bs + row * in_vec_size_g;
        float s = float(sl[0]);
        float b = float(bl[0]);
        for (int t = 0; t < VERIFY_T; ++t) {
          result[t][row] += qdot_exact(wl, x_thread[t], s, b, sums[t]);
        }
      }

    }

    threadgroup float local_values[VERIFY_T * NUM_SIMDGROUPS];
    threadgroup uint local_indices[VERIFY_T * NUM_SIMDGROUPS];
    for (int t = 0; t < VERIFY_T; ++t) {
      float best = -INFINITY;
      uint best_id = uint(out_row);
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        float r = simd_sum(result[t][row]);
        // Match the native projection's BF16 rounding before comparing logits.
        float value = float(T(r));
        uint id = uint(out_row + row);
        if (value > best || (value == best && id < best_id)) {
          best = value;
          best_id = id;
        }
      }
      if (simd_lid == 0) {
        local_values[t * NUM_SIMDGROUPS + simd_gid] = best;
        local_indices[t * NUM_SIMDGROUPS + simd_gid] = best_id;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lid == 0 && simd_gid == 0) {
      for (int t = 0; t < VERIFY_T; ++t) {
        float a = local_values[t * NUM_SIMDGROUPS];
        float b = local_values[t * NUM_SIMDGROUPS + 1];
        uint ia = local_indices[t * NUM_SIMDGROUPS];
        uint ib = local_indices[t * NUM_SIMDGROUPS + 1];
        bool choose_b = b > a || (b == a && ib < ia);
        int index = (int(b_idx) * VERIFY_T + t) * (N_SIZE / BN) + int(n_tile);
        maxima[index] = choose_b ? b : a;
        indices[index] = choose_b ? ib : ia;
      }
    }
