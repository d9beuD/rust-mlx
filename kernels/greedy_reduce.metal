// ArgMax comparison/tie convention follows Apple MLX0.32.2, MIT; see NOTICE.
    const uint position = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    float best = -INFINITY;
    uint best_id = 0;
    for (uint j = tid; j < PARTIALS; j += 256) {
      float value = maxima[position * PARTIALS + j];
      uint id = indices[position * PARTIALS + j];
      if (value > best || (value == best && id < best_id)) {
        best = value;
        best_id = id;
      }
    }
    for (uint offset = 16; offset > 0; offset /= 2) {
      float value = simd_shuffle_down(best, offset);
      uint id = simd_shuffle_down(best_id, offset);
      if (value > best || (value == best && id < best_id)) {
        best = value;
        best_id = id;
      }
    }
    threadgroup float local_values[8];
    threadgroup uint local_indices[8];
    if (lane == 0) {
      local_values[simd_group] = best;
      local_indices[simd_group] = best_id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
      best = lane < 8 ? local_values[lane] : -INFINITY;
      best_id = lane < 8 ? local_indices[lane] : 0;
      for (uint offset = 16; offset > 0; offset /= 2) {
        float value = simd_shuffle_down(best, offset);
        uint id = simd_shuffle_down(best_id, offset);
        if (value > best || (value == best && id < best_id)) {
          best = value;
          best_id = id;
        }
      }
      if (lane == 0) { tokens[position] = best_id; }
    }
