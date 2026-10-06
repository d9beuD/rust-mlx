"""Independent loop interchange; keep the credited original arithmetic unchanged."""
from pathlib import Path
p=Path('kernels/verify_qmv.metal');s=p.read_text()
s=s.replace('    float x_thread[VERIFY_T][VALUES_PER_THREAD];\n','')
start=s.index('      float sums[VERIFY_T];')
end=s.index('\n      ws +=',start)
replacement='''      for (int t = 0; t < VERIFY_T; ++t) {
        float x_thread[VALUES_PER_THREAD];
        float sum = load_vector_exact<T>(xk + t * K_SIZE, x_thread);
#pragma clang loop unroll_count(2)
        for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
          const device uint8_t* wl = ws + row * in_vec_size_w;
          const device T* sl = sc + row * in_vec_size_g;
          const device T* bl = bs + row * in_vec_size_g;
          float s = float(sl[0]);
          float b = float(bl[0]);
          result[t][row] += qdot_exact(wl, x_thread, s, b, sum);
        }
      }
'''
s=s[:start]+replacement+s[end:]
s='// Experimental loop interchange: one scaled row live at a time.\n'+s
Path('kernels/verify_qmv_streamed.metal').write_text(s)
