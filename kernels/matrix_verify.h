// Experimental affine TensorOps verifier. Independently expressed from the
// Apple MPP public API; native MLX remains the numerical reference.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

inline uint matrix_code(const device uint* w, int n, int k, int K, int bits) {
    const int words = K * bits / 32;
    const int bit = k * bits;
    const int word = bit / 32;
    const int shift = bit % 32;
    uint code = w[size_t(n) * words + word] >> shift;
    // Only access the next packed word when this valid code crosses it.
    if (shift + bits > 32) {
        code |= w[size_t(n) * words + word + 1] << (32 - shift);
    }
    return code & ((1u << bits) - 1u);
}

template <typename T>
inline T matrix_weight(const device uint* w, const device T* scales,
                       const device T* biases, int n, int k,
                       int K, int bits, int group) {
    const uint code = matrix_code(w, n, k, K, bits);
    const size_t metadata = size_t(n) * (K / group) + k / group;
    // This BF16/FP16 weight rounding is intentionally measured against QMV;
    // mathematically equivalent dequantization is not assumed bit-identical.
    return T(float(code) * float(scales[metadata]) + float(biases[metadata]));
}
