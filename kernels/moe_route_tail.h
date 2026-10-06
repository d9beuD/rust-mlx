// Adapted from MTPLX by Youssof Altoukhi, Apache-2.0, revision9882703f3105363ddc37eca9f97aa09a1d387112.
// mtplx/kernels/qwen4_m4_route.py; see third-party/MTPLX-{APACHE-2.0,NOTICE}.
// Modification: native router/shared projections remain separate; tail accepts1–8 independent rows.

    #include <metal_simdgroup>
    #include <metal_stdlib>
    using namespace metal;

    // Order-preserving map from a bfloat to a uint16 so that
    //   key(a) < key(b)  <=>  a < b
    // for every non-NaN bfloat.  Softmax outputs are finite and >= 0, so the
    // sign branch is only exercised by the tests' constructed inputs.
    inline uint route_sort_key(bfloat v, uint idx) {
        const ushort u = as_type<ushort>(v);
        const ushort k = (u & 0x8000) ? (ushort)(~u) : (ushort)(u | 0x8000);
        return ((uint)k << 16) | idx;
    }
