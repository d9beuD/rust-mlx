use mlx_rs::{Array, Dtype};
use rust_mlx::{gdn_kernel, qsa_kernel};
fn fixtures() -> std::collections::HashMap<String, Array> {
    Array::load_safetensors(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/native-kernels.safetensors"
    ))
    .unwrap()
}
fn exact(a: &Array, b: &Array) {
    let a = a.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
    a.eval().unwrap();
    b.eval().unwrap();
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.as_slice::<f32>(), b.as_slice::<f32>());
}
#[test]
fn recurrent_real_bf16_dimensions_and_nonzero_state() {
    let f = fixtures();
    let (y, s) = gdn_kernel::recurrent(
        &f["q"],
        &f["k"],
        &f["v"],
        &f["g"],
        &f["beta"],
        &f["initial"],
    )
    .unwrap();
    exact(&y, &f["y"]);
    exact(&s, &f["state"]);
}
#[test]
fn qsa_prefill_short_context_sentinels_causality_and_tail() {
    let f = fixtures();
    let y = qsa_kernel::attention(
        &f["qsa_q"],
        &f["qsa_k"],
        &f["qsa_v"],
        &f["qsa_blocks"],
        &f["qsa_ends"],
        4,
        0.0625,
    )
    .unwrap();
    exact(&y, &f["qsa_out"]);
}
#[test]
fn invalid_recurrent_shape_is_rejected() {
    let f = fixtures();
    assert!(
        gdn_kernel::recurrent(
            &f["q"],
            &f["k"],
            &f["v"],
            &f["g"],
            &f["beta"],
            &Array::from_f32(0.)
        )
        .is_err()
    );
}

#[test]
fn retained_recurrent_history_matches_final_state() {
    let f = fixtures();
    let (y, s, h) = gdn_kernel::recurrent_with_history(
        &f["q"],
        &f["k"],
        &f["v"],
        &f["g"],
        &f["beta"],
        &f["initial"],
    )
    .unwrap();
    exact(&y, &f["y"]);
    exact(&s, &f["state"]);
    exact(&h.reshape(s.shape()).unwrap(), &s);
}

#[test]
fn packed_real_bf16_matches_native_reduction() {
    let f = fixtures();
    for history in [false, true] {
        let out = gdn_kernel::packed(
            &f["q"],
            &f["k"],
            &f["v"],
            &f["g"],
            &f["beta"],
            &f["initial"],
            history,
        )
        .unwrap();
        exact(&out[0], &f["y"]);
        exact(&out[1], &f["state"]);
        if history {
            exact(&out[2].reshape(out[1].shape()).unwrap(), &out[1]);
        }
    }
}
#[test]
fn packed_matches_native_across_lengths_and_extreme_gates() {
    for t in [1, 4, 17, 129] {
        let q = Array::from_iter(
            (0..t * 2 * 128).map(|i| ((i as f32 * 0.17).sin()) * 0.01),
            &[1, t, 2, 128],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let k = Array::from_iter(
            (0..t * 2 * 128).map(|i| ((i as f32 * 0.11).cos()) * 0.09),
            &[1, t, 2, 128],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let v = Array::from_iter(
            (0..t * 6 * 16).map(|i| (i as f32 * 0.23).sin()),
            &[1, t, 6, 16],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let g = Array::from_iter(
            (0..t * 6).map(|i| [0., 1., 0.9, 1e-8, 0.5, 0.99][i as usize % 6]),
            &[1, t, 6],
        );
        let b = Array::from_iter(
            (0..t * 6).map(|i| [0f32, 1., 0.5, 0.99, 0.01, 0.75][i as usize % 6]),
            &[1, t, 6],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let s = Array::from_iter(
            (0..6 * 16 * 128).map(|i| (i as f32 * 0.013).cos() * 0.02),
            &[1, 6, 16, 128],
        );
        let (y, state, h) = gdn_kernel::recurrent_with_history(&q, &k, &v, &g, &b, &s).unwrap();
        let p = gdn_kernel::packed(&q, &k, &v, &g, &b, &s, true).unwrap();
        exact(&p[0], &y);
        exact(&p[1], &state);
        exact(&p[2], &h);
    }
}
