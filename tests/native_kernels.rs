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
