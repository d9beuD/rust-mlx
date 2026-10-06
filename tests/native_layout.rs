use anyhow::{Result, ensure};
use mlx_rs::{
    Array, Dtype,
    ops::{self, indexing::IndexOp},
};
use rust_mlx::{
    hybrid::{HybridConfig, MoE},
    moe_layout,
    weights::{Linear, Weights},
};

fn gather(l: &Linear, x: &Array, ids: &Array) -> Result<Array> {
    let q = l.quant.as_ref().unwrap();
    Ok(ops::gather_qmm(
        x,
        &l.weight,
        l.scales.as_ref().unwrap(),
        l.biases.as_ref(),
        None,
        ids,
        true,
        q.group_size,
        q.bits,
        false,
    )?)
}
fn exact(a: &Array, b: &Array) -> Result<()> {
    let a = a.as_dtype(Dtype::Float32)?.contiguous()?;
    let b = b.as_dtype(Dtype::Float32)?.contiguous()?;
    mlx_rs::transforms::eval([&a, &b])?;
    ensure!(
        a.shape() == b.shape() && a.as_slice::<f32>() == b.as_slice::<f32>(),
        "native packed projection differs"
    );
    Ok(())
}
#[test]
fn native_packed_gate_up_preserves_independent_projections_and_moe_outputs() -> Result<()> {
    let w = Weights::load(std::path::Path::new("tests/fixtures/hybrid"))?;
    let c: HybridConfig = serde_json::from_value(w.config["text_config"].clone())?;
    let mut m = MoE::load(&w, "language_model.model.layers.0.mlp", &c)?;
    assert!(!moe_layout::enabled());
    moe_layout::prepare(&mut m)?;
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        for t in [1, 2, 3, 4, 8] {
            let x = Array::from_iter(
                (0..t * c.hidden_size).map(|i| (i as f32 * 0.011).sin()),
                &[1, t, c.hidden_size],
            )
            .as_dtype(dtype)?;
            let ids = ops::argpartition_axis(&m.router.forward(&x)?, -m.top_k, -1)?.index((
                ..,
                ..,
                -m.top_k..,
            ));
            let xe = x.expand_dims(-2)?.expand_dims(-2)?;
            let gate = gather(&m.gate, &xe, &ids)?;
            let up = gather(&m.up, &xe, &ids)?;
            let pair = gather(m.gate_up.as_ref().unwrap(), &xe, &ids)?;
            let n = gate.shape()[4];
            exact(&gate, &pair.index((.., .., .., .., ..n)))?;
            exact(&up, &pair.index((.., .., .., .., n..)))?;
            moe_layout::set_enabled(false);
            let reference = m.forward(&x)?;
            let before = moe_layout::calls();
            moe_layout::set_enabled(true);
            let candidate = m.forward(&x)?;
            assert!(moe_layout::calls() > before);
            exact(&reference, &candidate)?;
            moe_layout::set_enabled(false);
        }
    }
    Ok(())
}
