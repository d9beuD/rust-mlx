//! Expert-major permutation must restore the canonical per-token reduction order.
use anyhow::{Result, ensure};
use mlx_rs::{Array, Dtype, ops::indexing::IndexOp};
use rust_mlx::{
    hybrid::{HybridConfig, MoE},
    verification,
    weights::Weights,
};

fn compare(m: &MoE, x: &Array) -> Result<()> {
    m.sorted_mode.set(false);
    let reference = verification::with_rows(|| m.forward(x))?;
    m.sorted_mode.set(true);
    let candidate = verification::with_rows(|| m.forward(x))?;
    let r = reference.as_dtype(Dtype::Float32)?.contiguous()?;
    let c = candidate.as_dtype(Dtype::Float32)?.contiguous()?;
    mlx_rs::transforms::eval([&r, &c])?;
    ensure!(
        r.shape() == c.shape() && r.as_slice::<f32>() == c.as_slice::<f32>(),
        "sorted expert output differs"
    );
    Ok(())
}

#[test]
fn sorted_experts_preserve_fixture_outputs() -> Result<()> {
    let w = Weights::load(std::path::Path::new("tests/fixtures/hybrid"))?;
    let c: HybridConfig = serde_json::from_value(w.config["text_config"].clone())?;
    let m = MoE::load(&w, "language_model.model.layers.0.mlp", &c)?;
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        for t in [2, 4, 8] {
            let x = Array::from_iter(
                (0..t * c.hidden_size).map(|i| (i as f32 * 0.011).sin()),
                &[1, t, c.hidden_size],
            )
            .as_dtype(dtype)?;
            compare(&m, &x)?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires actual target component oracle"]
fn sorted_experts_preserve_target_bf16_outputs() -> Result<()> {
    let p = std::env::var("RUST_MLX_TARGET_MODEL")?;
    let w = Weights::load(std::path::Path::new(&p))?;
    let c: HybridConfig = serde_json::from_value(w.config["text_config"].clone())?;
    let m = MoE::load(&w, "language_model.model.layers.0.mlp", &c)?;
    let oracle = Array::load_safetensors("results/target-layer0-oracle.safetensors")?;
    for t in [2, 4, 8] {
        let x = oracle["mlp_input"].index((.., ..t, ..)).contiguous()?;
        ensure!(x.dtype() == Dtype::Bfloat16, "actual BF16 shape required");
        compare(&m, &x)?;
        println!("SORTED_MOE_TARGET_EXACT {t}");
    }
    Ok(())
}
