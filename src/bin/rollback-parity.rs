use anyhow::{Result, ensure};
use mlx_rs::{Array, Dtype};
use rust_mlx::{
    hybrid::{HybridModel, LayerCache},
    verification,
    weights::Weights,
};
fn error(a: &Array, b: &Array) -> Result<f32> {
    let a = a.as_dtype(Dtype::Float32)?.contiguous()?;
    let b = b.as_dtype(Dtype::Float32)?.contiguous()?;
    mlx_rs::transforms::eval([&a, &b])?;
    ensure!(a.shape() == b.shape(), "shape mismatch");
    Ok(a.as_slice::<f32>()
        .iter()
        .zip(b.as_slice::<f32>())
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max))
}
fn main() -> Result<()> {
    let path =
        std::path::Path::new("/Users/d9beud/.lmstudio/models/d9beuD/Qwen3.8-Flash-Next-oQ4e-mtp");
    let w = Weights::load(path)?;
    let m = HybridModel::load(&w, path)?;
    let mut cache = m.make_cache();
    m.forward(
        &[7734, 264, 2716, 32671, 709, 421, 55288, 76938, 4947, 13],
        &mut cache,
    )?
    .0
    .eval()?;
    let base = cache;
    let tokens = [271, 248068, 198, 760];
    let mut reports = Vec::new();
    for keep in 0..=4 {
        let mut expected = base.clone();
        let mut candidate = base.clone();
        for &t in &tokens[..keep] {
            m.forward(&[t], &mut expected)?.0.eval()?;
        }
        verification::with_mode(|| m.forward(&tokens, &mut candidate))?
            .0
            .eval()?;
        candidate.commit_verified(&base, &tokens, keep, &m.config)?;
        ensure!(
            candidate.offset == expected.offset && candidate.history == expected.history,
            "CPU rollback mismatch"
        );
        let (e, _) = m.forward(&[1156], &mut expected)?;
        let (a, _) = m.forward(&[1156], &mut candidate)?;
        let le = error(&e, &a)?;
        let mut state = 0f32;
        for (a, b) in expected.layers.iter().zip(&candidate.layers) {
            match (a, b) {
                (LayerCache::Linear(a), LayerCache::Linear(b)) => {
                    state = state.max(error(a.state.as_ref().unwrap(), b.state.as_ref().unwrap())?);
                    state = state.max(error(a.conv.as_ref().unwrap(), b.conv.as_ref().unwrap())?);
                }
                (LayerCache::Full(a), LayerCache::Full(b)) => {
                    state = state.max(error(
                        a.kv.keys.as_ref().unwrap(),
                        b.kv.keys.as_ref().unwrap(),
                    )?);
                    state = state.max(error(
                        a.kv.values.as_ref().unwrap(),
                        b.kv.values.as_ref().unwrap(),
                    )?);
                }
                _ => anyhow::bail!("cache types"),
            }
        }
        for (a, b) in expected.ple.iter().zip(&candidate.ple) {
            if let (Some(a), Some(b)) = (&a.conv, &b.conv) {
                state = state.max(error(a, b)?);
            }
        }
        println!("keep={keep} logits={le} state={state}");
        ensure!(le == 0. && state == 0., "rollback is not exact");
        reports.push(serde_json::json!({"keep":keep,"logit_error":le,"state_error":state}));
    }
    std::fs::write(
        "results/rollback-parity.json",
        serde_json::to_vec_pretty(&reports)?,
    )?;
    println!("ROLLBACK_PARITY_PASSED");
    Ok(())
}
