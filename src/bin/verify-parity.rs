use anyhow::{Result, ensure};
use mlx_rs::{
    Array, Dtype,
    ops::indexing::{self, IndexOp},
};
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
    let (prompt, _) = m.forward(
        &[7734, 264, 2716, 32671, 709, 421, 55288, 76938, 4947, 13],
        &mut cache,
    )?;
    let mut tail = prompt.index((0, -1, ..));
    let mut records = Vec::new();
    for depth in [5, 6, 7, 8, 5, 6, 7, 8] {
        let mut batched = cache.clone();
        let mut tokens = Vec::new();
        let mut logits = Vec::new();
        let mut hidden = Vec::new();
        let started = std::time::Instant::now();
        for _ in 0..depth {
            let token = indexing::argmax(&tail, false)?.item_exact::<u32>();
            tokens.push(token);
            let (l, h) = m.forward(&[token], &mut cache)?;
            tail = l.index((0, -1, ..));
            tail.eval()?;
            logits.push(l);
            hidden.push(h);
        }
        let reference = started.elapsed().as_secs_f64();
        let started = std::time::Instant::now();
        let (l, h) = verification::with_mode(|| m.forward(&tokens, &mut batched))?;
        l.eval()?;
        let candidate = started.elapsed().as_secs_f64();
        let rl = mlx_rs::ops::concatenate(&logits, 1)?;
        let rh = mlx_rs::ops::concatenate(&hidden, 1)?;
        let le = error(&l, &rl)?;
        let he = error(&h, &rh)?;
        let mut state = 0f32;
        for (a, b) in cache.layers.iter().zip(&batched.layers) {
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
        println!(
            "depth={depth} logits={le} hidden={he} state={state} reference_ms={} candidate_ms={}",
            reference * 1e3,
            candidate * 1e3
        );
        records.push(serde_json::json!({"depth":depth,"logit_error":le,"hidden_error":he,"state_error":state,"reference_seconds":reference,"candidate_seconds":candidate,"tokens":tokens}));
        std::fs::write(
            "results/verify-parity.json",
            serde_json::to_vec_pretty(&records)?,
        )?;
        ensure!(
            le == 0. && he == 0. && state == 0.,
            "verification is not exact"
        );
    }
    println!("VERIFY_PARITY_PASSED");
    Ok(())
}
