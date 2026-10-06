//! Native concatenation versus block-backed target logits/hidden/full caches and rollback.
use anyhow::{Result, ensure};
use clap::Parser;
use mlx_rs::{
    Array, Dtype, ops,
    ops::indexing::{self, IndexOp},
};
use rust_mlx::{
    environment::BenchmarkEnvironment,
    hybrid::{HybridCache, HybridModel, LayerCache},
    kv_blocks, verification,
    weights::Weights,
};
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    prompt_ids: Option<PathBuf>,
}
fn error(a: &Array, b: &Array) -> Result<f32> {
    let a = a.as_dtype(Dtype::Float32)?.contiguous()?;
    let b = b.as_dtype(Dtype::Float32)?.contiguous()?;
    mlx_rs::transforms::eval([&a, &b])?;
    ensure!(a.shape() == b.shape(), "shape mismatch");
    ensure!(
        a.as_slice::<f32>()
            .iter()
            .chain(b.as_slice::<f32>())
            .all(|v| v.is_finite()),
        "non-finite parity input"
    );
    Ok(a.as_slice::<f32>()
        .iter()
        .zip(b.as_slice::<f32>())
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max))
}
fn optional(a: &Option<Array>, b: &Option<Array>) -> Result<f32> {
    match (a, b) {
        (Some(a), Some(b)) => error(a, b),
        (None, None) => Ok(0.),
        _ => anyhow::bail!("cache initialization differs"),
    }
}
fn cache_error(a: &HybridCache, b: &HybridCache) -> Result<f32> {
    ensure!(
        a.offset == b.offset && a.history == b.history,
        "CPU cache mismatch"
    );
    let mut max = 0f32;
    for (a, b) in a.layers.iter().zip(&b.layers) {
        match (a, b) {
            (LayerCache::Linear(a), LayerCache::Linear(b)) => {
                max = max
                    .max(optional(&a.state, &b.state)?)
                    .max(optional(&a.conv, &b.conv)?);
            }
            (LayerCache::Full(a), LayerCache::Full(b)) => {
                ensure!(a.kv.offset == b.kv.offset, "KV offset mismatch");
                max = max
                    .max(optional(&a.kv.keys, &b.kv.keys)?)
                    .max(optional(&a.kv.values, &b.kv.values)?)
                    .max(optional(&a.raw_keys, &b.raw_keys)?)
                    .max(optional(&a.blocks, &b.blocks)?);
            }
            _ => anyhow::bail!("cache kind mismatch"),
        }
    }
    for (a, b) in a.ple.iter().zip(&b.ple) {
        max = max.max(optional(&a.conv, &b.conv)?);
    }
    Ok(max)
}
fn main() -> Result<()> {
    let a = Args::parse();
    let environment = BenchmarkEnvironment::capture()?;
    let w = Weights::load(&a.model)?;
    let m = HybridModel::load(&w, &a.model)?;
    let prompt: Vec<u32> = if let Some(p) = a.prompt_ids {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(p)?)?;
        serde_json::from_value(if v.is_array() { v } else { v["prompt"].clone() })?
    } else {
        vec![7734, 264, 2716, 32671, 709, 421, 55288, 76938, 4947, 13]
    };
    let mut native = m.make_cache();
    let mut blocked = m.make_cache();
    let mut next = 0;
    for chunk in prompt.chunks(128) {
        kv_blocks::set_enabled(false);
        let (nl, nh) = m.forward(chunk, &mut native)?;
        nl.eval()?;
        kv_blocks::set_enabled(true);
        let (bl, bh) = m.forward(chunk, &mut blocked)?;
        ensure!(
            error(&nl, &bl)? == 0.
                && error(&nh, &bh)? == 0.
                && cache_error(&native, &blocked)? == 0.,
            "prefill differs"
        );
        next = indexing::argmax(nl.index((0, -1, ..)), false)?.item_exact::<u32>();
    }
    let mut records = Vec::new();
    for step in 0..16 {
        kv_blocks::set_enabled(false);
        let (nl, nh) = m.forward(&[next], &mut native)?;
        nl.eval()?;
        kv_blocks::set_enabled(true);
        let (bl, bh) = m.forward(&[next], &mut blocked)?;
        let le = error(&nl, &bl)?;
        let he = error(&nh, &bh)?;
        let ce = cache_error(&native, &blocked)?;
        ensure!(le == 0. && he == 0. && ce == 0., "decode differs");
        records.push(
            serde_json::json!({"step":step,"logit_error":le,"hidden_error":he,"state_error":ce}),
        );
        next = indexing::argmax(nl.index((0, -1, ..)), false)?.item_exact::<u32>();
    }
    let base_native = native;
    let base_blocked = blocked;
    let mut verifier = Vec::new();
    for depth in 2..=8 {
        let mut n = base_native.clone();
        let mut b = base_blocked.clone();
        let mut tokens = Vec::new();
        let mut logits = Vec::new();
        let mut hidden = Vec::new();
        let mut token = next;
        kv_blocks::set_enabled(false);
        for _ in 0..depth {
            tokens.push(token);
            let (l, h) = m.forward(&[token], &mut n)?;
            l.eval()?;
            token = indexing::argmax(l.index((0, -1, ..)), false)?.item_exact::<u32>();
            logits.push(l);
            hidden.push(h);
        }
        kv_blocks::set_enabled(true);
        let (l, h) = verification::with_mode(|| m.forward(&tokens, &mut b))?;
        let le = error(&l, &ops::concatenate(&logits, 1)?)?;
        let he = error(&h, &ops::concatenate(&hidden, 1)?)?;
        let ce = cache_error(&n, &b)?;
        ensure!(le == 0. && he == 0. && ce == 0., "verifier differs");
        for keep in 0..=depth {
            let mut n = base_native.clone();
            let mut c = b.clone();
            c.commit_verified(&base_blocked, &tokens, keep, &m.config)?;
            kv_blocks::set_enabled(false);
            for &token in &tokens[..keep] {
                m.forward(&[token], &mut n)?.0.eval()?;
            }
            let (nl, nh) = m.forward(&[next], &mut n)?;
            nl.eval()?;
            kv_blocks::set_enabled(true);
            let (cl, ch) = m.forward(&[next], &mut c)?;
            ensure!(
                error(&nl, &cl)? == 0. && error(&nh, &ch)? == 0. && cache_error(&n, &c)? == 0.,
                "rollback continuation differs"
            );
        }
        verifier.push(serde_json::json!({"depth":depth,"logit_error":le,"hidden_error":he,"state_error":ce,"rollback_prefixes":depth+1}));
    }
    kv_blocks::set_enabled(false);
    std::fs::write(
        a.output,
        serde_json::to_vec_pretty(
            &serde_json::json!({"environment":environment,"model":a.model,"quantization":w.config["quantization"],"prompt_ids":prompt,"sampler":"greedy","cache":"native concatenation versus block storage, fresh independent prefill; original snapshots retained","instrumented_timing_excluded":true,"decode":records,"verifier":verifier}),
        )?,
    )?;
    println!("KV_BLOCK_TARGET_LOGITS_HIDDEN_CACHES_ROLLBACK_EXACT");
    Ok(())
}
