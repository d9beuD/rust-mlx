//! Synchronized diagnostic attribution. These timers alter scheduling; use infer for speed.
use anyhow::Result;
use mlx_rs::{
    Array,
    ops::{
        self,
        indexing::{self, IndexOp},
    },
};
use rust_mlx::{
    environment::BenchmarkEnvironment,
    hybrid::{HybridAttention, HybridCache, HybridModel, LayerCache},
    weights::Weights,
};
use std::{collections::BTreeMap, time::Instant};
#[derive(Default, serde::Serialize)]
struct Timing {
    calls: usize,
    graph_seconds: f64,
    synchronized_seconds: f64,
}
fn phase<F: FnOnce() -> Result<Array>>(
    name: &str,
    times: &mut BTreeMap<String, Timing>,
    f: F,
) -> Result<Array> {
    let start = Instant::now();
    let y = f()?;
    let graph = start.elapsed().as_secs_f64();
    y.eval()?;
    let t = times.entry(name.into()).or_default();
    t.calls += 1;
    t.graph_seconds += graph;
    t.synchronized_seconds += start.elapsed().as_secs_f64();
    Ok(y)
}
fn step(
    m: &HybridModel,
    token: u32,
    c: &mut HybridCache,
    times: &mut BTreeMap<String, Timing>,
) -> Result<Array> {
    let mut h = phase("embedding", times, || {
        let h = m
            .embedding
            .embedding(&Array::from_slice(&[token], &[1, 1]))?;
        Ok(ops::broadcast_to(&h.expand_dims(2)?, &[1, 1, 4, 2560])?
            .reshape(&[1, 1, 10240])?
            .contiguous()?)
    })?;
    for (i, l) in m.layers.iter().enumerate() {
        if let Some(p) = &m.ple[i] {
            h = phase("ple", times, || {
                p.forward(&h, &[token], &c.history, &mut c.ple[i])
            })?;
        }
        let started = Instant::now();
        let (mixed, inj) = l.attn_hc.forward(&h)?;
        let graph = started.elapsed().as_secs_f64();
        mlx_rs::transforms::eval([&mixed, inj.as_ref().unwrap()])?;
        let t = times.entry("attention_hyper".into()).or_default();
        t.calls += 1;
        t.graph_seconds += graph;
        t.synchronized_seconds += started.elapsed().as_secs_f64();
        let branch = match (&l.attention, &mut c.layers[i]) {
            (HybridAttention::Linear(a), LayerCache::Linear(c)) => {
                phase("gdn", times, || a.forward(&mixed, c))?
            }
            (HybridAttention::Full(a), LayerCache::Full(c)) => {
                phase("full_attention", times, || a.forward(&mixed, c))?
            }
            _ => anyhow::bail!("cache mismatch"),
        };
        h = phase("attention_write", times, || {
            l.attn_hc.write(&h, &branch, inj.as_ref().unwrap())
        })?;
        let started = Instant::now();
        let (mixed, inj) = l.mlp_hc.forward(&h)?;
        let graph = started.elapsed().as_secs_f64();
        mlx_rs::transforms::eval([&mixed, inj.as_ref().unwrap()])?;
        let t = times.entry("mlp_hyper".into()).or_default();
        t.calls += 1;
        t.graph_seconds += graph;
        t.synchronized_seconds += started.elapsed().as_secs_f64();
        let branch = phase("moe", times, || l.moe.forward(&mixed))?;
        h = phase("mlp_write", times, || {
            l.mlp_hc.write(&h, &branch, inj.as_ref().unwrap())
        })?;
    }
    c.history.push(token);
    if c.history.len() > 2 {
        c.history.remove(0);
    }
    c.offset += 1;
    let h = phase("mixer", times, || Ok(m.mixer.forward(&h)?.0))?;
    phase("head", times, || m.head.forward(&h))
}
fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("model path required");
    let path = std::path::Path::new(&path);
    let w = Weights::load(path)?;
    let m = HybridModel::load(&w, path)?;
    mlx_rs::transforms::eval(w.tensors.values())?;
    let prompt = [7734, 264, 2716, 32671, 709, 421, 55288, 76938, 4947, 13];
    let mut c = m.make_cache();
    let (mut logits, _) = m.forward(&prompt, &mut c)?;
    for _ in 0..16 {
        let token = indexing::argmax(logits.index((0, -1, ..)), false)?.item_exact::<u32>();
        (logits, _) = m.forward(&[token], &mut c)?;
        logits.eval()?;
    }
    let mut times = BTreeMap::new();
    for _ in 0..16 {
        let token = indexing::argmax(logits.index((0, -1, ..)), false)?.item_exact::<u32>();
        logits = step(&m, token, &mut c, &mut times)?;
    }
    let result = serde_json::json!({"environment":BenchmarkEnvironment::capture()?,"tokens":16,"warning":"Synchronized per-phase diagnostic changes GPU scheduling. Timings are not natural-runtime percentages.","phases":times});
    std::fs::write(
        "results/target-profile.json",
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
