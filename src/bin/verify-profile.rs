//! Synchronized T4 diagnostics using actual MTP proposals, never throughput evidence.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use mlx_rs::{Array, Dtype, ops, ops::indexing::IndexOp};
use rust_mlx::{
    environment::BenchmarkEnvironment,
    hybrid::{HybridAttention, HybridCache, HybridModel, LayerCache},
    verification,
    weights::Weights,
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "results/gpu-draft-raw-ab-256.json")]
    proposals: PathBuf,
    #[arg(long, default_value = "results/verify-profile-round2.json")]
    output: PathBuf,
    #[arg(long, default_value_t = 12)]
    rounds: usize,
}
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
fn hyper(
    name: &str,
    times: &mut BTreeMap<String, Timing>,
    f: impl FnOnce() -> Result<(Array, Option<Array>)>,
) -> Result<(Array, Array)> {
    let start = Instant::now();
    let (mixed, inject) = f()?;
    let graph = start.elapsed().as_secs_f64();
    let inject = inject.context("missing hyper-connection injection")?;
    mlx_rs::transforms::eval([&mixed, &inject])?;
    let t = times.entry(name.into()).or_default();
    t.calls += 1;
    t.graph_seconds += graph;
    t.synchronized_seconds += start.elapsed().as_secs_f64();
    Ok((mixed, inject))
}
fn step(
    m: &HybridModel,
    tokens: &[u32],
    c: &mut HybridCache,
    times: &mut BTreeMap<String, Timing>,
    overlap: &mut Vec<Value>,
    round: usize,
) -> Result<Array> {
    let t = tokens.len() as i32;
    let mut h = phase("embedding", times, || {
        let h = m.embedding.embedding(&Array::from_slice(tokens, &[1, t]))?;
        Ok(ops::broadcast_to(
            &h.expand_dims(2)?,
            &[1, t, m.config.hc_count, m.config.hidden_size],
        )?
        .reshape(&[1, t, m.config.hc_count * m.config.hidden_size])?
        .contiguous()?)
    })?;
    for (i, l) in m.layers.iter().enumerate() {
        if let Some(p) = &m.ple[i] {
            h = phase("ple", times, || {
                p.forward(&h, tokens, &c.history, &mut c.ple[i])
            })?;
        }
        let (mixed, inject) = hyper("attention_hyper", times, || l.attn_hc.forward(&h))?;
        let branch = match (&l.attention, &mut c.layers[i]) {
            (HybridAttention::Linear(a), LayerCache::Linear(c)) => {
                phase("gdn", times, || a.forward(&mixed, c))?
            }
            (HybridAttention::Full(a), LayerCache::Full(c)) => {
                phase("qsa", times, || a.forward(&mixed, c))?
            }
            _ => anyhow::bail!("cache mismatch"),
        };
        h = phase("attention_write", times, || {
            l.attn_hc.write(&h, &branch, &inject)
        })?;
        let (mixed, inject) = hyper("mlp_hyper", times, || l.mlp_hc.forward(&h))?;
        // Observe exact expert selection separately from the timed MoE graph.
        let scores = ops::softmax_axis(&l.moe.router.forward(&mixed)?, -1, true)?;
        let ids = ops::argpartition_axis(&scores, -l.moe.top_k, -1)?
            .index((.., .., -l.moe.top_k..))
            .as_dtype(Dtype::Uint32)?
            .contiguous()?;
        ids.eval()?;
        let all_ids = ids.as_slice::<u32>();
        let unique = all_ids
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        overlap.push(json!({"round":round,"layer":i,"positions":t,"selected":all_ids.len(),"unique_experts":unique.len(),"ids":all_ids,"max_weight_read_reduction_if_perfect_reuse":1.0-unique.len() as f64/all_ids.len() as f64}));
        let branch = phase("moe", times, || l.moe.forward(&mixed))?;
        h = phase("mlp_write", times, || l.mlp_hc.write(&h, &branch, &inject))?;
    }
    c.history.extend_from_slice(tokens);
    let retained = m.config.ngram_size - 1;
    if c.history.len() > retained {
        c.history = c.history[c.history.len() - retained..].to_vec();
    }
    c.offset += t;
    let mixed = phase("mixer", times, || Ok(m.mixer.forward(&h)?.0))?;
    phase("head", times, || m.head.forward(&mixed))
}
fn main() -> Result<()> {
    let a = Args::parse();
    let report: Value = serde_json::from_slice(&std::fs::read(&a.proposals)?)?;
    let g = &report["runs"][0]["generation"];
    let prompt: Vec<u32> = serde_json::from_value(report["prompt_ids"].clone())?;
    let output: Vec<u32> = serde_json::from_value(g["tokens"].clone())?;
    let proposals: Vec<Vec<u32>> = serde_json::from_value(g["draft_tokens"].clone())?;
    let accepted: Vec<usize> = serde_json::from_value(g["acceptance"].clone())?;
    ensure!(
        a.rounds > 0 && a.rounds <= proposals.len(),
        "invalid diagnostic rounds"
    );
    let environment = BenchmarkEnvironment::capture()?;
    mlx_rs::memory::set_memory_limit(100 * 1024usize.pow(3))?;
    mlx_rs::memory::set_cache_limit(512 * 1024usize.pow(2))?;
    let w = Weights::load(&a.model)?;
    let m = HybridModel::load(&w, &a.model)?;
    mlx_rs::transforms::eval(w.tensors.values())?;
    let mut cache = m.make_cache();
    m.forward(&prompt, &mut cache)?.0.eval()?;
    let mut times = BTreeMap::new();
    let mut overlap = Vec::new();
    let mut offset = 0;
    for round in 0..a.rounds {
        let mut tokens = vec![output[offset]];
        tokens.extend_from_slice(&proposals[round]);
        let original = cache.clone();
        let mut reference = original.clone();
        let expected = verification::with_mode(|| m.forward(&tokens, &mut reference))?.0;
        expected.eval()?;
        let actual = verification::with_mode(|| {
            step(&m, &tokens, &mut cache, &mut times, &mut overlap, round)
        })?;
        let x = actual.as_dtype(Dtype::Float32)?.contiguous()?;
        let y = expected.as_dtype(Dtype::Float32)?.contiguous()?;
        mlx_rs::transforms::eval([&x, &y])?;
        ensure!(
            x.as_slice::<f32>() == y.as_slice::<f32>(),
            "diagnostic logits differ"
        );
        cache.commit_verified(&original, &tokens, accepted[round] + 1, &m.config)?;
        offset += accepted[round] + 1;
        eprintln!("VERIFY_PROFILE_ROUND_EXACT {round}");
    }
    let data = json!({"environment":environment,"model":a.model,"quantization":w.config["quantization"],"proposal_source":a.proposals,"prompt_ids":prompt,"rounds":a.rounds,"sampler":"greedy","batch_size":1,"mtp_depth":3,"prefix_cache":false,"warning":"Synchronized diagnostic changes scheduling. Phase timings and maximum theoretical weight reuse are not raw throughput gains; native MLX may already reuse cached weights.","logits_exact":true,"phases":times,"expert_overlap":overlap});
    std::fs::write(a.output, serde_json::to_vec_pretty(&data)?)?;
    println!("VERIFY_PROFILE_PASSED");
    Ok(())
}
