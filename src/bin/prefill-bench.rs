//! Long-prefill A/B, separated from decode and cached-prefix latency.
use anyhow::{Result, ensure};
use clap::Parser;
use rust_mlx::{
    environment::BenchmarkEnvironment,
    hybrid::{HybridAttention, HybridModel},
    speculative::{self, Options},
    weights::Weights,
};
use serde_json::json;
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    prompt_ids: PathBuf,
    #[arg(long, default_value_t = 4)]
    runs: usize,
    #[arg(long, default_value = "results/prefill-ab.json")]
    output: PathBuf,
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(a.runs > 0, "empty benchmark");
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&a.prompt_ids)?)?;
    let ids: Vec<u32> = serde_json::from_value(if v.is_array() { v } else { v["prompt"].clone() })?;
    let environment = BenchmarkEnvironment::capture()?;
    mlx_rs::memory::set_memory_limit(100 * 1024usize.pow(3))?;
    mlx_rs::memory::set_cache_limit(512 * 1024usize.pow(2))?;
    let w = Weights::load(&a.model)?;
    let m = HybridModel::load(&w, &a.model)?;
    mlx_rs::transforms::eval(w.tensors.values())?;
    let options = Options {
        max_tokens: 32,
        depth: 3,
        chunk: 128,
        eos: &[],
    };
    let mut reports = Vec::new();
    let mut expected = None;
    for cycle in 0..=a.runs {
        for index in 0..4 {
            let mode = if cycle % 2 == 0 { index } else { 3 - index };
            let ple = mode & 1 != 0;
            let packed = mode & 2 != 0;
            for p in m.ple.iter().flatten() {
                p.table.set_batch(ple);
            }
            for l in &m.layers {
                if let HybridAttention::Linear(g) = &l.attention {
                    g.packed_mode.set(packed);
                }
            }
            let prepared = speculative::prepare(&m, &ids, options.chunk)?;
            // Scope candidates to prefill; decode uses the same native reference.
            for p in m.ple.iter().flatten() {
                p.table.set_batch(false);
            }
            for l in &m.layers {
                if let HybridAttention::Linear(g) = &l.attention {
                    g.packed_mode.set(false);
                }
            }
            let g = speculative::generate_plain_prepared(&prepared, &options, |_| Ok(()))?;
            if let Some(e) = &expected {
                ensure!(&g.tokens == e, "prefill trajectory drift");
            } else {
                expected = Some(g.tokens.clone());
            }
            eprintln!(
                "cycle={cycle} ple={ple} packed={packed} prefill={:.3}s",
                prepared.prefill_seconds
            );
            if cycle > 0 {
                reports.push(json!({"cycle":cycle,"batched_ple":ple,"packed_gdn":packed,"prefill_seconds":prepared.prefill_seconds,"prefill_tokens_per_second":ids.len() as f64/prepared.prefill_seconds,"generation":g,"exact":true}));
                std::fs::write(
                    &a.output,
                    serde_json::to_vec_pretty(
                        &json!({"environment":environment,"model":a.model,"quantization":w.config["quantization"],"prompt_ids":ids,"runtime":{"chunk":128,"warmup":"full prompt plus32 output tokens for each of four modes","greedy":true,"mtp":false,"batch":1,"prefix_cache":false,"timing":"prepare target cache/logits/wide hidden, synchronized; decode separately"},"records":reports}),
                    )?,
                )?;
            }
        }
    }
    println!("PREFILL_AB_PASSED");
    Ok(())
}
