//! Actual checkpoint hyperconnection projection qualification and timing.
use anyhow::{Result, ensure};
use clap::Parser;
use mlx_rs::{Array, Dtype, ops::indexing::IndexOp};
use rust_mlx::{
    environment::BenchmarkEnvironment,
    hc_kernel,
    hybrid::{HybridConfig, HyperConnection},
    weights::Weights,
};
use std::{path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    model: PathBuf,
    #[arg(long, default_value = "results/hc-bench.json")]
    output: PathBuf,
}
fn exact(a: &Array, b: &Array) -> Result<()> {
    let a = a.as_dtype(Dtype::Float32)?.contiguous()?;
    let b = b.as_dtype(Dtype::Float32)?.contiguous()?;
    mlx_rs::transforms::eval([&a, &b])?;
    ensure!(
        a.shape() == b.shape() && a.as_slice::<f32>() == b.as_slice::<f32>(),
        "HC projection differs"
    );
    Ok(())
}
fn main() -> Result<()> {
    let a = Args::parse();
    let w = Weights::load(&a.model)?;
    let c: HybridConfig = serde_json::from_value(w.config["text_config"].clone())?;
    let f = Array::load_safetensors("results/target-layer0-oracle.safetensors")?;
    let mut reports = Vec::new();
    for (name, input) in [
        ("attn_hyper_connection", "input"),
        ("mlp_hyper_connection", "after_attention"),
    ] {
        let h = HyperConnection::load(
            &w,
            &format!("language_model.model.layers.0.{name}"),
            &c,
            true,
        )?;
        for t in [1, 4, 8] {
            let x = f[input].index((.., ..t, ..)).contiguous()?;
            x.eval()?;
            let run = |enabled| {
                hc_kernel::set_enabled(enabled);
                rust_mlx::verification::with_rows(|| h.forward_reference(&x))
            };
            let (r, ri) = run(false)?;
            let (v, vi) = run(true)?;
            exact(&r, &v)?;
            exact(&ri.unwrap(), &vi.unwrap())?;
            let mut samples = [Vec::new(), Vec::new()];
            for cycle in 0..110 {
                for candidate in if cycle % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let start = Instant::now();
                    let (y, gate) = run(candidate)?;
                    mlx_rs::transforms::eval([&y, &gate.unwrap()])?;
                    if cycle >= 10 {
                        samples[candidate as usize].push(start.elapsed().as_secs_f64());
                    }
                }
            }
            reports
                .push(serde_json::json!({"module":name,"tokens":t,"exact":true,"seconds":samples}));
        }
    }
    std::fs::write(
        a.output,
        serde_json::to_vec_pretty(
            &serde_json::json!({"environment":BenchmarkEnvironment::capture()?,"model":a.model,"timing":"full native HC normalization/down/activation/up/mix/injection graph + synchronous eval;10 warmups then100 alternating pairs","benchmarks":reports}),
        )?,
    )?;
    println!("HC_BENCH_PASSED");
    Ok(())
}
