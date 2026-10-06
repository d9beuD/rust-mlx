//! Actual BF16 lossless gate/up concatenation qualification and synchronized microbenchmark.
use anyhow::{Result, ensure};
use clap::Parser;
use mlx_rs::{Array, Dtype, ops::indexing::IndexOp};
use rust_mlx::{
    environment::BenchmarkEnvironment,
    hybrid::{HybridConfig, MoE},
    verification,
    weights::Weights,
};
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "results/native-layout-components.json")]
    output: PathBuf,
}
fn main() -> Result<()> {
    let a = Args::parse();
    let environment = BenchmarkEnvironment::capture()?;
    let w = Weights::load(&a.model)?;
    let c: HybridConfig = serde_json::from_value(w.config["text_config"].clone())?;
    let mut m = MoE::load(&w, "language_model.model.layers.0.mlp", &c)?;
    rust_mlx::moe_layout::prepare(&mut m)?;
    let oracle = Array::load_safetensors("results/target-layer0-oracle.safetensors")?;
    let mut records = Vec::new();
    for t in [1, 2, 3, 4, 8] {
        let x = oracle["mlp_input"].index((.., ..t, ..)).contiguous()?;
        ensure!(x.dtype() == Dtype::Bfloat16, "actual BF16 shape required");
        x.eval()?;
        let run = |sorted| {
            rust_mlx::moe_layout::set_enabled(sorted);
            verification::with_rows(|| m.forward(&x))
        };
        let r = run(false)?.as_dtype(Dtype::Float32)?.contiguous()?;
        let s = run(true)?.as_dtype(Dtype::Float32)?.contiguous()?;
        mlx_rs::transforms::eval([&r, &s])?;
        ensure!(
            r.as_slice::<f32>() == s.as_slice::<f32>(),
            "native packed gate/up MoE differs"
        );
        let mut times = [Vec::new(), Vec::new()];
        for cycle in 0..110 {
            for sorted in if cycle % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let started = Instant::now();
                run(sorted)?.eval()?;
                if cycle >= 10 {
                    times[sorted as usize].push(started.elapsed().as_secs_f64());
                }
            }
        }
        records.push(serde_json::json!({"positions":t,"shape":x.shape(),"dtype":"BF16","exact":true,"native_seconds":times[0],"packed_seconds":times[1]}));
    }
    let report = serde_json::json!({"environment":environment,"model":a.model,"quantization":w.config["quantization"],"module":"language_model.model.layers.0.mlp","sampler":"not applicable, fixed component input","cache":"component weights warm; no prefix cache","batch_size":1,"input":"synthetic sin(j*0.013) BF16 layer0 stages, actual checkpoint weights; results/target-layer0-oracle.safetensors:mlp_input","timing":"full router/gather/gate/up/SwiGLU/down/shared-expert graph and synchronized eval;10 alternating warmups then100 pairs","benchmarks":records});
    std::fs::write(a.output, serde_json::to_vec_pretty(&report)?)?;
    println!("NATIVE_LAYOUT_COMPONENTS_PASSED");
    Ok(())
}
