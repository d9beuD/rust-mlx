//! Actual mixed-BF16 routing qualification; native probabilities and expert IDs.
use anyhow::{Result, ensure};
use clap::Parser;
use mlx_rs::{
    Array, Dtype,
    ops::{self, indexing::IndexOp},
};
use rust_mlx::{
    environment::BenchmarkEnvironment,
    hybrid::{HybridConfig, MoE},
    moe_route,
    weights::Weights,
};
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

fn diff(a: &Array, b: &Array) -> Result<f32> {
    let a = a.as_dtype(Dtype::Float32)?.contiguous()?;
    let b = b.as_dtype(Dtype::Float32)?.contiguous()?;
    mlx_rs::transforms::eval([&a, &b])?;
    ensure!(a.shape() == b.shape(), "routing shape mismatch");
    Ok(a.as_slice::<f32>()
        .iter()
        .zip(b.as_slice::<f32>())
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max))
}

fn main() -> Result<()> {
    let a = Args::parse();
    let environment = BenchmarkEnvironment::capture()?;
    let weights = Weights::load(&a.model)?;
    let config: HybridConfig = serde_json::from_value(weights.config["text_config"].clone())?;
    let fixture = Array::load_safetensors("results/target-layer0-oracle.safetensors")?;
    let mut records = Vec::new();
    for layer in 0..48 {
        let m = MoE::load(
            &weights,
            &format!("language_model.model.layers.{layer}.mlp"),
            &config,
        )?;
        for rows in [1, 2, 3, 4, 8] {
            let x = fixture["mlp_input"].index((.., ..rows, ..)).contiguous()?;
            let logits = m.router.forward(&x)?;
            let shared = m.shared_gate.forward_rows(&x)?;
            mlx_rs::transforms::eval([&logits, &shared])?;
            let native = || -> Result<_> {
                let gates = ops::softmax_axis(&logits, -1, true)?;
                let ids = ops::argpartition_axis(&gates, -10, -1)?.index((.., .., -10..));
                let scores = gates.take_along_axis(&ids, -1)?;
                let scores = scores.divide(scores.sum_axis(-1, true)?)?;
                Ok((ids, scores, ops::sigmoid(&shared)?))
            };
            moe_route::set_enabled(true);
            let (ni, ns, nf) = native()?;
            let (ci, cs, cf) =
                moe_route::tail(&logits, &shared, 10)?.expect("actual supported geometry");
            let errors = [diff(&ni, &ci)?, diff(&ns, &cs)?, diff(&nf, &cf)?];
            let mut times = [Vec::new(), Vec::new()];
            if layer == 0 && errors == [0., 0., 0.] {
                for cycle in 0..110 {
                    for candidate in if cycle % 2 == 0 {
                        [false, true]
                    } else {
                        [true, false]
                    } {
                        let start = Instant::now();
                        let (i, s, f) = if candidate {
                            moe_route::tail(&logits, &shared, 10)?.unwrap()
                        } else {
                            native()?
                        };
                        mlx_rs::transforms::eval([&i, &s, &f])?;
                        if cycle >= 10 {
                            times[candidate as usize].push(start.elapsed().as_secs_f64());
                        }
                    }
                }
            }
            records.push(serde_json::json!({"layer":layer,"rows":rows,"logits_shape":logits.shape(),"dtype":"BF16","errors":{"ids":errors[0],"scores":errors[1],"shared_factor":errors[2]},"native_seconds":times[0],"candidate_seconds":times[1]}));
            if errors != [0., 0., 0.] {
                std::fs::write(
                    &a.output,
                    serde_json::to_vec_pretty(
                        &serde_json::json!({"model":a.model,"environment":environment,"records":records,"exact":false,"complete":false,"failure":"MTPLX routing tail differs from native MLX; reject before end-to-end"}),
                    )?,
                )?;
                anyhow::bail!("routing differs at layer{layer} rows{rows}: {errors:?}");
            }
        }
        eprintln!("ROUTE_LAYER_EXACT {layer}");
    }
    std::fs::write(
        a.output,
        serde_json::to_vec_pretty(
            &serde_json::json!({"model":a.model,"environment":environment,"quantization":weights.config["quantization"],"sampler":"component native probabilities/top10","cache":"component evaluated inputs; no prefix reuse","input":"actual layer0 BF16 mlp_input fixture reused across48 actual router banks; not48 natural layer activations","warmup_pairs":10,"measurement_pairs":100,"records":records,"exact":true,"complete":true}),
        )?,
    )?;
    println!("ACTUAL_ROUTE_TAIL_COMPONENTS_EXACT");
    Ok(())
}
