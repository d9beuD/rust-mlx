//! Captured real MoE/HC epilogue parity and warmed alternating component timings.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use mlx_rs::{Array, Dtype, ops::indexing::IndexOp};
use std::{path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    fixture: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 100)]
    repetitions: usize,
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(!a.output.exists() && a.repetitions > 0, "preserve output");
    let tensors = Array::load_safetensors(a.fixture.join("activations.safetensors"))?;
    let mut records = Vec::new();
    for layer in [0, 23, 47] {
        for rows in [1, 2, 3, 4, 8] {
            let read = |key: &str| -> Result<Array> {
                Ok(tensors
                    .get(&format!("layer{layer}.{key}"))
                    .context("fixture missing")?
                    .index((.., ..rows, ..))
                    .contiguous()?)
            };
            let routed = read("down")?;
            let shared = read("shared")?;
            let factor = read("factor")?;
            let residual = read("residual")?;
            let gate = read("injection")?;
            mlx_rs::transforms::eval([&routed, &shared, &factor, &residual, &gate])?;
            let run = |candidate: bool| -> Result<Array> {
                rust_mlx::moe_epilogue::set_enabled(candidate);
                if candidate {
                    return rust_mlx::moe_epilogue::apply(
                        &routed, &shared, &factor, &residual, &gate, 4,
                    )?
                    .context("candidate declined");
                }
                let branch = routed.add(shared.multiply(&factor)?)?;
                Ok(residual
                    .reshape(&[1, rows, 4, 2560])?
                    .add(branch.expand_dims(2)?.multiply(gate.expand_dims(-1)?)?)?
                    .reshape(&[1, rows, 10240])?)
            };
            let native = run(false)?.as_dtype(Dtype::Float32)?.contiguous()?;
            let actual = run(true)?.as_dtype(Dtype::Float32)?.contiguous()?;
            mlx_rs::transforms::eval([&native, &actual])?;
            ensure!(
                native.as_slice::<f32>() == actual.as_slice::<f32>(),
                "epilogue mismatch {layer}/{rows}"
            );
            for _ in 0..10 {
                run(false)?.eval()?;
                run(true)?.eval()?;
            }
            let mut samples = Vec::new();
            for repeat in 0..a.repetitions {
                let mut pair = [0.; 2];
                for candidate in if repeat % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let start = Instant::now();
                    run(candidate)?.eval()?;
                    pair[usize::from(candidate)] = start.elapsed().as_secs_f64();
                }
                samples.push(
                    serde_json::json!({"native_seconds":pair[0],"candidate_seconds":pair[1]}),
                );
            }
            records.push(serde_json::json!({"layer":layer,"rows":rows,"dtype":"BF16","exact":true,"pairs":samples}));
        }
    }
    std::fs::write(
        a.output,
        serde_json::to_vec_pretty(
            &serde_json::json!({"complete":true,"fixture_sha256":rust_mlx::resident_quant::sha256_file(&a.fixture.join("activations.safetensors"))?,"environment":rust_mlx::environment::BenchmarkEnvironment::capture()?,"records":records}),
        )?,
    )?;
    println!("EPILOGUE_COMPONENT_PARITY_AND_TIMING_COMPLETED");
    Ok(())
}
