//! Exact greedy head on the checkpoint's actual mixed-quantized vocabulary matrix.
use anyhow::{Result, ensure};
use clap::Parser;
use mlx_rs::{Array, Dtype};
use rust_mlx::{environment::BenchmarkEnvironment, greedy_head, weights::Weights};
use std::{path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    output: PathBuf,
}
fn main() -> Result<()> {
    let a = Args::parse();
    let environment = BenchmarkEnvironment::capture()?;
    let w = Weights::load(&a.model)?;
    let head = w.linear("language_model.lm_head")?;
    let mut records = Vec::new();
    for t in [1, 2, 4] {
        let x = Array::from_iter(
            (0..t * 2560).map(|j| (j as f32 * 0.017).cos()),
            &[1, t, 2560],
        )
        .as_dtype(Dtype::Bfloat16)?;
        x.eval()?;
        let run = |candidate| -> Result<Array> {
            greedy_head::set_enabled(candidate);
            Ok(greedy_head::greedy(&head, &x)?.contiguous()?)
        };
        let native = run(false)?;
        let candidate = run(true)?;
        mlx_rs::transforms::eval([&native, &candidate])?;
        ensure!(
            native.as_slice::<u32>() == candidate.as_slice::<u32>(),
            "head IDs differ"
        );
        let mut times = [Vec::new(), Vec::new()];
        for cycle in 0..110 {
            for mode in if cycle % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let start = Instant::now();
                run(mode)?.eval()?;
                if cycle >= 10 {
                    times[mode as usize].push(start.elapsed().as_secs_f64());
                }
            }
        }
        records.push(serde_json::json!({"positions":t,"shape":x.shape(),"dtype":"BF16","exact":true,"native_seconds":times[0],"candidate_seconds":times[1],"ids":native.as_slice::<u32>()}));
    }
    let report = serde_json::json!({"environment":environment,"model":a.model,"quantization":w.config["quantization"],"module":"language_model.lm_head","sampler":"greedy","reference":"current selected QMV full logits plus native argmax; native singleton atT1","cache":"component weights warm; no prefix cache","batch_size":1,"input":"cos(j*0.017) BF16, actual checkpoint weights","timing":"full head plus greedy reduction, synchronized eval;10 alternating warmups then100 pairs","benchmarks":records});
    std::fs::write(a.output, serde_json::to_vec_pretty(&report)?)?;
    println!("GREEDY_HEAD_COMPONENTS_PASSED");
    Ok(())
}
