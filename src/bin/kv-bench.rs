//! Actual KV geometry, preserved transactional snapshots, full update/eval timing.
use anyhow::{Result, ensure};
use clap::Parser;
use mlx_rs::{Array, Dtype, ops};
use rust_mlx::{dense::KvCache, environment::BenchmarkEnvironment, kv_blocks};
use std::{path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    output: PathBuf,
}
fn main() -> Result<()> {
    let a = Args::parse();
    let environment = BenchmarkEnvironment::capture()?;
    let mut records = Vec::new();
    for context in [256, 2048, 4096, 8192] {
        for t in [1, 4] {
            let initial = Array::from_iter(
                (0..2 * context * 256).map(|i| (i as f32 * 0.001).sin()),
                &[1, 2, context, 256],
            )
            .as_dtype(Dtype::Bfloat16)?;
            let x = ops::ones_dtype(&[1, 2, t, 256], Dtype::Bfloat16)?;
            mlx_rs::transforms::eval([&initial, &x])?;
            let mut bases = Vec::new();
            for mode in [false, true] {
                kv_blocks::set_enabled(mode);
                let mut c = KvCache::default();
                let (k, v) = c.update(initial.clone(), initial.clone())?;
                mlx_rs::transforms::eval([&k, &v])?;
                bases.push(c);
            }
            let run = |mode: bool| -> Result<Array> {
                kv_blocks::set_enabled(mode);
                let mut c = bases[mode as usize].clone();
                let (k, v) = c.update(x.clone(), x.clone())?;
                mlx_rs::transforms::eval([&k, &v])?;
                Ok(k)
            };
            let native = run(false)?.as_dtype(Dtype::Float32)?.contiguous()?;
            let candidate = run(true)?.as_dtype(Dtype::Float32)?.contiguous()?;
            mlx_rs::transforms::eval([&native, &candidate])?;
            ensure!(
                native.as_slice::<f32>() == candidate.as_slice::<f32>(),
                "KV update differs"
            );
            let mut times = [Vec::new(), Vec::new()];
            for cycle in 0..110 {
                for mode in if cycle % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let start = Instant::now();
                    run(mode)?;
                    if cycle >= 10 {
                        times[mode as usize].push(start.elapsed().as_secs_f64());
                    }
                }
            }
            records.push(serde_json::json!({"context":context,"positions":t,"shape":[1,2,context,256],"dtype":"BF16","exact":true,"native_seconds":times[0],"candidate_seconds":times[1]}));
        }
    }
    std::fs::write(
        a.output,
        serde_json::to_vec_pretty(
            &serde_json::json!({"environment":environment,"model_geometry":"Qwen3.8-Flash-Next-oQ4e-mtp full attention KV:2 heads x256 BF16","block":kv_blocks::BLOCK,"cache":"immutable original snapshot retained; cloned per update; no prefix reuse timing","timing":"complete graph/update/eval,10 alternating warmups and100 pairs per shape","benchmarks":records}),
        )?,
    )?;
    println!("KV_BLOCK_COMPONENTS_PASSED");
    Ok(())
}
