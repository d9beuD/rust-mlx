//! Equal-offset end-to-end decode throughput with fresh independent caches.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use mlx_rs::{
    Array,
    ops::indexing::{self, IndexOp},
};
use rust_mlx::{environment::BenchmarkEnvironment, hybrid::HybridModel, weights::Weights};
use serde_json::json;
use std::{path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,
    #[arg(long, default_value_t = 3)]
    runs: usize,
    #[arg(long, default_value = "results/batch-bench.json")]
    output: PathBuf,
    #[arg(long)]
    ab_qmv: bool,
    #[arg(long, conflicts_with_all = ["ab_qmv", "ab_hc"])]
    ab_stream_x: bool,
    #[arg(long, conflicts_with = "ab_qmv")]
    ab_hc: bool,
    #[arg(long, conflicts_with_all = ["ab_qmv", "ab_hc", "ab_stream_x"])]
    ab_gemv: bool,
    #[arg(long)]
    expected: Option<PathBuf>,
}
fn greedy(x: &Array) -> Result<u32> {
    Ok(indexing::argmax(x, false)?.item_exact::<u32>())
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(a.max_tokens > 1 && a.runs > 0, "invalid benchmark limits");
    let environment = BenchmarkEnvironment::capture()?;
    mlx_rs::memory::set_memory_limit(100 * 1024usize.pow(3))?;
    mlx_rs::memory::set_cache_limit(512 * 1024usize.pow(2))?;
    let w = Weights::load(&a.model)?;
    let m = HybridModel::load(&w, &a.model)?;
    mlx_rs::transforms::eval(w.tensors.values())?;
    let mut reports = Vec::new();
    let reference: Option<serde_json::Value> = a
        .expected
        .map(|p| -> Result<_> { Ok(serde_json::from_slice(&std::fs::read(p)?)?) })
        .transpose()?;
    for batch in [2, 4, 8] {
        let prompts = (0..batch)
            .map(|row| {
                vec![
                    7734,
                    264,
                    2716,
                    32671,
                    709,
                    421,
                    55288,
                    76938,
                    4947,
                    13 + row,
                ]
            })
            .collect::<Vec<_>>();
        let mut expected: Option<Vec<Vec<u32>>> = reference
            .as_ref()
            .map(|r| -> Result<_> {
                let record = r["records"]
                    .as_array()
                    .context("missing reference records")?
                    .iter()
                    .find(|r| r["batch"] == batch)
                    .context("missing reference batch")?;
                Ok(serde_json::from_value(record["tokens"].clone())?)
            })
            .transpose()?;
        for cycle in 0..=a.runs {
            for candidate in if cycle % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let batched = a.ab_qmv || a.ab_stream_x || a.ab_hc || a.ab_gemv || candidate;
                if a.ab_qmv {
                    rust_mlx::qmv_kernel::set_batch_enabled(candidate);
                }
                if a.ab_stream_x {
                    rust_mlx::qmv_kernel::set_batch_enabled(true);
                    rust_mlx::qmv_kernel::set_stream_x(candidate);
                }
                if a.ab_hc {
                    rust_mlx::hc_kernel::set_enabled(candidate);
                }
                if a.ab_gemv {
                    rust_mlx::gemv_kernel::set_enabled(candidate);
                }
                let mut caches = Vec::new();
                let gemv_start = rust_mlx::gemv_kernel::launches();
                let mut tokens = Vec::new();
                let started = Instant::now();
                for prompt in &prompts {
                    let mut cache = m.make_cache();
                    let (l, _) = m.forward(prompt, &mut cache)?;
                    tokens.push(vec![greedy(&l.index((0, -1, ..)))?]);
                    caches.push(cache);
                }
                let prefill_seconds = started.elapsed().as_secs_f64();
                let mut latencies = Vec::new();
                let started = Instant::now();
                for step in 1..a.max_tokens {
                    let tick = Instant::now();
                    if batched {
                        let input = tokens
                            .iter()
                            .map(|row| *row.last().unwrap())
                            .collect::<Vec<_>>();
                        let (l, _) = m.decode_batch(&input, &mut caches)?;
                        let next = indexing::argmax_axis(&l, -1, false)?.contiguous()?;
                        next.eval()?;
                        for (row, &token) in tokens.iter_mut().zip(next.as_slice::<u32>()) {
                            row.push(token);
                        }
                    } else {
                        for (row, cache) in tokens.iter_mut().zip(&mut caches) {
                            let input = *row.last().unwrap();
                            let (l, _) = m.forward(&[input], cache)?;
                            row.push(greedy(&l.index((0, -1, ..)))?);
                        }
                    }
                    latencies.push(tick.elapsed().as_secs_f64());
                    if step % 64 == 0 {
                        eprintln!("batch={batch} cycle={cycle} batched={batched} step={step}");
                    }
                }
                let decode_seconds = started.elapsed().as_secs_f64();
                let gemv_launches = rust_mlx::gemv_kernel::launches().wrapping_sub(gemv_start);
                if a.ab_gemv && candidate {
                    ensure!(gemv_launches > 0, "GEMV candidate was not engaged");
                }
                let aggregate_tokens_per_second =
                    batch as f64 * (a.max_tokens - 1) as f64 / decode_seconds;
                ensure!(
                    tokens.iter().all(|row| row.len() == a.max_tokens),
                    "batch output length"
                );
                if let Some(e) = &expected {
                    ensure!(&tokens == e, "batched/serial trajectory drift");
                } else {
                    expected = Some(tokens.clone());
                }
                eprintln!(
                    "batch={batch} cycle={cycle} batched={batched} candidate={candidate} aggregate_tps={aggregate_tokens_per_second:.2}"
                );
                if cycle > 0 {
                    reports.push(json!({"batch":batch,"cycle":cycle,"batched":batched,"qmv_batch":rust_mlx::qmv_kernel::batch_enabled(),"qmv_stream_x":rust_mlx::qmv_kernel::stream_x(),"hc_projection":rust_mlx::hc_kernel::enabled(),"shared_gemv":rust_mlx::gemv_kernel::enabled(),"gemv_narrow":rust_mlx::gemv_kernel::narrow(),"gemv_launches":gemv_launches,"candidate":candidate,"prompt_ids":prompts,"tokens":tokens,"prefill_seconds":prefill_seconds,"decode_seconds":decode_seconds,"aggregate_tokens_per_second":aggregate_tokens_per_second,"per_conversation_tokens_per_second":aggregate_tokens_per_second/batch as f64,"round_seconds":latencies,"exact":true,"peak_memory_bytes":mlx_rs::memory::peak_memory()?}));
                    std::fs::write(
                        &a.output,
                        serde_json::to_vec_pretty(
                            &json!({"environment":environment,"model":a.model,"quantization":w.config["quantization"],"runtime":{"greedy":true,"ignore_eos":true,"mtp":false,"ab_qmv":a.ab_qmv,"ab_stream_x":a.ab_stream_x,"ab_hc":a.ab_hc,"ab_gemv":a.ab_gemv,"prefix_cache":false,"caches":"fresh per mode/run, independent per row","warmup_tokens_per_row":a.max_tokens,"output_tokens_per_row":a.max_tokens,"timing":"aggregate B*(generated-1)/decode, graph+argmax+eval, no unused final forward; prefill separate"},"records":reports}),
                        )?,
                    )?;
                }
            }
        }
    }
    println!("BATCH_BENCH_PASSED");
    Ok(())
}
