use anyhow::{Result, ensure};
use clap::Parser;
use rust_mlx::{
    environment::BenchmarkEnvironment, hybrid::HybridModel, mtp::Mtp, speculative, weights::Weights,
};
use serde_json::json;
use std::{
    io::{self, Write},
    path::PathBuf,
};
#[derive(Parser)]
struct Args {
    /// Alternate a kernel candidate in the same process.
    #[arg(long,value_parser=["packed","moe"])]
    ab_kernel: Option<String>,
    #[arg(long)]
    model: PathBuf,
    #[arg(
        long,
        default_value = "Write a short Rust function that computes Fibonacci numbers."
    )]
    prompt: String,
    #[arg(long)]
    chat: bool,
    #[arg(long)]
    no_thinking: bool,
    #[arg(long, default_value = "xhigh")]
    reasoning_effort: String,
    #[arg(long)]
    prompt_ids: Option<PathBuf>,
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,
    #[arg(long, default_value_t = 3)]
    draft_depth: usize,
    #[arg(long)]
    sweep_depth: bool,
    #[arg(long, default_value_t = 3)]
    runs: usize,
    #[arg(long, default_value_t = 32)]
    warmup_tokens: usize,
    #[arg(long, default_value_t = 128)]
    prefill_chunk: usize,
    #[arg(long)]
    ignore_eos: bool,
    #[arg(long)]
    stream: bool,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    expected: Option<PathBuf>,
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(a.runs > 0, "runs must be positive");
    let environment = BenchmarkEnvironment::capture()?;
    mlx_rs::memory::set_memory_limit(100 * 1024usize.pow(3))?;
    mlx_rs::memory::set_cache_limit(512 * 1024usize.pow(2))?;
    let w = Weights::load(&a.model)?;
    let m = HybridModel::load(&w, &a.model)?;
    let draft = Mtp::load(&w, &m.config)?;
    mlx_rs::transforms::eval(w.tensors.values())?;
    let t = tokenizers::Tokenizer::from_file(a.model.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let ids = if let Some(p) = a.prompt_ids {
        serde_json::from_slice::<Vec<u32>>(&std::fs::read(p)?)?
    } else {
        rust_mlx::chat::encode_prompt(
            &t,
            &a.model,
            &a.prompt,
            a.chat,
            !a.no_thinking,
            &a.reasoning_effort,
        )?
    };
    let expected: Option<Vec<u32>> = a
        .expected
        .map(|p| -> Result<Vec<u32>> {
            let d: serde_json::Value = serde_json::from_slice(&std::fs::read(p)?)?;
            Ok(serde_json::from_value(d["runs"][0]["tokens"].clone())?)
        })
        .transpose()?;
    let eos = if a.ignore_eos {
        vec![]
    } else {
        vec![248044, 248046]
    };
    let mut records = Vec::new();
    let depths = if a.sweep_depth {
        (1..=7).collect::<Vec<_>>()
    } else {
        vec![a.draft_depth]
    };
    for cycle in 0..=a.runs {
        let mut order = depths.clone();
        if cycle % 2 == 0 {
            order.reverse();
        }
        let modes = if a.ab_kernel.is_some() {
            if cycle % 2 == 0 {
                vec![false, true]
            } else {
                vec![true, false]
            }
        } else {
            vec![false]
        };
        for candidate in modes {
            if let Some(kernel) = &a.ab_kernel {
                for layer in &m.layers {
                    if kernel == "moe" {
                        layer.moe.fused_mode.set(candidate);
                    } else if let rust_mlx::hybrid::HybridAttention::Linear(g) = &layer.attention {
                        g.packed_mode.set(candidate);
                    }
                }
                if kernel == "moe" {
                    draft.mlp.fused_mode.set(candidate);
                }
            }
            for depth in order.clone() {
                let run = cycle;

                let warm = run == 0;
                let mut decoder = t.decode_stream(true);
                let g = speculative::generate(
                    &m,
                    &draft,
                    &ids,
                    &speculative::Options {
                        max_tokens: if warm { a.warmup_tokens } else { a.max_tokens },
                        depth,
                        chunk: a.prefill_chunk,
                        eos: &eos,
                    },
                    |token| {
                        if a.stream
                            && !warm
                            && let Some(text) =
                                decoder.step(token).map_err(|e| anyhow::anyhow!("{e}"))?
                        {
                            print!("{text}");
                            io::stdout().flush()?;
                        }
                        Ok(())
                    },
                )?;
                let tps = if g.decode_seconds > 0. {
                    g.tokens.len().saturating_sub(1) as f64 / g.decode_seconds
                } else {
                    0.
                };
                let accepted = g.acceptance.iter().sum::<usize>();
                let drafted = g.draft_lengths.iter().sum::<usize>();
                eprintln!(
                    "{} {run}: {} output, {tps:.2} tok/s, accepted {accepted}/{drafted}",
                    if warm { "warmup" } else { "run" },
                    g.tokens.len()
                );
                if !warm {
                    if let Some(e) = &expected {
                        ensure!(
                            g.tokens == e[..g.tokens.len().min(e.len())],
                            "MTP trajectory differs from baseline"
                        );
                    }
                    records.push(json!({"run":run,"kernel_candidate":a.ab_kernel,"candidate_enabled":candidate,"draft_depth":depth,"decode_tokens_per_second":tps,"text":t.decode(&g.tokens,true).ok(),"generation":g,"peak_memory_bytes":mlx_rs::memory::peak_memory()?}));
                }
            }
        }
    }
    let report = json!({"environment":environment,"model":a.model,"quantization":w.config["quantization"],"runtime":{"mtp":true,"draft_depth":a.draft_depth,"sampler":"greedy","batch_size":1,"prefix_cache":false,"kv_cache":"fresh per run","warmup_tokens":a.warmup_tokens,"prefill_chunk":a.prefill_chunk,"ignore_eos":a.ignore_eos,"rate_definition":"generated tokens after the first divided by decode wall time, including MTP priming, draft, verification and cache synchronization"},"prompt_ids":ids,"runs":records});
    if let Some(p) = a.output {
        std::fs::write(p, serde_json::to_vec_pretty(&report)?)?;
    } else if !a.stream {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }
    Ok(())
}
