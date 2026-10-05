//! Equal timing definitions, fresh caches, and trajectory comparisons in one process.
use anyhow::{Result, ensure};
use clap::Parser;
use rust_mlx::{
    chat,
    environment::BenchmarkEnvironment,
    hybrid::HybridModel,
    mtp::Mtp,
    speculative::{self, Options},
    weights::Weights,
};
use serde_json::json;
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 256)]
    warmup_tokens: usize,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    prompts: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,
    #[arg(long, default_value_t = 3)]
    runs: usize,
    #[arg(long, default_value_t = 3)]
    depth: usize,
    #[arg(long)]
    chat: bool,
    #[arg(long)]
    no_thinking: bool,
    #[arg(long, default_value = "xhigh")]
    reasoning_effort: String,
    #[arg(long)]
    ignore_eos: bool,
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(a.runs > 0 && a.max_tokens > 0, "invalid benchmark limits");
    let environment = BenchmarkEnvironment::capture()?;
    mlx_rs::memory::set_memory_limit(100 * 1024usize.pow(3))?;
    mlx_rs::memory::set_cache_limit(512 * 1024usize.pow(2))?;
    let w = Weights::load(&a.model)?;
    let m = HybridModel::load(&w, &a.model)?;
    let draft = Mtp::load(&w, &m.config)?;
    mlx_rs::transforms::eval(w.tensors.values())?;
    let tokenizer = tokenizers::Tokenizer::from_file(a.model.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let prompts: Vec<String> = serde_json::from_slice(&std::fs::read(&a.prompts)?)?;
    let mut records = Vec::new();
    for prompt in prompts {
        let ids = chat::encode_prompt(
            &tokenizer,
            &a.model,
            &prompt,
            a.chat,
            !a.no_thinking,
            &a.reasoning_effort,
        )?;
        let mut expected = None;
        for cycle in 0..=a.runs {
            for mtp in if cycle % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let options = Options {
                    max_tokens: if cycle == 0 {
                        a.warmup_tokens.min(a.max_tokens)
                    } else {
                        a.max_tokens
                    },
                    depth: a.depth,
                    chunk: 128,
                    eos: if a.ignore_eos { &[] } else { &[248044, 248046] },
                };
                let g = if mtp {
                    speculative::generate(&m, &draft, &ids, &options, |_| Ok(()))?
                } else {
                    speculative::generate_plain(&m, &ids, &options, |_| Ok(()))?
                };
                let tps = if g.decode_seconds > 0. {
                    g.tokens.len().saturating_sub(1) as f64 / g.decode_seconds
                } else {
                    0.
                };
                eprintln!(
                    "prompt={} cycle={cycle} mtp={mtp} output={} tps={tps:.2}",
                    ids.len(),
                    g.tokens.len()
                );
                if cycle > 0 {
                    if let Some(e) = &expected {
                        ensure!(&g.tokens == e, "plain/MTP trajectory mismatch");
                    } else {
                        expected = Some(g.tokens.clone());
                    }
                    records.push(json!({"prompt":prompt,"prompt_ids":ids,"cycle":cycle,"mtp":mtp,"tokens_per_second":tps,"text":tokenizer.decode(&g.tokens,true).ok(),"generation":g,"peak_memory_bytes":mlx_rs::memory::peak_memory()?}));
                    std::fs::write(
                        &a.output,
                        serde_json::to_vec_pretty(
                            &json!({"environment":environment,"model":a.model,"quantization":w.config["quantization"],"runtime":{"chat":a.chat,"thinking":!a.no_thinking,"reasoning_effort":a.reasoning_effort,"ignore_eos":a.ignore_eos,"batch":1,"prefix_cache":false,"greedy":true,"warmup_tokens":a.warmup_tokens,"depth":a.depth,"timing":"(generated-1)/decode including all draft/verify/prime/sync; fresh caches, alternating modes, full requested warmup each mode"},"records":records}),
                        )?,
                    )?;
                }
            }
        }
    }
    println!("WORKLOAD_BENCH_PASSED");
    Ok(())
}
