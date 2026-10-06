//! Same-process adapter/native depth study with fresh caches and canonical target IDs.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use rust_mlx::{hybrid::HybridModel, mtp::Mtp, speculative, weights::Weights};
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    adapter: PathBuf,
    #[arg(long, conflicts_with = "prompts")]
    corpus: Option<PathBuf>,
    #[arg(long, conflicts_with = "corpus")]
    prompts: Option<PathBuf>,
    #[arg(long, default_value = "validation")]
    split: String,
    #[arg(long)]
    chat: bool,
    #[arg(long, value_delimiter = ',', default_value = "3")]
    depths: Vec<usize>,
    #[arg(long, default_value_t = 1)]
    runs: usize,
    #[arg(long, default_value_t = 64)]
    max_tokens: usize,
    #[arg(long, default_value_t = 32)]
    warmup_tokens: usize,
    #[arg(long)]
    ignore_eos: bool,
    #[arg(long)]
    output: PathBuf,
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(
        !a.output.exists()
            && a.runs > 0
            && a.warmup_tokens > 0
            && a.max_tokens > 0
            && !a.depths.is_empty()
            && a.depths.iter().all(|d| (1..=7).contains(d)),
        "invalid study/preserve prior output"
    );
    mlx_rs::memory::set_memory_limit(100 * 1024usize.pow(3))?;
    mlx_rs::memory::set_cache_limit(512 * 1024usize.pow(2))?;
    let w = Weights::load(&a.model)?;
    let m = HybridModel::load(&w, &a.model)?;
    let mut draft = Mtp::load(&w, &m.config)?;
    let started = std::time::Instant::now();
    draft.adapter = Some(rust_mlx::draft_adapter::DraftAdapter::load(
        &a.model,
        &a.adapter,
        m.config.hidden_size,
        m.config.hc_count,
    )?);
    let preparation = started.elapsed().as_secs_f64();
    draft.record_drafts.set(true);
    let prompts: Vec<(serde_json::Value, Vec<u32>)> = if let Some(path) = &a.corpus {
        let corpus: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        corpus["records"].as_array().context("corpus records")?.iter().filter(|r|r["split"].as_str()==Some(&a.split)).map(|r|Ok((serde_json::json!({"document_sha256":r["document_sha256"],"split":r["split"],"domain":r["domain"]}),serde_json::from_value(r["tokens"].clone())?))).collect::<Result<_>>()?
    } else {
        let text: Vec<String> = serde_json::from_slice(&std::fs::read(
            a.prompts.as_ref().context("provide corpus or prompts")?,
        )?)?;
        let tokenizer = tokenizers::Tokenizer::from_file(a.model.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        text.into_iter()
            .map(|s| {
                Ok((
                    serde_json::json!({"text":s,"chat":a.chat}),
                    rust_mlx::chat::encode_prompt(
                        &tokenizer, &a.model, &s, a.chat, false, "xhigh",
                    )?,
                ))
            })
            .collect::<Result<_>>()?
    };
    ensure!(!prompts.is_empty(), "empty split");
    let eos = if a.ignore_eos {
        vec![]
    } else {
        vec![248044, 248046]
    };
    let mut records = Vec::new();
    for (i, (metadata, ids)) in prompts.iter().enumerate() {
        let mut modes = vec![(false, 3)];
        modes.extend(a.depths.iter().map(|d| (true, *d)));
        let mut expected: Option<Vec<u32>> = None;
        for repetition in 0..=a.runs {
            let mut order = modes.clone();
            if (repetition + i) % 2 == 0 {
                order.reverse();
            }
            for (candidate, depth) in order {
                draft.adapter_enabled.set(candidate);
                let generation = speculative::generate(
                    &m,
                    &draft,
                    ids,
                    &speculative::Options {
                        max_tokens: if repetition == 0 {
                            a.warmup_tokens
                        } else {
                            a.max_tokens
                        },
                        depth,
                        chunk: 128,
                        eos: &eos,
                    },
                    |_| Ok(()),
                )?;
                if repetition == 0 {
                    continue;
                }
                if let Some(reference) = &expected {
                    ensure!(
                        &generation.tokens == reference,
                        "target trajectory changed prompt{i}/depth{depth}"
                    );
                } else {
                    expected = Some(generation.tokens.clone());
                }
                eprintln!(
                    "ADAPTER_STUDY prompt{i} rep{repetition} adapter{candidate} depth{depth} accepted{}/{}",
                    generation.acceptance.iter().sum::<usize>(),
                    generation.draft_lengths.iter().sum::<usize>()
                );
                records.push(serde_json::json!({"prompt":i,"metadata":metadata,"prompt_ids":ids,"repetition":repetition,"candidate":candidate,"depth":depth,"generation":generation,"peak_memory_bytes":mlx_rs::memory::peak_memory()?}));
                std::fs::write(
                    &a.output,
                    serde_json::to_vec_pretty(
                        &serde_json::json!({"complete":false,"records":records}),
                    )?,
                )?;
            }
        }
    }
    std::fs::write(
        a.output,
        serde_json::to_vec_pretty(
            &serde_json::json!({"complete":true,"environment":rust_mlx::environment::BenchmarkEnvironment::capture()?,"model":a.model,"mixed_quantization":w.config["quantization"],"adapter_sha256":rust_mlx::resident_quant::sha256_file(&a.adapter.join("adapter.safetensors"))?,"preparation_seconds":preparation,"sampler":"greedy","batch_size":1,"prefix_cache":false,"cache":"fresh per generation","warmup_tokens":a.warmup_tokens,"max_tokens":a.max_tokens,"ignore_eos":a.ignore_eos,"native_depth":3,"depths":a.depths,"records":records}),
        )?,
    )?;
    println!("ADAPTER_TARGET_IDS_AND_DEPTH_STUDY_COMPLETED");
    Ok(())
}
