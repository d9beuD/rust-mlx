//! Paired full-MTP evaluation of a model-scoped draft head; exact target oracle.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use rust_mlx::{
    draft_head::DraftHead,
    environment::BenchmarkEnvironment,
    hybrid::HybridModel,
    mtp::Mtp,
    speculative::{self, Options},
    weights::Weights,
};
use serde_json::json;
use std::{
    path::{Path, PathBuf},
    process::Command,
};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    prompts: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    head_bits: Option<i32>,
    #[arg(long)]
    vocab_ids: Option<PathBuf>,
    #[arg(long)]
    chat: bool,
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,
    #[arg(long, default_value_t = 256)]
    warmup_tokens: usize,
    #[arg(long, default_value_t = 4)]
    runs: usize,
    #[arg(long, default_value_t = 3)]
    depth: usize,
    #[arg(long)]
    ignore_eos: bool,
}
fn sha(path: &Path) -> Result<String> {
    let out = Command::new("shasum")
        .args(["-a", "256"])
        .arg(path)
        .output()?;
    ensure!(
        out.status.success(),
        "failed checksum for {}",
        path.display()
    );
    Ok(String::from_utf8(out.stdout)?
        .split_whitespace()
        .next()
        .context("empty checksum")?
        .into())
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(
        a.runs > 0 && a.max_tokens > 1 && a.warmup_tokens > 1,
        "invalid timing limits"
    );
    ensure!(
        a.warmup_tokens <= a.max_tokens,
        "warmup exceeds oracle length"
    );
    ensure!(
        a.head_bits.is_some() || a.vocab_ids.is_some(),
        "choose a draft candidate"
    );
    let environment = BenchmarkEnvironment::capture()?;
    let vocab_artifact = a
        .vocab_ids
        .as_ref()
        .map(|p| -> Result<serde_json::Value> {
            let value: serde_json::Value = serde_json::from_slice(&std::fs::read(p)?)?;
            ensure!(
                value["tokenizer_sha256"].as_str()
                    == Some(sha(&a.model.join("tokenizer.json"))?.as_str()),
                "draft vocabulary tokenizer checksum differs"
            );
            Ok(value)
        })
        .transpose()?;
    let rows: Vec<u32> = vocab_artifact
        .as_ref()
        .map(|v| serde_json::from_value(v["ids"].clone()))
        .transpose()?
        .unwrap_or_default();
    if vocab_artifact.as_ref().is_some_and(|v| {
        v["source"]["url"]
            .as_str()
            .is_some_and(|s| s.starts_with("https://github.com/youssofal/MTPLX/"))
    }) {
        eprintln!(
            "Powered by MTPLX — frequency membership artifact by Youssof Altoukhi, https://github.com/youssofal/mtplx"
        );
    }
    ensure!(
        a.vocab_ids.is_none() || !rows.is_empty(),
        "empty vocabulary artifact"
    );
    mlx_rs::memory::set_memory_limit(100 * 1024usize.pow(3))?;
    mlx_rs::memory::set_cache_limit(512 * 1024usize.pow(2))?;
    let w = Weights::load(&a.model)?;
    let m = HybridModel::load(&w, &a.model)?;
    let mut draft = Mtp::load(&w, &m.config)?;
    draft.record_drafts.set(true);
    draft.draft_head = Some(DraftHead::prepare(&m.head, a.head_bits, &rows)?);
    mlx_rs::transforms::eval(w.tensors.values())?;
    let head = draft.draft_head.as_ref().context("prepared head missing")?;
    let setup = json!({"bits":a.head_bits,"source_bits":m.head.quant.as_ref().map(|q|q.bits),
        "group_size":head.linear.quant.as_ref().map(|q|q.group_size),"rows":head.linear.weight.shape()[0],
        "preparation_seconds":head.preparation_seconds,"bytes":head.bytes,
        "amortization":"model-scoped, evaluated once before prompt processing; no prompt-fit rows",
        "requantization_source":"dequantized existing affine checkpoint; original BF16 weights unavailable",
        "vocab_artifact_sha256":a.vocab_ids.as_ref().map(|p|sha(p)).transpose()?,
        "vocab_source":vocab_artifact.as_ref().map(|v|&v["source"])});
    let source_files = [
        "src/draft_head.rs",
        "src/draft_vocab.rs",
        "src/mtp.rs",
        "src/speculative.rs",
        "src/bin/draft-study.rs",
    ]
    .into_iter()
    .map(|s| Ok((s, sha(Path::new(s))?)))
    .collect::<Result<std::collections::BTreeMap<_, _>>>()?;
    let t = tokenizers::Tokenizer::from_file(a.model.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let prompts: Vec<String> = serde_json::from_slice(&std::fs::read(&a.prompts)?)?;
    ensure!(!prompts.is_empty(), "empty prompt suite");
    let mut records = Vec::new();
    let mut oracles = Vec::new();
    for (pi, prompt) in prompts.iter().enumerate() {
        let ids = rust_mlx::chat::encode_prompt(&t, &a.model, prompt, a.chat, false, "xhigh")?;
        let opts = Options {
            max_tokens: a.max_tokens,
            depth: a.depth,
            chunk: 128,
            eos: if a.ignore_eos { &[] } else { &[248044, 248046] },
        };
        let oracle = speculative::generate_plain(&m, &ids, &opts, |_| Ok(()))?;
        oracles.push(json!({"prompt":pi,"tokens":oracle.tokens,"kind":"independent plain greedy target; not timed cohort"}));
        for cycle in 0..=a.runs {
            for candidate in if cycle % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                draft.draft_head_enabled.set(candidate);
                let limit = if cycle == 0 {
                    a.warmup_tokens
                } else {
                    a.max_tokens
                };
                let opts = Options {
                    max_tokens: limit,
                    ..opts
                };
                let g = speculative::generate(&m, &draft, &ids, &opts, |_| Ok(()))?;
                let compare = g.tokens.len().min(oracle.tokens.len());
                let exact = g.tokens[..compare] == oracle.tokens[..compare]
                    && (cycle == 0 || g.tokens.len() == oracle.tokens.len());
                let rate = g.tokens.len().saturating_sub(1) as f64 / g.decode_seconds;
                eprintln!(
                    "prompt={pi} cycle={cycle} candidate={candidate} tokens={} tps={rate:.2} accepted={}/{} exact={exact}",
                    g.tokens.len(),
                    g.acceptance.iter().sum::<usize>(),
                    g.draft_lengths.iter().sum::<usize>()
                );
                if cycle > 0 || !exact {
                    records.push(
                        json!({"prompt":pi,"prompt_ids":ids,"cycle":cycle,"candidate":candidate,
                        "exact_target_ids":exact,"tokens_per_second":rate,"generation":g,
                        "peak_memory_bytes":mlx_rs::memory::peak_memory()?}),
                    );
                    std::fs::write(
                        &a.output,
                        serde_json::to_vec_pretty(&json!({"environment":environment,
                        "model":a.model,"quantization":w.config["quantization"],"draft_head":setup,"source_file_sha256":source_files,
                        "runtime":{"mtp":true,"depth":a.depth,"sampler":"greedy","batch_size":1,"prefix_cache":false,
                            "max_tokens":a.max_tokens,"warmup_tokens":a.warmup_tokens,"runs":a.runs,"chat":a.chat,
                            "thinking":false,"ignore_eos":a.ignore_eos,"timing":"(generated-1)/decode includes all prime/draft/verify/sync; model-init head cost separate; alternating warmed modes"},
                        "oracles":oracles,"records":records,"complete":false,"qualified":false}))?,
                    )?;
                }
                ensure!(exact, "draft candidate changes canonical target trajectory");
            }
        }
    }
    let mut report: serde_json::Value = serde_json::from_slice(&std::fs::read(&a.output)?)?;
    report["complete"] = true.into();
    // Completion is not default promotion; the end-to-end/quality analyzer decides.
    std::fs::write(&a.output, serde_json::to_vec_pretty(&report)?)?;
    println!("DRAFT_STUDY_PASSED");
    Ok(())
}
