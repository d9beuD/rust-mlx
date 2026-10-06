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
    /// Compare MTP with shared-weight verifier projections disabled/enabled.
    #[arg(long)]
    ab_qmv: bool,
    #[arg(long, conflicts_with_all = ["ab_qmv", "ab_hc", "ab_stream_x"])]
    ab_shortlist: bool,
    #[arg(long, conflicts_with_all = ["ab_qmv", "ab_stream_x"])]
    ab_hc: bool,
    #[arg(long, conflicts_with = "ab_qmv")]
    ab_stream_x: bool,
    #[arg(long, conflicts_with_all = ["ab_qmv", "ab_shortlist", "ab_hc", "ab_stream_x"])]
    ab_gemv: bool,
    #[arg(long, conflicts_with_all = ["ab_qmv", "ab_shortlist", "ab_hc", "ab_stream_x", "ab_gemv"])]
    ab_adaptive_depth: bool,
    #[arg(long, conflicts_with_all = ["ab_qmv", "ab_shortlist", "ab_hc", "ab_stream_x", "ab_gemv", "ab_adaptive_depth"])]
    ab_adaptive_vocab: bool,
    #[arg(long, conflicts_with_all = ["ab_qmv", "ab_shortlist", "ab_hc", "ab_stream_x", "ab_gemv", "ab_adaptive_depth", "ab_adaptive_vocab"])]
    ab_adaptive: bool,
    #[arg(long, conflicts_with_all = ["ab_qmv", "ab_shortlist", "ab_hc", "ab_stream_x", "ab_gemv", "ab_adaptive_depth", "ab_adaptive_vocab", "ab_adaptive"])]
    ab_greedy_head: bool,
    #[arg(long, conflicts_with_all = ["ab_qmv", "ab_shortlist", "ab_hc", "ab_stream_x", "ab_gemv", "ab_adaptive_depth", "ab_adaptive_vocab", "ab_adaptive", "ab_greedy_head"])]
    ab_kv_blocks: bool,
    #[arg(long, default_value_t = 32768)]
    draft_vocab_limit: usize,
    #[arg(long, default_value_t = 0)]
    draft_vocab_refresh_rounds: usize,
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
    draft.record_drafts.set(true);
    draft
        .draft_vocab_refresh_rounds
        .set(a.draft_vocab_refresh_rounds);
    mlx_rs::transforms::eval(w.tensors.values())?;
    let tokenizer = tokenizers::Tokenizer::from_file(a.model.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let prompts: Vec<String> = serde_json::from_slice(&std::fs::read(&a.prompts)?)?;
    ensure!(
        !prompts.is_empty(),
        "benchmark requires at least one prompt"
    );
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
            for candidate in if cycle % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let mtp = a.ab_qmv
                    || a.ab_shortlist
                    || a.ab_hc
                    || a.ab_stream_x
                    || a.ab_gemv
                    || a.ab_adaptive_depth
                    || a.ab_adaptive_vocab
                    || a.ab_adaptive
                    || a.ab_greedy_head
                    || a.ab_kv_blocks
                    || candidate;
                if a.ab_kv_blocks {
                    rust_mlx::kv_blocks::set_enabled(candidate);
                }
                if a.ab_greedy_head {
                    rust_mlx::greedy_head::set_enabled(candidate);
                }
                if a.ab_qmv {
                    rust_mlx::qmv_kernel::set_enabled(candidate);
                }
                if a.ab_shortlist {
                    draft
                        .draft_vocab_limit
                        .set(if candidate { a.draft_vocab_limit } else { 0 });
                }
                if a.ab_hc {
                    rust_mlx::hc_kernel::set_enabled(candidate);
                }
                if a.ab_stream_x {
                    rust_mlx::qmv_kernel::set_stream_x(candidate);
                }
                if a.ab_gemv {
                    rust_mlx::gemv_kernel::set_enabled(candidate);
                }
                if a.ab_adaptive_depth || a.ab_adaptive {
                    draft.adaptive_depth.set(candidate);
                }
                if a.ab_adaptive_vocab || a.ab_adaptive {
                    draft.adaptive_vocab.set(candidate);
                    draft
                        .draft_vocab_limit
                        .set(if candidate { a.draft_vocab_limit } else { 0 });
                }
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
                let kv_start = rust_mlx::kv_blocks::calls();
                let head_start = rust_mlx::greedy_head::calls();
                let gemv_start = rust_mlx::gemv_kernel::launches();
                let g = if mtp {
                    speculative::generate(&m, &draft, &ids, &options, |_| Ok(()))?
                } else {
                    speculative::generate_plain(&m, &ids, &options, |_| Ok(()))?
                };
                let kv_calls = rust_mlx::kv_blocks::calls().wrapping_sub(kv_start);
                if a.ab_kv_blocks && candidate && g.tokens.len() > 1 {
                    ensure!(kv_calls > 0, "KV block candidate was not engaged");
                }
                let head_calls = rust_mlx::greedy_head::calls().wrapping_sub(head_start);
                if a.ab_greedy_head && candidate && g.tokens.len() > 1 {
                    ensure!(head_calls > 0, "greedy head candidate was not engaged");
                }
                let gemv_launches = rust_mlx::gemv_kernel::launches().wrapping_sub(gemv_start);
                if a.ab_gemv && candidate && g.tokens.len() > 1 {
                    ensure!(gemv_launches > 0, "GEMV candidate was not engaged");
                }
                let tps = if g.decode_seconds > 0. {
                    g.tokens.len().saturating_sub(1) as f64 / g.decode_seconds
                } else {
                    0.
                };
                eprintln!(
                    "prompt={} cycle={cycle} mtp={mtp} candidate={candidate} output={} tps={tps:.2}",
                    ids.len(),
                    g.tokens.len()
                );
                if cycle > 0 {
                    if let Some(e) = &expected {
                        ensure!(&g.tokens == e, "benchmark trajectory mismatch");
                    } else {
                        expected = Some(g.tokens.clone());
                    }
                    records.push(json!({"prompt":prompt,"prompt_ids":ids,"cycle":cycle,"mtp":mtp,"candidate":candidate,"kv_blocks":rust_mlx::kv_blocks::enabled(),"kv_block_calls":kv_calls,"greedy_head":rust_mlx::greedy_head::enabled(),"greedy_head_calls":head_calls,"adaptive_depth":draft.adaptive_depth.get(),"adaptive_depth_costs":draft.adaptive_depth_costs.get(),"adaptive_vocab":draft.adaptive_vocab.get(),"draft_vocab_limit":draft.draft_vocab_limit.get(),"draft_vocab_refresh_rounds":draft.draft_vocab_refresh_rounds.get(),"qmv":rust_mlx::qmv_kernel::enabled(),"hc_projection":rust_mlx::hc_kernel::enabled(),"shared_gemv":rust_mlx::gemv_kernel::enabled(),"gemv_narrow":rust_mlx::gemv_kernel::narrow(),"gemv_launches":gemv_launches,"qmv_stream_x":rust_mlx::qmv_kernel::stream_x(),"tokens_per_second":tps,"text":tokenizer.decode(&g.tokens,true).ok(),"generation":g,"peak_memory_bytes":mlx_rs::memory::peak_memory()?}));
                    std::fs::write(
                        &a.output,
                        serde_json::to_vec_pretty(
                            &json!({"environment":environment,"model":a.model,"quantization":w.config["quantization"],"runtime":{"chat":a.chat,"thinking":!a.no_thinking,"reasoning_effort":a.reasoning_effort,"ignore_eos":a.ignore_eos,"batch":1,"prefix_cache":false,"greedy":true,"ab_kv_blocks":a.ab_kv_blocks,"ab_greedy_head":a.ab_greedy_head,"ab_adaptive_depth":a.ab_adaptive_depth,"ab_adaptive_vocab":a.ab_adaptive_vocab,"ab_adaptive":a.ab_adaptive,"ab_qmv":a.ab_qmv,"ab_shortlist":a.ab_shortlist,"ab_hc":a.ab_hc,"ab_gemv":a.ab_gemv,"ab_stream_x":a.ab_stream_x,"warmup_tokens":a.warmup_tokens,"depth":a.depth,"timing":"(generated-1)/decode including all draft/verify/prime/sync; fresh caches, alternating modes, full requested warmup each mode"},"records":records}),
                        )?,
                    )?;
                }
            }
        }
    }
    println!("WORKLOAD_BENCH_PASSED");
    Ok(())
}
