use anyhow::{Context, Result, ensure};
use clap::Parser;
use mlx_rs::{
    Array,
    ops::indexing::{self, IndexOp},
};
use rust_mlx::{environment::BenchmarkEnvironment, hybrid::HybridModel, weights::Weights};
use serde_json::json;
use std::{
    io::{self, Write},
    path::PathBuf,
    time::Instant,
};
#[derive(Parser)]
struct Args {
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
    runs: usize,
    #[arg(long, default_value_t = 16)]
    warmup_tokens: usize,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    ignore_eos: bool,
    #[arg(long)]
    stream: bool,
    /// Alternate baseline and batched PLE in one process, with independent warmups.
    #[arg(long)]
    ab_ple: bool,
    #[arg(long, conflicts_with = "ab_ple")]
    ab_async: bool,
    #[arg(long,conflicts_with_all=["ab_ple","ab_async"])]
    ab_hyper: bool,
    #[arg(long,conflicts_with_all=["ab_ple","ab_async","ab_hyper"])]
    ab_packed: bool,
    #[arg(long, default_value_t = 128)]
    prefill_chunk: usize,
}
fn greedy(logits: &Array) -> Result<u32> {
    Ok(indexing::argmax(logits, false)?.item_exact::<u32>())
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(
        a.runs > 0 && a.max_tokens > 0 && a.prefill_chunk > 0,
        "runs/tokens/chunk must be positive"
    );
    let environment = BenchmarkEnvironment::capture()?;
    mlx_rs::memory::set_memory_limit(100 * 1024usize.pow(3))?;
    mlx_rs::memory::set_cache_limit(512 * 1024usize.pow(2))?;
    let load = Instant::now();
    let w = Weights::load(&a.model)?;
    let m = HybridModel::load(&w, &a.model)?;
    let mut tokenizer = None;
    let prompt: Vec<u32> = if let Some(p) = &a.prompt_ids {
        serde_json::from_slice(&std::fs::read(p)?)?
    } else {
        let t = tokenizers::Tokenizer::from_file(a.model.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let ids = rust_mlx::chat::encode_prompt(
            &t,
            &a.model,
            &a.prompt,
            a.chat,
            !a.no_thinking,
            &a.reasoning_effort,
        )?;
        tokenizer = Some(t);
        ids
    };
    ensure!(!prompt.is_empty(), "empty prompt");
    mlx_rs::transforms::eval(w.tensors.values())?;
    let load_seconds = load.elapsed().as_secs_f64();
    eprintln!(
        "weights ready: {:.3}s, {:.2} GiB active",
        load_seconds,
        mlx_rs::memory::active_memory()? as f64 / 1024f64.powi(3)
    );
    let eos: Vec<u32> = match w.config.get("eos_token_id") {
        Some(v) if v.is_array() => serde_json::from_value(v.clone())?,
        Some(v) => vec![v.as_u64().context("invalid eos")? as u32],
        None => vec![248044, 248046],
    };
    let mut records = Vec::new();
    let warmups = if a.ab_ple || a.ab_async || a.ab_hyper || a.ab_packed {
        2
    } else {
        1
    };
    for run in 0..a.runs + warmups {
        let warmup = run < warmups;
        let batch = if a.ab_ple {
            run % 2 == 1
        } else {
            std::env::var_os("RUST_MLX_BATCH_PLE").is_some()
        };
        for p in m.ple.iter().flatten() {
            p.table.set_batch(batch);
        }
        let async_layers = if a.ab_async {
            run % 2 == 1
        } else {
            std::env::var_os("RUST_MLX_ASYNC_LAYERS").is_some()
        };
        m.async_layers.set(async_layers);
        let compiled_hyper = if a.ab_hyper {
            run % 2 == 1
        } else {
            std::env::var_os("RUST_MLX_COMPILE_HYPER").is_some()
        };
        let packed_gdn = if a.ab_packed {
            run % 2 == 1
        } else {
            std::env::var_os("RUST_MLX_PACKED_GDN").is_some()
        };
        for l in &m.layers {
            if let rust_mlx::hybrid::HybridAttention::Linear(g) = &l.attention {
                g.packed_mode.set(packed_gdn);
            }
        }
        m.mixer.compiled_mode.set(compiled_hyper);
        for l in &m.layers {
            l.attn_hc.compiled_mode.set(compiled_hyper);
            l.mlp_hc.compiled_mode.set(compiled_hyper);
        }

        let limit = if warmup {
            a.warmup_tokens
        } else {
            a.max_tokens
        };
        let mut cache = m.make_cache();
        let mut tail = None;
        let prefill = Instant::now();
        for chunk in prompt.chunks(a.prefill_chunk) {
            let (logits, _) = m.forward(chunk, &mut cache)?;
            let last = logits.index((0, -1, ..));
            last.eval()?;
            tail = Some(last);
        }
        let mut tail = tail.context("no prefill output")?;
        let prefill_seconds = prefill.elapsed().as_secs_f64();
        let mut tokens = Vec::new();
        let mut latencies = Vec::new();
        let mut decoder = tokenizer.as_ref().map(|t| t.decode_stream(true));
        let mut emitted = String::new();
        let decode = Instant::now();
        for _ in 0..limit {
            let token = greedy(&tail)?;
            if !a.ignore_eos && eos.contains(&token) {
                break;
            }
            tokens.push(token);
            if a.stream
                && !warmup
                && let Some(d) = &mut decoder
                && let Some(s) = d.step(token).map_err(|e| anyhow::anyhow!("{e}"))?
            {
                emitted.push_str(&s);
                print!("{s}");
                io::stdout().flush()?;
            }
            let started = Instant::now();
            let (logits, _) = m.forward(&[token], &mut cache)?;
            tail = logits.index((0, -1, ..));
            tail.eval()?;
            latencies.push(started.elapsed().as_secs_f64());
        }
        let seconds = decode.elapsed().as_secs_f64();
        if a.stream
            && !warmup
            && let Some(t) = &tokenizer
        {
            let complete = t
                .decode(&tokens, true)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            print!(
                "{}",
                complete
                    .strip_prefix(&emitted)
                    .context("stream decoder changed emitted prefix")?
            );
            io::stdout().flush()?;
        }
        let tps = tokens.len() as f64 / seconds;
        eprintln!(
            "{} {run}: prompt={} output={} prefill={prefill_seconds:.3}s decode={tps:.2} tok/s",
            if warmup { "warmup" } else { "run" },
            prompt.len(),
            tokens.len()
        );
        if !warmup {
            records.push(json!({"run":run,"packed_gdn":packed_gdn,"compiled_hyper":compiled_hyper,"async_layers":async_layers,"ple_lookup":if batch{"batched candidate"}else{"per-row reference"},"prompt_tokens":prompt.len(),"generated_tokens":tokens.len(),"tokens":tokens,"text":tokenizer.as_ref().and_then(|t|t.decode(&tokens,true).ok()),"prefill_seconds":prefill_seconds,"decode_seconds":seconds,"decode_tokens_per_second":tps,"inter_token_seconds":latencies,"peak_memory_bytes":mlx_rs::memory::peak_memory()?}));
        }
    }
    let report = json!({"environment":environment,"model":a.model,"quantization":w.config["quantization"],"runtime":{"backend":"mlx","mtp":false,"temperature":0,"batch_size":1,"prefix_cache":false,"ab_ple":a.ab_ple,"weights_warm":true,"kv_cache":"fresh per run","ignore_eos":a.ignore_eos,"prefill_chunk":a.prefill_chunk,"warmup_tokens":a.warmup_tokens,"ple_lookup":if std::env::var_os("RUST_MLX_BATCH_PLE").is_some(){"batched candidate"}else{"per-row reference"},"gdn":if std::env::var_os("RUST_MLX_GDN_OPS").is_some(){"ops fallback"}else{"native reduction Metal"}},"prompt_ids":prompt,"load_seconds":load_seconds,"runs":records});
    if let Some(output) = a.output {
        std::fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    } else if !a.stream {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }
    Ok(())
}
