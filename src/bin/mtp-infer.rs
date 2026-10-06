use anyhow::{Context, Result, ensure};
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
    #[arg(long)]
    resident_overlay: Option<PathBuf>,
    #[arg(long)]
    draft_adapter: Option<PathBuf>,
    #[arg(long)]
    draft_head_bits: Option<i32>,
    #[arg(long)]
    draft_logit_bias: Option<PathBuf>,
    #[arg(long)]
    draft_vocab_ids: Option<PathBuf>,
    /// Lossless experimental expert gate/up row concatenation; model-init cost separate.
    #[arg(long)]
    native_gate_up: bool,
    /// Alternate a kernel candidate in the same process.
    #[arg(long,value_parser=["packed","moe","qmv","shortlist","hc","stream-x","gemv","gdn","gpu-draft","sorted-moe","adaptive-depth","adaptive-vocab","adaptive","greedy-head","kv-blocks","qmv-address","matrix-affine","matrix-packed","route-tail","ple-prepare","rope-ids","config-reuse","runtime-prepare","down-tail","down-packed","down-packed-vector","draft-bias","draft-adapter","moe-hc-epilogue"])]
    ab_kernel: Option<String>,
    #[arg(long)]
    gpu_draft: bool,
    #[arg(long)]
    greedy_head: bool,
    #[arg(long)]
    adaptive_depth: bool,
    #[arg(long)]
    adaptive_vocab: bool,
    /// Target calibration, seconds per full round at depths1/2/3.
    #[arg(
        long,
        value_delimiter = ',',
        num_args = 3,
        default_value = "0.031545832,0.038575983,0.046995903"
    )]
    adaptive_depth_costs: Vec<f64>,
    #[arg(long)]
    draft_vocab_limit: Option<usize>,
    #[arg(long, default_value_t = 0)]
    draft_vocab_refresh_rounds: usize,
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
    let mut w = Weights::load(&a.model)?;
    let resident_variant = a
        .resident_overlay
        .as_ref()
        .map(|p| rust_mlx::resident_quant::apply_overlay(&mut w, &a.model, p))
        .transpose()?;
    let mut m = HybridModel::load(&w, &a.model)?;
    let mut draft = Mtp::load(&w, &m.config)?;
    if let Some(path) = &a.draft_adapter {
        ensure!(
            a.draft_logit_bias.is_none()
                && a.draft_vocab_ids.is_none()
                && a.draft_head_bits.is_none()
                && !a.adaptive_vocab
                && a.draft_vocab_limit.unwrap_or(0) == 0,
            "adapter requires original full draft head"
        );
        draft.adapter = Some(rust_mlx::draft_adapter::DraftAdapter::load(
            &a.model,
            path,
            m.config.hidden_size,
            m.config.hc_count,
        )?);
        draft.adapter_enabled.set(true);
    }
    ensure!(
        a.ab_kernel.as_deref() != Some("draft-adapter") || draft.adapter.is_some(),
        "missing draft adapter"
    );
    let fixed_head = if a.draft_head_bits.is_some() || a.draft_vocab_ids.is_some() {
        ensure!(
            !a.adaptive_vocab && a.draft_vocab_limit.unwrap_or(0) == 0 && a.ab_kernel.is_none(),
            "fixed draft head incompatible with other draft experiments"
        );
        let rows: Vec<u32> = if let Some(path) = &a.draft_vocab_ids {
            let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
            if value["source"]["url"]
                .as_str()
                .is_some_and(|url| url.starts_with("https://github.com/youssofal/MTPLX/"))
            {
                eprintln!(
                    "Powered by MTPLX — frequency membership artifact by Youssof Altoukhi, https://github.com/youssofal/mtplx"
                );
            }
            ensure!(
                value["tokenizer_sha256"].as_str()
                    == Some(
                        rust_mlx::resident_quant::sha256_file(&a.model.join("tokenizer.json"))?
                            .as_str()
                    ),
                "vocabulary tokenizer mismatch"
            );
            let rows: Vec<u32> = serde_json::from_value(value["ids"].clone())?;
            ensure!(!rows.is_empty(), "empty fixed vocabulary");
            rows
        } else {
            Vec::new()
        };
        let head = rust_mlx::draft_head::DraftHead::prepare(&m.head, a.draft_head_bits, &rows)?;
        let report = json!({"bits":a.draft_head_bits,"rows":head.linear.weight.shape()[0],"bytes":head.bytes,"preparation_seconds":head.preparation_seconds,"vocab_ids":a.draft_vocab_ids,
            "vocab_artifact_sha256":a.draft_vocab_ids.as_ref().map(|p|rust_mlx::resident_quant::sha256_file(p)).transpose()?,
            "source":"requantized existing affine head; model-scoped fixed membership; target head unchanged"});
        draft.draft_head = Some(head);
        draft.draft_head_enabled.set(true);
        Some(report)
    } else {
        None
    };
    let bias_artifact = if let Some(path) = &a.draft_logit_bias {
        ensure!(
            fixed_head.is_none()
                && a.ab_kernel.as_deref() == Some("draft-bias")
                && !a.adaptive_vocab
                && a.draft_vocab_limit.unwrap_or(0) == 0,
            "bias pilot requires isolated full-vocabulary A/B"
        );
        let (head, metadata) = rust_mlx::draft_head::DraftHead::with_bias(&m.head, &a.model, path)?;
        draft.draft_head = Some(head);
        Some(metadata)
    } else {
        None
    };
    ensure!(
        a.ab_kernel.as_deref() != Some("draft-bias") || bias_artifact.is_some(),
        "missing bias artifact"
    );
    let down_layout = if matches!(
        a.ab_kernel.as_deref(),
        Some("down-packed" | "down-packed-vector")
    ) {
        Some(rust_mlx::moe_down::prepare_selected(&mut m)?)
    } else {
        None
    };
    let expert_layout = if a.native_gate_up {
        let report = rust_mlx::moe_layout::prepare_model(&mut m, &mut w)?;
        rust_mlx::moe_layout::set_enabled(true);
        Some(report)
    } else {
        None
    };
    draft.gpu_draft.set(a.gpu_draft);
    rust_mlx::greedy_head::set_enabled(a.greedy_head);
    draft.record_drafts.set(true);
    draft.adaptive_depth.set(a.adaptive_depth);
    draft.adaptive_vocab.set(a.adaptive_vocab);
    draft
        .adaptive_depth_costs
        .set(a.adaptive_depth_costs.as_slice().try_into()?);
    draft
        .draft_vocab_refresh_rounds
        .set(a.draft_vocab_refresh_rounds);
    if let Some(limit) = a.draft_vocab_limit {
        draft.draft_vocab_limit.set(limit);
    }
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
            if a.ab_kernel.as_deref() == Some("draft-adapter") {
                draft.adapter_enabled.set(candidate);
            }
            if a.ab_kernel.as_deref() == Some("moe-hc-epilogue") {
                rust_mlx::moe_epilogue::set_enabled(candidate);
            }
            rust_mlx::moe_down::configure(a.ab_kernel.as_deref(), candidate);
            if a.ab_kernel.as_deref() == Some("draft-bias") {
                draft.draft_head_enabled.set(candidate);
            }
            rust_mlx::runtime_prepare::configure(a.ab_kernel.as_deref(), candidate);
            if let Some(kernel) = &a.ab_kernel {
                if kernel == "route-tail" {
                    rust_mlx::moe_route::set_enabled(candidate);
                }
                if kernel == "matrix-affine" || kernel == "matrix-packed" {
                    rust_mlx::matrix_kernel::set_packed(kernel == "matrix-packed");
                    rust_mlx::matrix_kernel::set_enabled(candidate);
                }
                if kernel == "qmv-address" {
                    rust_mlx::qmv_kernel::set_address32(candidate);
                }
                if kernel == "kv-blocks" {
                    rust_mlx::kv_blocks::set_enabled(candidate);
                }
                if kernel == "greedy-head" {
                    rust_mlx::greedy_head::set_enabled(candidate);
                }
                if kernel == "qmv" {
                    rust_mlx::qmv_kernel::set_enabled(candidate);
                }
                if kernel == "hc" {
                    rust_mlx::hc_kernel::set_enabled(candidate);
                }
                if kernel == "stream-x" {
                    rust_mlx::qmv_kernel::set_stream_x(candidate);
                }
                if kernel == "gemv" {
                    rust_mlx::gemv_kernel::set_enabled(candidate);
                }
                if kernel == "gdn" {
                    rust_mlx::gdn_compiled::set_enabled(candidate);
                }
                if kernel == "gpu-draft" {
                    draft.gpu_draft.set(candidate);
                }
                if kernel == "adaptive-depth" || kernel == "adaptive" {
                    draft.adaptive_depth.set(candidate);
                }
                if kernel == "adaptive-vocab" || kernel == "adaptive" {
                    draft.adaptive_vocab.set(candidate);
                    draft.draft_vocab_limit.set(if candidate {
                        a.draft_vocab_limit.unwrap_or(32768)
                    } else {
                        0
                    });
                }
                if kernel == "shortlist" {
                    draft.draft_vocab_limit.set(if candidate {
                        a.draft_vocab_limit.unwrap_or(32768)
                    } else {
                        0
                    });
                }
                for layer in &m.layers {
                    if kernel == "sorted-moe" {
                        layer.moe.sorted_mode.set(candidate);
                    }
                    if kernel == "moe" {
                        layer.moe.fused_mode.set(candidate);
                    } else if kernel == "packed"
                        && let rust_mlx::hybrid::HybridAttention::Linear(g) = &layer.attention
                    {
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
                let mut emitted = String::new();
                let matrix_start = rust_mlx::matrix_kernel::calls();
                let down_start = rust_mlx::moe_down::calls();
                let epilogue_start = rust_mlx::moe_epilogue::calls();
                let runtime_start = rust_mlx::runtime_prepare::stats();
                let route_start = rust_mlx::moe_route::calls();
                let layout_start = rust_mlx::moe_layout::calls();
                let gemv_start = rust_mlx::gemv_kernel::launches();
                let gdn_start = rust_mlx::gdn_compiled::calls();
                let address_start = rust_mlx::qmv_kernel::address32_calls();
                let kv_start = rust_mlx::kv_blocks::calls();
                let head_start = rust_mlx::greedy_head::calls();
                let sorted_start = rust_mlx::hybrid::sorted_moe_calls();
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
                            emitted.push_str(&text);
                            print!("{text}");
                            io::stdout().flush()?;
                        }
                        Ok(())
                    },
                )?;
                let gemv_launches = rust_mlx::gemv_kernel::launches().wrapping_sub(gemv_start);
                let matrix_calls = rust_mlx::matrix_kernel::calls().wrapping_sub(matrix_start);
                let down_calls = rust_mlx::moe_down::calls().wrapping_sub(down_start);
                let epilogue_calls = rust_mlx::moe_epilogue::calls().wrapping_sub(epilogue_start);
                if candidate
                    && a.ab_kernel.as_deref() == Some("moe-hc-epilogue")
                    && g.tokens.len() > 1
                {
                    ensure!(epilogue_calls > 0, "epilogue did not engage");
                }
                if candidate
                    && matches!(
                        a.ab_kernel.as_deref(),
                        Some("down-tail" | "down-packed" | "down-packed-vector")
                    )
                    && g.tokens.len() > 1
                {
                    ensure!(down_calls > 0, "down candidate did not engage");
                }
                let runtime_calls = std::array::from_fn::<_, 3, _>(|i| {
                    rust_mlx::runtime_prepare::stats()[i].wrapping_sub(runtime_start[i])
                });
                if candidate && g.tokens.len() > 1 {
                    match a.ab_kernel.as_deref() {
                        Some("ple-prepare") => {
                            ensure!(runtime_calls[0] > 0, "PLE preparation did not engage")
                        }
                        Some("rope-ids") => {
                            ensure!(runtime_calls[1] > 0, "position reuse did not engage")
                        }
                        Some("config-reuse") => {
                            ensure!(runtime_calls[2] > 0, "config reuse did not engage")
                        }
                        Some("runtime-prepare") => ensure!(
                            runtime_calls.iter().all(|&c| c > 0),
                            "runtime preparation did not engage"
                        ),
                        _ => {}
                    }
                }
                let route_calls = rust_mlx::moe_route::calls().wrapping_sub(route_start);
                if a.ab_kernel.as_deref() == Some("route-tail") && candidate && g.tokens.len() > 1 {
                    ensure!(route_calls > 0, "routing candidate was not engaged");
                }
                let layout_calls = rust_mlx::moe_layout::calls().wrapping_sub(layout_start);
                if a.native_gate_up && g.tokens.len() > 1 {
                    ensure!(layout_calls > 0, "native gate/up candidate did not engage");
                }
                if matches!(
                    a.ab_kernel.as_deref(),
                    Some("matrix-affine" | "matrix-packed")
                ) && candidate
                    && g.tokens.len() > 1
                {
                    ensure!(matrix_calls > 0, "matrix candidate was not engaged");
                }
                let gdn_calls = rust_mlx::gdn_compiled::calls().wrapping_sub(gdn_start);
                let address_calls =
                    rust_mlx::qmv_kernel::address32_calls().wrapping_sub(address_start);
                if a.ab_kernel.as_deref() == Some("qmv-address") && candidate && g.tokens.len() > 1
                {
                    ensure!(address_calls > 0, "QMV address candidate was not engaged");
                }
                let kv_calls = rust_mlx::kv_blocks::calls().wrapping_sub(kv_start);
                if a.ab_kernel.as_deref() == Some("kv-blocks") && candidate && g.tokens.len() > 1 {
                    ensure!(kv_calls > 0, "KV block candidate was not engaged");
                }
                let head_calls = rust_mlx::greedy_head::calls().wrapping_sub(head_start);
                if a.ab_kernel.as_deref() == Some("greedy-head") && candidate && g.tokens.len() > 1
                {
                    ensure!(head_calls > 0, "greedy head candidate was not engaged");
                }
                let sorted_calls = rust_mlx::hybrid::sorted_moe_calls().wrapping_sub(sorted_start);
                if a.ab_kernel.as_deref() == Some("sorted-moe") && candidate && g.tokens.len() > 1 {
                    ensure!(sorted_calls > 0, "sorted MoE candidate was not engaged");
                }
                if a.ab_kernel.as_deref() == Some("gemv") && candidate && g.tokens.len() > 1 {
                    ensure!(gemv_launches > 0, "GEMV candidate was not engaged");
                }
                if a.ab_kernel.as_deref() == Some("gdn") && candidate && g.tokens.len() > 1 {
                    ensure!(gdn_calls > 0, "compiled GDN candidate was not engaged");
                }
                if a.stream && !warm {
                    let complete = t
                        .decode(&g.tokens, true)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    print!(
                        "{}",
                        complete
                            .strip_prefix(&emitted)
                            .context("stream decoder changed emitted prefix")?
                    );
                    io::stdout().flush()?;
                }
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
                        if g.tokens != e[..g.tokens.len().min(e.len())]
                            && let Some(p) = &a.output
                        {
                            // Preserve the actual failed trajectory before returning
                            // an error; this report is never a qualified A/B cohort.
                            std::fs::write(
                                p,
                                serde_json::to_vec_pretty(&json!({
                                    "environment":environment,"model":a.model,"quantization":w.config["quantization"],
                                    "prompt_ids":ids,"runtime":{"mtp":true,"draft_depth":depth,"sampler":"greedy","batch_size":1,"prefix_cache":false,"warmup_tokens":a.warmup_tokens,"ignore_eos":a.ignore_eos},
                                    "kernel_candidate":a.ab_kernel,"candidate_enabled":candidate,"matrix_calls":matrix_calls,"epilogue_calls":epilogue_calls,"down_calls":down_calls,"draft_adapter_enabled":draft.adapter_enabled.get(),"draft_head_enabled":draft.draft_head_enabled.get(),"route_tail_calls":route_calls,"runtime_prepare_calls":runtime_calls,
                                    "generation":g,"expected_tokens":e,"run":run,
                                    "first_mismatch":g.tokens.iter().zip(e).position(|(a,b)|a!=b),
                                    "complete":false,"qualified":false,"failure":"MTP trajectory differs from baseline"
                                }))?,
                            )?;
                        }
                        ensure!(
                            g.tokens == e[..g.tokens.len().min(e.len())],
                            "MTP trajectory differs from baseline"
                        );
                    }
                    records.push(json!({"epilogue_calls":epilogue_calls,"down_calls":down_calls,"draft_adapter_enabled":draft.adapter_enabled.get(),"draft_head_enabled":draft.draft_head_enabled.get(),"route_tail_calls":route_calls,"runtime_prepare_calls":runtime_calls,"native_gate_up":a.native_gate_up,"layout_calls":layout_calls,"run":run,"kernel_candidate":a.ab_kernel,"candidate_enabled":candidate,"adaptive_depth":draft.adaptive_depth.get(),"adaptive_depth_costs":draft.adaptive_depth_costs.get(),"adaptive_vocab":draft.adaptive_vocab.get(),"kv_blocks":rust_mlx::kv_blocks::enabled(),"kv_block_calls":kv_calls,"greedy_head":rust_mlx::greedy_head::enabled(),"greedy_head_calls":head_calls,"sorted_moe_calls":sorted_calls,"gpu_draft":draft.gpu_draft.get(),"draft_vocab_limit":draft.draft_vocab_limit.get(),"draft_vocab_refresh_rounds":draft.draft_vocab_refresh_rounds.get(),"qmv_address32_calls":address_calls,"qmv_address32":rust_mlx::qmv_kernel::address32(),"qmv":rust_mlx::qmv_kernel::enabled(),"hc_projection":rust_mlx::hc_kernel::enabled(),"shared_gemv":rust_mlx::gemv_kernel::enabled(),"gemv_narrow":rust_mlx::gemv_kernel::narrow(),"gemv_launches":gemv_launches,"matrix_enabled":rust_mlx::matrix_kernel::enabled(),"matrix_packed":rust_mlx::matrix_kernel::packed(),"matrix_calls":matrix_calls,"compiled_gdn":rust_mlx::gdn_compiled::enabled(),"compiled_gdn_calls":gdn_calls,"draft_depth":depth,"decode_tokens_per_second":tps,"text":t.decode(&g.tokens,true).ok(),"generation":g,"peak_memory_bytes":mlx_rs::memory::peak_memory()?}));
                }
            }
        }
    }
    let report = json!({"environment":environment,"model":a.model,"quantization":w.config["quantization"],"expert_layout":expert_layout,"resident_variant":resident_variant,"fixed_draft_head":fixed_head,"draft_adapter":a.draft_adapter,"draft_adapter_sha256":a.draft_adapter.as_ref().map(|p| rust_mlx::resident_quant::sha256_file(&p.join("adapter.safetensors"))).transpose()?,"draft_bias_artifact":bias_artifact,"down_layout":down_layout,"runtime":{"native_gate_up":a.native_gate_up,"mtp":true,"draft_depth":a.draft_depth,"sampler":"greedy","batch_size":1,"prefix_cache":false,"kv_cache":"fresh per run","warmup_tokens":a.warmup_tokens,"prefill_chunk":a.prefill_chunk,"ignore_eos":a.ignore_eos,"rate_definition":"generated tokens after the first divided by decode wall time, including MTP priming, draft, verification and cache synchronization"},"prompt_ids":ids,"runs":records});
    if let Some(p) = a.output {
        std::fs::write(p, serde_json::to_vec_pretty(&report)?)?;
    } else if !a.stream {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }
    Ok(())
}
