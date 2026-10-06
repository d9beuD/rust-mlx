//! Single-projection downstream sensitivity, using actual calibration token inputs, independent of held-out quality.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use mlx_rs::{Array, Dtype, ops::indexing};
use rust_mlx::{
    environment::BenchmarkEnvironment,
    hybrid::{HybridAttention, HybridModel},
    resident_quant::{apply_overlay, sha256_file},
    weights::{Linear, Weights},
};
use serde_json::json;
use std::{collections::HashSet, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    corpus: PathBuf,
    #[arg(long)]
    overlay: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 64)]
    tokens: usize,
}
fn projection<'a>(m: &'a mut HybridModel, name: &str) -> Result<&'a mut Linear> {
    let rest = name
        .strip_prefix("language_model.model.layers.")
        .context("projection prefix")?;
    let (i, p) = rest.split_once('.').context("projection layer")?;
    let l = m
        .layers
        .get_mut(i.parse::<usize>()?)
        .context("projection layer missing")?;
    match p {
        "mlp.shared_expert.gate_proj" => return Ok(&mut l.moe.shared.gate),
        "mlp.shared_expert.up_proj" => return Ok(&mut l.moe.shared.up),
        "mlp.shared_expert.down_proj" => return Ok(&mut l.moe.shared.down),
        _ => {}
    }
    match (&mut l.attention, p) {
        (HybridAttention::Linear(g), "linear_attn.in_proj_qkv") => Ok(&mut g.qkv),
        (HybridAttention::Linear(g), "linear_attn.in_proj_z") => Ok(&mut g.z),
        (HybridAttention::Linear(g), "linear_attn.out_proj") => Ok(&mut g.out),
        (HybridAttention::Full(g), "self_attn.q_proj") => Ok(&mut g.attention.q),
        (HybridAttention::Full(g), "self_attn.k_proj") => Ok(&mut g.attention.k),
        (HybridAttention::Full(g), "self_attn.v_proj") => Ok(&mut g.attention.v),
        (HybridAttention::Full(g), "self_attn.o_proj") => Ok(&mut g.attention.o),
        _ => anyhow::bail!("protected projection {name}"),
    }
}
fn logp(a: &Array) -> Result<Array> {
    let f = a.as_dtype(Dtype::Float32)?;
    Ok(f.subtract(f.logsumexp_axis(-1, true)?)?)
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(
        a.tokens >= 32 && !a.output.exists(),
        "invalid sensitivity output/tokens"
    );
    let env = BenchmarkEnvironment::capture()?;
    let corpus: serde_json::Value = serde_json::from_slice(&std::fs::read(&a.corpus)?)?;
    let selection = corpus["calibration"]
        .as_array()
        .context("calibration corpus missing")?;
    let heldout: HashSet<&str> = corpus["heldout"]
        .as_array()
        .context("held-out corpus missing")?
        .iter()
        .filter_map(|d| d["document_sha256"].as_str())
        .collect();
    let mut languages = HashSet::new();
    let docs = selection
        .iter()
        .filter(|d| languages.insert(d["language"].as_str().unwrap_or_default()))
        .collect::<Vec<_>>();
    ensure!(
        docs.len() == 7
            && docs.iter().all(|d| d["document_sha256"]
                .as_str()
                .is_some_and(|h| !heldout.contains(h))),
        "sensitivity corpus must be seven calibration languages disjoint from quality evaluation"
    );
    mlx_rs::memory::set_memory_limit(100 * 1024usize.pow(3))?;
    mlx_rs::memory::set_cache_limit(512 * 1024usize.pow(2))?;
    let w = Weights::load(&a.model)?;
    let mut m = HybridModel::load(&w, &a.model)?;
    let mut variant = Weights {
        tensors: w.tensors.clone(),
        config: w.config.clone(),
    };
    let overlay = apply_overlay(&mut variant, &a.model, &a.overlay)?;
    ensure!(
        overlay["corpus_sha256"].as_str() == Some(sha256_file(&a.corpus)?.as_str()),
        "sensitivity corpus mismatch"
    );
    rust_mlx::gdn_compiled::set_enabled(false);
    let mut samples = Vec::new();
    for d in &docs {
        let mut ids: Vec<u32> = serde_json::from_value(d["tokens"].clone())?;
        ids.truncate(a.tokens);
        ensure!(ids.len() >= 32, "sensitivity document too short");
        let mut cache = m.make_cache();
        let logits = m.forward(&ids[..ids.len() - 1], &mut cache)?.0;
        let bp = logp(&logits)?;
        let argmax = indexing::argmax_axis(&logits, -1, false)?;
        let labels = Array::from_slice(&ids[1..], &[1, ids.len() as i32 - 1, 1]);
        let nll = bp
            .take_along_axis(&labels, -1)?
            .mean(false)?
            .negative()?
            .item_exact::<f32>() as f64;
        mlx_rs::transforms::eval([&bp, &argmax])?;
        samples.push((ids, bp, argmax, labels, nll));
    }
    let modules = overlay["quantization"]
        .as_object()
        .context("overlay modules missing")?;
    let mut records = Vec::new();
    for name in modules.keys() {
        let candidate = variant.linear(name)?;
        let original = std::mem::replace(projection(&mut m, name)?, candidate);
        let measured = (|| -> Result<serde_json::Value> {
            let mut rows = Vec::new();
            let mut total = 0;
            let mut kl_sum = 0.;
            let mut loss_sum = 0.;
            for (i, (ids, bp, bi, labels, nll)) in samples.iter().enumerate() {
                let mut cache = m.make_cache();
                let logits = m.forward(&ids[..ids.len() - 1], &mut cache)?.0;
                let cp = logp(&logits)?;
                let kl = bp
                    .exp()?
                    .multiply(bp.subtract(&cp)?)?
                    .sum_axis(-1, false)?
                    .mean(false)?
                    .item_exact::<f32>() as f64;
                let candidate_nll = cp
                    .take_along_axis(labels, -1)?
                    .mean(false)?
                    .negative()?
                    .item_exact::<f32>() as f64;
                let agreement = indexing::argmax_axis(&logits, -1, false)?
                    .eq(bi)?
                    .as_dtype(Dtype::Float32)?
                    .mean(false)?
                    .item_exact::<f32>() as f64;
                ensure!(
                    kl.is_finite() && candidate_nll.is_finite(),
                    "nonfinite downstream sensitivity"
                );
                let predictions = ids.len() - 1;
                total += predictions;
                kl_sum += kl * predictions as f64;
                loss_sum += (candidate_nll - nll) * predictions as f64;
                rows.push(json!({"language":docs[i]["language"],"document_sha256":docs[i]["document_sha256"],
                    "predictions":predictions,"kl":kl,"original_nll":nll,"candidate_nll":candidate_nll,"argmax_agreement":agreement}));
            }
            Ok(
                json!({"projection":name,"predictions":total,"token_weighted_kl":kl_sum/total as f64,
                "token_weighted_nll_delta":loss_sum/total as f64,"documents":rows}),
            )
        })();
        // Restore even after a diagnostic failure; no state or weights escape.
        *projection(&mut m, name)? = original;
        let measured = measured?;
        eprintln!(
            "sensitivity {name} KL={} deltaNLL={}",
            measured["token_weighted_kl"], measured["token_weighted_nll_delta"]
        );
        records.push(measured);
        std::fs::write(
            &a.output,
            serde_json::to_vec_pretty(&json!({"environment":env,"model":a.model,
            "model_config_sha256":sha256_file(&a.model.join("config.json"))?,"corpus_sha256":overlay["corpus_sha256"],
            "overlay_sha256":sha256_file(&a.overlay.join("overlay.json"))?,"tokens_per_document":a.tokens,
            "scope":"one resident projection changed at a time; seven calibration documents disjoint from quality/timing; fresh native prefills; measures downstream sensitivity, not additive errors or a general task-quality guarantee",
            "records":records,"complete":records.len()==modules.len(),"quality_qualified":false,"performance_qualified":false}))?,
        )?;
    }
    println!("RESIDENT_SENSITIVITY_EVALUATED");
    Ok(())
}
