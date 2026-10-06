//! Held-out quality study. Native and approximate models load in separate processes.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use mlx_rs::{Array, Dtype, ops::indexing};
use rust_mlx::{
    environment::BenchmarkEnvironment,
    hybrid::HybridModel,
    resident_quant::{apply_overlay, sha256_file},
    speculative::{self, Options},
    weights::Weights,
};
use serde_json::json;
use std::{collections::HashSet, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    corpus: PathBuf,
    /// Create this directory for the original; read it for an overlay.
    #[arg(long)]
    reference: PathBuf,
    #[arg(long)]
    overlay: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
}
fn main() -> Result<()> {
    let a = Args::parse();
    let environment = BenchmarkEnvironment::capture()?;
    let corpus: serde_json::Value = serde_json::from_slice(&std::fs::read(&a.corpus)?)?;
    let docs = corpus["heldout"]
        .as_array()
        .context("held-out documents missing")?;
    let train: HashSet<&str> = corpus["calibration"]
        .as_array()
        .context("calibration missing")?
        .iter()
        .filter_map(|d| d["document_sha256"].as_str())
        .collect();
    ensure!(
        !docs.is_empty()
            && docs.iter().all(|d| d["document_sha256"]
                .as_str()
                .is_some_and(|h| !train.contains(h))),
        "held-out corpus overlaps calibration"
    );
    let identity = json!({"model_config_sha256":sha256_file(&a.model.join("config.json"))?,
        "tokenizer_sha256":sha256_file(&a.model.join("tokenizer.json"))?,
        "corpus_sha256":sha256_file(&a.corpus)?});
    let reference: Option<serde_json::Value> = if a.overlay.is_some() {
        let r: serde_json::Value =
            serde_json::from_slice(&std::fs::read(a.reference.join("reference.json"))?)?;
        ensure!(
            r["identity"] == identity && r["complete"] == true,
            "quality reference identity/completion mismatch"
        );
        Some(r)
    } else {
        ensure!(!a.reference.exists(), "reference directory exists");
        std::fs::create_dir_all(&a.reference)?;
        None
    };
    mlx_rs::memory::set_memory_limit(100 * 1024usize.pow(3))?;
    mlx_rs::memory::set_cache_limit(512 * 1024usize.pow(2))?;
    let mut w = Weights::load(&a.model)?;
    let variant = a
        .overlay
        .as_ref()
        .map(|p| apply_overlay(&mut w, &a.model, p))
        .transpose()?;
    if let Some(v) = &variant {
        ensure!(
            v["corpus_sha256"] == identity["corpus_sha256"],
            "calibration corpus mismatch"
        );
    }
    let m = HybridModel::load(&w, &a.model)?;
    let mut records = Vec::new();
    let mut languages = HashSet::new();
    for (i, doc) in docs.iter().enumerate() {
        let tokens: Vec<u32> = serde_json::from_value(doc["tokens"].clone())?;
        ensure!(tokens.len() >= 64, "quality document too short");
        let mut cache = m.make_cache();
        let logits = m.forward(&tokens[..tokens.len() - 1], &mut cache)?.0;
        logits.eval()?;
        let f = logits.as_dtype(Dtype::Float32)?;
        let logp = f.subtract(f.logsumexp_axis(-1, true)?)?;
        let labels = Array::from_slice(&tokens[1..], &[1, tokens.len() as i32 - 1, 1]);
        let nll = logp
            .take_along_axis(&labels, -1)?
            .mean(false)?
            .negative()?
            .item_exact::<f32>() as f64;
        let ids = indexing::argmax_axis(&logits, -1, false)?.contiguous()?;
        ids.eval()?;
        let original = if let Some(r) = &reference {
            ensure!(
                r["records"][i]["document_sha256"] == doc["document_sha256"],
                "reference document mismatch"
            );
            let path = a.reference.join(format!("{i}.safetensors"));
            ensure!(
                r["records"][i]["logits_sha256"].as_str() == Some(sha256_file(&path)?.as_str()),
                "reference logits changed"
            );
            let mut arrays = Array::load_safetensors(path)?;
            Some(
                arrays
                    .remove("logits")
                    .context("reference logits missing")?,
            )
        } else {
            let path = a.reference.join(format!("{i}.safetensors"));
            Array::save_safetensors([("logits", &logits)], None, &path)?;
            None
        };
        let metrics = if let Some(base) = &original {
            ensure!(
                base.shape() == logits.shape(),
                "quality logits shape mismatch"
            );
            let bf = base.as_dtype(Dtype::Float32)?;
            let bp = bf.subtract(bf.logsumexp_axis(-1, true)?)?;
            let kl = bp
                .exp()?
                .multiply(bp.subtract(&logp)?)?
                .sum_axis(-1, false)?
                .mean(false)?
                .item_exact::<f32>() as f64;
            let d = f.subtract(&bf)?;
            let native_ids = indexing::argmax_axis(base, -1, false)?;
            let agreement = ids
                .eq(&native_ids)?
                .as_dtype(Dtype::Float32)?
                .mean(false)?
                .item_exact::<f32>() as f64;
            json!({"kl_original_to_variant":kl,"argmax_agreement":agreement,
                "logits_max_abs_error":d.abs()?.max(false)?.item_exact::<f32>(),
                "logits_mean_squared_error":d.multiply(&d)?.mean(false)?.item_exact::<f32>(),
                "original_nll":reference.as_ref().unwrap()["records"][i]["nll"],
                "variant_nll":nll})
        } else {
            serde_json::Value::Null
        };
        ensure!(nll.is_finite(), "nonfinite held-out loss");
        let continuation = if languages.insert(
            doc["language"]
                .as_str()
                .context("language missing")?
                .to_owned(),
        ) {
            let g = speculative::generate_plain(
                &m,
                &tokens[..32],
                &Options {
                    max_tokens: 64,
                    depth: 3,
                    chunk: 128,
                    eos: &[],
                },
                |_| Ok(()),
            )?;
            let comparison=reference.as_ref().map(|r|->Result<serde_json::Value>{
                let baseline:Vec<u32>=serde_json::from_value(r["records"][i]["continuation"]["tokens"].clone())?;
                Ok(json!({"identical":baseline==g.tokens,"first_mismatch":baseline.iter().zip(&g.tokens).position(|(x,y)|x!=y)}))
            }).transpose()?;
            json!({"tokens":g.tokens,"prompt_length":32,"max_tokens":64,"sampler":"greedy","comparison":comparison})
        } else {
            serde_json::Value::Null
        };
        eprintln!(
            "heldout doc={i} language={} nll={nll:.6} metrics={metrics}",
            doc["language"]
        );
        records.push(json!({"document_sha256":doc["document_sha256"],"language":doc["language"],
            "tokens":tokens.len(),"predictions":tokens.len()-1,"nll":nll,"metrics":metrics,
            "continuation":continuation,
            "logits_sha256":if reference.is_none(){Some(sha256_file(&a.reference.join(format!("{i}.safetensors")))?)}else{None}}));
        let report = json!({"environment":environment,"identity":identity,"model":a.model,"variant":variant,
            "scope":"28 public-preview held-out documents, seven languages; teacher-forced prefill and seven64-token plain greedy continuations; limited quality evidence, not a broad task benchmark",
            "records":records,"complete":i+1==docs.len(),"quality_qualified":false,"performance_qualified":false});
        std::fs::write(&a.output, serde_json::to_vec_pretty(&report)?)?;
        if reference.is_none() {
            std::fs::write(
                a.reference.join("reference.json"),
                serde_json::to_vec_pretty(&report)?,
            )?;
        }
    }
    println!("RESIDENT_QUALITY_EVALUATED");
    Ok(())
}
