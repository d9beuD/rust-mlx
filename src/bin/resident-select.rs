//! Preserve sensitive projections according to an independent calibration study.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use rust_mlx::resident_quant::sha256_file;
use serde_json::json;
use std::{
    collections::{BTreeMap, HashSet},
    path::PathBuf,
};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    source: PathBuf,
    #[arg(long)]
    sensitivity: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    report: PathBuf,
    #[arg(long, default_value_t = 32)]
    keep_projections: usize,
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(
        !a.output.exists() && !a.report.exists(),
        "preserve existing variant/report"
    );
    let source_path = a.source.join("overlay.json");
    let mut overlay: serde_json::Value = serde_json::from_slice(&std::fs::read(&source_path)?)?;
    let study: serde_json::Value = serde_json::from_slice(&std::fs::read(&a.sensitivity)?)?;
    ensure!(
        study["complete"] == true
            && study["overlay_sha256"].as_str() == Some(sha256_file(&source_path)?.as_str())
            && study["corpus_sha256"] == overlay["corpus_sha256"],
        "sensitivity source/completion mismatch"
    );
    let modules = overlay["quantization"]
        .as_object()
        .context("source modules missing")?;
    let rows = study["records"]
        .as_array()
        .context("sensitivity records missing")?;
    let mut scores = BTreeMap::new();
    for r in rows {
        let name = r["projection"].as_str().context("projection missing")?;
        let score = r["token_weighted_kl"]
            .as_f64()
            .context("sensitivity KL missing")?;
        ensure!(
            score.is_finite()
                && modules.contains_key(name)
                && scores.insert(name.to_owned(), score).is_none(),
            "invalid/duplicate sensitivity score"
        );
    }
    ensure!(
        scores.len() == modules.len() && a.keep_projections < scores.len(),
        "sensitivity coverage/selection count"
    );
    let mut ranked = scores.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|(a, x), (b, y)| y.total_cmp(x).then(a.cmp(b)));
    let preserved: HashSet<String> = ranked
        .iter()
        .take(a.keep_projections)
        .map(|(n, _)| n.clone())
        .collect();
    let tensor_path = a.source.join("tensors.safetensors");
    ensure!(
        overlay["tensors_sha256"].as_str() == Some(sha256_file(&tensor_path)?.as_str()),
        "source tensors changed"
    );
    let mut arrays = mlx_rs::Array::load_safetensors(&tensor_path)?;
    for name in &preserved {
        overlay["quantization"]
            .as_object_mut()
            .context("quantization metadata")?
            .remove(name);
        overlay["receipts"]
            .as_object_mut()
            .context("receipt metadata")?
            .remove(name);
        for suffix in ["weight", "scales", "biases"] {
            ensure!(
                arrays.remove(&format!("{name}.{suffix}")).is_some(),
                "selected tensor missing"
            );
        }
    }
    ensure!(!arrays.is_empty(), "selection removed all projections");
    std::fs::create_dir_all(&a.output)?;
    let output_tensors = a.output.join("tensors.safetensors");
    mlx_rs::Array::save_safetensors(arrays.iter(), None, &output_tensors)?;
    overlay["tensors_sha256"] = sha256_file(&output_tensors)?.into();
    overlay["selection"] = json!({"keep_original_projections":a.keep_projections,
        "preserved":ranked.iter().take(a.keep_projections).collect::<Vec<_>>(),
        "sensitivity_sha256":sha256_file(&a.sensitivity)?,"source_overlay_sha256":sha256_file(&source_path)?,
        "method":"preserve highest downstream single-projection KL on seven calibration documents; quality held-out documents and timing prompts are independent; not an additive error bound"});
    overlay["quality_qualified"] = false.into();
    overlay["performance_qualified"] = false.into();
    std::fs::write(
        a.output.join("overlay.json"),
        serde_json::to_vec_pretty(&overlay)?,
    )?;
    std::fs::write(a.report, serde_json::to_vec_pretty(&overlay)?)?;
    println!("RESIDENT_MIXED_OVERLAY_CREATED");
    Ok(())
}
