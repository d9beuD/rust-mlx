//! Create a distinct diagonal-calibrated affine overlay; never rewrite the checkpoint.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use rust_mlx::{
    calibration::{self, ActivationMoments},
    environment::BenchmarkEnvironment,
    hybrid::HybridModel,
    resident_quant,
    weights::Weights,
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    process::Command,
};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    corpus: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    moments: Option<PathBuf>,
    #[arg(long, default_value_t = 4)]
    bits: i32,
    #[arg(long)]
    shared_only: bool,
}
fn sha(p: &Path) -> Result<String> {
    let out = Command::new("shasum").args(["-a", "256"]).arg(p).output()?;
    ensure!(out.status.success(), "checksum failed");
    Ok(String::from_utf8(out.stdout)?
        .split_whitespace()
        .next()
        .context("empty checksum")?
        .into())
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!([4, 5, 6].contains(&a.bits), "resident bits must be4/5/6");
    ensure!(
        !a.output.exists(),
        "overlay destination already exists; choose a fresh directory"
    );
    let env = BenchmarkEnvironment::capture()?;
    let config_sha = sha(&a.model.join("config.json"))?;
    let corpus_sha = sha(&a.corpus)?;
    let mut w = Weights::load(&a.model)?;
    let m = HybridModel::load(&w, &a.model)?;
    // Collection is diagnostic: force transparent Rust boundaries rather than
    // reusing compiled closures that would hide projection input observations.
    rust_mlx::gdn_compiled::set_enabled(false);
    for l in &m.layers {
        l.attn_hc.compiled_mode.set(false);
        l.mlp_hc.compiled_mode.set(false);
    }
    let stats: BTreeMap<String, ActivationMoments> = if let Some(path) = &a.moments {
        let saved: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        ensure!(
            saved["model_config_sha256"].as_str() == Some(config_sha.as_str())
                && saved["corpus_sha256"].as_str() == Some(corpus_sha.as_str()),
            "calibration moments source mismatch"
        );
        serde_json::from_value(saved["moments"].clone())?
    } else {
        let corpus: serde_json::Value = serde_json::from_slice(&std::fs::read(&a.corpus)?)?;
        let docs = corpus["calibration"]
            .as_array()
            .context("calibration corpus missing")?;
        ensure!(!docs.is_empty(), "empty calibration corpus");
        let stats = calibration::capture(&m, |m| {
            for (i, d) in docs.iter().enumerate() {
                let ids: Vec<u32> = serde_json::from_value(d["tokens"].clone())?;
                ensure!(ids.len() >= 32, "calibration document too short");
                let mut cache = m.make_cache();
                m.forward(&ids, &mut cache)?.0.eval()?;
                eprintln!(
                    "calibration document={i} tokens={} language={}",
                    ids.len(),
                    d["language"]
                );
            }
            Ok(())
        })?;
        let path = Path::new("results/decode3-resident-moments.json");
        ensure!(
            !path.exists(),
            "fresh calibration would overwrite existing moments; use --moments or preserve the old file"
        );
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&json!({"environment":env,"model_config_sha256":config_sha,
            "corpus_sha256":corpus_sha,"model":a.model,"moments":stats,
            "scope":"original checkpoint BF16 resident inputs; no timing prompts; diagonal second moments only"}))?,
        )?;
        stats
    };
    let mut arrays = BTreeMap::new();
    let mut quantization = BTreeMap::new();
    let mut receipts = BTreeMap::new();
    for (name, moments) in &stats {
        let original = w.linear(name)?;
        if original
            .quant
            .as_ref()
            .context("missing quantization")?
            .bits
            <= a.bits
            || (a.shared_only && !name.contains(".shared_expert."))
        {
            continue;
        }
        let (l, receipt) = resident_quant::calibrate(&original, moments, a.bits)?;
        let q = l
            .quant
            .as_ref()
            .context("calibrated quantization missing")?;
        quantization.insert(
            name.clone(),
            json!({"bits":q.bits,"group_size":q.group_size,"mode":q.mode}),
        );
        for (suffix, array) in [
            ("weight", l.weight),
            ("scales", l.scales.context("scales missing")?),
            ("biases", l.biases.context("biases missing")?),
        ] {
            let key = format!("{name}.{suffix}");
            w.tensors.insert(key.clone(), array.clone());
            arrays.insert(key, array);
        }
        eprintln!(
            "calibrated {name} {}->{} bits, clipped_groups={}/{}",
            receipt.source_bits, a.bits, receipt.clipped_groups, receipt.total_groups
        );
        receipts.insert(name.clone(), receipt);
    }
    ensure!(!arrays.is_empty(), "no projections selected for variant");
    std::fs::create_dir_all(&a.output)?;
    let tensor_path = a.output.join("tensors.safetensors");
    mlx_rs::Array::save_safetensors(arrays.iter(), None, &tensor_path)?;
    let report = json!({"environment":env,"model":a.model,"model_config_sha256":config_sha,
        "corpus_sha256":corpus_sha,"tokenizer_sha256":sha(&a.model.join("tokenizer.json"))?,
        "kind":"resident_affine_variant","type":"approximate target, original checkpoint unchanged",
        "bits":a.bits,"shared_only":a.shared_only,"quantization":quantization,"receipts":receipts,
        "tensors_sha256":sha(&tensor_path)?,"tensors_file":"tensors.safetensors",
        "calibration":"per-group clipping grid minimizes diagonal activation-weighted BF16 dequant reconstruction error; no full covariance/GPTQ or original BF16 weights",
        "protected":"routed expert banks, router, norms, HC, PLE, embedding, target head, GDN a/b and private MTP",
        "quality_qualified":false,"performance_qualified":false});
    std::fs::write(
        a.output.join("overlay.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    std::fs::write(
        format!(
            "results/decode3-resident-q{}-{}.json",
            a.bits,
            if a.shared_only { "shared" } else { "resident" }
        ),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("RESIDENT_OVERLAY_CREATED");
    Ok(())
}
