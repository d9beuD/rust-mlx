//! Teacher-forced draft distillation pilot; full greedy verification remains required.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use mlx_rs::{Array, ops::indexing::IndexOp};
use rust_mlx::{
    hybrid::HybridModel, mtp::Mtp, qsa::QsaCache, resident_quant::sha256_file, weights::Weights,
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
    destination: PathBuf,
    #[arg(long)]
    output: PathBuf,
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(!a.destination.exists(), "preserve prior logits");
    let corpus: serde_json::Value = serde_json::from_slice(&std::fs::read(&a.corpus)?)?;
    let mut languages = HashSet::new();
    let fit = corpus["calibration"]
        .as_array()
        .context("fit missing")?
        .iter()
        .filter(|d| languages.insert(d["language"].as_str().unwrap_or("unknown")))
        .collect::<Vec<_>>();
    let heldout = corpus["heldout"].as_array().context("heldout missing")?;
    let hashes = fit
        .iter()
        .filter_map(|d| d["document_sha256"].as_str())
        .collect::<HashSet<_>>();
    ensure!(
        heldout.iter().all(|d| d["document_sha256"]
            .as_str()
            .is_some_and(|h| !hashes.contains(h))),
        "fit/heldout overlap"
    );
    std::fs::create_dir_all(&a.destination)?;
    mlx_rs::memory::set_memory_limit(100 * 1024usize.pow(3))?;
    mlx_rs::memory::set_cache_limit(512 * 1024usize.pow(2))?;
    let w = Weights::load(&a.model)?;
    let m = HybridModel::load(&w, &a.model)?;
    let draft = Mtp::load(&w, &m.config)?;
    let mut records = Vec::new();
    for (split, docs) in [("fit", fit), ("heldout", heldout.iter().collect())] {
        for (i, doc) in docs.iter().enumerate() {
            let tokens: Vec<u32> = serde_json::from_value(doc["tokens"].clone())?;
            ensure!(tokens.len() >= 64, "short document");
            let mut cache = m.make_cache();
            let (teacher, hidden) = m.forward(&tokens[..64], &mut cache)?;
            let emb = m
                .embedding
                .embedding(&Array::from_slice(&tokens[1..64], &[1, 63]))?;
            let (mixed, _) = draft.forward(
                &emb,
                &hidden.index((.., ..63, ..)),
                &mut QsaCache::default(),
                0,
            )?;
            // Hidden at t plus the observed token at t+1 predicts t+2.
            // Fixed positions32..47 use a target teacher distribution at33..48.
            let student = m
                .head
                .forward(&mixed.index((.., 32..48, ..)))?
                .contiguous()?;
            let teacher = teacher.index((.., 33..49, ..)).contiguous()?;
            mlx_rs::transforms::eval([&student, &teacher])?;
            let path = a.destination.join(format!("{split}-{i}.safetensors"));
            Array::save_safetensors([("draft", &student), ("teacher", &teacher)], None, &path)?;
            records.push(json!({"split":split,"document_sha256":doc["document_sha256"],"language":doc["language"],"prompt_ids":tokens[..64],"path":path,"sha256":sha256_file(&path)?,"shape":student.shape(),"dtype":"BF16"}));
            std::fs::write(
                &a.output,
                serde_json::to_vec_pretty(&json!({"complete":false,"records":records}))?,
            )?;
            eprintln!("DRAFT_COLLECT {split} {i}");
        }
    }
    std::fs::write(
        &a.output,
        serde_json::to_vec_pretty(
            &json!({"model":a.model,"environment":rust_mlx::environment::BenchmarkEnvironment::capture()?,"quantization":w.config["quantization"],"corpus_sha256":sha256_file(&a.corpus)?,"config_sha256":sha256_file(&a.model.join("config.json"))?,"tokenizer_sha256":sha256_file(&a.model.join("tokenizer.json"))?,"records":records,"complete":true,"warning":"teacher-forced private MTP contexts; not natural generation acceptance or a broad quality certificate"}),
        )?,
    )?;
    println!("DRAFT_PILOT_COLLECTION_PASSED");
    Ok(())
}
