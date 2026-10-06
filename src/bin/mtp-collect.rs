//! On-policy target trajectories and frozen private MTP features; not serving.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use mlx_rs::{Array, ops::indexing::IndexOp};
use rust_mlx::{
    draft_adapter::DraftAdapter, hybrid::HybridModel, mtp::Mtp, qsa::QsaCache, weights::Weights,
};
use serde_json::json;
use std::path::PathBuf;
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
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(
        a.max_tokens > 0 && a.max_tokens <= 256 && !a.destination.exists(),
        "invalid collection/preserve destination"
    );
    let spec: serde_json::Value = serde_json::from_slice(&std::fs::read(&a.corpus)?)?;
    let rows = spec["records"].as_array().context("missing records")?;
    let mut documents = std::collections::HashSet::new();
    for row in rows {
        ensure!(
            documents.insert(
                row["document_sha256"]
                    .as_str()
                    .context("missing document hash")?
            ),
            "duplicate document"
        );
    }
    mlx_rs::memory::set_memory_limit(100 * 1024usize.pow(3))?;
    mlx_rs::memory::set_cache_limit(512 * 1024usize.pow(2))?;
    let w = Weights::load(&a.model)?;
    let m = HybridModel::load(&w, &a.model)?;
    let draft = Mtp::load(&w, &m.config)?;
    ensure!(
        spec["tokenizer_sha256"].as_str()
            == Some(
                rust_mlx::resident_quant::sha256_file(&a.model.join("tokenizer.json"))?.as_str()
            ),
        "tokenizer mismatch"
    );
    std::fs::create_dir_all(&a.destination)?;
    let mut head = vec![
        ("weight", &m.head.weight),
        ("scales", m.head.scales.as_ref().context("head scales")?),
        ("biases", m.head.biases.as_ref().context("head biases")?),
    ];
    if let Some(bias) = &m.head.bias {
        head.push(("bias", bias));
    }
    let head_path = a.destination.join("head.safetensors");
    Array::save_safetensors(head, None, &head_path)?;
    let mut report = json!({"complete":false,"model":a.model,"config_sha256":rust_mlx::resident_quant::sha256_file(&a.model.join("config.json"))?,"tokenizer_sha256":spec["tokenizer_sha256"],"corpus_sha256":rust_mlx::resident_quant::sha256_file(&a.corpus)?,"head_path":head_path,"head_sha256":rust_mlx::resident_quant::sha256_file(&head_path)?,"head_quantization":{"bits":m.head.quant.as_ref().context("head quantization")?.bits,"group_size":m.head.quant.as_ref().context("head quantization")?.group_size},"hidden":m.config.hidden_size,"hc":m.config.hc_count,"environment":rust_mlx::environment::BenchmarkEnvironment::capture()?,"records":[]});
    for (i, row) in rows.iter().enumerate() {
        let prompt: Vec<u32> = serde_json::from_value(row["tokens"].clone())?;
        ensure!(!prompt.is_empty(), "empty prompt");
        let mut cache = m.make_cache();
        let (logits, wide) = m.forward(&prompt, &mut cache)?;
        let mut token = greedy(&logits.index((0, -1, ..)))?;
        let mut shifted = prompt[1..].to_vec();
        shifted.push(token);
        let emb = m
            .embedding
            .embedding(&Array::from_slice(&shifted, &[1, shifted.len() as i32]))?;
        let mut private = QsaCache::default();
        let (mixed, _) = draft.forward(&emb, &wide, &mut private, 0)?;
        let mut mixed = mixed.index((.., -1.., ..));
        let mut previous = wide.index((.., -1.., ..));
        let mut embedding = emb.index((.., -1.., ..));
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut labels = Vec::new();
        let mut trajectory = vec![token];
        for step in 0..a.max_tokens {
            let features = DraftAdapter::features(&mixed, &previous, &embedding)?
                .reshape(&[1, -1])?
                .contiguous()?;
            let output = mixed.reshape(&[1, -1])?.contiguous()?;
            let (target, hidden) = m.forward(&[token], &mut cache)?;
            let next = greedy(&target.index((0, -1, ..)))?;
            mlx_rs::transforms::eval([&features, &output, &hidden])?;
            xs.push(features);
            ys.push(output);
            labels.push(next);
            trajectory.push(next);
            if step + 1 < a.max_tokens {
                previous = hidden;
                embedding = m
                    .embedding
                    .embedding(&Array::from_slice(&[next], &[1, 1]))?;
                mixed = draft
                    .forward(
                        &embedding,
                        &previous,
                        &mut private,
                        (prompt.len() + step) as i32,
                    )?
                    .0;
            }
            token = next;
        }
        let x = mlx_rs::ops::concatenate(&xs, 0)?;
        let mixed = mlx_rs::ops::concatenate(&ys, 0)?;
        let labels = Array::from_slice(&labels, &[labels.len() as i32]);
        let path = a.destination.join(format!("trajectory-{i}.safetensors"));
        Array::save_safetensors(
            [("features", &x), ("mixed", &mixed), ("labels", &labels)],
            None,
            &path,
        )?;
        report["records"].as_array_mut().unwrap().push(json!({"split":row["split"],"domain":row["domain"],"document_sha256":row["document_sha256"],"prompt_ids":prompt,"tokens":trajectory,"predictions":a.max_tokens,"path":path,"sha256":rust_mlx::resident_quant::sha256_file(&path)?}));
        std::fs::write(&a.output, serde_json::to_vec_pretty(&report)?)?;
        eprintln!("MTP_COLLECT {i}/{} {}", rows.len(), row["split"]);
    }
    report["complete"] = true.into();
    std::fs::write(&a.output, serde_json::to_vec_pretty(&report)?)?;
    println!("MTP_TRAJECTORY_COLLECTION_COMPLETED");
    Ok(())
}

fn greedy(x: &Array) -> Result<u32> {
    let id = mlx_rs::ops::indexing::argmax(x, false)?;
    id.eval()?;
    Ok(id.item_exact::<u32>())
}
