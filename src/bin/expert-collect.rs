//! Capture real selected expert inputs from an exact eight-position target verifier.
use anyhow::{Result, ensure};
use clap::Parser;
use mlx_rs::ops::indexing::{self, IndexOp};
use rust_mlx::{hybrid::HybridModel, weights::Weights};
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    destination: PathBuf,
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(!a.destination.exists(), "preserve fixture");
    mlx_rs::memory::set_memory_limit(100 * 1024usize.pow(3))?;
    mlx_rs::memory::set_cache_limit(512 * 1024usize.pow(2))?;
    let w = Weights::load(&a.model)?;
    let m = HybridModel::load(&w, &a.model)?;
    let prompt = [7734, 264, 2716, 32671, 709, 421, 55288, 76938, 4947, 13];
    let mut cache = m.make_cache();
    let (l, _) = m.forward(&prompt, &mut cache)?;
    let base = cache.clone();
    let mut logits = l.index((0, -1, ..));
    let mut tokens = Vec::new();
    for _ in 0..8 {
        let token = indexing::argmax(&logits, false)?.item_exact::<u32>();
        tokens.push(token);
        logits = m.forward(&[token], &mut cache)?.0.index((0, -1, ..));
        logits.eval()?;
    }
    let mut cache = base;
    rust_mlx::expert_capture::start();
    let result = rust_mlx::verification::with_mode(|| m.forward(&tokens, &mut cache));
    let (count, arrays) = rust_mlx::expert_capture::finish();
    result?.0.eval()?;
    ensure!(
        count == 48 && arrays.len() == 27,
        "incomplete expert capture {count}"
    );
    std::fs::create_dir_all(&a.destination)?;
    let path = a.destination.join("activations.safetensors");
    mlx_rs::Array::save_safetensors(arrays.iter().map(|(k, v)| (k.as_str(), v)), None, &path)?;
    std::fs::write(
        a.destination.join("metadata.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"model":a.model,"config_sha256":rust_mlx::resident_quant::sha256_file(&a.model.join("config.json"))?,"fixture_sha256":rust_mlx::resident_quant::sha256_file(&path)?,"prompt_ids":prompt,"tokens":tokens,"layers":[0,23,47],"rows":8,"environment":rust_mlx::environment::BenchmarkEnvironment::capture()?,"complete":true}),
        )?,
    )?;
    println!("EXPERT_ACTIVATIONS_COLLECTED");
    Ok(())
}
