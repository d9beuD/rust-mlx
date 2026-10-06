use anyhow::{Context, Result, ensure};
use clap::Parser;
use mlx_rs::{
    Dtype,
    ops::indexing::{self, IndexOp},
};
use rust_mlx::{hybrid::HybridModel, weights::Weights};
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    oracle: PathBuf,
    #[arg(long, default_value_t = 1e-4)]
    max_error: f32,
    #[arg(long)]
    dump: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    prefill_chunk: usize,
}
fn main() -> Result<()> {
    let a = Args::parse();
    let w = Weights::load(&a.model)?;
    let m = HybridModel::load(&w, &a.model)?;
    let o: serde_json::Value = serde_json::from_slice(&std::fs::read(a.oracle)?)?;
    let prompt: Vec<u32> = serde_json::from_value(o["prompt"].clone())?;
    let expected: Vec<u32> = serde_json::from_value(o["tokens"].clone())?;
    let reference: Vec<f32> = serde_json::from_value(o["prefill_logits"].clone())?;
    let mut cache = m.make_cache();
    let chunk = if a.prefill_chunk == 0 {
        prompt.len()
    } else {
        a.prefill_chunk
    };
    let mut logits = None;
    for ids in prompt.chunks(chunk) {
        let (l, _) = m.forward(ids, &mut cache)?;
        l.eval()?;
        logits = Some(l);
    }
    let logits = logits.context("empty prompt")?;
    let mut tail = logits.index((0, -1, ..));
    let f32 = tail.as_dtype(Dtype::Float32)?;
    f32.eval()?;
    let actual = f32.as_slice::<f32>();
    ensure!(actual.len() == reference.len(), "vocabulary size mismatch");
    ensure!(
        actual.iter().chain(&reference).all(|v| v.is_finite()),
        "non-finite oracle parity input"
    );
    if let Some(p) = a.dump {
        std::fs::write(p, serde_json::to_vec(actual)?)?;
    }
    let max_abs = actual
        .iter()
        .zip(&reference)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    let mean_abs = actual
        .iter()
        .zip(&reference)
        .map(|(x, y)| (x - y).abs() as f64)
        .sum::<f64>()
        / actual.len() as f64;
    eprintln!("prefill logits max_absolute_error={max_abs} mean_absolute_error={mean_abs}");
    ensure!(
        max_abs <= a.max_error,
        "prefill error {max_abs} exceeds {}",
        a.max_error
    );
    for (i, &e) in expected.iter().enumerate() {
        let token = indexing::argmax(&tail, false)?.item_exact::<u32>();
        ensure!(token == e, "token mismatch at {i}: {token} vs {e}");
        let (logits, _) = m.forward(&[token], &mut cache)?;
        tail = logits.index((0, -1, ..));
    }
    println!(
        "HYBRID_PARITY_PASSED tokens={} max_absolute_error={max_abs}",
        expected.len()
    );
    Ok(())
}
