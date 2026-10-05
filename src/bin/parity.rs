use anyhow::{Result, ensure};
use clap::Parser;
use mlx_rs::{
    Dtype,
    ops::{self, indexing::IndexOp},
};
use rust_mlx::{
    dense::{DenseModel, KvCache},
    weights::Weights,
};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    oracle: PathBuf,
}
fn main() -> Result<()> {
    let a = Args::parse();
    let w = Weights::load(&a.model)?;
    let m = DenseModel::load(&w)?;
    let o: serde_json::Value = serde_json::from_slice(&std::fs::read(a.oracle)?)?;
    let prompt: Vec<u32> = serde_json::from_value(o["prompt"].clone())?;
    let expected: Vec<u32> = serde_json::from_value(o["tokens"].clone())?;
    let ref_logits: Vec<f32> = serde_json::from_value(o["prefill_logits"].clone())?;
    let mut cache = vec![KvCache::default(); m.layers.len()];
    let logits = m.forward(&prompt, &mut cache)?;
    let tail = logits.index((0, -1, ..)).as_dtype(Dtype::Float32)?;
    tail.eval()?;
    let actual = tail.as_slice::<f32>();
    let max_abs = actual
        .iter()
        .zip(&ref_logits)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    ensure!(actual.len() == ref_logits.len(), "logit size mismatch");
    ensure!(max_abs < 1e-4, "max logit error {max_abs}");
    let mut logits = tail;
    for (i, &expected) in expected.iter().enumerate() {
        let t = ops::indexing::argmax(&logits, false)?.item_exact::<u32>();
        ensure!(t == expected, "token mismatch at {i}: {t} vs {expected}");
        logits = m.forward(&[t], &mut cache)?.index((0, -1, ..));
    }
    println!(
        "DENSE_PARITY_PASSED tokens={} max_absolute_error={max_abs}",
        expected.len()
    );
    Ok(())
}
