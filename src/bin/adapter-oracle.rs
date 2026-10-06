//! File-based offline Python/Rust residual oracle; no model loading or serving.
use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    fixture: PathBuf,
    #[arg(long)]
    output: PathBuf,
}
fn main() -> Result<()> {
    let a = Args::parse();
    let mut t = mlx_rs::Array::load_safetensors(&a.fixture)?;
    let weights = t.remove("a").context("a")?;
    let b = t.remove("b").context("b")?;
    let mixed = t.get("mixed").context("mixed")?;
    let previous = t.get("previous").context("previous")?;
    let embedding = t.get("embedding").context("embedding")?;
    let adapter = rust_mlx::draft_adapter::DraftAdapter::new(
        weights,
        b,
        mixed.shape()[2],
        previous.shape()[2] / mixed.shape()[2],
    )?;
    let actual = adapter.apply(mixed, previous, embedding)?;
    mlx_rs::Array::save_safetensors([("actual", &actual)], None, &a.output)?;
    println!("ADAPTER_ORACLE_EXECUTED");
    Ok(())
}
