//! Small component capture; never capture all 68GiB target GPU weights.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use mlx_rs::{Array, ops::indexing::IndexOp};
use rust_mlx::{environment::BenchmarkEnvironment, metal::Capture, qmv_kernel, weights::Weights};
use serde_json::json;
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    output: PathBuf,
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(
        std::env::var("MTL_CAPTURE_ENABLED").ok().as_deref() == Some("1"),
        "launch with MTL_CAPTURE_ENABLED=1"
    );
    mlx_rs::memory::set_cache_limit(0)?;
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(a.model.join("config.json"))?)?;
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(
        a.model.join("model.safetensors.index.json"),
    )?)?;
    let mut files = HashSet::new();
    for name in ["weight", "scales", "biases"] {
        files.insert(
            index["weight_map"][format!("language_model.lm_head.{name}")]
                .as_str()
                .context("head shard missing")?
                .to_owned(),
        );
    }
    let mut tensors = HashMap::new();
    for file in files {
        for (name, tensor) in Array::load_safetensors(a.model.join(file))? {
            if name.starts_with("language_model.lm_head.") {
                tensors.insert(name, tensor);
            }
        }
    }
    let w = Weights { tensors, config };
    let head = w.linear("language_model.lm_head")?;
    mlx_rs::transforms::eval(w.tensors.values())?;
    let f = Array::load_safetensors("results/target-layer0-oracle.safetensors")?;
    let x = f["mlp_input"].index((.., ..4, ..)).contiguous()?;
    x.eval()?;
    drop(f);
    for _ in 0..4 {
        for candidate in [false, true] {
            qmv_kernel::set_enabled(candidate);
            head.forward_rows(&x)?.eval()?;
        }
    }
    mlx_rs::memory::clear_cache()?;
    let active = mlx_rs::memory::active_memory()?;
    ensure!(
        active < 2 * 1024usize.pow(3),
        "component capture exceeds2GiB resource budget"
    );
    let capture = Capture::start(&a.output)?;
    for candidate in [false, true] {
        qmv_kernel::set_enabled(candidate);
        head.forward_rows(&x)?.eval()?;
    }
    capture.finish()?;
    std::fs::write(
        a.output.with_extension("json"),
        serde_json::to_vec_pretty(
            &json!({"environment":BenchmarkEnvironment::capture()?,"model":a.model,"component":"actual 8-bit vocabulary head T4, native gather then shared-weight QMV","active_memory_bytes":active,"capture":a.output,"completed":true,"warning":"GPU component capture; instrumentation changes timings. Not a full-model hotspot profile."}),
        )?,
    )?;
    println!("COMPONENT_CAPTURE_SAVED");
    Ok(())
}
