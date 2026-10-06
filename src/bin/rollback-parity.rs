use anyhow::{Result, ensure};
use clap::Parser;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    resident_overlay: Option<std::path::PathBuf>,
    #[arg(long)]
    native_gate_up: bool,
    #[arg(long)]
    matrix_packed: bool,
    #[arg(long)]
    kv_blocks: bool,
    #[arg(long)]
    sorted_moe: bool,
    #[arg(
        long,
        default_value = "/Users/d9beud/.lmstudio/models/d9beuD/Qwen3.8-Flash-Next-oQ4e-mtp"
    )]
    model: std::path::PathBuf,
    #[arg(long, default_value = "results/rollback-parity.json")]
    output: std::path::PathBuf,
}
use mlx_rs::{Array, Dtype};
use rust_mlx::{
    hybrid::{HybridModel, LayerCache},
    verification,
    weights::Weights,
};
fn error(a: &Array, b: &Array) -> Result<f32> {
    let a = a.as_dtype(Dtype::Float32)?.contiguous()?;
    let b = b.as_dtype(Dtype::Float32)?.contiguous()?;
    mlx_rs::transforms::eval([&a, &b])?;
    ensure!(a.shape() == b.shape(), "shape mismatch");
    ensure!(
        a.as_slice::<f32>()
            .iter()
            .chain(b.as_slice::<f32>())
            .all(|v| v.is_finite()),
        "non-finite parity input"
    );
    Ok(a.as_slice::<f32>()
        .iter()
        .zip(b.as_slice::<f32>())
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max))
}
fn main() -> Result<()> {
    let args = Args::parse();
    rust_mlx::matrix_kernel::set_enabled(args.matrix_packed);
    rust_mlx::matrix_kernel::set_packed(args.matrix_packed);
    rust_mlx::kv_blocks::set_enabled(args.kv_blocks);
    let path = args.model.as_path();
    let mut w = Weights::load(path)?;
    if let Some(overlay) = &args.resident_overlay {
        rust_mlx::resident_quant::apply_overlay(&mut w, path, overlay)?;
    }
    let mut m = HybridModel::load(&w, path)?;
    if args.native_gate_up {
        rust_mlx::moe_layout::prepare_model(&mut m, &mut w)?;
    }
    for layer in &m.layers {
        layer.moe.sorted_mode.set(args.sorted_moe);
    }
    let mut cache = m.make_cache();
    m.forward(
        &[7734, 264, 2716, 32671, 709, 421, 55288, 76938, 4947, 13],
        &mut cache,
    )?
    .0
    .eval()?;
    let base = cache;
    let tokens = [271, 248068, 198, 760];
    let mut reports = Vec::new();
    for keep in 0..=4 {
        rust_mlx::moe_layout::set_enabled(false);
        let mut expected = base.clone();
        let mut candidate = base.clone();
        for &t in &tokens[..keep] {
            m.forward(&[t], &mut expected)?.0.eval()?;
        }
        let layout_before = rust_mlx::moe_layout::calls();
        rust_mlx::moe_layout::set_enabled(args.native_gate_up);
        verification::with_mode(|| m.forward(&tokens, &mut candidate))?
            .0
            .eval()?;
        candidate.commit_verified(&base, &tokens, keep, &m.config)?;
        ensure!(
            candidate.offset == expected.offset && candidate.history == expected.history,
            "CPU rollback mismatch"
        );
        rust_mlx::moe_layout::set_enabled(false);
        let (e, _) = m.forward(&[1156], &mut expected)?;
        rust_mlx::moe_layout::set_enabled(args.native_gate_up);
        let (a, _) = m.forward(&[1156], &mut candidate)?;
        let le = error(&e, &a)?;
        let mut state = 0f32;
        for (a, b) in expected.layers.iter().zip(&candidate.layers) {
            match (a, b) {
                (LayerCache::Linear(a), LayerCache::Linear(b)) => {
                    state = state.max(error(a.state.as_ref().unwrap(), b.state.as_ref().unwrap())?);
                    state = state.max(error(a.conv.as_ref().unwrap(), b.conv.as_ref().unwrap())?);
                }
                (LayerCache::Full(a), LayerCache::Full(b)) => {
                    ensure!(a.kv.offset == b.kv.offset, "KV rollback offset mismatch");
                    state = state.max(error(
                        a.kv.keys.as_ref().unwrap(),
                        b.kv.keys.as_ref().unwrap(),
                    )?);
                    state = state.max(error(
                        a.kv.values.as_ref().unwrap(),
                        b.kv.values.as_ref().unwrap(),
                    )?);
                    for (x, y) in [(&a.raw_keys, &b.raw_keys), (&a.blocks, &b.blocks)] {
                        match (x, y) {
                            (Some(x), Some(y)) => state = state.max(error(x, y)?),
                            (None, None) => {}
                            _ => anyhow::bail!("QSA rollback cache initialization differs"),
                        }
                    }
                }
                _ => anyhow::bail!("cache types"),
            }
        }
        for (a, b) in expected.ple.iter().zip(&candidate.ple) {
            if let (Some(a), Some(b)) = (&a.conv, &b.conv) {
                state = state.max(error(a, b)?);
            }
        }
        println!("keep={keep} logits={le} state={state}");
        if args.native_gate_up {
            ensure!(
                rust_mlx::moe_layout::calls() > layout_before,
                "native gate/up rollback did not engage"
            );
        }
        ensure!(le == 0. && state == 0., "rollback is not exact");
        reports.push(serde_json::json!({"resident_overlay":args.resident_overlay,"native_gate_up":args.native_gate_up,"layout_calls":rust_mlx::moe_layout::calls(),"matrix_packed":args.matrix_packed,"matrix_calls":rust_mlx::matrix_kernel::calls(),"sorted_moe":args.sorted_moe,"keep":keep,"logit_error":le,"state_error":state}));
    }
    std::fs::write(&args.output, serde_json::to_vec_pretty(&reports)?)?;
    println!("ROLLBACK_PARITY_PASSED");
    Ok(())
}
