use anyhow::{Result, ensure};
use clap::Parser;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    resident_overlay: Option<std::path::PathBuf>,
    #[arg(long)]
    native_gate_up: bool,
    #[arg(long)]
    matrix_affine: bool,
    #[arg(long, conflicts_with = "matrix_affine")]
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
    #[arg(long, default_value = "results/verify-parity.json")]
    output: std::path::PathBuf,
}
use mlx_rs::{
    Array, Dtype,
    ops::indexing::{self, IndexOp},
};
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
    rust_mlx::matrix_kernel::set_enabled(args.matrix_affine || args.matrix_packed);
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
    let (prompt, _) = m.forward(
        &[7734, 264, 2716, 32671, 709, 421, 55288, 76938, 4947, 13],
        &mut cache,
    )?;
    let mut tail = prompt.index((0, -1, ..));
    let mut records = Vec::new();
    for depth in [2, 3, 4, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7, 8] {
        rust_mlx::moe_layout::set_enabled(false);
        let mut batched = cache.clone();
        let mut tokens = Vec::new();
        let mut logits = Vec::new();
        let mut hidden = Vec::new();
        let started = std::time::Instant::now();
        for _ in 0..depth {
            let token = indexing::argmax(&tail, false)?.item_exact::<u32>();
            tokens.push(token);
            let (l, h) = m.forward(&[token], &mut cache)?;
            tail = l.index((0, -1, ..));
            tail.eval()?;
            logits.push(l);
            hidden.push(h);
        }
        let reference = started.elapsed().as_secs_f64();
        let matrix_before = rust_mlx::matrix_kernel::calls();
        let layout_before = rust_mlx::moe_layout::calls();
        rust_mlx::moe_layout::set_enabled(args.native_gate_up);
        let started = std::time::Instant::now();
        let (l, h) = verification::with_mode(|| m.forward(&tokens, &mut batched))?;
        l.eval()?;
        let candidate = started.elapsed().as_secs_f64();
        let rl = mlx_rs::ops::concatenate(&logits, 1)?;
        let rh = mlx_rs::ops::concatenate(&hidden, 1)?;
        let le = error(&l, &rl)?;
        let he = error(&h, &rh)?;
        let mut state = 0f32;
        ensure!(
            cache.offset == batched.offset && cache.history == batched.history,
            "CPU verifier cache mismatch"
        );
        for (a, b) in cache.layers.iter().zip(&batched.layers) {
            match (a, b) {
                (LayerCache::Linear(a), LayerCache::Linear(b)) => {
                    state = state.max(error(a.state.as_ref().unwrap(), b.state.as_ref().unwrap())?);
                    state = state.max(error(a.conv.as_ref().unwrap(), b.conv.as_ref().unwrap())?);
                }
                (LayerCache::Full(a), LayerCache::Full(b)) => {
                    ensure!(a.kv.offset == b.kv.offset, "KV verifier offset mismatch");
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
                            _ => anyhow::bail!("QSA verifier cache initialization differs"),
                        }
                    }
                }
                _ => anyhow::bail!("cache types"),
            }
        }
        for (a, b) in cache.ple.iter().zip(&batched.ple) {
            match (&a.conv, &b.conv) {
                (Some(x), Some(y)) => state = state.max(error(x, y)?),
                (None, None) => {}
                _ => anyhow::bail!("PLE verifier cache initialization differs"),
            }
        }
        println!(
            "depth={depth} logits={le} hidden={he} state={state} reference_ms={} candidate_ms={}",
            reference * 1e3,
            candidate * 1e3
        );
        records.push(serde_json::json!({"resident_overlay":args.resident_overlay,"native_gate_up":args.native_gate_up,"layout_calls":rust_mlx::moe_layout::calls(),"matrix_affine":args.matrix_affine,"matrix_packed":args.matrix_packed,"matrix_calls":rust_mlx::matrix_kernel::calls(),"sorted_moe":args.sorted_moe,"depth":depth,"logit_error":le,"hidden_error":he,"state_error":state,"reference_seconds":reference,"candidate_seconds":candidate,"tokens":tokens}));
        std::fs::write(&args.output, serde_json::to_vec_pretty(&records)?)?;
        if args.native_gate_up {
            ensure!(
                rust_mlx::moe_layout::calls() > layout_before,
                "native gate/up candidate did not engage"
            );
        }
        if (args.matrix_affine || args.matrix_packed) && (2..=4).contains(&depth) {
            ensure!(
                rust_mlx::matrix_kernel::calls() > matrix_before,
                "matrix candidate did not engage; cooperative inputs fall back under shader validation"
            );
        }
        ensure!(
            le == 0. && he == 0. && state == 0.,
            "verification is not exact"
        );
    }
    println!("VERIFY_PARITY_PASSED");
    Ok(())
}
