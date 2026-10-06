use anyhow::{Result, ensure};
use clap::Parser;
use mlx_rs::{Array, Dtype, ops::indexing::IndexOp};
use rust_mlx::{environment::BenchmarkEnvironment, qmv_kernel, weights::Weights};
use std::time::Instant;
#[derive(Parser)]
struct Args {
    model: std::path::PathBuf,
    #[arg(long)]
    ab_stream_x: bool,
    #[arg(long, conflicts_with = "ab_stream_x")]
    ab_gemv: bool,
    #[arg(long, requires = "ab_gemv")]
    gemv_narrow: bool,
    #[arg(long, default_value = "results/qmv-bench.json")]
    output: std::path::PathBuf,
}
fn exact(a: &Array, b: &Array) -> Result<()> {
    let a = a.as_dtype(Dtype::Float32)?.contiguous()?;
    let b = b.as_dtype(Dtype::Float32)?.contiguous()?;
    mlx_rs::transforms::eval([&a, &b])?;
    ensure!(
        a.shape() == b.shape() && a.as_slice::<f32>() == b.as_slice::<f32>(),
        "projection differs"
    );
    Ok(())
}
fn main() -> Result<()> {
    let args = Args::parse();
    let path = args.model;
    let w = Weights::load(&path)?;
    let f = Array::load_safetensors("results/target-layer0-oracle.safetensors")?;
    let mut reports = Vec::new();
    let set_mode = |candidate| {
        if args.ab_gemv {
            rust_mlx::gemv_kernel::set_enabled(candidate);
            rust_mlx::gemv_kernel::set_narrow(args.gemv_narrow);
        } else {
            qmv_kernel::set_enabled(args.ab_stream_x || candidate);
            qmv_kernel::set_stream_x(args.ab_stream_x && candidate);
        }
    };
    for (name, input) in [
        ("language_model.model.layers.0.mlp.gate", "mlp_input"),
        ("language_model.lm_head", "mlp_input"),
        (
            "language_model.model.layers.0.attn_hyper_connection.input_mix_weight_down",
            "input",
        ),
        (
            "language_model.model.layers.0.linear_attn.in_proj_qkv",
            "mixed",
        ),
        (
            "language_model.model.layers.0.linear_attn.in_proj_z",
            "mixed",
        ),
        (
            "language_model.model.layers.0.linear_attn.in_proj_a",
            "mixed",
        ),
        ("language_model.model.layers.3.self_attn.q_proj", "mixed"),
    ] {
        if args.ab_gemv != name.ends_with(".mlp.gate") {
            continue;
        }
        let l = w.linear(name)?;
        if args.ab_gemv {
            l.weight.eval()?;
        }
        for t in [2, 4, 8] {
            let x = f[input].index((.., ..t, ..)).contiguous()?;
            x.eval()?;
            set_mode(false);
            let a = l.forward_rows(&x)?;
            set_mode(true);
            let launches = rust_mlx::gemv_kernel::launches();
            let b = l.forward_rows(&x)?;
            if args.ab_gemv {
                ensure!(
                    rust_mlx::gemv_kernel::launches() == launches + 1,
                    "GEMV candidate not engaged"
                );
            }
            exact(&a, &b)?;
            let mut samples = [Vec::new(), Vec::new()];
            for i in 0..110 {
                for candidate in if i % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    set_mode(candidate);
                    let start = Instant::now();
                    l.forward_rows(&x)?.eval()?;
                    if i >= 10 {
                        samples[candidate as usize].push(start.elapsed().as_secs_f64());
                    }
                }
            }
            reports.push(serde_json::json!({"operation":if args.ab_gemv{"shared-weight unquantized router GEMV"}else{"shared-weight verifier QMV"},"module":name,"tokens":t,"weight_shape":l.weight.shape(),"input_shape":x.shape(),"weight_dtype":format!("{:?}",l.weight.dtype()),"input_dtype":format!("{:?}",x.dtype()),"quantization":l.quant.as_ref().map(|q|serde_json::json!({"bits":q.bits,"group_size":q.group_size,"mode":q.mode})),"exact":true,"seconds":samples}));
        }
    }
    std::fs::write(
        args.output,
        serde_json::to_vec_pretty(
            &serde_json::json!({"environment":BenchmarkEnvironment::capture()?,"model":path,"ab_stream_x":args.ab_stream_x,"ab_gemv":args.ab_gemv,"gemv_narrow":args.gemv_narrow,"timing":"wall graph+sync eval, ten warmup then100 alternating pairs; real BF16 layer0 input and checkpoint projections","benchmarks":reports}),
        )?,
    )?;
    println!("QMV_BENCH_SAVED");
    Ok(())
}
