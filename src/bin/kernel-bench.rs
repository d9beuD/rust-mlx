use anyhow::{Result, ensure};
use mlx_rs::{
    Array, Dtype,
    ops::{self, indexing::IndexOp},
};
use rust_mlx::{
    environment::BenchmarkEnvironment,
    gdn_kernel,
    hybrid::{HybridConfig, MoE},
    weights::Weights,
};
use std::time::Instant;
fn exact(a: &Array, b: &Array) -> Result<()> {
    let a = a.as_dtype(Dtype::Float32)?.contiguous()?;
    let b = b.as_dtype(Dtype::Float32)?.contiguous()?;
    mlx_rs::transforms::eval([&a, &b])?;
    ensure!(
        a.shape() == b.shape() && a.as_slice::<f32>() == b.as_slice::<f32>(),
        "kernel output differs"
    );
    Ok(())
}
fn main() -> Result<()> {
    let f = Array::load_safetensors("tests/fixtures/native-kernels.safetensors")?;
    let mut reports = Vec::new();
    for t in [1, 4, 128] {
        for history in [false, true] {
            let q = ops::tile(&f["q"], &[1, t, 1, 1])?.contiguous()?;
            let k = ops::tile(&f["k"], &[1, t, 1, 1])?.contiguous()?;
            let v = ops::tile(&f["v"], &[1, t, 1, 1])?.contiguous()?;
            let g = ops::tile(&f["g"], &[1, t, 1])?.contiguous()?;
            let b = ops::tile(&f["beta"], &[1, t, 1])?.contiguous()?;
            let s = &f["initial"];
            mlx_rs::transforms::eval([&q, &k, &v, &g, &b, s])?;
            let run = |candidate| -> Result<Vec<Array>> {
                if candidate {
                    gdn_kernel::packed(&q, &k, &v, &g, &b, s, history)
                } else if history {
                    let (y, s, h) = gdn_kernel::recurrent_with_history(&q, &k, &v, &g, &b, s)?;
                    Ok(vec![y, s, h])
                } else {
                    let (y, s) = gdn_kernel::recurrent(&q, &k, &v, &g, &b, s)?;
                    Ok(vec![y, s])
                }
            };
            for (a, b) in run(false)?.iter().zip(run(true)?) {
                exact(a, &b)?;
            }
            let mut samples = [Vec::new(), Vec::new()];
            for i in 0..110 {
                for candidate in if i % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let start = Instant::now();
                    mlx_rs::transforms::eval(run(candidate)?.iter())?;
                    if i >= 10 {
                        samples[candidate as usize].push(start.elapsed().as_secs_f64());
                    }
                }
            }
            reports.push(serde_json::json!({"operation":"packed GDN","tokens":t,"history":history,"shape":[1,t,48,128,128],"input":"repeat of independent real-weight BF16 fixture, nonzero initial state","exact":true,"seconds":samples}));
        }
    }
    if let Some(path) = std::env::args().nth(1) {
        let w = Weights::load(std::path::Path::new(&path))?;
        let c: HybridConfig = serde_json::from_value(w.config["text_config"].clone())?;
        let m = MoE::load(&w, "language_model.model.layers.0.mlp", &c)?;
        let f = Array::load_safetensors("results/target-layer0-oracle.safetensors")?;
        for t in [1, 4, 8] {
            let x = f["mlp_input"].index((.., ..t, ..)).contiguous()?;
            x.eval()?;
            m.fused_mode.set(false);
            let a = m.forward(&x)?;
            m.fused_mode.set(true);
            let b = m.forward(&x)?;
            exact(&a, &b)?;
            let mut samples = [Vec::new(), Vec::new()];
            for i in 0..110 {
                for candidate in if i % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    m.fused_mode.set(candidate);
                    let start = Instant::now();
                    m.forward(&x)?.eval()?;
                    if i >= 10 {
                        samples[candidate as usize].push(start.elapsed().as_secs_f64());
                    }
                }
            }
            reports.push(serde_json::json!({"operation":"fused MoE gate/up","tokens":t,"shape":[1,t,2560],"input":"independent real layer0 mlp_input","exact":true,"seconds":samples}));
        }
    }
    std::fs::write(
        "results/kernel-bench.json",
        serde_json::to_vec_pretty(
            &serde_json::json!({"environment":BenchmarkEnvironment::capture()?,"timing":"wall graph+sync eval; ten warmup then 100 alternating pairs per configuration","benchmarks":reports}),
        )?,
    )?;
    println!("KERNEL_BENCH_SAVED");
    Ok(())
}
