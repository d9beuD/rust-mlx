use anyhow::{Result, ensure};
use mlx_rs::{Array, Dtype};
use rust_mlx::{
    environment::BenchmarkEnvironment,
    gdn_kernel,
    hybrid::gdn_reference,
    ngram::{NGramHasher, NGramTable},
    weights::Weights,
};
use std::{path::Path, time::Instant};
fn stats(v: &[f64]) -> serde_json::Value {
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    serde_json::json!({"samples":v,"mean_seconds":mean,"standard_deviation_seconds":sd,"repetitions":v.len()})
}
fn exact(a: &Array, b: &Array) -> Result<()> {
    let a = a.as_dtype(Dtype::Float32)?.contiguous()?;
    let b = b.as_dtype(Dtype::Float32)?.contiguous()?;
    mlx_rs::transforms::eval([&a, &b])?;
    ensure!(
        a.as_slice::<f32>() == b.as_slice::<f32>(),
        "candidate changed lookup output"
    );
    Ok(())
}
fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("model path");
    let path = Path::new(&path);
    let w = Weights::load(path)?;
    let p = "language_model.model.layers.1.ple.ple_embedding";
    let hash = NGramHasher::load(&w, p, 3, 8, 248044)?;
    let table = NGramTable::load(path, &format!("{p}.ngram_embedding"), &w.config)?;
    let mut reports = Vec::new();
    for len in [1, 128] {
        let ids = (0..len).map(|i| 7734 + i as u32 * 37).collect::<Vec<_>>();
        let rows = hash.rows(&ids, &[709, 421])?;
        let shape = [1, len, 2560];
        let a = table.gather_reference(&rows, &shape)?;
        let b = table
            .gather_batch(&rows, &shape)?
            .expect("uniform affine checkpoint");
        exact(&a, &b)?;
        let mut base = Vec::new();
        let mut batch = Vec::new();
        for i in 0..110 {
            for candidate in if i % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let start = Instant::now();
                let y = if candidate {
                    table.gather_batch(&rows, &shape)?.unwrap()
                } else {
                    table.gather_reference(&rows, &shape)?
                };
                y.eval()?;
                let t = start.elapsed().as_secs_f64();
                if i >= 10 {
                    if candidate {
                        batch.push(t)
                    } else {
                        base.push(t)
                    }
                }
            }
        }
        reports.push(serde_json::json!({"operation":"mmap PLE gather","tokens":len,"rows":rows.len(),"exact":true,"baseline":stats(&base),"candidate":stats(&batch)}));
    }
    let f = Array::load_safetensors("tests/fixtures/native-kernels.safetensors")?;
    let mut base = Vec::new();
    let mut native = Vec::new();
    for i in 0..110 {
        for candidate in if i % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let start = Instant::now();
            let (y, s) = if candidate {
                gdn_kernel::recurrent(
                    &f["q"],
                    &f["k"],
                    &f["v"],
                    &f["g"],
                    &f["beta"],
                    &f["initial"],
                )?
            } else {
                gdn_reference(
                    &f["q"],
                    &f["k"],
                    &f["v"],
                    &f["g"],
                    &f["beta"],
                    &f["initial"],
                )?
            };
            mlx_rs::transforms::eval([&y, &s])?;
            let t = start.elapsed().as_secs_f64();
            if i >= 10 {
                if candidate {
                    native.push(t)
                } else {
                    base.push(t)
                }
            }
        }
    }
    reports.push(serde_json::json!({"operation":"GDN decode actual dimensions","shape":[1,1,48,128,128],"note":"Native Metal matches oracle exactly; ops fallback has a different floating-point reduction order.","ops":stats(&base),"native_metal":stats(&native)}));
    let report = serde_json::json!({"environment":BenchmarkEnvironment::capture()?,"model":path,"quantization":w.config["quantization"],"benchmarks":reports,"timing":"wall-clock construction plus synchronized eval; alternated; 10 warmup pairs then 100 recorded pairs; same static rows warm in OS page cache"});
    std::fs::write(
        "results/microbench.json",
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("MICROBENCH_SAVED");
    Ok(())
}
