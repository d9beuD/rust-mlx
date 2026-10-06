//! Actual affine g64 expert-down fusion and lossless four-output packing experiment.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use mlx_rs::{
    Array, Dtype,
    ops::{self, indexing::IndexOp},
};
use rust_mlx::{
    hybrid::{HybridConfig, MoE},
    metal::{Kernel, Launch},
    weights::{Linear, Weights},
};
use serde_json::json;
use std::{path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    allow_inexact: bool,
}
fn gather(l: &Linear, x: &Array, ids: &Array) -> Result<Array> {
    let q = l.quant.as_ref().context("quant missing")?;
    Ok(ops::gather_qmm(
        x,
        &l.weight,
        l.scales.as_ref().context("scales")?,
        l.biases.as_ref(),
        None,
        ids,
        true,
        q.group_size,
        q.bits,
        false,
    )?)
}
fn error(a: &Array, b: &Array) -> Result<f32> {
    let a = a.as_dtype(Dtype::Float32)?.contiguous()?;
    let b = b.as_dtype(Dtype::Float32)?.contiguous()?;
    mlx_rs::transforms::eval([&a, &b])?;
    ensure!(a.shape() == b.shape(), "shape");
    ensure!(
        a.as_slice::<f32>()
            .iter()
            .chain(b.as_slice::<f32>())
            .all(|x| x.is_finite()),
        "nonfinite"
    );
    Ok(a.as_slice::<f32>()
        .iter()
        .zip(b.as_slice::<f32>())
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max))
}
fn main() -> Result<()> {
    let a = Args::parse();
    let environment = rust_mlx::environment::BenchmarkEnvironment::capture()?;
    let w = Weights::load(&a.model)?;
    let c: HybridConfig = serde_json::from_value(w.config["text_config"].clone())?;
    let fixture = Array::load_safetensors("results/target-layer0-oracle.safetensors")?;
    eprintln!("Powered by MTPLX — https://github.com/youssofal/mtplx");
    let inputs = [
        "routed_h",
        "weights",
        "scales",
        "biases",
        "expert_ids",
        "route_scores",
    ];
    let k = Kernel::with_header(
        "rust_mlx_down_tail",
        &inputs,
        &["routed_down"],
        include_str!("../../kernels/moe_down_tail.metal"),
        include_str!("../../kernels/moe_down_tail.h"),
    )?;
    let packed_k = Kernel::with_header(
        "rust_mlx_down_word_packed",
        &inputs,
        &["routed_down"],
        include_str!("../../kernels/moe_down_packed.metal"),
        include_str!("../../kernels/moe_down_packed.h"),
    )?;
    let vector_k = Kernel::with_header(
        "rust_mlx_down_word_vector",
        &inputs,
        &["routed_down"],
        include_str!("../../kernels/moe_down_vector.metal"),
        include_str!("../../kernels/moe_down_vector.h"),
    )?;
    let mut records = Vec::new();
    let mut exact = true;
    for layer in [0, 23, 47] {
        let m = MoE::load(&w, &format!("language_model.model.layers.{layer}.mlp"), &c)?;
        let q = m.down.quant.as_ref().context("down quant")?;
        ensure!(
            q.bits == 4
                && q.group_size == 64
                && q.mode == "affine"
                && m.down.weight.shape() == [512, 2560, 80],
            "unsupported actual bank"
        );
        let started = Instant::now();
        let packed = m
            .down
            .weight
            .reshape(&[512, 640, 4, 80])?
            .transpose_axes(&[0, 1, 3, 2])?
            .contiguous()?;
        packed.eval()?;
        let preparation = started.elapsed().as_secs_f64();
        let unpacked = packed
            .transpose_axes(&[0, 1, 3, 2])?
            .reshape(m.down.weight.shape())?
            .contiguous()?;
        ensure!(
            error(&unpacked, &m.down.weight)? == 0.,
            "word packing loses codes"
        );
        for rows in [1, 2, 3, 4, 8] {
            let x = fixture["mlp_input"].index((.., ..rows, ..)).contiguous()?;
            let gates = ops::softmax_axis(&m.router.forward(&x)?, -1, true)?;
            let ids = ops::argpartition_axis(&gates, -10, -1)?
                .index((.., .., -10..))
                .contiguous()?;
            let scores = gates.take_along_axis(&ids, -1)?;
            let scores = scores.divide(scores.sum_axis(-1, true)?)?.contiguous()?;
            let xe = x.expand_dims(-2)?.expand_dims(-2)?;
            let routed = rust_mlx::compiled::swiglu(
                &gather(&m.gate, &xe, &ids)?,
                &gather(&m.up, &xe, &ids)?,
            )?
            .contiguous()?;
            let scales = m.down.scales.as_ref().context("scales missing")?;
            let biases = m.down.biases.as_ref().context("biases missing")?;
            mlx_rs::transforms::eval([&routed, &ids, &scores, scales, biases, &m.down.weight])?;
            let native = || -> Result<Array> {
                Ok(gather(&m.down, &routed, &ids)?
                    .squeeze_axes(&[-2])?
                    .multiply(scores.expand_dims(-1)?)?
                    .sum_axis(-2, false)?)
            };
            let run = |kernel: &Kernel, weight: &Array| -> Result<Array> {
                Ok(kernel
                    .launch(Launch {
                        inputs: &[&routed, weight, scales, biases, &ids, &scores],
                        templates: &[],
                        outputs: &[(&[1, rows, 2560], Dtype::Bfloat16)],
                        grid: [(2560 / 8) * 64, rows, 1],
                        group: [64, 1, 1],
                    })?
                    .remove(0))
            };
            let n = native()?;
            let down = run(&k, &m.down.weight)?;
            let p = run(&packed_k, &packed)?;
            let v = run(&vector_k, &packed)?;
            let errors = [
                error(&n, &down)?,
                error(&n, &p)?,
                error(&down, &p)?,
                error(&n, &v)?,
            ];
            exact &= errors[0] == 0. && errors[1] == 0. && errors[3] == 0.;
            let mut samples: [Vec<f64>; 4] = Default::default();
            if errors[0] == 0. && errors[1] == 0. && errors[3] == 0. && layer == 0 {
                for cycle in 0..110 {
                    for kind in if cycle % 2 == 0 {
                        [0, 1, 2, 3]
                    } else {
                        [3, 2, 1, 0]
                    } {
                        let start = Instant::now();
                        let y = match kind {
                            0 => native()?,
                            1 => run(&k, &m.down.weight)?,
                            2 => run(&packed_k, &packed)?,
                            _ => run(&vector_k, &packed)?,
                        };
                        y.eval()?;
                        if cycle >= 10 {
                            samples[kind].push(start.elapsed().as_secs_f64());
                        }
                    }
                }
            }
            records.push(json!({"layer":layer,"rows":rows,"ids":ids.as_slice::<u32>(),"routed_shape":routed.shape(),"weight_shape":m.down.weight.shape(),"packed_shape":packed.shape(),"code_bytes":packed.nbytes(),"retained_original_bytes":m.down.weight.nbytes(),"packing_seconds":preparation,"native_down_max_error":errors[0],"native_packed_max_error":errors[1],"packing_arithmetic_max_error":errors[2],"native_vector_max_error":errors[3],"native_seconds":samples[0],"down_seconds":samples[1],"packed_seconds":samples[2],"vector_seconds":samples[3]}));
            std::fs::write(
                &a.output,
                serde_json::to_vec_pretty(&json!({"complete":false,"records":records}))?,
            )?;
            eprintln!("DOWN_COMPONENT {layer} {rows} {errors:?}");
        }
    }
    std::fs::write(
        &a.output,
        serde_json::to_vec_pretty(
            &json!({"model":a.model,"environment":environment,"quantization":w.config["quantization"],"complete":true,"native_exact":exact,"records":records,"input":"synthetic sin(j*.013) BF16 layer0 component fixture reused on3 real expert banks; full-model qualification remains required","format":"lossless [512,2560,80] uint32 -> [512,640,80,4] interleaving; scale/bias untouched; no double quantization","comparison":"Native gather_qmm plus BF16 weighted reduction vs MTPLX g64 fused down; packed and unpacked custom versions use identical scalar/reduction arithmetic","memory":"benchmark retains original plus one packed bank for reference; production ownership/conversion cost not qualified","warmup_pairs":10,"measurement_pairs":100,"sampler":"component native softmax/top10","cache":"evaluated components; no prefix caching"}),
        )?,
    )?;
    ensure!(
        exact || a.allow_inexact,
        "down component differs; reject full-model promotion"
    );
    println!("DOWN_FORMAT_COMPONENT_STUDY_COMPLETED");
    Ok(())
}
