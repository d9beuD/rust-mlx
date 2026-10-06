//! Actual checkpoint TensorOps feasibility and paired component benchmark.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use mlx_rs::{Array, Dtype, ops::indexing::IndexOp};
use rust_mlx::{
    environment::BenchmarkEnvironment,
    matrix_kernel::{self, InputMode},
    weights::Weights,
};
use serde_json::json;
use std::{collections::HashMap, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 30)]
    repetitions: usize,
    #[arg(long)]
    quick: bool,
    #[arg(long)]
    affine: bool,
    #[arg(long)]
    compact: bool,
    #[arg(long)]
    packed: bool,
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        matrix_kernel::supported(),
        "these experiments require Apple M5 and macOS27"
    );
    ensure!(
        args.repetitions >= 2,
        "at least two alternating pairs required"
    );
    let environment = BenchmarkEnvironment::capture()?;
    let identities = std::process::Command::new("shasum")
        .args([
            "-a",
            "256",
            "src/matrix_kernel.rs",
            "src/bin/matrix-bench.rs",
            "src/weights.rs",
            "kernels/matrix_verify.h",
            "kernels/matrix_verify.metal",
            "kernels/matrix_affine.metal",
            "kernels/matrix_packed.metal",
            "scripts/target_components.py",
            "results/target-layer0-oracle.safetensors",
        ])
        .output()?;
    ensure!(
        identities.status.success(),
        "cannot identify component sources"
    );
    let source_file_sha256 = String::from_utf8(identities.stdout)?;
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(args.model.join("config.json"))?)?;
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(
        args.model.join("model.safetensors.index.json"),
    )?)?;
    let fixture = Array::load_safetensors("results/target-layer0-oracle.safetensors")?;
    let cases = if args.quick && (args.affine || args.packed) {
        vec![("language_model.model.layers.3.self_attn.q_proj", "mixed")]
    } else if args.quick {
        vec![(
            "language_model.model.layers.0.linear_attn.in_proj_qkv",
            "mixed",
        )]
    } else {
        vec![
            (
                "language_model.model.layers.0.linear_attn.in_proj_qkv",
                "mixed",
            ),
            (
                "language_model.model.layers.0.linear_attn.in_proj_z",
                "mixed",
            ),
            ("language_model.model.layers.3.self_attn.q_proj", "mixed"),
            (
                "language_model.model.layers.0.attn_hyper_connection.input_mix_weight_down",
                "input",
            ),
            ("language_model.lm_head", "mlp_input"),
        ]
    };
    let mut records = Vec::new();
    for (prefix, input) in cases {
        let filename = index["weight_map"][format!("{prefix}.weight")]
            .as_str()
            .context("missing checkpoint module")?;
        let shard = Array::load_safetensors(args.model.join(filename))?;
        let tensors: HashMap<_, _> = shard
            .into_iter()
            .filter(|(name, _)| name.starts_with(&format!("{prefix}.")))
            .collect();
        let weights = Weights {
            tensors,
            config: config.clone(),
        };
        let linear = weights.linear(prefix)?;
        for t in [2, 3, 4] {
            let x = fixture[input].index((.., ..t, ..)).contiguous()?;
            x.eval()?;
            // Singleton native QMV preserves the target verifier arithmetic.
            let native = mlx_rs::ops::concatenate(
                &(0..t)
                    .map(|r| linear.forward(&x.index((.., r..r + 1, ..))))
                    .collect::<Result<Vec<_>>>()?,
                1,
            )?;
            let native_f = native.as_dtype(Dtype::Float32)?.contiguous()?;
            native_f.eval()?;
            for splits in if args.packed {
                if args.quick {
                    vec![8]
                } else {
                    vec![2, 4, 8, 16]
                }
            } else if args.affine {
                vec![if args.compact { 8 } else { 32 }]
            } else if args.quick {
                vec![4]
            } else {
                vec![1, 2, 4, 8]
            } {
                let modes = if args.packed {
                    vec![InputMode::Packed]
                } else if args.affine {
                    vec![if args.compact {
                        InputMode::AffineCompact
                    } else {
                        InputMode::AffineRegisters
                    }]
                } else if args.compact {
                    vec![InputMode::CompactRegisters]
                } else {
                    vec![InputMode::Staged, InputMode::Registers]
                };
                for mode in modes {
                    let Some(output) = matrix_kernel::project(&linear, &x, mode, splits)? else {
                        continue;
                    };
                    let y = output.as_dtype(Dtype::Float32)?.contiguous()?;
                    y.eval()?;
                    let a = native_f.as_slice::<f32>();
                    let b = y.as_slice::<f32>();
                    ensure!(
                        a.len() == b.len() && b.iter().all(|v| v.is_finite()),
                        "invalid matrix output"
                    );
                    let max_error = a
                        .iter()
                        .zip(b)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0., f32::max);
                    let different = a.iter().zip(b).filter(|(a, b)| a != b).count();
                    let mut samples = [Vec::new(), Vec::new()];
                    for cycle in 0..args.repetitions + 5 {
                        for candidate in if cycle % 2 == 0 {
                            [false, true]
                        } else {
                            [true, false]
                        } {
                            let start = Instant::now();
                            let out = if candidate {
                                matrix_kernel::project(&linear, &x, mode, splits)?
                                    .context("shape changed")?
                            } else {
                                linear.forward_rows(&x)?
                            };
                            out.eval()?;
                            if cycle >= 5 {
                                samples[candidate as usize].push(start.elapsed().as_secs_f64());
                            }
                        }
                    }
                    eprintln!(
                        "{prefix} T{t} {mode:?} split{splits} exact={} different={different}/{} max={max_error}",
                        different == 0,
                        a.len()
                    );
                    records.push(json!({"module":prefix,"input":input,"rows":t,"input_shape":x.shape(),"weight_shape":linear.weight.shape(),
                        "bits":linear.quant.as_ref().map(|q|q.bits),"group_size":linear.quant.as_ref().map(|q|q.group_size),
                        "input_dtype":format!("{:?}",x.dtype()),"mode":format!("{mode:?}"),"splits":splits,"exact":different==0,
                        "different_elements":different,"elements":a.len(),"max_abs_error":max_error,
                        "original_seconds":samples[0],"candidate_seconds":samples[1]}));
                    std::fs::write(
                        &args.output,
                        serde_json::to_vec_pretty(
                            &json!({"environment":environment,"model":args.model,
                        "quantization":config["quantization"],"sampler":"none; projection comparison","mtp":"T2/T3/T4 component only",
                        "prefix_cache":false,"batch_size":1,"repetitions":args.repetitions,"warmup_pairs":5,
                        "reference_exactness":"native singleton MLX QMV","reference_timing":"current selected forward_rows, shared QMV where eligible",
                        "input_source":"target checkpoint layer0 stages from synthetic sin(j*0.013) BF16[1,10,10240]; cross-module probes reuse compatible stage values; no token prompt or generated output",
                        "source_file_sha256":source_file_sha256,"synthetic_positions":10,
                        "prompt_ids":null,"prompt_length":null,"output_ids":null,"output_length":null,
                        "instrumented":std::env::var("MTL_SHADER_VALIDATION").is_ok_and(|v|v=="1"),
                        "benchmarks":records,"complete":false}),
                        )?,
                    )?;
                }
            }
        }
    }
    let mut report: serde_json::Value = serde_json::from_slice(&std::fs::read(&args.output)?)?;
    report["complete"] = json!(true);
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    println!("MATRIX_COMPONENT_STUDY_COMPLETE");
    Ok(())
}
