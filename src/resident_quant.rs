//! Separate approximate resident-weight variants, calibrated with diagonal moments.
//! Existing checkpoint codes are dequantized; this is explicitly double quantization.
use crate::{
    calibration::ActivationMoments,
    weights::{Linear, Quantization},
};
use anyhow::{Context, Result, ensure};
use mlx_rs::{Array, Dtype, ops, ops::indexing::IndexOp};
use serde::Serialize;

pub fn sha256_file(path: &std::path::Path) -> Result<String> {
    let out = std::process::Command::new("shasum")
        .args(["-a", "256"])
        .arg(path)
        .output()?;
    ensure!(
        out.status.success(),
        "checksum failed for {}",
        path.display()
    );
    Ok(String::from_utf8(out.stdout)?
        .split_whitespace()
        .next()
        .context("empty checksum")?
        .to_owned())
}

/// Apply a research overlay atomically after validating every projection.
/// Only the named resident projections may change; no checkpoint is written.
pub fn apply_overlay(
    w: &mut crate::weights::Weights,
    model: &std::path::Path,
    directory: &std::path::Path,
) -> Result<serde_json::Value> {
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("overlay.json"))?)?;
    ensure!(
        report["kind"] == "resident_affine_variant"
            && report["tensors_file"] == "tensors.safetensors",
        "unsupported overlay schema"
    );
    ensure!(
        report["model_config_sha256"].as_str()
            == Some(sha256_file(&model.join("config.json"))?.as_str())
            && report["tokenizer_sha256"].as_str()
                == Some(sha256_file(&model.join("tokenizer.json"))?.as_str()),
        "overlay configuration/tokenizer mismatch"
    );
    let tensor_path = directory.join("tensors.safetensors");
    ensure!(
        report["tensors_sha256"].as_str() == Some(sha256_file(&tensor_path)?.as_str()),
        "overlay tensor checksum mismatch"
    );
    let arrays = Array::load_safetensors(&tensor_path)?;
    let modules = report["quantization"]
        .as_object()
        .context("overlay quantization missing")?;
    ensure!(
        !modules.is_empty() && arrays.len() == 3 * modules.len(),
        "overlay tensor count"
    );
    let mut config = w.config.clone();
    let config_key = if config.get("quantization").is_some() {
        "quantization"
    } else {
        "quantization_config"
    };
    for (name, metadata) in modules {
        let rest = name
            .strip_prefix("language_model.model.layers.")
            .context("overlay prefix")?;
        let (layer, projection) = rest.split_once('.').context("overlay projection prefix")?;
        let layer: usize = layer.parse().context("overlay layer index")?;
        let layer_count = w.config["text_config"]["num_hidden_layers"]
            .as_u64()
            .or_else(|| w.config["num_hidden_layers"].as_u64())
            .context("layer count missing")? as usize;
        ensure!(
            layer < layer_count
                && [
                    "mlp.shared_expert.gate_proj",
                    "mlp.shared_expert.up_proj",
                    "mlp.shared_expert.down_proj",
                    "linear_attn.in_proj_qkv",
                    "linear_attn.in_proj_z",
                    "linear_attn.out_proj",
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj"
                ]
                .contains(&projection),
            "protected overlay projection {name}"
        );
        let old = w.linear(name)?;
        let old_q = old.quant.as_ref().context("overlay source unquantized")?;
        let new_q: Quantization = serde_json::from_value(metadata.clone())?;
        ensure!(
            new_q.mode == "affine"
                && [4, 5, 6].contains(&new_q.bits)
                && new_q.bits < old_q.bits
                && new_q.group_size == old_q.group_size
                && old.weight.ndim() == 2
                && old.bias.is_none(),
            "overlay source geometry {name}"
        );
        ensure!(
            report["receipts"][name]["source_bits"].as_i64() == Some(old_q.bits as i64),
            "overlay receipt source bits {name}"
        );
        let n = old.weight.shape()[0];
        let k = old.weight.shape()[1] * 32 / old_q.bits;
        let code = arrays
            .get(&format!("{name}.weight"))
            .context("overlay codes missing")?;
        ensure!(
            code.dtype() == Dtype::Uint32 && code.shape() == [n, k * new_q.bits / 32],
            "overlay code shape {name}"
        );
        for suffix in ["scales", "biases"] {
            let array = arrays
                .get(&format!("{name}.{suffix}"))
                .context("overlay affine metadata missing")?;
            ensure!(
                array.dtype() == Dtype::Bfloat16 && array.shape() == [n, k / new_q.group_size],
                "overlay affine metadata shape {name}"
            );
        }
        config[config_key][name] = metadata.clone();
    }
    // Every array was accounted for by the exact count and checked required keys.
    w.tensors.extend(arrays);
    w.config = config;
    Ok(report)
}

#[derive(Serialize)]
pub struct CalibrationReceipt {
    pub source_bits: i32,
    pub bits: i32,
    pub group_size: i32,
    pub calibration_rows: usize,
    pub source_bytes: usize,
    pub candidate_bytes: usize,
    pub weighted_squared_error: f64,
    pub native_minmax_squared_error: f64,
    pub weighted_source_energy: f64,
    pub clipped_groups: usize,
    pub total_groups: usize,
    pub grid: Vec<f32>,
}
pub fn calibrate(
    source: &Linear,
    moments: &ActivationMoments,
    bits: i32,
) -> Result<(Linear, CalibrationReceipt)> {
    let q = source
        .quant
        .as_ref()
        .context("resident quantization missing")?;
    ensure!(
        source.weight.ndim() == 2
            && source.weight.dtype() == Dtype::Uint32
            && q.mode == "affine"
            && [5, 6, 8].contains(&q.bits)
            && [32, 64, 128].contains(&q.group_size)
            && [4, 5, 6].contains(&bits)
            && bits < q.bits,
        "unsupported resident requantization"
    );
    let sc = source.scales.as_ref().context("resident scales missing")?;
    let bs = source.biases.as_ref().context("resident biases missing")?;
    ensure!(
        sc.dtype() == Dtype::Bfloat16 && bs.dtype() == Dtype::Bfloat16 && source.bias.is_none(),
        "only BF16 affine resident projections are calibrated"
    );
    let k = source.weight.shape()[1] * 32 / q.bits;
    let n = source.weight.shape()[0];
    let groups = k / q.group_size;
    let words = q.group_size * bits / 32;
    ensure!(
        n > 0 && k > 0 && sc.shape() == [n, groups] && bs.shape() == sc.shape(),
        "invalid resident weight metadata"
    );
    ensure!(
        moments.rows > 0
            && moments.sum_squares.len() == k as usize
            && k % q.group_size == 0
            && q.group_size * bits % 32 == 0
            && moments
                .sum_squares
                .iter()
                .all(|v| v.is_finite() && *v >= 0.),
        "resident calibration geometry"
    );
    let energy = moments
        .sum_squares
        .iter()
        .map(|x| (*x / moments.rows as f64).max(1e-12) as f32)
        .collect::<Vec<_>>();
    ensure!(
        energy.iter().all(|e| e.is_finite()),
        "nonfinite calibration energy"
    );
    let energy = Array::from_slice(&energy, &[1, groups, q.group_size]);
    let grid = vec![1., 0.995, 0.99, 0.98, 0.95, 0.9];
    let mut receipt = CalibrationReceipt {
        source_bits: q.bits,
        bits,
        group_size: q.group_size,
        calibration_rows: moments.rows,
        source_bytes: source.weight.nbytes() + sc.nbytes() + bs.nbytes(),
        candidate_bytes: 0,
        weighted_squared_error: 0.,
        native_minmax_squared_error: 0.,
        weighted_source_energy: 0.,
        clipped_groups: 0,
        total_groups: n as usize * groups as usize,
        grid: grid.clone(),
    };
    let mut chunks = Vec::new();
    for start in (0..n).step_by(512) {
        let end = (start + 512).min(n);
        let rows = end - start;
        let original = ops::dequantize(
            source.weight.index((start..end, ..)),
            sc.index((start..end, ..)),
            Some(&bs.index((start..end, ..))),
            q.group_size,
            q.bits,
        )?;
        let grouped = original
            .as_dtype(Dtype::Float32)?
            .reshape(&[rows, groups, q.group_size])?;
        let low = grouped.min_axis(-1, true)?;
        let high = grouped.max_axis(-1, true)?;
        let center = low.add(&high)?.multiply(Array::from_f32(0.5))?;
        let radius = high.subtract(&low)?.multiply(Array::from_f32(0.5))?;
        let error = |w: &Array, s: &Array, b: &Array| -> Result<Array> {
            let reconstructed = ops::dequantize(w, s, Some(b), q.group_size, bits)?
                .as_dtype(Dtype::Float32)?
                .reshape(&[rows, groups, q.group_size])?;
            let difference = reconstructed.subtract(&grouped)?;
            Ok(difference
                .multiply(&difference)?
                .multiply(&energy)?
                .sum_axis(-1, false)?)
        };
        let (mut best_w, mut best_s, mut best_b) = ops::quantize(&original, q.group_size, bits)?;
        let mut best_error = error(&best_w, &best_s, &best_b)?;
        let mut selected = ops::zeros_dtype(&[rows, groups], Dtype::Int32)?;
        receipt.native_minmax_squared_error += best_error.sum(false)?.item_exact::<f32>() as f64;
        receipt.weighted_source_energy += grouped
            .multiply(&grouped)?
            .multiply(&energy)?
            .sum(false)?
            .item_exact::<f32>() as f64;
        for (i, &alpha) in grid.iter().enumerate().skip(1) {
            let extent = radius.multiply(Array::from_f32(alpha))?;
            let clip = ops::minimum(
                ops::maximum(&grouped, center.subtract(&extent)?)?,
                center.add(&extent)?,
            )?
            .reshape(&[rows, k])?
            .as_dtype(Dtype::Bfloat16)?;
            let (w, s, b) = ops::quantize(clip, q.group_size, bits)?;
            let candidate_error = error(&w, &s, &b)?;
            let mask = candidate_error.lt(&best_error)?;
            best_w = ops::select(
                &mask.expand_dims(-1)?,
                w.reshape(&[rows, groups, words])?,
                best_w.reshape(&[rows, groups, words])?,
            )?
            .reshape(&[rows, k * bits / 32])?;
            best_s = ops::select(&mask, &s, &best_s)?;
            best_b = ops::select(&mask, &b, &best_b)?;
            best_error = ops::select(&mask, &candidate_error, &best_error)?;
            selected = ops::select(&mask, Array::from_int(i as i32), &selected)?;
            mlx_rs::transforms::eval([&best_w, &best_s, &best_b, &best_error, &selected])?;
        }
        receipt.weighted_squared_error += best_error.sum(false)?.item_exact::<f32>() as f64;
        receipt.clipped_groups += selected
            .gt(Array::from_int(0))?
            .as_dtype(Dtype::Int32)?
            .sum(false)?
            .item_exact::<i32>() as usize;
        chunks.push((best_w, best_s, best_b));
    }
    ensure!(
        receipt.weighted_squared_error.is_finite()
            && receipt.weighted_squared_error
                <= receipt.native_minmax_squared_error * 1.000001 + 1e-12,
        "calibration increased measured weighted error"
    );
    let l = Linear {
        weight: ops::concatenate(&chunks.iter().map(|c| &c.0).collect::<Vec<_>>(), 0)?,
        scales: Some(ops::concatenate(
            &chunks.iter().map(|c| &c.1).collect::<Vec<_>>(),
            0,
        )?),
        biases: Some(ops::concatenate(
            &chunks.iter().map(|c| &c.2).collect::<Vec<_>>(),
            0,
        )?),
        bias: None,
        quant: Some(Quantization {
            bits,
            group_size: q.group_size,
            mode: "affine".into(),
        }),
    };
    mlx_rs::transforms::eval([
        &l.weight,
        l.scales.as_ref().unwrap(),
        l.biases.as_ref().unwrap(),
    ])?;
    receipt.candidate_bytes = l.weight.nbytes()
        + l.scales.as_ref().unwrap().nbytes()
        + l.biases.as_ref().unwrap().nbytes();
    Ok((l, receipt))
}
