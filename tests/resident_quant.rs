use anyhow::Result;
use mlx_rs::{Array, Dtype, ops, ops::indexing::IndexOp};
use rust_mlx::{
    calibration::ActivationMoments,
    resident_quant, verification,
    weights::{Linear, Quantization},
};
#[test]
fn calibrated_group_selection_improves_proxy_and_preserves_native_variant_rows() -> Result<()> {
    let (n, k) = (513, 128);
    let original = Array::from_iter((0..n * k).map(|i| (i as f32 * 0.17).sin() * 0.3), &[n, k])
        .as_dtype(Dtype::Bfloat16)?;
    let (weight, scales, biases) = ops::quantize(&original, 64, 8)?;
    let source = Linear {
        weight,
        scales: Some(scales),
        biases: Some(biases),
        bias: None,
        quant: Some(Quantization {
            bits: 8,
            group_size: 64,
            mode: "affine".into(),
        }),
    };
    let moments = ActivationMoments {
        rows: 8,
        sum_squares: (0..k)
            .map(|i| if i % 64 < 8 { 800. } else { 0.01 })
            .collect(),
    };
    for bits in [4, 5, 6] {
        let (candidate, receipt) = resident_quant::calibrate(&source, &moments, bits)?;
        assert!(receipt.weighted_squared_error <= receipt.native_minmax_squared_error * 1.000001);
        assert!(receipt.clipped_groups > 0);
        assert_eq!(candidate.quant.as_ref().unwrap().bits, bits);
        let x = Array::from_iter((0..3 * k).map(|i| (i as f32 * 0.019).cos()), &[1, 3, k])
            .as_dtype(Dtype::Bfloat16)?;
        let actual = verification::with_rows(|| candidate.forward(&x))?
            .as_dtype(Dtype::Float32)?
            .contiguous()?;
        let mut expected = Vec::new();
        for row in 0..3 {
            expected.push(ops::quantized_matmul(
                x.index((.., row..row + 1, ..)),
                &candidate.weight,
                candidate.scales.as_ref().unwrap(),
                candidate.biases.as_ref(),
                true,
                64,
                bits,
            )?);
        }
        let expected = ops::concatenate(&expected, 1)?
            .as_dtype(Dtype::Float32)?
            .contiguous()?;
        mlx_rs::transforms::eval([&actual, &expected])?;
        assert_eq!(actual.as_slice::<f32>(), expected.as_slice::<f32>());
        let ref_w = ops::dequantize(
            &source.weight,
            source.scales.as_ref().unwrap(),
            source.biases.as_ref(),
            64,
            8,
        )?
        .as_dtype(Dtype::Float32)?;
        let new_w = ops::dequantize(
            &candidate.weight,
            candidate.scales.as_ref().unwrap(),
            candidate.biases.as_ref(),
            64,
            bits,
        )?
        .as_dtype(Dtype::Float32)?;
        let d = ref_w.subtract(new_w)?;
        let energy = Array::from_iter(
            moments.sum_squares.iter().map(|x| (*x / 8.) as f32),
            &[1, k],
        );
        let measured = d
            .multiply(&d)?
            .multiply(energy)?
            .sum(false)?
            .item_exact::<f32>() as f64;
        assert!((measured - receipt.weighted_squared_error).abs() <= measured * 1e-6);
    }
    assert_eq!(source.quant.as_ref().unwrap().bits, 8);
    let bad = ActivationMoments {
        rows: 1,
        sum_squares: vec![f64::NAN; k as usize],
    };
    assert!(resident_quant::calibrate(&source, &bad, 4).is_err());
    assert!(resident_quant::calibrate(&source, &moments, 8).is_err());
    Ok(())
}

#[test]
fn overlay_roundtrip_and_protected_failure_are_atomic() -> Result<()> {
    use rust_mlx::{
        resident_quant::{apply_overlay, sha256_file},
        weights::Weights,
    };
    use serde_json::json;
    use std::collections::HashMap;
    let path = std::env::temp_dir().join(format!(
        "rust-mlx-overlay-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    std::fs::create_dir(&path)?;
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(path.clone());
    let name = "language_model.model.layers.0.mlp.shared_expert.gate_proj";
    let source = Array::from_iter(
        (0..512 * 128).map(|i| (i as f32 * 0.013).sin()),
        &[512, 128],
    )
    .as_dtype(Dtype::Bfloat16)?;
    let (weight, scales, biases) = ops::quantize(&source, 64, 8)?;
    let linear = Linear {
        weight,
        scales: Some(scales),
        biases: Some(biases),
        bias: None,
        quant: Some(Quantization {
            bits: 8,
            group_size: 64,
            mode: "affine".into(),
        }),
    };
    let config = json!({"text_config":{"num_hidden_layers":1},"quantization":{"bits":8,"group_size":64,"mode":"affine"}});
    std::fs::write(path.join("config.json"), serde_json::to_vec(&config)?)?;
    std::fs::write(path.join("tokenizer.json"), b"{}")?;
    let (candidate, receipt) = resident_quant::calibrate(
        &linear,
        &ActivationMoments {
            rows: 1,
            sum_squares: vec![1.; 128],
        },
        4,
    )?;
    let tensors = HashMap::from([
        (format!("{name}.weight"), linear.weight),
        (format!("{name}.scales"), linear.scales.unwrap()),
        (format!("{name}.biases"), linear.biases.unwrap()),
    ]);
    let mut w = Weights { tensors, config };
    let arrays = HashMap::from([
        (format!("{name}.weight"), candidate.weight),
        (format!("{name}.scales"), candidate.scales.unwrap()),
        (format!("{name}.biases"), candidate.biases.unwrap()),
    ]);
    Array::save_safetensors(arrays.iter(), None, path.join("tensors.safetensors"))?;
    let mut report = json!({"kind":"resident_affine_variant","tensors_file":"tensors.safetensors",
        "model_config_sha256":sha256_file(&path.join("config.json"))?,"tokenizer_sha256":sha256_file(&path.join("tokenizer.json"))?,
        "tensors_sha256":sha256_file(&path.join("tensors.safetensors"))?,
        "quantization":{name:{"bits":4,"group_size":64,"mode":"affine"}},"receipts":{name:receipt}});
    let good = report.clone();
    report["quantization"] = json!({"language_model.model.layers.0.mlp.switch_mlp.gate_proj":{"bits":4,"group_size":64,"mode":"affine"}});
    std::fs::write(path.join("overlay.json"), serde_json::to_vec(&report)?)?;
    assert!(apply_overlay(&mut w, &path, &path).is_err());
    assert_eq!(w.linear(name)?.quant.unwrap().bits, 8);
    report = good;
    std::fs::write(path.join("overlay.json"), serde_json::to_vec(&report)?)?;
    apply_overlay(&mut w, &path, &path)?;
    assert_eq!(w.linear(name)?.quant.unwrap().bits, 4);
    assert_eq!(w.linear(name)?.weight.shape(), [512, 16]);
    report["tensors_sha256"] = "bad".into();
    std::fs::write(path.join("overlay.json"), serde_json::to_vec(&report)?)?;
    assert!(apply_overlay(&mut w, &path, &path).is_err());
    assert_eq!(w.linear(name)?.quant.unwrap().bits, 4);
    Ok(())
}
