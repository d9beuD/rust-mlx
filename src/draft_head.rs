//! Model-scoped draft-only head copies. The target head is never replaced.
use crate::weights::{Linear, Quantization};
use anyhow::{Context, Result, ensure};
use mlx_rs::{Array, ops};

pub struct DraftHead {
    pub linear: Linear,
    pub rows: Option<Vec<u32>>,
    pub ids: Option<Array>,
    pub preparation_seconds: f64,
    pub bytes: usize,
    pub bits: Option<i32>,
}
impl DraftHead {
    /// Draft-only small distillation artifact, full vocabulary and source codes.
    pub fn with_bias(
        source: &Linear,
        model: &std::path::Path,
        artifact: &std::path::Path,
    ) -> Result<(Self, serde_json::Value)> {
        ensure!(source.weight.ndim() == 2, "draft source must be a matrix");
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(artifact.join("metadata.json"))?)?;
        let path = artifact.join("bias.safetensors");
        ensure!(
            metadata["complete"] == true && metadata["bias_dtype"] == "BF16",
            "incomplete bias artifact"
        );
        ensure!(
            metadata["config_sha256"].as_str()
                == Some(crate::resident_quant::sha256_file(&model.join("config.json"))?.as_str())
                && metadata["tokenizer_sha256"].as_str()
                    == Some(
                        crate::resident_quant::sha256_file(&model.join("tokenizer.json"))?.as_str()
                    )
                && metadata["bias_sha256"].as_str()
                    == Some(crate::resident_quant::sha256_file(&path)?.as_str()),
            "bias model/artifact identity mismatch"
        );
        let mut tensors = Array::load_safetensors(path)?;
        let bias = tensors.remove("bias").context("missing draft bias")?;
        ensure!(
            tensors.is_empty()
                && bias.shape() == [source.weight.shape()[0]]
                && bias.dtype() == mlx_rs::Dtype::Bfloat16,
            "bad bias shape/dtype"
        );
        bias.eval()?;
        let f = bias.as_dtype(mlx_rs::Dtype::Float32)?.contiguous()?;
        f.eval()?;
        ensure!(
            f.as_slice::<f32>()
                .iter()
                .all(|v| v.is_finite() && v.abs() <= 0.5),
            "bias outside declared finite bound"
        );
        let mut head = Self::prepare(source, None, &[])?;
        head.linear.bias = Some(if let Some(original) = &head.linear.bias {
            original.add(&bias)?
        } else {
            bias.clone()
        });
        head.linear.bias.as_ref().unwrap().eval()?;
        head.bytes += bias.nbytes();
        Ok((head, metadata))
    }

    /// Gather a fixed vocabulary and optionally requantize a BF16-dequantized
    /// copy in bounded chunks. These are independent model-init costs, not a
    /// prompt-specific shortlist or a change to the canonical target tensors.
    pub fn prepare(source: &Linear, bits: Option<i32>, rows: &[u32]) -> Result<Self> {
        let started = std::time::Instant::now();
        ensure!(source.weight.ndim() == 2, "draft head must be a matrix");
        ensure!(
            bits.is_none_or(|b| [4, 6].contains(&b)),
            "draft bits must be4 or6"
        );
        let n = source.weight.shape()[0];
        let mut rows = rows.to_vec();
        ensure!(
            rows.iter().all(|&r| r < n as u32),
            "draft vocabulary ID out of range"
        );
        rows.sort_unstable();
        rows.dedup();
        let rows = if rows.is_empty() || rows.len() == n as usize {
            None
        } else {
            Some(rows)
        };
        let ids = rows
            .as_ref()
            .map(|r| Array::from_slice(r, &[r.len() as i32]));
        let take = |a: &Array| -> Result<Array> {
            Ok(if let Some(ids) = &ids {
                a.take_axis(ids, 0)?
            } else {
                a.clone()
            })
        };
        let optional =
            |a: &Option<Array>| -> Result<Option<Array>> { a.as_ref().map(take).transpose() };
        let mut linear = Linear {
            weight: take(&source.weight)?,
            scales: optional(&source.scales)?,
            biases: optional(&source.biases)?,
            bias: optional(&source.bias)?,
            quant: source.quant.clone(),
        };
        if let Some(bits) = bits {
            use mlx_rs::ops::indexing::IndexOp;
            let q = linear
                .quant
                .as_ref()
                .context("draft requantization requires affine source")?;
            ensure!(q.mode == "affine", "non-affine draft source");
            let sc = linear.scales.as_ref().context("draft scales missing")?;
            let bs = linear.biases.as_ref().context("draft biases missing")?;
            let mut chunks = Vec::new();
            for start in (0..linear.weight.shape()[0]).step_by(8192) {
                let end = (start + 8192).min(linear.weight.shape()[0]);
                let w = ops::dequantize(
                    linear.weight.index((start..end, ..)),
                    sc.index((start..end, ..)),
                    Some(&bs.index((start..end, ..))),
                    q.group_size,
                    q.bits,
                )?;
                let (w, s, b) = ops::quantize(w, q.group_size, bits)?;
                mlx_rs::transforms::eval([&w, &s, &b])?;
                chunks.push((w, s, b));
            }
            linear.weight = ops::concatenate(&chunks.iter().map(|c| &c.0).collect::<Vec<_>>(), 0)?;
            linear.scales = Some(ops::concatenate(
                &chunks.iter().map(|c| &c.1).collect::<Vec<_>>(),
                0,
            )?);
            linear.biases = Some(ops::concatenate(
                &chunks.iter().map(|c| &c.2).collect::<Vec<_>>(),
                0,
            )?);
            linear.quant = Some(Quantization {
                bits,
                group_size: q.group_size,
                mode: "affine".into(),
            });
        }
        let mut arrays = vec![&linear.weight];
        arrays.extend(linear.scales.iter());
        arrays.extend(linear.biases.iter());
        arrays.extend(linear.bias.iter());
        arrays.extend(ids.iter());
        mlx_rs::transforms::eval(arrays.iter().copied())?;
        let bytes = arrays.iter().map(|a| a.nbytes()).sum();
        Ok(Self {
            linear,
            rows,
            ids,
            preparation_seconds: started.elapsed().as_secs_f64(),
            bytes,
            bits,
        })
    }
}

#[cfg(test)]
mod bias_tests {
    use super::*;
    #[test]
    fn bias_artifact_is_bounded_bound_to_model_and_keeps_target_unchanged() {
        let temp = std::env::temp_dir().join(format!(
            "rust-mlx-draft-bias-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&temp).unwrap();
        std::fs::write(temp.join("config.json"), b"{}").unwrap();
        std::fs::write(temp.join("tokenizer.json"), b"{\"tokenizer\":1}").unwrap();
        let source = Linear {
            weight: Array::from_slice(
                &[
                    1f32, 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
                ],
                &[4, 4],
            ),
            scales: None,
            biases: None,
            bias: None,
            quant: None,
        };
        let x = Array::from_slice(&[1f32, 2., 3., 4.], &[1, 1, 4]);
        let good = Array::from_slice(&[0.25f32, -0.5, 0.5, 0.125], &[4])
            .as_dtype(mlx_rs::Dtype::Bfloat16)
            .unwrap();
        let write = |bias: &Array, corrupt: bool| {
            let weights = temp.join("bias.safetensors");
            Array::save_safetensors([("bias", bias)], None, &weights).unwrap();
            let metadata = serde_json::json!({"complete":true,"bias_dtype":"BF16","config_sha256":crate::resident_quant::sha256_file(&temp.join("config.json")).unwrap(),"tokenizer_sha256":crate::resident_quant::sha256_file(&temp.join("tokenizer.json")).unwrap(),"bias_sha256":if corrupt {"wrong".into()} else {crate::resident_quant::sha256_file(&weights).unwrap()}});
            std::fs::write(
                temp.join("metadata.json"),
                serde_json::to_vec(&metadata).unwrap(),
            )
            .unwrap();
        };
        write(&good, false);
        let (head, _) = DraftHead::with_bias(&source, &temp, &temp).unwrap();
        let candidate = head.linear.forward(&x).unwrap();
        candidate.eval().unwrap();
        assert_eq!(candidate.as_slice::<f32>(), &[1.25, 1.5, 3.5, 4.125]);
        let original = source.forward(&x).unwrap();
        original.eval().unwrap();
        assert_eq!(original.as_slice::<f32>(), &[1., 2., 3., 4.]);
        for bias in [
            Array::from_slice(&[1f32, 0., 0., 0.], &[4])
                .as_dtype(mlx_rs::Dtype::Bfloat16)
                .unwrap(),
            Array::from_slice(&[f32::NAN, 0., 0., 0.], &[4])
                .as_dtype(mlx_rs::Dtype::Bfloat16)
                .unwrap(),
            Array::from_slice(&[0f32; 3], &[3])
                .as_dtype(mlx_rs::Dtype::Bfloat16)
                .unwrap(),
            Array::from_slice(&[0f32; 4], &[4]),
        ] {
            write(&bias, false);
            assert!(DraftHead::with_bias(&source, &temp, &temp).is_err());
        }
        write(&good, true);
        assert!(DraftHead::with_bias(&source, &temp, &temp).is_err());
        write(&good, false);
        std::fs::write(temp.join("config.json"), b"{\"changed\":true}").unwrap();
        assert!(DraftHead::with_bias(&source, &temp, &temp).is_err());
        std::fs::remove_dir_all(temp).unwrap();
    }
}
