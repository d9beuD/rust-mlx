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
