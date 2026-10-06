//! Experimental draft-only shortlist. Target verification remains full-vocabulary.
use crate::weights::Linear;
use anyhow::{Context, Result, ensure};
use mlx_rs::{Array, ops::indexing};
pub(crate) struct DraftVocabulary<'a> {
    full: &'a Linear,
    selected: Option<(Linear, Vec<u32>)>,
    fixed: Option<&'a crate::draft_head::DraftHead>,
}

pub(crate) enum DraftToken {
    Cpu(u32),
    Gpu(Array),
}

impl<'a> DraftVocabulary<'a> {
    pub(crate) fn prepared(head: &'a crate::draft_head::DraftHead) -> Self {
        Self {
            full: &head.linear,
            selected: None,
            fixed: Some(head),
        }
    }
    pub(crate) fn size(&self) -> usize {
        self.selected
            .as_ref()
            .map_or(self.full.weight.shape()[0] as usize, |(_, rows)| rows.len())
    }
    pub(crate) fn new(
        full: &'a Linear,
        limit: usize,
        prompt: &[u32],
        eos: &[u32],
        ranked: &[u32],
    ) -> Result<Self> {
        ensure!(
            full.weight.ndim() == 2,
            "invalid draft vocabulary head rank"
        );
        let n = full.weight.shape()[0] as usize;
        ensure!(n > 0, "empty draft vocabulary head");
        if limit == 0 || limit >= n {
            return Ok(Self {
                full,
                selected: None,
                fixed: None,
            });
        }
        let mut rows = (0..limit as u32)
            .chain(n.saturating_sub(1024) as u32..n as u32)
            .chain(
                prompt
                    .iter()
                    .chain(eos)
                    .chain(ranked)
                    .copied()
                    .filter(|&id| id < n as u32),
            )
            .collect::<Vec<_>>();
        rows.sort_unstable();
        rows.dedup();
        if rows.len() == n {
            return Ok(Self {
                full,
                selected: None,
                fixed: None,
            });
        }
        let ids = Array::from_slice(&rows, &[rows.len() as i32]);
        let take = |x: &Option<Array>| -> Result<Option<Array>> {
            x.as_ref()
                .map(|x| x.take_axis(&ids, 0).map_err(Into::into))
                .transpose()
        };
        let selected = Linear {
            weight: full.weight.take_axis(&ids, 0)?,
            scales: take(&full.scales)?,
            biases: take(&full.biases)?,
            bias: take(&full.bias)?,
            quant: full.quant.clone(),
        };
        Ok(Self {
            full,
            selected: Some((selected, rows)),
            fixed: None,
        })
    }
    pub(crate) fn greedy(&self, x: &Array) -> Result<u32> {
        let head = self.selected.as_ref().map(|(l, _)| l).unwrap_or(self.full);
        let local = if crate::greedy_head::enabled() {
            crate::greedy_head::greedy(head, x)?.item_exact::<u32>()
        } else {
            indexing::argmax(head.forward(x)?, false)?.item_exact::<u32>()
        };
        if let Some((_, rows)) = &self.selected {
            rows.get(local as usize)
                .copied()
                .context("draft local ID outside shortlist")
        } else if let Some(rows) = self.fixed.and_then(|h| h.rows.as_ref()) {
            rows.get(local as usize)
                .copied()
                .context("draft ID outside fixed vocabulary")
        } else {
            Ok(local)
        }
    }

    pub(crate) fn greedy_token(&self, x: &Array, gpu: bool) -> Result<DraftToken> {
        if !gpu {
            return Ok(DraftToken::Cpu(self.greedy(x)?));
        }
        let head = self.selected.as_ref().map(|(l, _)| l).unwrap_or(self.full);
        let local = if crate::greedy_head::enabled() {
            crate::greedy_head::greedy(head, x)?
        } else {
            indexing::argmax(head.forward(x)?, false)?
        };
        let token = if let Some((_, rows)) = &self.selected {
            Array::from_slice(rows, &[rows.len() as i32]).take(&local)?
        } else if let Some(ids) = self.fixed.and_then(|h| h.ids.as_ref()) {
            ids.take(&local)?
        } else {
            local
        };
        Ok(DraftToken::Gpu(token.reshape(&[1, 1])?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::Quantization;
    use mlx_rs::{Dtype, ops};
    #[test]
    fn fixed_head_matches_native_requantization_and_global_mapping() {
        use crate::draft_head::DraftHead;
        let (n, k) = (8224, 64);
        let source = Array::from_iter((0..n * k).map(|i| (i as f32 * 0.19).sin() * 0.15), &[n, k])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let (w, s, b) = ops::quantize(&source, 64, 8).unwrap();
        let full = Linear {
            weight: w,
            scales: Some(s),
            biases: Some(b),
            bias: None,
            quant: Some(Quantization {
                bits: 8,
                group_size: 64,
                mode: "affine".into(),
            }),
        };
        let x = Array::from_iter((0..k).map(|i| (i as f32 * 0.11).cos()), &[1, 1, k])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        for bits in [4, 6] {
            let copy = DraftHead::prepare(&full, Some(bits), &[]).unwrap();
            assert!(copy.rows.is_none());
            let native = ops::dequantize(
                &full.weight,
                full.scales.as_ref().unwrap(),
                full.biases.as_ref(),
                64,
                8,
            )
            .unwrap();
            let (w, s, b) = ops::quantize(native, 64, bits).unwrap();
            let expected = ops::quantized_matmul(&x, &w, &s, Some(&b), true, 64, bits)
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap()
                .contiguous()
                .unwrap();
            let actual = copy
                .linear
                .forward(&x)
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap()
                .contiguous()
                .unwrap();
            mlx_rs::transforms::eval([&actual, &expected]).unwrap();
            assert_eq!(actual.as_slice::<f32>(), expected.as_slice::<f32>());
        }
        let copy = DraftHead::prepare(&full, None, &[8223, 12, 97, 12]).unwrap();
        assert_eq!(copy.rows.as_ref().unwrap(), &[12, 97, 8223]);
        let vocab = DraftVocabulary::prepared(&copy);
        let native = full
            .forward(&x)
            .unwrap()
            .take_axis(copy.ids.as_ref().unwrap(), -1)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
            .contiguous()
            .unwrap();
        let actual = copy
            .linear
            .forward(&x)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
            .contiguous()
            .unwrap();
        mlx_rs::transforms::eval([&native, &actual]).unwrap();
        assert_eq!(native.as_slice::<f32>(), actual.as_slice::<f32>());
        let local = indexing::argmax(&native, false)
            .unwrap()
            .item_exact::<u32>();
        let expected = copy.rows.as_ref().unwrap()[local as usize];
        assert_eq!(vocab.greedy(&x).unwrap(), expected);
        let DraftToken::Gpu(id) = vocab.greedy_token(&x, true).unwrap() else {
            panic!("expected GPU token")
        };
        assert_eq!(id.item_exact::<u32>(), expected);
        assert!(DraftHead::prepare(&full, Some(5), &[]).is_err());
        assert!(DraftHead::prepare(&full, None, &[n as u32]).is_err());
        assert_eq!(full.quant.as_ref().unwrap().bits, 8);
    }
    #[test]
    fn mixed_quantized_shortlist_matches_selected_full_logits_and_global_ids() {
        for bits in [4, 5, 6, 8] {
            let (n, k) = (3072, 512);
            let w = Array::from_iter((0..n * k).map(|i| (i as f32 * 0.031).sin() * 0.05), &[n, k])
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let (weight, scales, biases) = ops::quantize(w, 64, bits).unwrap();
            let full = Linear {
                weight,
                scales: Some(scales),
                biases: Some(biases),
                bias: Some(
                    Array::from_iter((0..n).map(|i| (i as f32 * 0.017).cos() * 0.01), &[n])
                        .as_dtype(Dtype::Bfloat16)
                        .unwrap(),
                ),
                quant: Some(Quantization {
                    bits,
                    group_size: 64,
                    mode: "affine".into(),
                }),
            };
            let shortlist =
                DraftVocabulary::new(&full, 16, &[1234, 1234, 1633], &[2000], &[]).unwrap();
            let (head, rows) = shortlist.selected.as_ref().unwrap();
            assert!(rows.contains(&1234) && rows.contains(&1633) && rows.contains(&2000));
            assert_eq!(rows.len(), 1043);
            let x = Array::from_iter((0..k).map(|i| (i as f32 * 0.11).cos()), &[1, 1, k])
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let expected = full
                .forward(&x)
                .unwrap()
                .take_axis(Array::from_slice(rows, &[rows.len() as i32]), -1)
                .unwrap();
            let actual = head.forward(&x).unwrap();
            let expected = expected
                .as_dtype(Dtype::Float32)
                .unwrap()
                .contiguous()
                .unwrap();
            let actual = actual
                .as_dtype(Dtype::Float32)
                .unwrap()
                .contiguous()
                .unwrap();
            mlx_rs::transforms::eval([&expected, &actual]).unwrap();
            assert_eq!(actual.as_slice::<f32>(), expected.as_slice::<f32>());
            let local = indexing::argmax(&expected, false)
                .unwrap()
                .item_exact::<u32>();
            assert_eq!(shortlist.greedy(&x).unwrap(), rows[local as usize]);
            let DraftToken::Gpu(token) = shortlist.greedy_token(&x, true).unwrap() else {
                panic!("GPU drafting must return a tensor");
            };
            assert_eq!(token.shape(), &[1, 1]);
            assert_eq!(token.item_exact::<u32>(), rows[local as usize]);
            assert!(
                DraftVocabulary::new(&full, 0, &[], &[], &[])
                    .unwrap()
                    .selected
                    .is_none()
            );
            assert!(
                DraftVocabulary::new(&full, n as usize, &[], &[], &[])
                    .unwrap()
                    .selected
                    .is_none()
            );
        }
    }
}
