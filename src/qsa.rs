use crate::{
    dense::{Attention, KvCache},
    hybrid::{HybridConfig, plus_one},
    weights::{Linear, Weights},
};
use anyhow::{Result, ensure};
use mlx_rs::{
    Array, Dtype, fast,
    ops::{self, indexing::IndexOp},
};

pub struct Qsa {
    pub attention: Attention,
    pub index: Linear,
    pub qscale: Array,
    pub kscale: Array,
    pub heads: i32,
    pub dim: i32,
    pub budget: i32,
    pub ratio: i32,
    pub rotary: i32,
    pub theta: f32,
    pub eps: f32,
}
#[derive(Default, Clone)]
pub struct QsaCache {
    pub kv: KvCache,
    pub raw_keys: Option<Array>,
    pub blocks: Option<Array>,
}
impl Qsa {
    pub fn load(w: &Weights, p: &str, c: &HybridConfig) -> Result<Self> {
        let theta = c.rope_parameters["rope_theta"].as_f64().unwrap_or(1e7) as f32;
        let rotary = (c.head_dim as f64
            * c.rope_parameters["partial_rotary_factor"]
                .as_f64()
                .unwrap_or(0.25)) as i32;
        Ok(Self {
            attention: Attention {
                q: w.linear(&format!("{p}.q_proj"))?,
                k: w.linear(&format!("{p}.k_proj"))?,
                v: w.linear(&format!("{p}.v_proj"))?,
                o: w.linear(&format!("{p}.o_proj"))?,
                qnorm: plus_one(&w.tensor(&format!("{p}.q_norm.weight"))?)?,
                knorm: plus_one(&w.tensor(&format!("{p}.k_norm.weight"))?)?,
                heads: c.num_attention_heads,
                kv_heads: c.num_key_value_heads,
                head_dim: c.head_dim,
                eps: c.rms_norm_eps,
                theta,
                rotary_dim: rotary,
                gated: true,
            },
            index: w.linear(&format!("{p}.indexer.index_qk_proj"))?,
            qscale: plus_one(&w.tensor(&format!("{p}.indexer.q_layernorm.weight"))?)?,
            kscale: plus_one(&w.tensor(&format!("{p}.indexer.k_layernorm.weight"))?)?,
            heads: c.indexer_n_heads,
            dim: c.indexer_head_dim,
            budget: c.indexer_budget,
            ratio: c.indexer_compress_ratio,
            rotary,
            theta,
            eps: c.rms_norm_eps,
        })
    }
    pub fn forward(&self, x: &Array, cache: &mut QsaCache) -> Result<Array> {
        let (b, t) = (x.shape()[0], x.shape()[1]);
        let offset = cache.kv.offset;
        let qk = self
            .index
            .forward(x)?
            .reshape(&[b, t, self.heads + 1, self.dim])?;
        let raw = qk.index((.., .., self.heads, ..));
        let raw = if let Some(old) = &cache.raw_keys {
            ops::concatenate(&[old, &raw], 1)?
        } else {
            raw
        };
        let len = raw.shape()[1];
        cache.raw_keys = Some(raw.clone());
        let nblocks = len / self.ratio;
        let topk = self.budget / self.ratio;
        if nblocks <= topk {
            return self.attention.forward(x, &mut cache.kv);
        }
        ensure!(self.rotary <= self.dim, "indexer rotary dimension");
        let q = fast::rms_norm(
            qk.index((.., .., ..self.heads, ..)),
            Some(&self.qscale),
            self.eps,
        )?
        .as_dtype(x.dtype())?
        .transpose_axes(&[0, 2, 1, 3])?;
        let q = fast::rope(&q, self.rotary, false, self.theta, 1., offset, None)?;
        let old_count = cache.blocks.as_ref().map_or(0, |a| a.shape()[2]);
        let new = raw
            .index((.., old_count * self.ratio..nblocks * self.ratio, ..))
            .as_dtype(Dtype::Float32)?
            .reshape(&[b, nblocks - old_count, self.ratio, self.dim])?
            .mean_axis(2, false)?
            .as_dtype(raw.dtype())?;
        let new = fast::rms_norm(&new, Some(&self.kscale), self.eps)?
            .as_dtype(raw.dtype())?
            .expand_dims(1)?;
        let new = fast::rope(
            &new,
            self.rotary,
            false,
            self.theta,
            self.ratio as f32,
            old_count,
            None,
        )?;
        let blocks = if let Some(old) = &cache.blocks {
            if nblocks > old_count {
                ops::concatenate(&[old, &new], 2)?
            } else {
                old.clone()
            }
        } else {
            new
        };
        cache.blocks = Some(blocks.clone());
        let raw_scores = q.as_dtype(Dtype::Float32)?.matmul(
            &blocks
                .as_dtype(Dtype::Float32)?
                .transpose_axes(&[0, 1, 3, 2])?,
        )?;
        let scores = ops::maximum(&raw_scores, &Array::from_f32(0.))?
            .sum_axis(1, false)?
            .divide(&Array::from_f32((self.dim as f32).sqrt()))?;
        let ends = Array::from_iter(offset + 1..offset + t + 1, &[1, t, 1]);
        let complete = ends.floor_divide(&Array::from_int(self.ratio))?;
        let block_ids = Array::from_iter(0..nblocks, &[1, 1, nblocks]);
        let valid = block_ids.lt(&complete)?;
        let scores = ops::select(&valid, &scores, &Array::from_f32(f32::NEG_INFINITY))?;
        let selected = ops::argpartition_axis(&scores, -topk, -1)?.index((.., .., -topk..));
        let selected_mask = block_ids
            .reshape(&[1, 1, nblocks, 1])?
            .eq(&selected.reshape(&[b, t, 1, topk])?)?
            .any_axis(-1, false)?;
        let token_blocks =
            Array::from_iter((0..len).map(|i| (i / self.ratio).min(nblocks - 1)), &[len]);
        let selected_mask = selected_mask.take_axis(&token_blocks, 2)?;
        let token_ids = Array::from_iter(0..len, &[1, 1, len]);
        let tail = token_ids.ge(&complete.multiply(&Array::from_int(self.ratio))?)?;
        let causal = token_ids.lt(&ends)?;
        let sparse = selected_mask.logical_or(&tail)?.logical_and(&causal)?;
        let mask =
            ops::select(&complete.gt(&Array::from_int(topk))?, &sparse, &causal)?.expand_dims(1)?;
        self.attention.forward_mask(x, &mut cache.kv, Some(&mask))
    }
}
