use crate::{
    dense::{scalar_like, silu},
    hybrid::{HybridConfig, grouped_norm},
    ngram::{NGramHasher, NGramTable},
    weights::{Linear, Weights},
};
use anyhow::Result;
use mlx_rs::{
    Array,
    ops::{self, indexing::IndexOp},
};
use std::path::Path;

pub struct Ple {
    pub key: Linear,
    pub value: Linear,
    pub norm_key: Array,
    pub norm_query: Array,
    pub norm_conv: Array,
    pub conv: Array,
    pub table: NGramTable,
    pub table_scale: Option<Array>,
    pub hash: NGramHasher,
    pub hc: i32,
    pub hidden: i32,
    pub eps: f32,
    pub kernel: i32,
    pub dilation: i32,
    pub embed: i32,
}
#[derive(Clone, Default)]
pub struct PleCache {
    pub conv: Option<Array>,
}
impl Ple {
    pub fn load(w: &Weights, path: &Path, p: &str, c: &HybridConfig) -> Result<Self> {
        let eos = w.config["text_config"]["eos_token_id"]
            .as_u64()
            .unwrap_or(248044) as u32;
        let ep = format!("{p}.ple_embedding");
        Ok(Self {
            key: w.linear(&format!("{p}.key_proj"))?,
            value: w.linear(&format!("{p}.value_proj"))?,
            norm_key: w.tensor(&format!("{p}.norm_key.weight"))?,
            norm_query: w.tensor(&format!("{p}.norm_query.weight"))?,
            norm_conv: w.tensor(&format!("{p}.norm_conv.weight"))?,
            conv: w.tensor(&format!("{p}.conv1d.weight"))?,
            table_scale: w
                .tensors
                .get(&format!("{ep}.ngram_embedding.weight_scale"))
                .cloned(),
            table: NGramTable::load(path, &format!("{ep}.ngram_embedding"), &w.config)?,
            hash: NGramHasher::load(w, &ep, c.ngram_size, c.heads_per_ngram, eos)?,
            hc: c.hc_count,
            hidden: c.hidden_size,
            eps: c.rms_norm_eps,
            kernel: c.ple_conv_kernel_size,
            dilation: c.ngram_size as i32,
            embed: c.ple_embed_dim,
        })
    }
    pub fn forward(
        &self,
        x: &Array,
        tokens: &[u32],
        history: &[u32],
        cache: &mut PleCache,
    ) -> Result<Array> {
        let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let rows = self.hash.rows(tokens, history)?;
        let mut emb = self
            .table
            .gather(&rows, &[b, t, self.embed])?
            .as_dtype(x.dtype())?;
        if let Some(scale) = &self.table_scale {
            emb = emb.multiply(scale)?;
        }
        let key = grouped_norm(&self.key.forward(&emb)?, &self.norm_key, self.hc, self.eps)?
            .reshape(&[b, t, self.hc, self.hidden])?;
        let query = grouped_norm(x, &self.norm_query, self.hc, self.eps)?.reshape(&[
            b,
            t,
            self.hc,
            self.hidden,
        ])?;
        let gate = key
            .multiply(&query)?
            .sum_axis(-1, true)?
            .divide(scalar_like(&key, (self.hidden as f32).sqrt())?)?;
        let gate = ops::sign(&gate)?
            .multiply(&ops::maximum(gate.abs()?, scalar_like(&gate, 1e-6)?)?.sqrt()?)?;
        let gated = ops::sigmoid(&gate)?
            .multiply(&self.value.forward(&emb)?.expand_dims(2)?)?
            .reshape(&[b, t, d])?;
        let normed = grouped_norm(&gated, &self.norm_conv, self.hc, self.eps)?;
        let history_len = (self.kernel - 1) * self.dilation;
        let old = cache
            .conv
            .clone()
            .unwrap_or(ops::zeros_dtype(&[b, history_len, d], x.dtype())?);
        let inp = ops::concatenate(&[&old, &normed], 1)?;
        cache.conv = Some(inp.index((.., inp.shape()[1] - history_len.., ..)));
        let conv = silu(&ops::conv1d(&inp, &self.conv, 1, 0, self.dilation, d)?)?;
        Ok(x.add(gated.add(&conv)?)?)
    }
}
