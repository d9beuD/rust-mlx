//! Equal-offset decode batches. Each row owns an independent recurrent/KV/PLE state.
use crate::{
    hybrid::{HybridAttention, HybridCache, HybridModel, LayerCache},
    verification,
};
use anyhow::{Result, ensure};
use mlx_rs::{
    Array,
    ops::{self, indexing::IndexOp},
};
fn join<'a>(arrays: impl Iterator<Item = Option<&'a Array>>) -> Result<Option<Array>> {
    let arrays: Vec<_> = arrays.collect();
    let some: Vec<_> = arrays.iter().copied().flatten().collect();
    if some.is_empty() {
        return Ok(None);
    }
    ensure!(
        some.len() == arrays.len(),
        "batch has inconsistent initialized caches"
    );
    let shape = some[0].shape();
    let dtype = some[0].dtype();
    ensure!(
        some.iter()
            .all(|a| a.shape() == shape && a.dtype() == dtype && a.shape()[0] == 1),
        "batch cache shapes differ"
    );
    Ok(Some(ops::concatenate(&some, 0)?))
}
fn split(x: &Option<Array>, row: i32) -> Option<Array> {
    x.as_ref().map(|a| a.index((row..row + 1, ..)))
}
impl HybridModel {
    pub fn decode_batch(
        &self,
        tokens: &[u32],
        caches: &mut [HybridCache],
    ) -> Result<(Array, Array)> {
        ensure!(
            !tokens.is_empty() && tokens.len() == caches.len(),
            "batch tokens/caches mismatch"
        );
        ensure!(
            tokens.len() <= 32 && tokens.iter().all(|&t| t < self.config.vocab_size as u32),
            "invalid batch size or vocabulary token"
        );
        let offset = caches[0].offset;
        ensure!(
            caches.iter().all(|c| c.offset == offset
                && c.layers.len() == self.layers.len()
                && c.ple.len() == self.layers.len()),
            "decode batch requires equal-offset complete caches"
        );
        let mut merged = self.make_cache();
        merged.offset = offset;
        for (i, layer) in merged.layers.iter_mut().enumerate() {
            match layer {
                LayerCache::Linear(dst) => {
                    let src = caches
                        .iter()
                        .map(|c| match &c.layers[i] {
                            LayerCache::Linear(s) => Ok(s),
                            _ => Err(anyhow::anyhow!("cache kind mismatch")),
                        })
                        .collect::<Result<Vec<_>>>()?;
                    dst.conv = join(src.iter().map(|c| c.conv.as_ref()))?;
                    dst.state = join(src.iter().map(|c| c.state.as_ref()))?;
                }
                LayerCache::Full(dst) => {
                    let src = caches
                        .iter()
                        .map(|c| match &c.layers[i] {
                            LayerCache::Full(s) => Ok(s),
                            _ => Err(anyhow::anyhow!("cache kind mismatch")),
                        })
                        .collect::<Result<Vec<_>>>()?;
                    ensure!(
                        src.iter()
                            .all(|c| c.kv.offset == offset && c.kv.rope_offset.is_none()),
                        "invalid batch attention offset"
                    );
                    dst.kv.offset = offset;
                    dst.kv.keys = join(src.iter().map(|c| c.kv.keys.as_ref()))?;
                    dst.kv.values = join(src.iter().map(|c| c.kv.values.as_ref()))?;
                    dst.raw_keys = join(src.iter().map(|c| c.raw_keys.as_ref()))?;
                    dst.blocks = join(src.iter().map(|c| c.blocks.as_ref()))?;
                }
            }
            merged.ple[i].conv = join(caches.iter().map(|c| c.ple[i].conv.as_ref()))?;
        }
        let batch = tokens.len() as i32;
        let (logits, hidden) = verification::with_rows(|| -> Result<_> {
            let ids = Array::from_slice(tokens, &[batch, 1]);
            let h = self.embedding.embedding(&ids)?;
            let mut h = ops::broadcast_to(
                &h.expand_dims(2)?,
                &[batch, 1, self.config.hc_count, self.config.hidden_size],
            )?
            .reshape(&[batch, 1, self.config.hc_count * self.config.hidden_size])?
            .contiguous()?;
            for (i, l) in self.layers.iter().enumerate() {
                if let Some(p) = &self.ple[i] {
                    let embeddings = tokens
                        .iter()
                        .zip(caches.iter())
                        .map(|(&t, c)| p.embedding_tokens(&[t], &c.history, h.dtype()))
                        .collect::<Result<Vec<_>>>()?;
                    let emb = ops::concatenate(&embeddings, 0)?;
                    h = p.forward_embedding(&h, &emb, &mut merged.ple[i])?;
                }
                let (mixed, inject) = l.attn_hc.forward(&h)?;
                let branch = match (&l.attention, &mut merged.layers[i]) {
                    (HybridAttention::Linear(a), LayerCache::Linear(c)) => a.forward(&mixed, c)?,
                    (HybridAttention::Full(a), LayerCache::Full(c)) => a.forward(&mixed, c)?,
                    _ => anyhow::bail!("cache kind mismatch"),
                };
                h = l.attn_hc.write(
                    &h,
                    &branch,
                    inject
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("missing attention injection"))?,
                )?;
                let (mixed, inject) = l.mlp_hc.forward(&h)?;
                h = l.mlp_hc.write(
                    &h,
                    &l.moe.forward(&mixed)?,
                    inject
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("missing MLP injection"))?,
                )?;
            }
            let mixed = self.mixer.forward(&h)?.0;
            Ok((self.head.forward(&mixed)?, h))
        })?;
        // Evaluate before committing: GPU errors leave every caller's cache intact.
        logits.eval()?;
        for (row, (cache, &token)) in caches.iter_mut().zip(tokens).enumerate() {
            let row = row as i32;
            for (dst, src) in cache.layers.iter_mut().zip(&merged.layers) {
                match (dst, src) {
                    (LayerCache::Linear(d), LayerCache::Linear(s)) => {
                        d.conv = split(&s.conv, row);
                        d.state = split(&s.state, row);
                        d.verified_states = None;
                        d.verified_conv = None;
                    }
                    (LayerCache::Full(d), LayerCache::Full(s)) => {
                        d.kv.keys = split(&s.kv.keys, row);
                        d.kv.values = split(&s.kv.values, row);
                        d.kv.offset = offset + 1;
                        d.raw_keys = split(&s.raw_keys, row);
                        d.blocks = split(&s.blocks, row);
                    }
                    _ => unreachable!("checked before graph construction"),
                }
            }
            for (dst, src) in cache.ple.iter_mut().zip(&merged.ple) {
                dst.conv = split(&src.conv, row);
                dst.verified_conv = None;
            }
            cache.offset = offset + 1;
            cache.history.push(token);
            let retained = self.config.ngram_size - 1;
            if cache.history.len() > retained {
                cache.history = cache.history[cache.history.len() - retained..].to_vec();
            }
        }
        Ok((logits, hidden))
    }
}
