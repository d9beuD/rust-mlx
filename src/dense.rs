use crate::weights::{Linear, Weights};
use anyhow::{Result, ensure};
use mlx_rs::{
    Array, fast,
    ops::{
        self,
        indexing::{IndexOp, TryIndexMutOp},
    },
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct DenseConfig {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: usize,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: Option<f32>,
    pub rope_parameters: Option<serde_json::Value>,
    pub tie_word_embeddings: bool,
    pub vocab_size: i32,
}

#[derive(Default, Clone)]
pub struct KvCache {
    pub keys: Option<Array>,
    pub values: Option<Array>,
    pub offset: i32,
    pub rope_offset: Option<i32>,
    backing: Option<(Array, Array)>,
}
impl KvCache {
    pub(crate) fn reset_storage(&mut self) {
        self.backing = None;
    }
    pub fn trim(&mut self, end: i32) -> Result<()> {
        ensure!((0..=self.offset).contains(&end), "invalid KV trim");
        self.keys = self.keys.as_ref().map(|k| k.index((.., .., ..end, ..)));
        self.values = self.values.as_ref().map(|v| v.index((.., .., ..end, ..)));
        self.offset = end;
        Ok(())
    }
    pub fn update(&mut self, k: Array, v: Array) -> Result<(Array, Array)> {
        ensure!(
            k.ndim() == 4 && v.shape() == k.shape() && v.dtype() == k.dtype(),
            "incompatible KV update"
        );
        let t = k.shape()[2];
        let compatible = |old: &Option<Array>, new: &Array| {
            old.as_ref().is_none_or(|a| {
                a.dtype() == new.dtype()
                    && a.shape()[0] == new.shape()[0]
                    && a.shape()[1] == new.shape()[1]
                    && a.shape()[3] == new.shape()[3]
            })
        };
        if crate::kv_blocks::enabled()
            && k.shape()[0] == 1
            && compatible(&self.keys, &k)
            && compatible(&self.values, &v)
        {
            let end = self.offset + t;
            let old = self.backing.take();
            let (mut kb, mut vb) =
                if let Some((kb, vb)) = old.filter(|(kb, _)| kb.shape()[2] >= end) {
                    (kb, vb)
                } else {
                    let capacity = ((end + crate::kv_blocks::BLOCK - 1) / crate::kv_blocks::BLOCK)
                        * crate::kv_blocks::BLOCK;
                    let mut shape = k.shape().to_vec();
                    shape[2] = capacity;
                    let mut kb = ops::zeros_dtype(&shape, k.dtype())?;
                    let mut vb = ops::zeros_dtype(&shape, v.dtype())?;
                    if let (Some(keys), Some(values)) = (&self.keys, &self.values) {
                        kb.try_index_mut((.., .., ..self.offset, ..), keys)?;
                        vb.try_index_mut((.., .., ..self.offset, ..), values)?;
                    }
                    (kb, vb)
                };
            // Functional slice updates: aliases in prompt/verification snapshots remain immutable.
            kb.try_index_mut((.., .., self.offset..end, ..), &k)?;
            vb.try_index_mut((.., .., self.offset..end, ..), &v)?;
            let keys = kb.index((.., .., ..end, ..));
            let values = vb.index((.., .., ..end, ..));
            self.backing = Some((kb, vb));
            self.keys = Some(keys.clone());
            self.values = Some(values.clone());
            self.offset = end;
            crate::kv_blocks::record();
            return Ok((keys, values));
        }
        self.reset_storage();
        let k = if let Some(old) = &self.keys {
            ops::concatenate(&[old, &k], 2)?
        } else {
            k
        };
        let v = if let Some(old) = &self.values {
            ops::concatenate(&[old, &v], 2)?
        } else {
            v
        };
        self.offset += t;
        self.keys = Some(k.clone());
        self.values = Some(v.clone());
        Ok((k, v))
    }
}
pub struct Attention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub qnorm: Array,
    pub knorm: Array,
    pub heads: i32,
    pub kv_heads: i32,
    pub head_dim: i32,
    pub eps: f32,
    pub theta: f32,
    pub rotary_dim: i32,
    pub gated: bool,
}
impl Attention {
    pub fn forward(&self, x: &Array, cache: &mut KvCache) -> Result<Array> {
        self.forward_mask(x, cache, None)
    }
    pub fn forward_mask(
        &self,
        x: &Array,
        cache: &mut KvCache,
        explicit_mask: Option<&Array>,
    ) -> Result<Array> {
        self.forward_selection(x, cache, explicit_mask, None)
    }
    pub fn forward_selection(
        &self,
        x: &Array,
        cache: &mut KvCache,
        explicit_mask: Option<&Array>,
        selection: Option<(&Array, &Array, i32)>,
    ) -> Result<Array> {
        let (b, t) = (x.shape()[0], x.shape()[1]);
        let q = self.q.forward(x)?;
        let (q, gate) = if self.gated {
            let qg = q.reshape(&[b, t, self.heads, 2, self.head_dim])?;
            (
                qg.index((.., .., .., 0, ..)),
                Some(qg.index((.., .., .., 1, ..))),
            )
        } else {
            (q.reshape(&[b, t, self.heads, self.head_dim])?, None)
        };
        let k = self
            .k
            .forward(x)?
            .reshape(&[b, t, self.kv_heads, self.head_dim])?;
        let v = self
            .v
            .forward(x)?
            .reshape(&[b, t, self.kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let q = if self.gated {
            crate::compiled::norm(&q, &self.qnorm, self.eps)?
        } else {
            fast::rms_norm(&q, Some(&self.qnorm), self.eps)?
        }
        .transpose_axes(&[0, 2, 1, 3])?;
        let k = if self.gated {
            crate::compiled::norm(&k, &self.knorm, self.eps)?
        } else {
            fast::rms_norm(&k, Some(&self.knorm), self.eps)?
        }
        .transpose_axes(&[0, 2, 1, 3])?;
        let q = if self.gated {
            crate::rope::text(
                &q,
                self.rotary_dim,
                self.theta,
                cache.rope_offset.unwrap_or(cache.offset),
                1,
            )?
        } else {
            fast::rope(
                &q,
                self.rotary_dim,
                false,
                self.theta,
                1.,
                cache.offset,
                None,
            )?
        };
        let k = if self.gated {
            crate::rope::text(
                &k,
                self.rotary_dim,
                self.theta,
                cache.rope_offset.unwrap_or(cache.offset),
                1,
            )?
        } else {
            fast::rope(
                &k,
                self.rotary_dim,
                false,
                self.theta,
                1.,
                cache.offset,
                None,
            )?
        };
        let (k, v) = cache.update(k, v)?;
        let mask = explicit_mask
            .map(fast::ScaledDotProductAttentionMask::Array)
            .or_else(|| (t > 1).then_some(fast::ScaledDotProductAttentionMask::Causal));
        let output = if crate::verification::active() && t > 1 {
            let old = k.shape()[2] - t;
            let mut ys = Vec::new();
            for i in 0..t {
                let qq = q.index((.., .., i..i + 1, ..));
                let kk = k.index((.., .., ..old + i + 1, ..));
                let vv = v.index((.., .., ..old + i + 1, ..));
                let mm = explicit_mask.map(|m| m.index((.., .., i..i + 1, ..old + i + 1)));
                ys.push(fast::scaled_dot_product_attention(
                    &qq,
                    &kk,
                    &vv,
                    (self.head_dim as f32).powf(-0.5),
                    mm.as_ref().map(fast::ScaledDotProductAttentionMask::Array),
                    None,
                )?);
            }
            ops::concatenate(&ys, 2)?
        } else if let Some((blocks, ends, ratio)) = selection {
            crate::qsa_kernel::attention(
                &q,
                &k,
                &v,
                blocks,
                ends,
                ratio,
                (self.head_dim as f32).powf(-0.5),
            )?
        } else {
            fast::scaled_dot_product_attention(
                &q,
                &k,
                &v,
                (self.head_dim as f32).powf(-0.5),
                mask,
                None,
            )?
        };
        let mut out =
            output
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, t, self.heads * self.head_dim])?;
        if let Some(gate) = gate {
            out = out.multiply(ops::sigmoid(gate.reshape(&[
                b,
                t,
                self.heads * self.head_dim,
            ])?)?)?;
        }
        self.o.forward(&out)
    }
}
pub struct Mlp {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}
/// Rust scalar arrays are strongly typed; Python scalar literals are weakly typed.
/// Preserve the activation dtype instead of silently promoting bf16 graphs to fp32.
pub fn scalar_like(x: &Array, value: f32) -> Result<Array> {
    Ok(Array::from_f32(value).as_dtype(x.dtype())?)
}
pub fn silu(x: &Array) -> Result<Array> {
    Ok(crate::compiled::silu(x)?)
}
impl Mlp {
    pub fn load(w: &Weights, p: &str) -> Result<Self> {
        Ok(Self {
            gate: w.linear(&format!("{p}.gate_proj"))?,
            up: w.linear(&format!("{p}.up_proj"))?,
            down: w.linear(&format!("{p}.down_proj"))?,
        })
    }
    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.down.forward(&crate::compiled::swiglu(
            &self.gate.forward(x)?,
            &self.up.forward(x)?,
        )?)
    }
}
pub struct DenseLayer {
    pub attention: Attention,
    pub mlp: Mlp,
    pub norm1: Array,
    pub norm2: Array,
}
pub struct DenseModel {
    pub config: DenseConfig,
    pub embedding: Linear,
    pub head: Option<Linear>,
    pub norm: Array,
    pub layers: Vec<DenseLayer>,
}
impl DenseModel {
    pub fn load(w: &Weights) -> Result<Self> {
        let c: DenseConfig = serde_json::from_value(w.config.clone())?;
        ensure!(c.model_type == "qwen3", "dense path only supports qwen3");
        ensure!(
            c.hidden_size > 0
                && c.vocab_size > 0
                && c.num_attention_heads % c.num_key_value_heads == 0,
            "invalid dense configuration"
        );
        let theta = c
            .rope_theta
            .or_else(|| {
                c.rope_parameters
                    .as_ref()?
                    .get("rope_theta")?
                    .as_f64()
                    .map(|v| v as f32)
            })
            .unwrap_or(1e6);
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for i in 0..c.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let a = format!("{p}.self_attn");
            layers.push(DenseLayer {
                attention: Attention {
                    q: w.linear(&format!("{a}.q_proj"))?,
                    k: w.linear(&format!("{a}.k_proj"))?,
                    v: w.linear(&format!("{a}.v_proj"))?,
                    o: w.linear(&format!("{a}.o_proj"))?,
                    qnorm: w.tensor(&format!("{a}.q_norm.weight"))?,
                    knorm: w.tensor(&format!("{a}.k_norm.weight"))?,
                    heads: c.num_attention_heads,
                    kv_heads: c.num_key_value_heads,
                    head_dim: c.head_dim,
                    eps: c.rms_norm_eps,
                    theta,
                    rotary_dim: c.head_dim,
                    gated: false,
                },
                mlp: Mlp::load(w, &format!("{p}.mlp"))?,
                norm1: w.tensor(&format!("{p}.input_layernorm.weight"))?,
                norm2: w.tensor(&format!("{p}.post_attention_layernorm.weight"))?,
            });
        }
        let head = if c.tie_word_embeddings {
            None
        } else {
            Some(w.linear("lm_head")?)
        };
        Ok(Self {
            config: c,
            embedding: w.linear("model.embed_tokens")?,
            head,
            norm: w.tensor("model.norm.weight")?,
            layers,
        })
    }
    pub fn forward(&self, ids: &[u32], cache: &mut [KvCache]) -> Result<Array> {
        ensure!(!ids.is_empty(), "empty input");
        ensure!(cache.len() == self.layers.len(), "cache layer mismatch");
        ensure!(
            ids.iter().all(|&id| id < self.config.vocab_size as u32),
            "token outside vocabulary"
        );
        let ids = Array::from_slice(ids, &[1, ids.len() as i32]);
        let mut h = self.embedding.embedding(&ids)?;
        for (l, c) in self.layers.iter().zip(cache.iter_mut()) {
            h = h.add(&l.attention.forward(
                &fast::rms_norm(&h, Some(&l.norm1), self.config.rms_norm_eps)?,
                c,
            )?)?;
            h = h.add(&l.mlp.forward(&fast::rms_norm(
                &h,
                Some(&l.norm2),
                self.config.rms_norm_eps,
            )?)?)?;
        }
        let h = fast::rms_norm(&h, Some(&self.norm), self.config.rms_norm_eps)?;
        if let Some(head) = &self.head {
            head.forward(&h)
        } else {
            self.embedding.forward(&h)
        }
    }
}
