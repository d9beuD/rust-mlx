use crate::{
    dense::{Mlp, silu},
    weights::{Linear, Weights},
};
use anyhow::{Context, Result, ensure};
use mlx_rs::{
    Array, Dtype, fast,
    ops::{self, indexing::IndexOp},
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct HybridConfig {
    pub hidden_size: i32,
    pub num_hidden_layers: usize,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub hc_count: i32,
    pub rms_norm_eps: f32,
    pub linear_num_key_heads: i32,
    pub linear_num_value_heads: i32,
    pub linear_key_head_dim: i32,
    pub linear_value_head_dim: i32,
    pub linear_conv_kernel_dim: i32,
    pub num_experts_per_tok: i32,
    pub layer_types: Vec<String>,
    pub vocab_size: i32,
    pub indexer_budget: i32,
    pub indexer_compress_ratio: i32,
    pub indexer_n_heads: i32,
    pub indexer_head_dim: i32,
    pub rope_parameters: serde_json::Value,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub ple_conv_kernel_size: i32,
    pub ple_embed_dim: i32,
}
pub fn plus_one(w: &Array) -> Result<Array> {
    Ok(w.as_dtype(Dtype::Float32)?.add(&Array::from_f32(1.))?)
}
pub fn grouped_norm(x: &Array, scale: &Array, hc: i32, eps: f32) -> Result<Array> {
    let shape = x.shape();
    let d = shape[2] / hc;
    let y = x
        .as_dtype(Dtype::Float32)?
        .reshape(&[shape[0], shape[1], hc, d])?;
    Ok(fast::rms_norm(&y, None, eps)?
        .multiply(&scale.reshape(&[hc, d])?)?
        .reshape(shape)?
        .as_dtype(x.dtype())?)
}
pub struct HyperConnection {
    pub scale: Array,
    pub down: Linear,
    pub up: Linear,
    pub inject: Option<Linear>,
    pub hc: i32,
    pub eps: f32,
}
impl HyperConnection {
    pub fn load(w: &Weights, p: &str, c: &HybridConfig, inject: bool) -> Result<Self> {
        Ok(Self {
            scale: plus_one(&w.tensor(&format!("{p}.hc_norm.weight"))?)?,
            down: w.linear(&format!("{p}.input_mix_weight_down"))?,
            up: w.linear(&format!("{p}.input_mix_weight_up"))?,
            inject: if inject {
                Some(w.linear(&format!("{p}.block_inject_weight"))?)
            } else {
                None
            },
            hc: c.hc_count,
            eps: c.rms_norm_eps,
        })
    }
    pub fn forward(&self, x: &Array) -> Result<(Array, Option<Array>)> {
        let xn = grouped_norm(x, &self.scale, self.hc, self.eps)?;
        let lo = silu(
            &self
                .down
                .forward(&xn)?
                .divide(&Array::from_f32(self.hc as f32))?,
        )?;
        let gate = ops::sigmoid(&self.up.forward(&lo)?)?;
        let s = x.shape();
        let mixed = xn
            .multiply(&gate)?
            .reshape(&[s[0], s[1], self.hc, s[2] / self.hc])?
            .mean_axis(2, false)?;
        let inject = self
            .inject
            .as_ref()
            .map(|i| -> Result<Array> {
                Ok(
                    ops::sigmoid(&i.forward(&xn)?.divide(&Array::from_f32(self.hc as f32))?)?
                        .multiply(&Array::from_f32(2.))?,
                )
            })
            .transpose()?;
        Ok((mixed, inject))
    }
    pub fn write(&self, x: &Array, y: &Array, gate: &Array) -> Result<Array> {
        let s = x.shape();
        Ok(x.reshape(&[s[0], s[1], self.hc, s[2] / self.hc])?
            .add(&y.expand_dims(2)?.multiply(&gate.expand_dims(-1)?)?)?
            .reshape(s)?)
    }
}
pub struct MoE {
    pub router: Linear,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
    pub shared: Mlp,
    pub shared_gate: Linear,
    pub top_k: i32,
}
impl MoE {
    pub fn load(w: &Weights, p: &str, c: &HybridConfig) -> Result<Self> {
        Ok(Self {
            router: w.linear(&format!("{p}.gate"))?,
            gate: w.linear(&format!("{p}.switch_mlp.gate_proj"))?,
            up: w.linear(&format!("{p}.switch_mlp.up_proj"))?,
            down: w.linear(&format!("{p}.switch_mlp.down_proj"))?,
            shared: Mlp::load(w, &format!("{p}.shared_expert"))?,
            shared_gate: w.linear(&format!("{p}.shared_expert_gate"))?,
            top_k: c.num_experts_per_tok,
        })
    }
    fn gather(l: &Linear, x: &Array, ids: &Array) -> Result<Array> {
        let q = l
            .quant
            .as_ref()
            .context("unquantized experts not supported on this path")?;
        Ok(ops::gather_qmm(
            x,
            &l.weight,
            l.scales.as_ref().context("missing scales")?,
            l.biases.as_ref(),
            None,
            ids,
            true,
            q.group_size,
            q.bits,
            false,
        )?)
    }
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let gates = ops::softmax_axis(&self.router.forward(x)?, -1, true)?;
        let ids = ops::argpartition_axis(&gates, -self.top_k, -1)?.index((.., .., -self.top_k..));
        let scores = gates.take_along_axis(&ids, -1)?;
        let scores = scores.divide(&scores.sum_axis(-1, true)?)?;
        let xe = x.expand_dims(-2)?.expand_dims(-2)?;
        let gate = Self::gather(&self.gate, &xe, &ids)?;
        let up = Self::gather(&self.up, &xe, &ids)?;
        let y =
            Self::gather(&self.down, &silu(&gate)?.multiply(&up)?, &ids)?.squeeze_axes(&[-2])?;
        let y = y.multiply(&scores.expand_dims(-1)?)?.sum_axis(-2, false)?;
        Ok(y.add(
            &self
                .shared
                .forward(x)?
                .multiply(&ops::sigmoid(&self.shared_gate.forward(x)?)?)?,
        )?)
    }
}
#[derive(Default, Clone)]
pub struct GdnCache {
    pub conv: Option<Array>,
    pub state: Option<Array>,
}
pub struct Gdn {
    pub qkv: Linear,
    pub z: Linear,
    pub a: Linear,
    pub b: Linear,
    pub out: Linear,
    pub conv: Array,
    pub alog: Array,
    pub dt: Array,
    pub norm: Array,
    pub hk: i32,
    pub hv: i32,
    pub dk: i32,
    pub dv: i32,
    pub kernel: i32,
    pub eps: f32,
}
impl Gdn {
    pub fn load(w: &Weights, p: &str, c: &HybridConfig) -> Result<Self> {
        Ok(Self {
            qkv: w.linear(&format!("{p}.in_proj_qkv"))?,
            z: w.linear(&format!("{p}.in_proj_z"))?,
            a: w.linear(&format!("{p}.in_proj_a"))?,
            b: w.linear(&format!("{p}.in_proj_b"))?,
            out: w.linear(&format!("{p}.out_proj"))?,
            conv: w.tensor(&format!("{p}.conv1d.weight"))?,
            alog: w.tensor(&format!("{p}.A_log"))?.as_dtype(Dtype::Float32)?,
            dt: w.tensor(&format!("{p}.dt_bias"))?,
            norm: w.tensor(&format!("{p}.norm.weight"))?,
            hk: c.linear_num_key_heads,
            hv: c.linear_num_value_heads,
            dk: c.linear_key_head_dim,
            dv: c.linear_value_head_dim,
            kernel: c.linear_conv_kernel_dim,
            eps: c.rms_norm_eps,
        })
    }
    pub fn forward(&self, x: &Array, cache: &mut GdnCache) -> Result<Array> {
        let (batch, t) = (x.shape()[0], x.shape()[1]);
        let kd = self.hk * self.dk;
        let vd = self.hv * self.dv;
        let cd = kd * 2 + vd;
        let qkv = self.qkv.forward(x)?;
        let old = cache
            .conv
            .clone()
            .unwrap_or(ops::zeros_dtype(&[batch, self.kernel - 1, cd], x.dtype())?);
        let inp = ops::concatenate(&[&old, &qkv], 1)?;
        cache.conv = Some(inp.index((.., inp.shape()[1] - self.kernel + 1.., ..)));
        let convolved = silu(&ops::conv1d(&inp, &self.conv, 1, 0, 1, cd)?)?;
        let q = convolved
            .index((.., .., ..kd))
            .reshape(&[batch, t, self.hk, self.dk])?;
        let k = convolved
            .index((.., .., kd..kd * 2))
            .reshape(&[batch, t, self.hk, self.dk])?;
        let v = convolved
            .index((.., .., kd * 2..))
            .reshape(&[batch, t, self.hv, self.dv])?;
        let eps = Array::from_f32(1e-6);
        let q = q
            .multiply(&q.square()?.sum_axis(-1, true)?.add(&eps)?.rsqrt()?)?
            .multiply(&Array::from_f32((self.dk as f32).powf(-0.5)))?;
        let k = k.multiply(&k.square()?.sum_axis(-1, true)?.add(&eps)?.rsqrt()?)?;
        let beta = ops::sigmoid(&self.b.forward(x)?)?;
        let a = self.a.forward(x)?.add(&self.dt)?;
        let softplus = ops::logaddexp(&a, &Array::from_f32(0.))?;
        let g = self.alog.exp()?.negative()?.multiply(&softplus)?.exp()?;
        let state = cache.state.clone().unwrap_or(ops::zeros_dtype(
            &[batch, self.hv, self.dv, self.dk],
            Dtype::Float32,
        )?);
        let (y, state) = gdn_reference(&q, &k, &v, &g, &beta, &state)?;
        cache.state = Some(state);
        let z = self
            .z
            .forward(x)?
            .reshape(&[batch, t, self.hv, self.dv])?
            .as_dtype(Dtype::Float32)?;
        let y = fast::rms_norm(&y, Some(&self.norm), self.eps)?
            .as_dtype(Dtype::Float32)?
            .multiply(&ops::sigmoid(&z)?)?
            .as_dtype(x.dtype())?;
        self.out.forward(&y.reshape(&[batch, t, vd])?)
    }
}
pub fn gdn_reference(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> Result<(Array, Array)> {
    let (batch, t, hk, dk) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let hv = v.shape()[2];
    let dv = v.shape()[3];
    ensure!(hv % hk == 0, "GDN head ratio");
    let ids = Array::from_iter((0..hv).map(|h| h / (hv / hk)), &[hv]);
    let q = q.take_axis(&ids, 2)?;
    let k = k.take_axis(&ids, 2)?;
    let mut state = state.clone();
    let mut ys = Vec::with_capacity(t as usize);
    for ti in 0..t {
        let q = q
            .index((.., ti, .., ..))
            .as_dtype(Dtype::Float32)?
            .reshape(&[batch, hv, 1, dk])?;
        let k = k
            .index((.., ti, .., ..))
            .as_dtype(Dtype::Float32)?
            .reshape(&[batch, hv, 1, dk])?;
        state = state.multiply(&g.index((.., ti, ..)).reshape(&[batch, hv, 1, 1])?)?;
        let memory = state.multiply(&k)?.sum_axis(-1, false)?;
        let delta = v
            .index((.., ti, .., ..))
            .subtract(&memory)?
            .multiply(&beta.index((.., ti, ..)).reshape(&[batch, hv, 1])?)?;
        state = state.add(&k.multiply(&delta.reshape(&[batch, hv, dv, 1])?)?)?;
        ys.push(
            state
                .multiply(&q)?
                .sum_axis(-1, false)?
                .as_dtype(v.dtype())?,
        );
    }
    Ok((ops::stack(&ys, 1)?, state))
}

pub struct HybridLayer {
    pub attn_hc: HyperConnection,
    pub mlp_hc: HyperConnection,
    pub attention: HybridAttention,
    pub moe: MoE,
}
pub enum HybridAttention {
    Linear(Box<Gdn>),
    Full(Box<crate::qsa::Qsa>),
}
#[derive(Clone)]
pub enum LayerCache {
    Linear(GdnCache),
    Full(crate::qsa::QsaCache),
}
pub struct HybridModel {
    pub config: HybridConfig,
    pub embedding: Linear,
    pub head: Linear,
    pub mixer: HyperConnection,
    pub layers: Vec<HybridLayer>,
    pub ple: Vec<Option<crate::ple::Ple>>,
}
impl HybridModel {
    pub fn load(w: &Weights, path: &std::path::Path) -> Result<Self> {
        let c: HybridConfig = serde_json::from_value(w.config["text_config"].clone())?;
        ensure!(
            c.layer_types.len() == c.num_hidden_layers,
            "layer count mismatch"
        );
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for i in 0..c.num_hidden_layers {
            let p = format!("language_model.model.layers.{i}");
            let attention = if c.layer_types[i] == "linear_attention" {
                HybridAttention::Linear(Box::new(Gdn::load(w, &format!("{p}.linear_attn"), &c)?))
            } else {
                let a = format!("{p}.self_attn");
                HybridAttention::Full(Box::new(crate::qsa::Qsa::load(w, &a, &c)?))
            };
            layers.push(HybridLayer {
                attn_hc: HyperConnection::load(w, &format!("{p}.attn_hyper_connection"), &c, true)?,
                mlp_hc: HyperConnection::load(w, &format!("{p}.mlp_hyper_connection"), &c, true)?,
                attention,
                moe: MoE::load(w, &format!("{p}.mlp"), &c)?,
            });
        }
        let ple = (0..c.num_hidden_layers)
            .map(|i| {
                if w.tensors.contains_key(&format!(
                    "language_model.model.layers.{i}.ple.key_proj.weight"
                )) {
                    crate::ple::Ple::load(
                        w,
                        path,
                        &format!("language_model.model.layers.{i}.ple"),
                        &c,
                    )
                    .map(Some)
                } else {
                    Ok(None)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            ple,
            embedding: w.linear("language_model.model.embed_tokens")?,
            head: w.linear("language_model.lm_head")?,
            mixer: HyperConnection::load(
                w,
                "language_model.model.hyper_connection_mixer",
                &c,
                false,
            )?,
            config: c,
            layers,
        })
    }
    pub fn make_cache(&self) -> HybridCache {
        HybridCache {
            layers: self
                .layers
                .iter()
                .map(|l| match l.attention {
                    HybridAttention::Linear(_) => LayerCache::Linear(GdnCache::default()),
                    HybridAttention::Full(_) => LayerCache::Full(crate::qsa::QsaCache::default()),
                })
                .collect(),
            ple: vec![crate::ple::PleCache::default(); self.layers.len()],
            history: Vec::new(),
            offset: 0,
        }
    }
    pub fn forward(&self, tokens: &[u32], cache: &mut HybridCache) -> Result<(Array, Array)> {
        ensure!(!tokens.is_empty(), "empty input");
        ensure!(
            tokens.iter().all(|&t| t < self.config.vocab_size as u32),
            "token outside vocabulary"
        );
        ensure!(
            cache.layers.len() == self.layers.len(),
            "cache layer mismatch"
        );
        let t = tokens.len() as i32;
        let ids = Array::from_slice(tokens, &[1, t]);
        let h = self.embedding.embedding(&ids)?;
        let mut h = ops::broadcast_to(
            &h.expand_dims(2)?,
            &[1, t, self.config.hc_count, self.config.hidden_size],
        )?
        .reshape(&[1, t, self.config.hc_count * self.config.hidden_size])?
        .contiguous()?;
        for (i, l) in self.layers.iter().enumerate() {
            if let Some(ple) = &self.ple[i] {
                h = ple.forward(&h, tokens, &cache.history, &mut cache.ple[i])?;
            }
            let (mixed, inject) = l.attn_hc.forward(&h)?;
            let branch = match (&l.attention, &mut cache.layers[i]) {
                (HybridAttention::Linear(a), LayerCache::Linear(c)) => a.forward(&mixed, c)?,
                (HybridAttention::Full(a), LayerCache::Full(c)) => a.forward(&mixed, c)?,
                _ => anyhow::bail!("cache type mismatch at {i}"),
            };
            h = l.attn_hc.write(
                &h,
                &branch,
                inject.as_ref().context("missing attention inject")?,
            )?;
            let (mixed, inject) = l.mlp_hc.forward(&h)?;
            h = l.mlp_hc.write(
                &h,
                &l.moe.forward(&mixed)?,
                inject.as_ref().context("missing MLP inject")?,
            )?;
        }
        cache.history.extend_from_slice(tokens);
        if cache.history.len() > self.config.ngram_size - 1 {
            cache.history =
                cache.history[cache.history.len() - self.config.ngram_size + 1..].to_vec();
        }
        cache.offset += t;
        let mixed = self.mixer.forward(&h)?.0;
        Ok((self.head.forward(&mixed)?, h))
    }
}

#[derive(Clone)]
pub struct HybridCache {
    pub layers: Vec<LayerCache>,
    pub ple: Vec<crate::ple::PleCache>,
    pub history: Vec<u32>,
    pub offset: i32,
}
