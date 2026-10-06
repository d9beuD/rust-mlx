use crate::{
    dense::{Mlp, scalar_like, silu},
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
pub fn grouped_norm(x: &Array, scale: &Array, hc: i32, eps: f32) -> Result<Array> {
    let shape = x.shape();
    let d = shape[2] / hc;
    let y = x.reshape(&[shape[0], shape[1], hc, d])?;
    Ok(crate::compiled::norm(&y, &scale.reshape(&[hc, d])?, eps)?.reshape(shape)?)
}
pub struct HyperConnection {
    pub compiled_mode: std::cell::Cell<bool>,
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
            compiled_mode: std::cell::Cell::new(
                std::env::var_os("RUST_MLX_COMPILE_HYPER").is_some(),
            ),
            scale: w.tensor(&format!("{p}.hc_norm.weight"))?,
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
        if self.compiled_mode.get() {
            return crate::hyper_compiled::forward(self, x);
        }
        self.forward_reference(x)
    }
    pub fn forward_reference(&self, x: &Array) -> Result<(Array, Option<Array>)> {
        let xn = grouped_norm(x, &self.scale, self.hc, self.eps)?;
        if crate::hc_kernel::enabled()
            && self.hc == 4
            && let Some(i) = &self.inject
            && let Some((down, inj)) = crate::hc_kernel::project(&self.down, i, &xn)?
        {
            let lo = crate::compiled::activate(&down, self.hc)?;
            let mixed = crate::compiled::mix(&self.up.forward(&lo)?, &xn, self.hc)?;
            return Ok((mixed, Some(crate::compiled::injection(&inj, self.hc)?)));
        }
        let lo = crate::compiled::activate(&self.down.forward(&xn)?, self.hc)?;
        let mixed = crate::compiled::mix(&self.up.forward(&lo)?, &xn, self.hc)?;
        let inject = self
            .inject
            .as_ref()
            .map(|i| -> Result<Array> {
                Ok(crate::compiled::injection(&i.forward_rows(&xn)?, self.hc)?)
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
    pub down_packed: Option<Array>,
    /// Lossless native gather of concatenated gate/up expert rows, experimental.
    pub gate_up: Option<Linear>,
    pub fused_mode: std::cell::Cell<bool>,
    /// Experimental expert-major ordering for decode-equivalent verifier rows.
    pub sorted_mode: std::cell::Cell<bool>,
    pub router: Linear,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
    pub shared: Mlp,
    pub shared_gate: Linear,
    pub top_k: i32,
}
thread_local! {
    static SORTED_MOE_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}
pub fn sorted_moe_calls() -> u64 {
    SORTED_MOE_CALLS.with(std::cell::Cell::get)
}
impl MoE {
    pub fn load(w: &Weights, p: &str, c: &HybridConfig) -> Result<Self> {
        Ok(Self {
            gate_up: None,
            down_packed: None,
            fused_mode: std::cell::Cell::new(std::env::var_os("RUST_MLX_FUSED_MOE").is_some()),
            sorted_mode: std::cell::Cell::new(false),
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
        Self::gather_sorted(l, x, ids, false)
    }
    fn gather_sorted(l: &Linear, x: &Array, ids: &Array, sorted: bool) -> Result<Array> {
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
            sorted,
        )?)
    }
    fn sorted_experts(&self, x: &Array, ids: &Array) -> Result<Array> {
        // Expert-major gather/restore follows MLX-LM SwitchGLU; see NOTICE.
        let flat = ids.reshape(&[-1])?;
        let order = ops::argsort(&flat)?;
        let inverse = ops::argsort(&order)?;
        let sorted_ids = flat.take(&order)?;
        let rows = order.floor_divide(Array::from_slice(&[self.top_k as u32], &[]))?;
        let grouped = x
            .reshape(&[-1, x.shape()[2]])?
            .take_axis(&rows, 0)?
            .expand_dims(-2)?;
        let gate = Self::gather_sorted(&self.gate, &grouped, &sorted_ids, true)?;
        let up = Self::gather_sorted(&self.up, &grouped, &sorted_ids, true)?;
        let down = Self::gather_sorted(
            &self.down,
            &crate::compiled::swiglu(&gate, &up)?,
            &sorted_ids,
            true,
        )?;
        Ok(down.take_axis(&inverse, 0)?.reshape(&[
            x.shape()[0],
            x.shape()[1],
            self.top_k,
            1,
            x.shape()[2],
        ])?)
    }
    pub fn forward(&self, x: &Array) -> Result<Array> {
        ensure!(
            self.gate.weight.ndim() == 3
                && self.up.weight.ndim() == 3
                && self.down.weight.ndim() == 3
                && self.gate.weight.shape()[0] == self.router.weight.shape()[0]
                && self.up.weight.shape()[0] == self.router.weight.shape()[0]
                && self.down.weight.shape()[0] == self.router.weight.shape()[0],
            "expert/router count mismatch"
        );
        let logits = self.router.forward(x)?;
        let route = if crate::moe_route::enabled()
            && x.dtype() == mlx_rs::Dtype::Bfloat16
            && x.shape()[0] == 1
            && (1..=8).contains(&x.shape()[1])
            && self.top_k == 10
            && self.router.weight.shape()[0] == 512
        {
            crate::moe_route::tail(&logits, &self.shared_gate.forward_rows(x)?, self.top_k)?
        } else {
            None
        };
        let (ids, scores, shared_factor) = if let Some((ids, scores, factor)) = route {
            (ids, scores, Some(factor))
        } else {
            let gates = ops::softmax_axis(&logits, -1, true)?;
            let ids =
                ops::argpartition_axis(&gates, -self.top_k, -1)?.index((.., .., -self.top_k..));
            let scores = gates.take_along_axis(&ids, -1)?;
            let scores = scores.divide(&scores.sum_axis(-1, true)?)?;
            (ids, scores, None)
        };
        let y = if self.sorted_mode.get() && crate::verification::rows() && x.shape()[1] > 1 {
            SORTED_MOE_CALLS.with(|calls| calls.set(calls.get().wrapping_add(1)));
            (self.sorted_experts(x, &ids)?, false)
        } else {
            let xe = x.expand_dims(-2)?.expand_dims(-2)?;
            let packed = if crate::moe_layout::enabled() {
                self.gate_up
                    .as_ref()
                    .map(|l| -> Result<_> {
                        let y = Self::gather(l, &xe, &ids)?;
                        let n = y.shape()[y.ndim() - 1] / 2;
                        crate::moe_layout::record_call();
                        Ok((
                            y.index((.., .., .., .., ..n)),
                            y.index((.., .., .., .., n..)),
                        ))
                    })
                    .transpose()?
            } else {
                None
            };
            let fused = if packed.is_some() {
                packed
            } else if self.fused_mode.get() {
                crate::moe_kernel::gate_up(&self.up, &self.gate, x, &ids)?
            } else {
                None
            };
            let (gate, up) = if let Some(pair) = fused {
                pair
            } else {
                (
                    Self::gather(&self.gate, &xe, &ids)?,
                    Self::gather(&self.up, &xe, &ids)?,
                )
            };
            let routed = crate::compiled::swiglu(&gate, &up)?;
            if let Some(y) = crate::moe_down::reduce(
                &self.down,
                &routed,
                &ids,
                &scores,
                self.down_packed.as_ref(),
            )? {
                (y, true)
            } else {
                (Self::gather(&self.down, &routed, &ids)?, false)
            }
        };
        let (y, reduced) = y;
        let y = if reduced {
            y
        } else {
            y.squeeze_axes(&[-2])?
                .multiply(&scores.expand_dims(-1)?)?
                .sum_axis(-2, false)?
        };
        let shared_factor = match shared_factor {
            Some(factor) => factor,
            None => ops::sigmoid(&self.shared_gate.forward_rows(x)?)?,
        };
        Ok(y.add(&self.shared.forward(x)?.multiply(&shared_factor)?)?)
    }
}
#[derive(Default, Clone)]
pub struct GdnCache {
    pub conv: Option<Array>,
    pub state: Option<Array>,
    pub verified_states: Option<Array>,
    pub verified_conv: Option<Array>,
}
pub struct Gdn {
    pub packed_mode: std::cell::Cell<bool>,
    pub qkv: Linear,
    pub z: Linear,
    pub a: Linear,
    pub b: Linear,
    pub out: Linear,
    pub conv: Array,
    pub decode_conv: Array,
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
            packed_mode: std::cell::Cell::new(std::env::var_os("RUST_MLX_PACKED_GDN").is_some()),
            qkv: w.linear(&format!("{p}.in_proj_qkv"))?,
            z: w.linear(&format!("{p}.in_proj_z"))?,
            a: w.linear(&format!("{p}.in_proj_a"))?,
            b: w.linear(&format!("{p}.in_proj_b"))?,
            out: w.linear(&format!("{p}.out_proj"))?,
            conv: crate::conv_weights::guarded(&w.tensor(&format!("{p}.conv1d.weight"))?)?,
            decode_conv: w
                .tensor(&format!("{p}.conv1d.weight"))?
                .index((.., .., 0))
                .t()
                .as_dtype(Dtype::Float32)?,
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
        if crate::gdn_compiled::enabled()
            && let Some(y) = crate::gdn_compiled::forward(self, x, cache)?
        {
            return Ok(y);
        }
        self.forward_reference(x, cache)
    }
    pub(crate) fn forward_reference(&self, x: &Array, cache: &mut GdnCache) -> Result<Array> {
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
        cache.verified_conv = if crate::verification::active() {
            Some(inp.clone())
        } else {
            None
        };
        cache.conv = Some(inp.index((.., inp.shape()[1] - self.kernel + 1.., ..)));
        let convolved = silu(&if crate::verification::active()
            && matches!(self.conv.dtype(), Dtype::Bfloat16 | Dtype::Float16)
        {
            let mut ys = Vec::new();
            for i in 0..t {
                ys.push(crate::compiled::decode_conv(
                    &inp.index((.., i..i + self.kernel, ..)),
                    &self.decode_conv,
                )?);
            }
            ops::concatenate(&ys, 1)?
        } else if t == 1 && matches!(self.conv.dtype(), Dtype::Bfloat16 | Dtype::Float16) {
            crate::compiled::decode_conv(&inp, &self.decode_conv)?
        } else {
            ops::conv1d(&inp, &self.conv, 1, 0, 1, cd)?
        })?;
        let q = convolved
            .index((.., .., ..kd))
            .reshape(&[batch, t, self.hk, self.dk])?;
        let k = convolved
            .index((.., .., kd..kd * 2))
            .reshape(&[batch, t, self.hk, self.dk])?;
        let v = convolved
            .index((.., .., kd * 2..))
            .reshape(&[batch, t, self.hv, self.dv])?;
        let eps = scalar_like(&q, 1e-6)?;
        let q = q
            .multiply(&q.square()?.sum_axis(-1, true)?.add(&eps)?.rsqrt()?)?
            .multiply(scalar_like(&q, (self.dk as f32).powf(-0.5))?)?;
        let k = k.multiply(&k.square()?.sum_axis(-1, true)?.add(&eps)?.rsqrt()?)?;
        let beta = ops::sigmoid(&self.b.forward_rows(x)?)?;
        let g = crate::compiled::decay(&self.alog, &self.a.forward_rows(x)?, &self.dt)?;
        let state = cache.state.clone().unwrap_or(ops::zeros_dtype(
            &[batch, self.hv, self.dv, self.dk],
            Dtype::Float32,
        )?);
        let (y, state) = if self.packed_mode.get() && self.dk == 128 && self.dv % 8 == 0 {
            let history = crate::verification::active();
            let mut out = crate::gdn_kernel::packed(&q, &k, &v, &g, &beta, &state, history)?;
            if history {
                cache.verified_states = out.pop();
            }
            let s = out.pop().context("packed state")?;
            let y = out.pop().context("packed output")?;
            (y, s)
        } else if crate::verification::active() {
            let (y, s, states) =
                crate::gdn_kernel::recurrent_with_history(&q, &k, &v, &g, &beta, &state)?;
            cache.verified_states = Some(states);
            (y, s)
        } else if std::env::var_os("RUST_MLX_GDN_OPS").is_some() {
            gdn_reference(&q, &k, &v, &g, &beta, &state)?
        } else {
            crate::gdn_kernel::recurrent(&q, &k, &v, &g, &beta, &state)?
        };
        if !crate::verification::active() {
            cache.verified_states = None;
        }
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
    pub async_layers: std::cell::Cell<bool>,
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
            async_layers: std::cell::Cell::new(std::env::var_os("RUST_MLX_ASYNC_LAYERS").is_some()),
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
        let (mixed, hidden) = self.forward_hidden(tokens, cache)?;
        Ok((self.head.forward(&mixed)?, hidden))
    }
    /// Complete target body and caches, before the vocabulary projection.
    pub fn forward_hidden(
        &self,
        tokens: &[u32],
        cache: &mut HybridCache,
    ) -> Result<(Array, Array)> {
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
        // Preparation reads the same immutable CPU history and table rows as the
        // native path. MLX owns asynchronous evaluation; arrays stay on this
        // owning thread and no cache state is changed before the PLE layer.
        let prepared = if crate::runtime_prepare::ple_enabled() && t <= 8 {
            let embeddings = self
                .ple
                .iter()
                .map(|p| {
                    p.as_ref()
                        .map(|p| p.embedding_tokens(tokens, &cache.history, h.dtype()))
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?;
            let work = embeddings
                .iter()
                .filter_map(Option::as_ref)
                .collect::<Vec<_>>();
            if !work.is_empty() {
                mlx_rs::transforms::async_eval(work)?;
                crate::runtime_prepare::record_ple();
            }
            Some(embeddings)
        } else {
            None
        };
        for (i, l) in self.layers.iter().enumerate() {
            if let Some(ple) = &self.ple[i] {
                h = if let Some(emb) = prepared.as_ref().and_then(|p| p[i].as_ref()) {
                    ple.forward_embedding(&h, emb, &mut cache.ple[i])?
                } else {
                    ple.forward(&h, tokens, &cache.history, &mut cache.ple[i])?
                };
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
            if self.async_layers.get() {
                mlx_rs::transforms::async_eval([&h])?;
            }
        }
        cache.history.extend_from_slice(tokens);
        if cache.history.len() > self.config.ngram_size - 1 {
            cache.history =
                cache.history[cache.history.len() - self.config.ngram_size + 1..].to_vec();
        }
        cache.offset += t;
        let mixed = self.mixer.forward(&h)?.0;
        Ok((mixed, h))
    }
}

#[derive(Clone)]
pub struct HybridCache {
    pub layers: Vec<LayerCache>,
    pub ple: Vec<crate::ple::PleCache>,
    pub history: Vec<u32>,
    pub offset: i32,
}
impl HybridCache {
    /// Commit an accepted prefix of a completed verification graph. The original
    /// snapshot supplies CPU token history; recurrent states come from the exact
    /// history kernel rather than an inverse update or a lossy reconstruction.
    pub fn commit_verified(
        &mut self,
        original: &Self,
        tokens: &[u32],
        keep: usize,
        c: &HybridConfig,
    ) -> Result<()> {
        ensure!(
            keep <= tokens.len() && self.offset == original.offset + tokens.len() as i32,
            "invalid verification transaction"
        );
        if keep == 0 {
            *self = original.clone();
            return Ok(());
        }
        let keep = keep as i32;
        let end = original.offset + keep;
        for layer in &mut self.layers {
            match layer {
                LayerCache::Linear(l) => {
                    l.state = Some(
                        l.verified_states
                            .as_ref()
                            .context("missing verified recurrent states")?
                            .index((.., keep - 1, .., .., ..)),
                    );
                    l.conv = Some(
                        l.verified_conv
                            .as_ref()
                            .context("missing verified convolution")?
                            .index((.., keep..keep + c.linear_conv_kernel_dim - 1, ..)),
                    );
                    l.verified_states = None;
                    l.verified_conv = None;
                }
                LayerCache::Full(l) => {
                    l.kv.trim(end)?;
                    l.raw_keys = l.raw_keys.as_ref().map(|v| v.index((.., ..end, ..)));
                    l.blocks = l
                        .blocks
                        .as_ref()
                        .map(|v| v.index((.., .., ..end / c.indexer_compress_ratio, ..)));
                }
            }
        }
        let history_len = (c.ple_conv_kernel_size - 1) * c.ngram_size as i32;
        for p in &mut self.ple {
            if let Some(v) = &p.verified_conv {
                p.conv = Some(v.index((.., keep..keep + history_len, ..)));
            }
            p.verified_conv = None;
        }
        self.history = original.history.clone();
        self.history.extend_from_slice(&tokens[..keep as usize]);
        let retained = c.ngram_size - 1;
        if self.history.len() > retained {
            self.history = self.history[self.history.len() - retained..].to_vec();
        }
        self.offset = end;
        Ok(())
    }
}
