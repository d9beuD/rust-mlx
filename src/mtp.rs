//! Native checkpoint MTP head. Greedy speculation must verify against the target.
use crate::{
    compiled,
    hybrid::{HybridConfig, HyperConnection, MoE},
    qsa::{Qsa, QsaCache},
    weights::{Linear, Weights},
};
use anyhow::{Context, Result, ensure};
use mlx_rs::Array;
pub struct Mtp {
    /// Experimental lazy draft chain; target verification still consumes CPU IDs.
    pub gpu_draft: std::cell::Cell<bool>,
    /// CLI diagnostics only; serving does not retain private proposal IDs.
    pub record_drafts: std::cell::Cell<bool>,
    pub adaptive_depth: std::cell::Cell<bool>,
    pub adaptive_depth_costs: std::cell::Cell<[f64; 3]>,
    pub adaptive_vocab: std::cell::Cell<bool>,
    /// Draft-only low-ID shortlist size; zero retains the complete vocabulary.
    pub draft_vocab_limit: std::cell::Cell<usize>,
    pub draft_vocab_refresh_rounds: std::cell::Cell<usize>,
    pub embedding_norm: Array,
    pub hidden_norm: Array,
    pub fc_embedding: Linear,
    pub fc_hidden: Linear,
    pub attention: Qsa,
    pub attention_hyper: HyperConnection,
    pub mlp_hyper: HyperConnection,
    pub mlp: MoE,
    pub mixer: HyperConnection,
    pub hidden: i32,
    pub hc: i32,
    pub eps: f32,
}
impl Mtp {
    pub fn load(w: &Weights, c: &HybridConfig) -> Result<Self> {
        let p = "mtp.layers.0";
        Ok(Self {
            gpu_draft: std::cell::Cell::new(false),
            record_drafts: std::cell::Cell::new(false),
            adaptive_depth: std::cell::Cell::new(false),
            adaptive_depth_costs: std::cell::Cell::new(crate::draft_policy::TARGET_ROUND_COSTS),
            adaptive_vocab: std::cell::Cell::new(false),
            draft_vocab_limit: std::cell::Cell::new(0),
            draft_vocab_refresh_rounds: std::cell::Cell::new(0),
            embedding_norm: w.tensor("mtp.pre_fc_norm_embedding.weight")?,
            hidden_norm: w.tensor("mtp.pre_fc_norm_hidden.weight")?,
            fc_embedding: w.linear("mtp.fc_embedding")?,
            fc_hidden: w.linear("mtp.fc_hidden")?,
            attention: Qsa::load(w, &format!("{p}.self_attn"), c)?,
            attention_hyper: HyperConnection::load(
                w,
                &format!("{p}.attn_hyper_connection"),
                c,
                true,
            )?,
            mlp_hyper: HyperConnection::load(w, &format!("{p}.mlp_hyper_connection"), c, true)?,
            mlp: MoE::load(w, &format!("{p}.mlp"), c)?,
            mixer: HyperConnection::load(w, "mtp.hyper_connection_mixer", c, false)?,
            hidden: c.hidden_size,
            hc: c.hc_count,
            eps: c.rms_norm_eps,
        })
    }
    pub fn forward(
        &self,
        embedding: &Array,
        previous_hidden: &Array,
        cache: &mut QsaCache,
        position: i32,
    ) -> Result<(Array, Array)> {
        ensure!(
            embedding.ndim() == 3
                && previous_hidden.ndim() == 3
                && previous_hidden.shape()[2] == self.hc * self.hidden
                && embedding.shape()[2] == self.hidden
                && embedding.shape()[..2] == previous_hidden.shape()[..2],
            "MTP input shape mismatch"
        );
        let s = previous_hidden.shape();
        let e = self.fc_embedding.forward(&compiled::norm(
            embedding,
            &self.embedding_norm,
            self.eps,
        )?)?;
        // This normalization is global across all HC streams, before independent projection.
        let h = compiled::norm(previous_hidden, &self.hidden_norm, self.eps)?.reshape(&[
            s[0],
            s[1],
            self.hc,
            self.hidden,
        ])?;
        let h = self.fc_hidden.forward(&h)?;
        let mut h = e.expand_dims(2)?.add(h)?.reshape(s)?;
        cache.kv.rope_offset = Some(position);
        let (mixed, gate) = self.attention_hyper.forward(&h)?;
        let branch = self.attention.forward(&mixed, cache)?;
        h = self.attention_hyper.write(
            &h,
            &branch,
            gate.as_ref().context("MTP attention injection")?,
        )?;
        let (mixed, gate) = self.mlp_hyper.forward(&h)?;
        h = self.mlp_hyper.write(
            &h,
            &self.mlp.forward(&mixed)?,
            gate.as_ref().context("MTP MLP injection")?,
        )?;
        Ok((self.mixer.forward(&h)?.0, h))
    }
}
