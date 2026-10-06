//! Greedy MTP with target verification and exact prefix cache commit.
use crate::{
    draft_policy::{DepthPolicy, VocabularyPolicy},
    draft_vocab::DraftToken,
    hybrid::{HybridCache, HybridModel},
    mtp::Mtp,
    qsa::QsaCache,
    verification,
};
use anyhow::{Result, ensure};
use mlx_rs::{
    Array,
    ops::{
        self,
        indexing::{self, IndexOp},
    },
};
use serde::Serialize;
use std::collections::VecDeque;
use std::time::Instant;

/// Immutable exact-prompt snapshot. Every continuation owns a cache clone.
#[derive(Clone)]
pub struct PreparedPrompt<'m> {
    model: &'m HybridModel,
    tokens: Vec<u32>,
    cache: HybridCache,
    logits: Array,
    hidden: Array,
    pub prefill_seconds: f64,
}
impl PreparedPrompt<'_> {
    /// Independent handles for a non-speculative continuation scheduler.
    pub fn decode_state(&self) -> (HybridCache, Array) {
        (self.cache.clone(), self.logits.clone())
    }
}
pub fn prepare<'m>(
    model: &'m HybridModel,
    tokens: &[u32],
    chunk: usize,
) -> Result<PreparedPrompt<'m>> {
    ensure!(
        !tokens.is_empty() && chunk > 0,
        "invalid prompt preparation"
    );
    let started = Instant::now();
    let mut cache = model.make_cache();
    let mut hidden = Vec::new();
    let mut logits = None;
    for ids in tokens.chunks(chunk) {
        let (l, h) = model.forward(ids, &mut cache)?;
        let l = l.index((0, -1, ..));
        l.eval()?;
        logits = Some(l);
        hidden.push(h);
    }
    let hidden = ops::concatenate(&hidden, 1)?;
    hidden.eval()?;
    Ok(PreparedPrompt {
        model,
        tokens: tokens.to_vec(),
        cache,
        logits: logits.expect("validated nonempty prompt"),
        hidden,
        prefill_seconds: started.elapsed().as_secs_f64(),
    })
}
/// Model-scoped LRU, bounded by entry count and total prompt tokens.
pub struct PrefixCache<'m> {
    model: &'m HybridModel,
    entries: VecDeque<PreparedPrompt<'m>>,
    max_entries: usize,
    max_tokens: usize,
}
impl<'m> PrefixCache<'m> {
    pub fn new(model: &'m HybridModel, max_entries: usize, max_tokens: usize) -> Self {
        Self {
            model,
            entries: VecDeque::new(),
            max_entries,
            max_tokens,
        }
    }
    pub fn get_or_prepare(
        &mut self,
        tokens: &[u32],
        chunk: usize,
    ) -> Result<(PreparedPrompt<'m>, bool)> {
        ensure!(
            !tokens.is_empty() && chunk > 0,
            "invalid prompt preparation"
        );
        if let Some(i) = self.entries.iter().position(|p| p.tokens == tokens) {
            let p = self.entries.remove(i).expect("existing LRU index");
            self.entries.push_back(p.clone());
            return Ok((p, true));
        }
        let p = prepare(self.model, tokens, chunk)?;
        if self.max_entries > 0 && tokens.len() <= self.max_tokens {
            while self.entries.len() >= self.max_entries
                || self.entries.iter().map(|p| p.tokens.len()).sum::<usize>() + tokens.len()
                    > self.max_tokens
            {
                self.entries.pop_front();
            }
            self.entries.push_back(p.clone());
        }
        Ok((p, false))
    }
    pub fn clear(&mut self) {
        self.entries.clear();
    }
    pub fn prepare_uncached(&self, tokens: &[u32], chunk: usize) -> Result<PreparedPrompt<'m>> {
        prepare(self.model, tokens, chunk)
    }
}
#[derive(Serialize)]
pub struct Generation {
    pub tokens: Vec<u32>,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    pub draft_seconds: f64,
    pub verify_seconds: f64,
    pub synchronize_draft_seconds: f64,
    pub acceptance: Vec<usize>,
    pub draft_lengths: Vec<usize>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub draft_tokens: Vec<Vec<u32>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub draft_vocab_sizes: Vec<usize>,
    pub round_seconds: Vec<f64>,
}
fn greedy(x: &Array) -> Result<u32> {
    Ok(indexing::argmax(x, false)?.item_exact::<u32>())
}
pub struct Options<'a> {
    pub max_tokens: usize,
    pub depth: usize,
    pub chunk: usize,
    pub eos: &'a [u32],
}
/// Ordinary greedy generation with the same timing convention as MTP.
pub fn generate_plain(
    m: &HybridModel,
    prompt: &[u32],
    options: &Options<'_>,
    emit: impl FnMut(u32) -> Result<()>,
) -> Result<Generation> {
    ensure!(
        !prompt.is_empty() && options.max_tokens > 0 && options.chunk > 0,
        "invalid generation request"
    );
    let prepared = prepare(m, prompt, options.chunk)?;
    let mut result = generate_plain_prepared(&prepared, options, emit)?;
    result.prefill_seconds += prepared.prefill_seconds;
    Ok(result)
}
pub fn generate_plain_prepared(
    prompt: &PreparedPrompt<'_>,
    options: &Options<'_>,
    mut emit: impl FnMut(u32) -> Result<()>,
) -> Result<Generation> {
    ensure!(options.max_tokens > 0, "invalid generation limit");
    let prefill = Instant::now();
    let m = prompt.model;
    let mut cache = prompt.cache.clone();
    let mut token = greedy(&prompt.logits)?;
    let mut result = Generation {
        tokens: Vec::new(),
        prefill_seconds: prefill.elapsed().as_secs_f64(),
        decode_seconds: 0.,
        draft_seconds: 0.,
        verify_seconds: 0.,
        synchronize_draft_seconds: 0.,
        acceptance: Vec::new(),
        draft_lengths: Vec::new(),
        draft_tokens: Vec::new(),
        draft_vocab_sizes: Vec::new(),
        round_seconds: Vec::new(),
    };
    if options.eos.contains(&token) {
        return Ok(result);
    }
    emit(token)?;
    result.tokens.push(token);
    let decode = Instant::now();
    while result.tokens.len() < options.max_tokens {
        token = if crate::greedy_head::enabled() {
            let (mixed, _) = m.forward_hidden(&[token], &mut cache)?;
            crate::greedy_head::greedy(&m.head, &mixed)?.item_exact::<u32>()
        } else {
            let (l, _) = m.forward(&[token], &mut cache)?;
            greedy(&l.index((0, -1, ..)))?
        };
        if options.eos.contains(&token) {
            break;
        }
        emit(token)?;
        result.tokens.push(token);
    }
    result.decode_seconds = if result.tokens.len() > 1 {
        decode.elapsed().as_secs_f64()
    } else {
        0.
    };
    Ok(result)
}
pub fn generate(
    m: &HybridModel,
    draft: &Mtp,
    prompt: &[u32],
    options: &Options<'_>,
    emit: impl FnMut(u32) -> Result<()>,
) -> Result<Generation> {
    ensure!(
        options.max_tokens > 0 && (1..=7).contains(&options.depth),
        "invalid MTP generation request"
    );
    let prepared = prepare(m, prompt, options.chunk)?;
    let mut result = generate_prepared(&prepared, draft, options, emit)?;
    result.prefill_seconds += prepared.prefill_seconds;
    Ok(result)
}
pub fn generate_prepared(
    prepared: &PreparedPrompt<'_>,
    draft: &Mtp,
    options: &Options<'_>,
    mut emit: impl FnMut(u32) -> Result<()>,
) -> Result<Generation> {
    let m = prepared.model;
    let prompt = &prepared.tokens;
    let Options {
        max_tokens,
        depth,
        chunk,
        eos,
    } = *options;
    ensure!(
        !prompt.is_empty() && max_tokens > 0 && (1..=7).contains(&depth) && chunk > 0,
        "invalid MTP generation request"
    );
    let started = Instant::now();
    let mut cache = prepared.cache.clone();
    let mut bonus = greedy(&prepared.logits)?;
    let prefill_seconds = started.elapsed().as_secs_f64();
    let mut result = Generation {
        tokens: Vec::new(),
        prefill_seconds,
        decode_seconds: 0.,
        draft_seconds: 0.,
        verify_seconds: 0.,
        synchronize_draft_seconds: 0.,
        acceptance: Vec::new(),
        draft_lengths: Vec::new(),
        draft_tokens: Vec::new(),
        draft_vocab_sizes: Vec::new(),
        round_seconds: Vec::new(),
    };
    if eos.contains(&bonus) {
        return Ok(result);
    }
    result.tokens.push(bonus);
    emit(bonus)?;
    if max_tokens == 1 {
        return Ok(result);
    }
    let decode = Instant::now();
    let mut ranked = Vec::new();
    let limit = draft.draft_vocab_limit.get();
    if limit > 0 && limit < m.config.vocab_size as usize {
        let count = 4096.min(m.config.vocab_size);
        let ids = ops::argpartition_axis(&prepared.logits, -count, -1)?
            .index(-count..)
            .contiguous()?;
        ids.eval()?;
        ranked = ids.as_slice::<u32>().to_vec();
    }
    let mut draft_head = crate::draft_vocab::DraftVocabulary::new(
        &m.head,
        draft.draft_vocab_limit.get(),
        prompt,
        eos,
        &ranked,
    )?;
    let full_head = crate::draft_vocab::DraftVocabulary::new(&m.head, 0, prompt, eos, &[])?;
    let costs = draft.adaptive_depth_costs.get();
    ensure!(
        costs.iter().all(|c| c.is_finite() && *c > 0.0),
        "invalid depth calibration"
    );
    let mut depth_policy = DepthPolicy::new(depth, draft.adaptive_depth.get(), costs);
    let mut vocab_policy = VocabularyPolicy::default();
    let h = &prepared.hidden;
    let mut shifted = prompt[1..].to_vec();
    shifted.push(bonus);
    let mut dc = QsaCache::default();
    let emb = m
        .embedding
        .embedding(&Array::from_slice(&shifted, &[1, shifted.len() as i32]))?;
    let start = Instant::now();
    let (mixed, wide) = draft.forward(&emb, h, &mut dc, 0)?;
    let mut dh = wide.index((.., -1.., ..));
    let gpu_draft = draft.gpu_draft.get();
    let mut seed = draft_head.greedy_token(&mixed.index((.., -1.., ..)), gpu_draft)?;
    result.synchronize_draft_seconds += start.elapsed().as_secs_f64();
    while result.tokens.len() < max_tokens {
        let round = Instant::now();
        let k = depth_policy.choose().min(max_tokens - result.tokens.len());
        let active_head = if vocab_policy.use_full() {
            &full_head
        } else {
            &draft_head
        };
        let start = Instant::now();
        let mut snapshots = vec![dc.clone()];
        let mut hh = dh.clone();
        let drafted = match seed {
            DraftToken::Cpu(token) => {
                let mut ids = vec![token];
                for _ in 1..k {
                    let token = *ids.last().unwrap();
                    let emb = m
                        .embedding
                        .embedding(&Array::from_slice(&[token], &[1, 1]))?;
                    let pos = dc.kv.offset;
                    let (x, h) = draft.forward(&emb, &hh, &mut dc, pos)?;
                    ids.push(active_head.greedy(&x)?);
                    hh = h;
                    snapshots.push(dc.clone());
                }
                ids
            }
            DraftToken::Gpu(token) => {
                let mut ids = vec![token];
                for _ in 1..k {
                    let emb = m.embedding.embedding(ids.last().unwrap())?;
                    let pos = dc.kv.offset;
                    let (x, h) = draft.forward(&emb, &hh, &mut dc, pos)?;
                    let DraftToken::Gpu(token) = active_head.greedy_token(&x, true)? else {
                        unreachable!("GPU draft token requested");
                    };
                    ids.push(token);
                    hh = h;
                    snapshots.push(dc.clone());
                }
                let ids = ops::concatenate(&ids, 1)?.contiguous()?;
                ids.eval()?;
                ids.as_slice::<u32>().to_vec()
            }
        };
        result.draft_seconds += start.elapsed().as_secs_f64();
        if draft.record_drafts.get() {
            result.draft_tokens.push(drafted.clone());
            result.draft_vocab_sizes.push(active_head.size());
        }
        let mut verify = vec![bonus];
        verify.extend_from_slice(&drafted);
        let original = cache.clone();
        let start = Instant::now();
        let (mut l, mixed, h, sampled) = if crate::greedy_head::enabled() {
            let (mixed, hidden) =
                verification::with_mode(|| m.forward_hidden(&verify, &mut cache))?;
            let sampled = crate::greedy_head::greedy(&m.head, &mixed)?.contiguous()?;
            (None, Some(mixed), hidden, sampled)
        } else {
            let (l, hidden) = verification::with_mode(|| m.forward(&verify, &mut cache))?;
            let sampled = indexing::argmax_axis(&l, -1, false)?.contiguous()?;
            (Some(l), None, hidden, sampled)
        };
        sampled.eval()?;
        let predicted = sampled.as_slice::<u32>();
        let accepted = drafted
            .iter()
            .zip(predicted)
            .take_while(|(a, b)| a == b)
            .count();
        let next = predicted[accepted];
        result.verify_seconds += start.elapsed().as_secs_f64();
        result.acceptance.push(accepted);
        result.draft_lengths.push(k);
        depth_policy.observe(k, accepted);
        let mut ending = false;
        let mut emitted = 0;
        for &token in drafted[..accepted].iter().chain(std::iter::once(&next)) {
            if eos.contains(&token) {
                ending = true;
                break;
            }
            if result.tokens.len() == max_tokens {
                ending = true;
                break;
            }
            result.tokens.push(token);
            emit(token)?;
            emitted += 1;
        }
        let committed = if ending {
            emitted.min(accepted) + 1
        } else {
            accepted + 1
        };
        cache.commit_verified(&original, &verify, committed, &m.config)?;
        if ending || result.tokens.len() == max_tokens {
            result.round_seconds.push(round.elapsed().as_secs_f64());
            break;
        }
        bonus = next;
        let start = Instant::now();
        let refresh = draft.draft_vocab_refresh_rounds.get();
        let adaptive_refresh =
            draft.adaptive_vocab.get() && vocab_policy.observe(accepted, result.acceptance.len());
        if limit > 0
            && limit < m.config.vocab_size as usize
            && (adaptive_refresh
                || (refresh > 0 && result.acceptance.len().is_multiple_of(refresh)))
        {
            let count = 4096.min(m.config.vocab_size);
            if l.is_none() {
                l = Some(
                    m.head
                        .forward_rows(mixed.as_ref().expect("greedy target retains mixed"))?,
                );
            }
            let ids = ops::argpartition_axis(
                l.as_ref()
                    .expect("full ranking logits")
                    .index((0, accepted as i32, ..)),
                -count,
                -1,
            )?
            .index(-count..)
            .contiguous()?;
            ids.eval()?;
            let mut context = prompt.clone();
            if draft.adaptive_vocab.get() {
                context.extend_from_slice(&result.tokens[result.tokens.len().saturating_sub(32)..]);
            }
            draft_head = crate::draft_vocab::DraftVocabulary::new(
                &m.head,
                limit,
                &context,
                eos,
                ids.as_slice::<u32>(),
            )?;
        }
        let kept = accepted.min(k - 1);
        dc = snapshots[kept].clone();
        let mut sync_tokens = drafted[kept..accepted].to_vec();
        sync_tokens.push(bonus);
        let sync_hidden = h.index((.., kept as i32..accepted as i32 + 1, ..));
        let emb = m.embedding.embedding(&Array::from_slice(
            &sync_tokens,
            &[1, sync_tokens.len() as i32],
        ))?;
        let pos = dc.kv.offset;
        let (x, wide) = draft.forward(&emb, &sync_hidden, &mut dc, pos)?;
        dh = wide.index((.., -1.., ..));
        let active_head = if vocab_policy.use_full() {
            &full_head
        } else {
            &draft_head
        };
        seed = active_head.greedy_token(&x.index((.., -1.., ..)), gpu_draft)?;
        result.synchronize_draft_seconds += start.elapsed().as_secs_f64();
        result.round_seconds.push(round.elapsed().as_secs_f64());
    }
    result.decode_seconds = decode.elapsed().as_secs_f64();
    Ok(result)
}
