//! Greedy MTP with target verification and exact prefix cache commit.
use crate::{hybrid::HybridModel, mtp::Mtp, qsa::QsaCache, verification};
use anyhow::{Result, ensure};
use mlx_rs::{
    Array,
    ops::{
        self,
        indexing::{self, IndexOp},
    },
};
use serde::Serialize;
use std::time::Instant;
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
    mut emit: impl FnMut(u32) -> Result<()>,
) -> Result<Generation> {
    ensure!(
        !prompt.is_empty() && options.max_tokens > 0 && options.chunk > 0,
        "invalid generation request"
    );
    let mut cache = m.make_cache();
    let prefill = Instant::now();
    let mut tail = None;
    for ids in prompt.chunks(options.chunk) {
        let (l, _) = m.forward(ids, &mut cache)?;
        let l = l.index((0, -1, ..));
        l.eval()?;
        tail = Some(l);
    }
    let mut token = greedy(&tail.expect("nonempty prompt"))?;
    let mut result = Generation {
        tokens: Vec::new(),
        prefill_seconds: prefill.elapsed().as_secs_f64(),
        decode_seconds: 0.,
        draft_seconds: 0.,
        verify_seconds: 0.,
        synchronize_draft_seconds: 0.,
        acceptance: Vec::new(),
        draft_lengths: Vec::new(),
        round_seconds: Vec::new(),
    };
    if options.eos.contains(&token) {
        return Ok(result);
    }
    emit(token)?;
    result.tokens.push(token);
    let decode = Instant::now();
    while result.tokens.len() < options.max_tokens {
        let (l, _) = m.forward(&[token], &mut cache)?;
        token = greedy(&l.index((0, -1, ..)))?;
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
    mut emit: impl FnMut(u32) -> Result<()>,
) -> Result<Generation> {
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
    let mut cache = m.make_cache();
    let mut hidden = Vec::new();
    let mut logits = None;
    for ids in prompt.chunks(chunk) {
        let (l, h) = m.forward(ids, &mut cache)?;
        l.eval()?;
        logits = Some(l.index((0, -1, ..)));
        hidden.push(h);
    }
    let mut bonus = greedy(&logits.expect("nonempty prompt"))?;
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
    let h = ops::concatenate(&hidden, 1)?;
    let mut shifted = prompt[1..].to_vec();
    shifted.push(bonus);
    let mut dc = QsaCache::default();
    let emb = m
        .embedding
        .embedding(&Array::from_slice(&shifted, &[1, shifted.len() as i32]))?;
    let start = Instant::now();
    let (mixed, wide) = draft.forward(&emb, &h, &mut dc, 0)?;
    let mut dh = wide.index((.., -1.., ..));
    let mut seed = greedy(&m.head.forward(&mixed.index((.., -1.., ..)))?)?;
    result.synchronize_draft_seconds += start.elapsed().as_secs_f64();
    while result.tokens.len() < max_tokens {
        let round = Instant::now();
        let k = depth.min(max_tokens - result.tokens.len());
        let start = Instant::now();
        let mut drafted = vec![seed];
        let mut snapshots = vec![dc.clone()];
        let mut hh = dh.clone();
        for _ in 1..k {
            let token = *drafted.last().unwrap();
            let emb = m
                .embedding
                .embedding(&Array::from_slice(&[token], &[1, 1]))?;
            let pos = dc.kv.offset;
            let (x, h) = draft.forward(&emb, &hh, &mut dc, pos)?;
            let token = greedy(&m.head.forward(&x)?)?;
            hh = h;
            drafted.push(token);
            snapshots.push(dc.clone());
        }
        result.draft_seconds += start.elapsed().as_secs_f64();
        let mut verify = vec![bonus];
        verify.extend_from_slice(&drafted);
        let original = cache.clone();
        let start = Instant::now();
        let (l, h) = verification::with_mode(|| m.forward(&verify, &mut cache))?;
        let sampled = indexing::argmax_axis(&l, -1, false)?.contiguous()?;
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
        seed = greedy(&m.head.forward(&x.index((.., -1.., ..)))?)?;
        result.synchronize_draft_seconds += start.elapsed().as_secs_f64();
        result.round_seconds.push(round.elapsed().as_secs_f64());
    }
    result.decode_seconds = decode.elapsed().as_secs_f64();
    Ok(result)
}
