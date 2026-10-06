//! Cooperative plain decode: merge only equal-offset rows, never share a cache.
use super::{Args, Job};
use anyhow::{Context, Result, ensure};
use mlx_rs::ops::indexing;
use rust_mlx::{
    chat::ChatTemplate,
    hybrid::{HybridCache, HybridModel},
    speculative::{Generation, PrefixCache},
};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokenizers::tokenizer::{
    DecodeStream, DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper,
    PreTokenizerWrapper,
};
use tokio::sync::mpsc;
type Decoder<'a> = DecodeStream<
    'a,
    ModelWrapper,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;
struct Row<'a> {
    job: Job,
    cache: HybridCache,
    next: u32,
    tokens: Vec<u32>,
    prompt_tokens: usize,
    max: usize,
    decoder: Decoder<'a>,
    output: String,
    pending: VecDeque<String>,
    done: bool,
    decode: Instant,
    prefill_seconds: f64,
    prefix_hit: bool,
    created: u64,
    model_id: String,
}
impl<'a> Row<'a> {
    fn new(
        job: Job,
        tokenizer: &'a tokenizers::Tokenizer,
        template: &ChatTemplate,
        prefixes: &mut PrefixCache<'_>,
        args: &Args,
        model_id: &str,
    ) -> Result<Self> {
        ensure!(!job.events.is_closed(), "client disconnected");
        let r = &job.request;
        ensure!(
            !r.mtp.unwrap_or(false),
            "cooperative batching supports mtp:false only"
        );
        let prompt = if job.chat {
            template.render(
                r.messages.as_ref().unwrap(),
                r.enable_thinking,
                &r.reasoning_effort,
            )?
        } else {
            r.prompt.as_ref().unwrap().clone()
        };
        let ids = tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .to_vec();
        let max = r.max_tokens.or(r.max_completion_tokens).unwrap_or(256);
        ensure!(
            !ids.is_empty() && ids.len() + max <= args.max_context,
            "prompt plus completion exceeds max_context ({})",
            args.max_context
        );
        let tick = Instant::now();
        let (prepared, hit) = if r.prefix_cache.unwrap_or(true) {
            prefixes.get_or_prepare(&ids, 128)?
        } else {
            (prefixes.prepare_uncached(&ids, 128)?, false)
        };
        let (cache, l) = prepared.decode_state();
        let next = indexing::argmax(l, false)?.item_exact::<u32>();
        let mut row = Self {
            job,
            cache,
            next,
            tokens: Vec::new(),
            prompt_tokens: ids.len(),
            max,
            decoder: tokenizer.decode_stream(true),
            output: String::new(),
            pending: VecDeque::new(),
            done: false,
            decode: Instant::now(),
            prefill_seconds: tick.elapsed().as_secs_f64(),
            prefix_hit: hit,
            created: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            model_id: model_id.into(),
        };
        if row.job.chat && row.job.request.stream {
            row.pending.push_back(
                row.base(json!({"index":0,"delta":{"role":"assistant"},"finish_reason":null}))
                    .to_string(),
            );
        }
        row.emit(next, tokenizer)?;
        Ok(row)
    }
    fn base(&self, choice: Value) -> Value {
        json!({"id":self.job.id,"object":if self.job.chat{"chat.completion.chunk"}else{"text_completion"},"created":self.created,"model":self.model_id,"choices":[choice]})
    }
    fn emit(&mut self, token: u32, tokenizer: &tokenizers::Tokenizer) -> Result<()> {
        if [248044, 248046].contains(&token) {
            return self.finish("stop", tokenizer);
        }
        self.next = token;
        self.tokens.push(token);
        if let Some(text) = self
            .decoder
            .step(token)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            && self.job.request.stream
        {
            self.output.push_str(&text);
            let choice = if self.job.chat {
                json!({"index":0,"delta":{"content":text},"finish_reason":null})
            } else {
                json!({"index":0,"text":text,"finish_reason":null})
            };
            self.pending.push_back(self.base(choice).to_string());
        }
        if self.tokens.len() == self.max {
            self.finish("length", tokenizer)?;
        }
        Ok(())
    }
    fn finish(&mut self, reason: &str, tokenizer: &tokenizers::Tokenizer) -> Result<()> {
        self.done = true;
        if self.job.request.stream {
            let complete = tokenizer
                .decode(&self.tokens, true)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let suffix = complete
                .strip_prefix(&self.output)
                .context("stream decoder changed emitted prefix")?;
            if !suffix.is_empty() {
                let choice = if self.job.chat {
                    json!({"index":0,"delta":{"content":suffix},"finish_reason":null})
                } else {
                    json!({"index":0,"text":suffix,"finish_reason":null})
                };
                self.pending.push_back(self.base(choice).to_string());
            }
        }
        let generation = Generation {
            tokens: self.tokens.clone(),
            prefill_seconds: self.prefill_seconds,
            decode_seconds: if self.tokens.len() > 1 {
                self.decode.elapsed().as_secs_f64()
            } else {
                0.
            },
            draft_seconds: 0.,
            verify_seconds: 0.,
            synchronize_draft_seconds: 0.,
            acceptance: Vec::new(),
            draft_lengths: Vec::new(),
            draft_tokens: Vec::new(),
            draft_vocab_sizes: Vec::new(),
            round_seconds: Vec::new(),
        };
        let mut metrics = serde_json::to_value(&generation)?;
        metrics["prefix_cache_hit"] = json!(self.prefix_hit);
        metrics["scheduler"] = json!(
            "cooperative plain; equal-offset groups; decode wall includes scheduling and other prefills"
        );
        let mut response = if self.job.request.stream {
            self.base(if self.job.chat {
                json!({"index":0,"delta":{},"finish_reason":reason})
            } else {
                json!({"index":0,"text":"","finish_reason":reason})
            })
        } else {
            let text = tokenizer
                .decode(&self.tokens, true)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut value=self.base(if self.job.chat{json!({"index":0,"message":{"role":"assistant","content":text},"finish_reason":reason})}
                else{json!({"index":0,"text":text,"finish_reason":reason})});
            value["object"] = json!(if self.job.chat {
                "chat.completion"
            } else {
                "text_completion"
            });
            value
        };
        response["usage"] = json!({"prompt_tokens":self.prompt_tokens,"completion_tokens":self.tokens.len(),
            "total_tokens":self.prompt_tokens+self.tokens.len(),"prompt_tokens_details":{"cached_tokens":if self.prefix_hit{self.prompt_tokens}else{0}}});
        response["rust_mlx"] = metrics;
        self.pending.push_back(response.to_string());
        if self.job.request.stream {
            self.pending.push_back("[DONE]".into());
        }
        eprintln!(
            "{} cooperative prompt={} output={} prefix_hit={}",
            self.job.id,
            self.prompt_tokens,
            self.tokens.len(),
            self.prefix_hit
        );
        Ok(())
    }
    fn fail(&mut self, error: impl std::fmt::Display) {
        self.done = true;
        self.pending.push_back(
            json!({"error":{"message":error.to_string(),"type":"inference_error"}}).to_string(),
        );
        if self.job.request.stream {
            self.pending.push_back("[DONE]".into());
        }
    }
    fn flush(&mut self) -> bool {
        while let Some(event) = self.pending.pop_front() {
            match self.job.events.try_send(event) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(event)) => {
                    self.pending.push_front(event);
                    break;
                }
                Err(mpsc::error::TrySendError::Closed(_)) => return false,
            }
        }
        !self.job.events.is_closed() && !(self.done && self.pending.is_empty())
    }
}
pub(super) fn serve(
    args: &Args,
    model: &HybridModel,
    tokenizer: &tokenizers::Tokenizer,
    template: &ChatTemplate,
    prefixes: &mut PrefixCache<'_>,
    model_id: &str,
    mut jobs: mpsc::Receiver<Job>,
) -> Result<()> {
    let mut rows: Vec<Row<'_>> = Vec::new();
    let mut disconnected = false;
    loop {
        rows.retain_mut(Row::flush);
        if rows.is_empty() {
            if disconnected {
                break;
            }
            let Some(job) = jobs.blocking_recv() else {
                break;
            };
            admit(
                job, &mut rows, tokenizer, template, prefixes, args, model_id,
            );
        }
        while rows.len() < args.batch_size && !disconnected {
            match jobs.try_recv() {
                Ok(job) => admit(
                    job, &mut rows, tokenizer, template, prefixes, args, model_id,
                ),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => disconnected = true,
            }
        }
        rows.retain_mut(Row::flush);
        // Smallest position catches up to longer prompts, forming compatible groups.
        let offset = rows
            .iter()
            .filter(|r| !r.done && r.pending.is_empty())
            .map(|r| r.cache.offset)
            .min();
        let Some(offset) = offset else {
            std::thread::sleep(Duration::from_millis(2));
            continue;
        };
        let indices = rows
            .iter()
            .enumerate()
            .filter(|(_, r)| !r.done && r.pending.is_empty() && r.cache.offset == offset)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        let tokens = indices.iter().map(|&i| rows[i].next).collect::<Vec<_>>();
        let mut caches = indices
            .iter()
            .map(|&i| rows[i].cache.clone())
            .collect::<Vec<_>>();
        let result = (|| -> Result<Vec<u32>> {
            let (l, _) = if indices.len() == 1 {
                model.forward(&tokens, &mut caches[0])?
            } else {
                model.decode_batch(&tokens, &mut caches)?
            };
            let next = indexing::argmax_axis(&l, -1, false)?.contiguous()?;
            next.eval()?;
            Ok(next.as_slice::<u32>().to_vec())
        })();
        match result {
            Ok(next) => {
                ensure!(next.len() == indices.len(), "scheduler output row mismatch");
                for ((&i, cache), token) in indices.iter().zip(caches).zip(next) {
                    rows[i].cache = cache;
                    if let Err(e) = rows[i].emit(token, tokenizer) {
                        rows[i].fail(e);
                    }
                }
            }
            Err(e) => {
                for &i in &indices {
                    rows[i].fail(&e);
                }
            }
        }
    }
    Ok(())
}
fn admit<'a>(
    job: Job,
    rows: &mut Vec<Row<'a>>,
    tokenizer: &'a tokenizers::Tokenizer,
    template: &ChatTemplate,
    prefixes: &mut PrefixCache<'_>,
    args: &Args,
    model_id: &str,
) {
    if job.events.is_closed() {
        return;
    }
    let events = job.events.clone();
    let stream = job.request.stream;
    match Row::new(job, tokenizer, template, prefixes, args, model_id) {
        Ok(row) => rows.push(row),
        Err(e) => {
            let _ = events.try_send(
                json!({"error":{"message":format!("{e:#}"),"type":"inference_error"}}).to_string(),
            );
            if stream {
                let _ = events.try_send("[DONE]".into());
            }
        }
    }
}
