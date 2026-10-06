use anyhow::{Result, ensure};
use clap::Parser;
use mlx_rs::{
    Array, Dtype,
    ops::{
        self,
        indexing::{self, IndexOp},
    },
};
use rust_mlx::{
    environment::BenchmarkEnvironment,
    hybrid::{HybridCache, HybridModel, LayerCache},
    weights::Weights,
};
use std::{path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    kv_blocks: bool,
    #[arg(long, default_value = "tests/fixtures/hybrid")]
    model: PathBuf,
    #[arg(long, default_value_t = 16)]
    steps: usize,
    #[arg(long, default_value = "results/batch-parity.json")]
    output: PathBuf,
    #[arg(long)]
    prompt_ids: Option<PathBuf>,
}
fn error(a: &Array, b: &Array) -> Result<f32> {
    let a = a.as_dtype(Dtype::Float32)?.contiguous()?;
    let b = b.as_dtype(Dtype::Float32)?.contiguous()?;
    mlx_rs::transforms::eval([&a, &b])?;
    ensure!(a.shape() == b.shape(), "shape mismatch");
    ensure!(
        a.as_slice::<f32>()
            .iter()
            .chain(b.as_slice::<f32>())
            .all(|v| v.is_finite()),
        "non-finite parity input"
    );
    Ok(a.as_slice::<f32>()
        .iter()
        .zip(b.as_slice::<f32>())
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max))
}
fn optional(a: &Option<Array>, b: &Option<Array>) -> Result<f32> {
    match (a, b) {
        (Some(a), Some(b)) => error(a, b),
        (None, None) => Ok(0.),
        _ => anyhow::bail!("cache initialization differs"),
    }
}
fn cache_error(a: &HybridCache, b: &HybridCache) -> Result<f32> {
    ensure!(
        a.offset == b.offset && a.history == b.history,
        "CPU cache mismatch"
    );
    let mut max = 0f32;
    for (a, b) in a.layers.iter().zip(&b.layers) {
        match (a, b) {
            (LayerCache::Linear(a), LayerCache::Linear(b)) => {
                max = max
                    .max(optional(&a.state, &b.state)?)
                    .max(optional(&a.conv, &b.conv)?);
            }
            (LayerCache::Full(a), LayerCache::Full(b)) => {
                ensure!(a.kv.offset == b.kv.offset, "KV offset mismatch");
                max = max
                    .max(optional(&a.kv.keys, &b.kv.keys)?)
                    .max(optional(&a.kv.values, &b.kv.values)?)
                    .max(optional(&a.raw_keys, &b.raw_keys)?)
                    .max(optional(&a.blocks, &b.blocks)?);
            }
            _ => anyhow::bail!("cache kind mismatch"),
        }
    }
    for (a, b) in a.ple.iter().zip(&b.ple) {
        max = max.max(optional(&a.conv, &b.conv)?);
    }
    Ok(max)
}
fn main() -> Result<()> {
    let a = Args::parse();
    rust_mlx::kv_blocks::set_enabled(a.kv_blocks);
    ensure!(a.steps > 0, "parity requires at least one decode step");
    let w = Weights::load(&a.model)?;
    let m = HybridModel::load(&w, &a.model)?;
    mlx_rs::transforms::eval(w.tensors.values())?;
    let environment = BenchmarkEnvironment::capture()?;
    let base_prompt: Vec<u32> = if let Some(path) = &a.prompt_ids {
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        serde_json::from_value(if value.is_array() {
            value
        } else {
            value["prompt"].clone()
        })?
    } else if m.config.vocab_size > 1000 {
        vec![7734, 264, 2716, 32671, 709, 421, 55288, 76938, 4947, 13]
    } else {
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    };
    ensure!(!base_prompt.is_empty(), "empty parity prompt");
    let mut records = Vec::new();
    for batch in [2, 4, 8] {
        let mut reference = Vec::new();
        let mut next = Vec::new();
        for row in 0..batch {
            let mut cache = m.make_cache();
            let mut prompt = base_prompt.clone();
            *prompt.last_mut().unwrap() =
                (*prompt.last().unwrap() + row as u32) % m.config.vocab_size as u32;
            let mut tail = None;
            for chunk in prompt.chunks(128) {
                let (l, _) = m.forward(chunk, &mut cache)?;
                l.eval()?;
                tail = Some(l);
            }
            let l = tail.expect("validated nonempty prompt");
            next.push(indexing::argmax(l.index((0, -1, ..)), false)?.item_exact::<u32>());
            reference.push(cache);
        }
        let mut candidate = reference.clone();
        let mut seconds = [Vec::new(), Vec::new()];
        let mut tokens = Vec::new();
        for step in 0..a.steps {
            let ids = next.clone();
            tokens.push(ids.clone());
            let mut logits = Vec::new();
            let mut hidden = Vec::new();
            let start = Instant::now();
            for (cache, &token) in reference.iter_mut().zip(&ids) {
                let (l, h) = m.forward(&[token], cache)?;
                l.eval()?;
                next[logits.len()] =
                    indexing::argmax(l.index((0, -1, ..)), false)?.item_exact::<u32>();
                logits.push(l);
                hidden.push(h);
            }
            seconds[0].push(start.elapsed().as_secs_f64());
            let start = Instant::now();
            let (l, h) = m.decode_batch(&ids, &mut candidate)?;
            seconds[1].push(start.elapsed().as_secs_f64());
            let le = error(&l, &ops::concatenate(&logits, 0)?)?;
            let he = error(&h, &ops::concatenate(&hidden, 0)?)?;
            let mut ce = 0f32;
            for (r, c) in reference.iter().zip(&candidate) {
                ce = ce.max(cache_error(r, c)?);
            }
            println!("batch={batch} step={step} logits={le} hidden={he} cache={ce}");
            ensure!(le == 0. && he == 0. && ce == 0., "batch numerics differ");
        }
        records.push(serde_json::json!({"batch":batch,"steps":a.steps,"tokens":tokens,"sequential_batch_seconds":seconds[0],"batched_seconds":seconds[1],"logit_error":0,"hidden_error":0,"cache_error":0}));
        std::fs::write(
            &a.output,
            serde_json::to_vec_pretty(
                &serde_json::json!({"environment":environment,"model":a.model,"quantization":w.config["quantization"],"base_prompt_ids":base_prompt,"prefill_chunk":128,"note":"teacher-forced greedy B1 versus equal-offset batched decode, synchronized diagnostic includes error checks between rounds; not a serving-throughput benchmark","records":records}),
            )?,
        )?;
    }
    let mut wrong = [m.make_cache(), m.make_cache()];
    wrong[1].offset = 1;
    ensure!(
        m.decode_batch(&[1, 2], &mut wrong).is_err(),
        "unequal offsets accepted"
    );
    ensure!(
        m.decode_batch(&[m.config.vocab_size as u32, 1], &mut wrong)
            .is_err(),
        "invalid token accepted"
    );
    println!("BATCH_PARITY_PASSED");
    Ok(())
}
