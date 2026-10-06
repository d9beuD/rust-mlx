//! Immutable prompt snapshots, MTP reuse and LRU eviction qualification.
use anyhow::{Result, ensure};
use clap::Parser;
use rust_mlx::{
    environment::BenchmarkEnvironment,
    hybrid::HybridModel,
    mtp::Mtp,
    speculative::{self, Options, PrefixCache},
    weights::Weights,
};
use serde_json::json;
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    prompt_ids: Option<PathBuf>,
    #[arg(long, default_value_t = 32)]
    max_tokens: usize,
    #[arg(long, default_value = "results/prefix-parity.json")]
    output: PathBuf,
}
fn main() -> Result<()> {
    let a = Args::parse();
    mlx_rs::memory::set_memory_limit(100 * 1024usize.pow(3))?;
    mlx_rs::memory::set_cache_limit(512 * 1024usize.pow(2))?;
    let w = Weights::load(&a.model)?;
    let m = HybridModel::load(&w, &a.model)?;
    let draft = Mtp::load(&w, &m.config)?;
    mlx_rs::transforms::eval(w.tensors.values())?;
    let prompt: Vec<u32> = if let Some(path) = a.prompt_ids {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        serde_json::from_value(if v.is_array() { v } else { v["prompt"].clone() })?
    } else {
        vec![7734, 264, 2716, 32671, 709, 421, 55288, 76938, 4947, 13]
    };
    ensure!(!prompt.is_empty(), "empty prompt");
    let options = Options {
        max_tokens: a.max_tokens,
        depth: 3,
        chunk: 128,
        eos: &[],
    };
    let expected = speculative::generate_plain(&m, &prompt, &options, |_| Ok(()))?;
    let mut cache = PrefixCache::new(&m, 2, prompt.len() * 2);
    let mut reports = Vec::new();
    for cycle in 0..3 {
        let started = Instant::now();
        let (prepared, hit) = cache.get_or_prepare(&prompt, 128)?;
        let lookup_seconds = started.elapsed().as_secs_f64();
        ensure!(hit == (cycle > 0), "unexpected cache hit");
        for mtp in [false, true] {
            let generation = if mtp {
                speculative::generate_prepared(&prepared, &draft, &options, |_| Ok(()))?
            } else {
                speculative::generate_plain_prepared(&prepared, &options, |_| Ok(()))?
            };
            ensure!(
                generation.tokens == expected.tokens,
                "cached continuation drift"
            );
            reports.push(json!({"cycle":cycle,"prefix_hit":hit,"lookup_seconds":lookup_seconds,"mtp":mtp,"generation":generation,"exact":true}));
        }
    }
    let mut alternate = prompt.clone();
    *alternate.last_mut().unwrap() = (*alternate.last().unwrap() + 1) % m.config.vocab_size as u32;
    let (p, hit) = cache.get_or_prepare(&alternate, 128)?;
    ensure!(!hit, "distinct prompt hit");
    let fresh = speculative::generate_plain(&m, &alternate, &options, |_| Ok(()))?;
    let reused = speculative::generate_prepared(&p, &draft, &options, |_| Ok(()))?;
    ensure!(fresh.tokens == reused.tokens, "alternate prompt drift");
    // Accessing the first entry makes the alternate least recent.
    ensure!(cache.get_or_prepare(&prompt, 128)?.1, "lost first prompt");
    let mut third = alternate.clone();
    *third.last_mut().unwrap() = (*third.last().unwrap() + 1) % m.config.vocab_size as u32;
    ensure!(!cache.get_or_prepare(&third, 128)?.1, "third prompt hit");
    ensure!(
        cache.get_or_prepare(&prompt, 128)?.1,
        "LRU evicted recent prompt"
    );
    ensure!(
        !cache.get_or_prepare(&alternate, 128)?.1,
        "LRU retained evicted prompt"
    );
    cache.clear();
    ensure!(
        !cache.get_or_prepare(&prompt, 128)?.1,
        "clear retained entry"
    );
    ensure!(
        cache.get_or_prepare(&[], 128).is_err(),
        "accepted empty prompt"
    );
    let mut disabled = PrefixCache::new(&m, 0, 0);
    for _ in 0..2 {
        ensure!(
            !disabled.get_or_prepare(&prompt, 128)?.1,
            "disabled cache hit"
        );
    }
    let mut bounded = PrefixCache::new(&m, 2, prompt.len() - 1);
    for _ in 0..2 {
        ensure!(
            !bounded.get_or_prepare(&prompt, 128)?.1,
            "oversize prompt cached"
        );
    }
    std::fs::write(
        a.output,
        serde_json::to_vec_pretty(
            &json!({"environment":BenchmarkEnvironment::capture()?,"model":a.model,"quantization":w.config["quantization"],"prompt_ids":prompt,"runtime":{"greedy":true,"depth":3,"batch":1,"ignore_eos":true,"prefix_cache":"exact complete prompt; model-scoped LRU"},"records":reports,"alternate_exact":true,"lru_eviction":true,"clear":true,"disabled_cache":true,"token_bound":true}),
        )?,
    )?;
    println!("PREFIX_PARITY_PASSED");
    Ok(())
}
